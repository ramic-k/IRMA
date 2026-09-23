"""Direct-geometry chopper resolution (``irma.spectra.chopper_resolution``).

``test_pychop_reference_json`` replays a captured Mantid PyChop dump
(``tests/chopper_reference/pychop_reference.json``, regenerated with
``dump_pychop_reference.py``) for all eight instruments; the regression pins
check each instrument more tightly. The config/instrument wiring of
``resolution_model='chopper'`` is at the bottom.
"""
import json
import os

import numpy as np
import pytest

from irma.spectra import chopper_resolution as cr

_ALL_INSTRUMENTS = ["ARCS", "CNCS", "HYSPEC", "LET", "MAPS", "MARI",
                    "MERLIN", "SEQUOIA"]


# ---- instrument database / lookup -------------------------------------------
def test_available_instruments_and_packages():
    assert cr.available_instruments() == _ALL_INSTRUMENTS    # sorted, all 8
    arcs = cr.available_packages("ARCS")
    assert "ARCS-700-1.5-AST" in arcs and len(arcs) >= 5
    seq = cr.available_packages("SEQUOIA")
    assert "High-Resolution" in seq and "High-Flux" in seq
    # disk machines expose a single resolution mode each
    assert cr.available_packages("CNCS") == ["Standard"]
    assert cr.available_packages("LET") == ["High-Resolution"]


def test_every_instrument_has_a_default_frequency_in_range():
    for inst in cr.available_instruments():
        f = cr.default_frequency(inst)
        assert isinstance(f, float) and 0.0 < f <= cr._lookup(inst)["max_frequency"]


def test_every_instrument_has_a_plausible_detector_coverage():
    """default_coverage returns the PyChop tthlims (min,max) 2theta band for the
    2-D map mask: a valid, ordered angular range strictly inside (0, 180) deg."""
    for inst in cr.available_instruments():
        lo, hi = cr.default_coverage(inst)
        assert 0.0 < lo < hi < 180.0, (inst, lo, hi)
    with pytest.raises(ValueError):
        cr.default_coverage("NOT-AN-INSTRUMENT")


def test_lookup_is_case_insensitive():
    a = cr.instrument_geometry("arcs", "ARCS-700-1.5-AST")
    b = cr.instrument_geometry("ARCS", "ARCS-700-1.5-AST")
    assert a["x0"] == b["x0"] == 11.61
    assert a["package"] == "ARCS-700-1.5-AST"


def test_lookup_rejects_unknown_instrument_and_package():
    with pytest.raises(ValueError):
        cr.instrument_geometry("NOT-AN-INSTRUMENT", "x")
    with pytest.raises(ValueError):
        cr.instrument_geometry("ARCS", "not-a-package")
    # a known disk instrument with a bogus mode still rejects
    with pytest.raises(ValueError):
        cr.instrument_geometry("CNCS", "not-a-mode")


# ---- direct_resolution_fwhm --------------------------------------------------
def _fwhm(Etrans, Ei, freq, instrument, package):
    g = cr.instrument_geometry(instrument, package)
    return np.atleast_1d(
        cr.direct_resolution_fwhm(np.atleast_1d(Etrans), Ei=Ei, frequency=freq, geom=g))


def test_forbidden_energy_transfer_is_nan():
    """E >= Ei (Ef <= 0) is kinematically forbidden -> NaN, not a crash."""
    g = cr.instrument_geometry("ARCS", "ARCS-700-1.5-AST")
    fw = cr.direct_resolution_fwhm(np.array([300.0, 350.0]), Ei=300.0,
                                   frequency=600.0, geom=g)
    assert np.all(~np.isfinite(fw))


# ---- regression pins (these ARE the PyChop-validated numbers) ---------------
@pytest.mark.parametrize("instrument,package,Ei,freq,Etrans,expect", [
    ("ARCS", "ARCS-700-1.5-AST", 300.0, 600.0, 0.0, 12.838600),
    ("ARCS", "ARCS-700-1.5-AST", 300.0, 600.0, 150.0, 6.681232),
    ("ARCS", "ARCS-700-1.5-AST", 100.0, 300.0, 0.0, 5.635810),
    ("SEQUOIA", "High-Resolution", 120.0, 600.0, 0.0, 3.159493),
    # Fermi: MAPS / MARI / MERLIN / HYSPEC
    ("MAPS", "A", 400.0, 400.0, 0.0, 11.421691),
    ("MAPS", "A", 400.0, 400.0, 200.0, 7.498260),
    ("MARI", "A", 180.0, 400.0, 0.0, 4.120581),
    ("MERLIN", "S", 80.0, 400.0, 0.0, 4.489467),
    ("HYSPEC", "OnlyOne", 35.0, 180.0, 0.0, 2.534201),
    # disk: CNCS / LET (incl. a high-energy-transfer point each)
    ("CNCS", "Standard", 25.0, 300.0, 0.0, 2.129937),
    ("CNCS", "Standard", 25.0, 300.0, 18.0, 0.539803),
    ("LET", "High-Resolution", 8.0, 240.0, 0.0, 0.233083),
    ("LET", "High-Resolution", 8.0, 240.0, 6.0, 0.085734),
])
def test_resolution_regression_values(instrument, package, Ei, freq, Etrans, expect):
    got = _fwhm(Etrans, Ei, freq, instrument, package)[0]
    assert got == pytest.approx(expect, rel=1e-5)


# ---- chopper_sigma_of_E: the convolution-facing wrapper ----------------------
def test_chopper_sigma_is_fwhm_over_2355():
    g = cr.instrument_geometry("ARCS", "ARCS-700-1.5-AST")
    E = np.linspace(0.0, 200.0, 25)
    sig = cr.chopper_sigma_of_E(E, Ei=300.0, instrument="ARCS",
                                package="ARCS-700-1.5-AST", frequency=600.0)
    fw = cr.direct_resolution_fwhm(E, Ei=300.0, frequency=600.0, geom=g)
    assert np.allclose(sig, fw / cr.SIGMA2FWHM, rtol=1e-9)
    assert np.all(sig > 0.0)


def test_chopper_sigma_gain_side_falls_back_to_elastic():
    """E<0 (energy gain) clips to the elastic width, never NaN."""
    sig = cr.chopper_sigma_of_E(np.array([-50.0, -10.0, 0.0]), Ei=300.0,
                                instrument="ARCS", package="ARCS-700-1.5-AST",
                                frequency=600.0)
    assert np.all(np.isfinite(sig)) and np.all(sig > 0.0)
    assert sig[0] == pytest.approx(sig[2], rel=1e-9)   # gain clamps to elastic


def test_chopper_raises_when_not_transmitting():
    """A curved slot that closes for (Ei, freq) yields NaN everywhere -> raise."""
    with pytest.raises(ValueError, match="no\n?\\s*transmission|transmission"):
        cr.chopper_sigma_of_E(np.array([0.0, 0.3]), Ei=1.0, instrument="ARCS",
                              package="ARCS-700-0.5-AST", frequency=600.0)


# ---- frequency coercion / guards --------------------------------------------
def test_direct_resolution_accepts_disk_frequency_list():
    """A caller may hand the raw PyChop [resolution, frame] frequency list
    straight through; only the resolution-disk (first) element drives the burst,
    so the list and its first element give identical resolution."""
    g = cr.instrument_geometry("CNCS", "Standard")
    Et = np.linspace(0.0, 0.9 * 12.0, 8)
    as_list = cr.direct_resolution_fwhm(Et, Ei=12.0, frequency=[300, 60], geom=g)
    as_scalar = cr.direct_resolution_fwhm(Et, Ei=12.0, frequency=300.0, geom=g)
    assert np.allclose(as_list, as_scalar, rtol=0, atol=0)


@pytest.mark.parametrize("bad", [0.0, [0, 60]])
def test_chopper_frequency_must_be_positive(bad):
    g = cr.instrument_geometry("ARCS", "ARCS-700-1.5-AST")
    with pytest.raises(ValueError):
        cr.direct_resolution_fwhm(np.array([0.0]), Ei=300.0, frequency=bad, geom=g)


def test_moderator_table_extrapolation_warns():
    """A table moderator queried below its shortest tabulated wavelength
    (very high Ei) flat-clamps -- it must warn, not silently mislead."""
    mod = cr.INSTRUMENT_DB["HYSPEC"]["moderator"]      # table starts at 1.189 A
    lam_min = mod["lam"][0]
    Ei_too_high = cr.E2L / (0.5 * lam_min) ** 2         # lambda well below table
    with pytest.warns(UserWarning, match="outside this range"):
        cr._moderator_fwhm_us(Ei_too_high, mod)
    # in-range query does NOT warn
    import warnings as _w
    with _w.catch_warnings():
        _w.simplefilter("error")
        cr._moderator_fwhm_us(20.0, mod)               # ~2 A, inside the table


# ---- PyChop reference --------------------------------------------------------
_REFERENCE_JSON = os.path.join(os.path.dirname(__file__),
                               "chopper_reference", "pychop_reference.json")


def test_pychop_reference_json():
    """Replay a captured Mantid PyChop dump (no PyChop dependency) across all
    eight instruments. This is the CI gate for the reimplementation: every
    record must agree with the reference to <1.2% over the full energy-transfer
    range (worst observed 0.21%, on CNCS at high transfer)."""
    records = json.load(open(_REFERENCE_JSON))
    seen = set()
    worst = 0.0
    for r in records:
        inst, pkg, freq = r["instrument"], r["package"], r["frequency"]
        Ei = r["Ei"]
        Et = np.asarray(r["Etrans"], float)
        ref = np.asarray(r["dE"], float)
        pkg = pkg if pkg else cr.available_packages(inst)[0]
        g = cr.instrument_geometry(inst, pkg)
        got = cr.direct_resolution_fwhm(Et, Ei=Ei, frequency=freq, geom=g)
        m = np.isfinite(ref) & (ref > 0)
        rel = np.abs(got[m] - ref[m]) / ref[m]
        worst = max(worst, float(np.nanmax(rel)))
        assert np.all(rel < 0.012), (inst, pkg, Ei, float(np.nanmax(rel)))
        seen.add(inst)
    assert seen == set(_ALL_INSTRUMENTS), seen   # every instrument exercised
    assert worst < 0.012


# ---- config + instrument wiring of resolution_model='chopper' ---------------
def _direct_chopper_cfg(**chopper):
    from irma.spectra.config import SpectraConfig
    spec = {"instrument": "ARCS", "package": "ARCS-700-1.5-AST", "frequency": 600.0}
    spec.update(chopper)
    return SpectraConfig.from_dict({
        "material": {"phonopy_yaml": "x.yaml",
                     "scatterers": [{"symbol": "C", "sigma_bound_b": 5.55,
                                     "awr": 11.898, "b_coh_fm": 6.646}]},
        "instrument": {"geometry": "direct", "e_fixed_meV": 300.0,
                       "angles_deg": [30.0, 60.0],
                       "resolution_model": "chopper", "chopper_spec": spec},
    })


def test_config_accepts_chopper_spec_and_round_trips():
    from irma.spectra.config import validate
    cfg = _direct_chopper_cfg()
    validate(cfg)
    assert cfg.instrument.resolution_model == "chopper"
    cs = cfg.to_dict()["instrument"]["chopper_spec"]
    assert cs["instrument"] == "ARCS" and cs["frequency"] == 600.0


def test_config_rejects_chopper_on_indirect_geometry():
    from irma.spectra.config import SpectraConfig, validate, SpectraConfigError
    cfg = SpectraConfig.from_dict({
        "material": {"phonopy_yaml": "x.yaml",
                     "scatterers": [{"symbol": "C", "sigma_bound_b": 5.55,
                                     "awr": 11.898, "b_coh_fm": 6.646}]},
        "instrument": {"geometry": "indirect", "e_fixed_meV": 3.5,
                       "angles_deg": [45.0], "resolution_model": "chopper",
                       "chopper_spec": {"instrument": "ARCS",
                                        "package": "ARCS-700-1.5-AST",
                                        "frequency": 600.0}},
    })
    with pytest.raises(SpectraConfigError, match="direct"):
        validate(cfg)


def test_config_rejects_bad_chopper_package_and_frequency():
    from irma.spectra.config import validate, SpectraConfigError
    with pytest.raises(SpectraConfigError):
        validate(_direct_chopper_cfg(package="no-such-package"))
    with pytest.raises(SpectraConfigError):
        validate(_direct_chopper_cfg(frequency=0.0))


def test_instrument_width_source_is_chopper_callable():
    from irma.spectra.instruments import direct
    ins = direct(300.0, [30.0, 60.0], resolution_model="chopper",
                 chopper_spec={"instrument": "ARCS",
                               "package": "ARCS-700-1.5-AST", "frequency": 600.0})
    w = ins.width_source()
    assert callable(w)
    sig = w(np.array([0.0, 100.0]))
    assert np.all(np.isfinite(sig)) and np.all(sig > 0.0)
