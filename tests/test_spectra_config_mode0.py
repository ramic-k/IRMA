"""irma.spectra.config mode-0 (DOS) path -- schema + validator + end-to-end run.

Pins the inelastic_mode=0 contract: (1) the new Scatterer DOS fields
(dos_file/dos_unit/multiplicity) round-trip through dump/load; (2) the validator
requires a dos_file + awr + sigma_bound_b per scatterer and accepts a config with
NO phonopy_yaml/mesh; (3) run_spectra wires DOS files straight through
compute_mode0_sqe to a finite SpectrumResult with no phonopy/engine. CI-safe
(writes tiny DOS files to tmp_path; no phonopy).
"""
import numpy as np
import pytest

from irma.spectra.config import (
    SpectraConfig, Scatterer, SpectraConfigError, load, dump, validate,
    run_spectra, _assemble_dos_species,
)
from irma.spectra.cli import parse_scatterer, format_scatterer


# ---- scatterer line round-trip (CLI/GUI shared parser) ----------------------
@pytest.mark.parametrize("line", [
    "C,5.551,11.898,6.646,0.001",                       # full
    "C,5.551,11.898",                                   # bare
    "H,80.27,0.999,-3.74,80.26",                        # negative b_coh
    "X,1.0,2.0,,0.5",                                   # b_coh ABSENT, sigma_inc present
    "Fe,12.2,55.3,9.45,0.4,dos=fe.txt,mult=2,pos=0:0:0;0.5:0.5:0.5",
])
def test_scatterer_line_round_trips(line):
    d = parse_scatterer(line)
    s = Scatterer(**d)
    # the absent-b_coh case must NOT shift sigma_inc into b_coh
    if line.startswith("X"):
        assert s.b_coh_fm is None and s.sigma_inc_b == 0.5
    # re-parse the formatted form -> identical dict
    assert parse_scatterer(format_scatterer(s)) == d


def _write_dos(tmp_path, name, w_max_meV=40.0, n=121):
    """A toy Debye-like DOS (freq_meV, intensity), zero above w_max."""
    rows = "".join(f"{w} {((w / w_max_meV) ** 2 if w <= w_max_meV else 0.0)}\n"
                   for w in range(0, n))
    p = tmp_path / name
    p.write_text("# freq_meV  dos\n" + rows)
    return str(p)


def _mode0_dict(tmp_path, **over):
    dos = _write_dos(tmp_path, "h.dos")
    d = {
        "material": {
            "temperature_K": 300.0,
            "scatterers": [{"symbol": "H", "dos_file": dos, "dos_unit": "meV",
                            "awr": 0.9991673, "sigma_bound_b": 80.27,
                            "multiplicity": 2}],
        },
        "physics": {"inelastic_mode": 0, "elastic": False,
                    "max_phonon_order": 100},
        "grid": {"e_min_meV": 0.0, "e_max_meV": 150.0, "de_meV": 1.0,
                 "dq_max_invA": 0.1},
        "instrument": {"geometry": "vision", "e_fixed_meV": 3.5,
                       "angles_deg": [45.0, 135.0]},
    }
    for sec, kv in over.items():
        d.setdefault(sec, {}).update(kv)
    return d


# ---- schema round-trip ------------------------------------------------------
@pytest.mark.parametrize("suffix", [".yaml", ".json"])
def test_mode0_scatterer_fields_round_trip(tmp_path, suffix):
    cfg = SpectraConfig.from_dict(_mode0_dict(tmp_path))
    p = dump(cfg, tmp_path / f"cfg{suffix}")
    back = load(p)
    assert back == cfg
    s = back.material.scatterers[0]
    assert s.dos_file.endswith("h.dos") and s.dos_unit == "meV" and s.multiplicity == 2
    assert back.material.phonopy_yaml is None      # DOS path needs no yaml


# ---- validator: mode-0 branch ----------------------------------------------
def test_mode0_accepts_config_without_phonopy_yaml(tmp_path):
    cfg = SpectraConfig.from_dict(_mode0_dict(tmp_path))
    assert validate(cfg) is cfg                            # no phonopy_yaml/mesh needed


def test_mode0_missing_dos_file_raises(tmp_path):
    d = _mode0_dict(tmp_path)
    d["material"]["scatterers"][0].pop("dos_file")
    with pytest.raises(SpectraConfigError):
        validate(SpectraConfig.from_dict(d))


@pytest.mark.parametrize("drop", ["awr", "sigma_bound_b"])
def test_mode0_missing_xs_or_mass_raises(tmp_path, drop):
    d = _mode0_dict(tmp_path)
    d["material"]["scatterers"][0].pop(drop)
    with pytest.raises(SpectraConfigError):
        validate(SpectraConfig.from_dict(d))


def test_mode0_bad_multiplicity_raises(tmp_path):
    d = _mode0_dict(tmp_path)
    d["material"]["scatterers"][0]["multiplicity"] = 0
    with pytest.raises(SpectraConfigError):
        validate(SpectraConfig.from_dict(d))


def test_mode0_no_scatterers_raises(tmp_path):
    d = _mode0_dict(tmp_path)
    d["material"]["scatterers"] = []
    with pytest.raises(SpectraConfigError):
        validate(SpectraConfig.from_dict(d))


# ---- _assemble_dos_species --------------------------------------------------
def test_assemble_dos_species_reads_files(tmp_path):
    cfg = SpectraConfig.from_dict(_mode0_dict(tmp_path))
    sp = _assemble_dos_species(cfg.material)
    assert len(sp) == 1
    s0 = sp[0]
    assert s0["symbol"] == "H" and s0["multiplicity"] == 2
    assert s0["awr"] == pytest.approx(0.9991673) and s0["sigma_bound_b"] == pytest.approx(80.27)
    # uniform omega grid in eV, non-negative DOS with a positive band
    assert s0["omega_ev"][0] == 0.0 and np.all(np.diff(s0["omega_ev"]) > 0)
    assert np.all(s0["rho"] >= 0.0) and s0["rho"].max() > 0


# ---- end-to-end run_spectra (no engine/phonopy) -----------------------------
def test_mode0_run_spectra_end_to_end(tmp_path):
    cfg = SpectraConfig.from_dict(_mode0_dict(tmp_path))
    res = run_spectra(cfg, progress=lambda *a, **k: None)
    assert np.all(np.isfinite(res.I_inelastic)) and res.I_inelastic.max() > 0
    assert res.metadata["sqe_key"] == "mode0_dos"
    assert res.metadata["inelastic_mode"] == 0
    # mode-0 + physics.elastic=False -> no elastic line
    assert res.metadata["elastic"] is False


def test_mode0_run_spectra_multispecies_is_atom_weighted_average(tmp_path):
    """PER-ATOM: two species in one config == the atom-weighted average of each
    run alone, (mult_H I_H + mult_C I_C)/N -- not a plain sum (old per-cell)."""
    dosH = _write_dos(tmp_path, "h2.dos", w_max_meV=40)
    dosC = _write_dos(tmp_path, "c2.dos", w_max_meV=30)
    H = {"symbol": "H", "dos_file": dosH, "awr": 0.999, "sigma_bound_b": 80.0, "multiplicity": 2}
    C = {"symbol": "C", "dos_file": dosC, "awr": 11.9, "sigma_bound_b": 5.55, "multiplicity": 1}
    base = _mode0_dict(tmp_path)
    base["material"]["scatterers"] = [H, C]
    both = run_spectra(SpectraConfig.from_dict(base), progress=lambda *a, **k: None)

    only_h = _mode0_dict(tmp_path); only_h["material"]["scatterers"] = [H]
    only_c = _mode0_dict(tmp_path); only_c["material"]["scatterers"] = [C]
    rh = run_spectra(SpectraConfig.from_dict(only_h), progress=lambda *a, **k: None)
    rc = run_spectra(SpectraConfig.from_dict(only_c), progress=lambda *a, **k: None)
    assert np.allclose(both.I_inelastic,
                       (2 * rh.I_inelastic + 1 * rc.I_inelastic) / 3.0,
                       rtol=1e-9, atol=1e-30)


def test_mode0_elastic_without_crystal_builds_incoherent_only(tmp_path):
    """physics.elastic=True (kind 'both') but no material.lattice -> the
    incoherent line is still built lattice-free, with a NOTE that the coherent
    Bragg peaks need the lattice (QA4 F31)."""
    d = _mode0_dict(tmp_path, physics={"elastic": True})
    d["material"]["scatterers"][0]["sigma_inc_b"] = 80.26
    msgs = []
    res = run_spectra(SpectraConfig.from_dict(d),
                      progress=lambda *a, **k: msgs.append(" ".join(map(str, a))))
    assert res.metadata["elastic"] is True
    assert any("material.lattice" in m for m in msgs)        # the NOTE
    assert res.I_elastic.max() > 0.0                         # H line present


def test_mode0_incoherent_elastic_needs_explicit_sigma_inc(tmp_path):
    """QA4 F11: a missing sigma_inc_b must be a loud config error when the
    incoherent elastic line is requested, not a silent zero channel."""
    from irma.spectra.config import validate, SpectraConfigError
    d = _mode0_dict(tmp_path, physics={"elastic": True})     # kind 'both'
    with pytest.raises(SpectraConfigError, match="sigma_inc_b"):
        validate(SpectraConfig.from_dict(d))


# ---- mode-0 coherent elastic (lattice + positions) --------------------------
def _mode0_crystal_dict(tmp_path):
    d = _mode0_dict(tmp_path, physics={"elastic": True, "elastic_kind": "both"})
    d["material"]["lattice"] = [2.866, 2.866, 2.866, 90.0, 90.0, 90.0]
    s = d["material"]["scatterers"][0]
    s["b_coh_fm"] = 9.45
    s["sigma_inc_b"] = 0.4
    s["multiplicity"] = 2
    s["positions"] = [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]
    return d


@pytest.mark.parametrize("suffix", [".yaml", ".json"])
def test_mode0_lattice_positions_round_trip(tmp_path, suffix):
    cfg = SpectraConfig.from_dict(_mode0_crystal_dict(tmp_path))
    back = load(dump(cfg, tmp_path / f"cfg{suffix}"))
    assert back == cfg
    assert back.material.lattice == [2.866, 2.866, 2.866, 90.0, 90.0, 90.0]
    assert back.material.scatterers[0].positions == [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]


def test_mode0_elastic_end_to_end(tmp_path):
    cfg = SpectraConfig.from_dict(_mode0_crystal_dict(tmp_path))
    res = run_spectra(cfg, progress=lambda *a, **k: None)
    assert res.metadata["elastic"] is True
    assert res.metadata["n_bragg_edges"] > 0
    assert res.metadata["elastic_kind"] == "both"
    assert np.all(np.isfinite(res.I_elastic)) and res.I_elastic.max() > 0


# ---- dos_source validation (no phonopy load needed) -------------------------
def test_dos_source_phonopy_without_yaml_raises(tmp_path):
    d = _mode0_dict(tmp_path, physics={"dos_source": "phonopy"})
    # phonopy source ignores dos_file but needs phonopy_yaml + mesh
    d["material"].pop("phonopy_yaml", None)
    with pytest.raises(SpectraConfigError):
        validate(SpectraConfig.from_dict(d))


def test_dos_source_phonopy_allows_missing_dos_file(tmp_path):
    d = _mode0_dict(tmp_path, physics={"dos_source": "phonopy"})
    d["material"]["phonopy_yaml"] = "graphite.yaml"
    d["material"]["mesh"] = [8, 8, 8]
    d["material"]["scatterers"][0].pop("dos_file")        # not needed for phonopy source
    assert validate(SpectraConfig.from_dict(d)) is not None


def test_bad_dos_source_raises(tmp_path):
    d = _mode0_dict(tmp_path, physics={"dos_source": "hdf5"})
    with pytest.raises(SpectraConfigError):
        validate(SpectraConfig.from_dict(d))


def test_dos_source_round_trips(tmp_path):
    d = _mode0_dict(tmp_path, physics={"dos_source": "phonopy"})
    d["material"]["phonopy_yaml"] = "graphite.yaml"
    cfg = SpectraConfig.from_dict(d)
    assert load(dump(cfg, tmp_path / "c.yaml")) == cfg
    assert cfg.physics.dos_source == "phonopy"


@pytest.mark.parametrize("mut", [
    lambda d: d["material"].__setitem__("lattice", [2.8, 2.8, 2.8, 90.0, 90.0]),     # 5 entries
    lambda d: d["material"].__setitem__("lattice", [-2.8, 2.8, 2.8, 90, 90, 90]),    # a<=0
    lambda d: d["material"].__setitem__("lattice", [2.8, 2.8, 2.8, 0.0, 90, 90]),    # angle 0
    lambda d: d["material"]["scatterers"][0].pop("positions"),                       # no positions
    lambda d: d["material"]["scatterers"][0].pop("b_coh_fm"),                        # no b_coh
    lambda d: d["material"]["scatterers"][0].__setitem__("multiplicity", 3),         # mult != len(pos)
])
def test_mode0_bad_crystal_rejected(tmp_path, mut):
    d = _mode0_crystal_dict(tmp_path)
    mut(d)
    with pytest.raises(SpectraConfigError):
        validate(SpectraConfig.from_dict(d))
