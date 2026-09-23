"""Spectra 1-D/2-D parity: chopper+lorentzian refusal, the direct poly default
width, and the map coverage following resolution_model."""
import numpy as np
import pytest

pytest.importorskip("scipy")  # the spectra forward model needs scipy

from irma.spectra.forward import compute_sqe_map, compute_spectrum  # noqa: E402

_CHOP = {"instrument": "ARCS", "package": "ARCS-100-1.5-AST", "frequency": 600.0}


@pytest.mark.parametrize("compute", [compute_sqe_map, compute_spectrum])
def test_chopper_lorentzian_rejected_at_both_entry_points(compute):
    with pytest.raises(ValueError, match="lorentzian"):
        compute(
            geometry="direct", phonopy_yaml=None, temperature_k=296.0, mesh=None,
            sab_mass_ratio=1.0, sab_sigma_barn=1.0, e_fixed_meV=100.0,
            resolution_model="chopper", resolution_shape="lorentzian",
            chopper_spec=_CHOP, progress=lambda *a, **k: None)


# mode-0 DOS fixture (CI-safe, no phonopy) -- mirrors tests/test_spectra_map_elastic.py
def _carbon():
    omega = np.arange(400) * 0.0005
    w = omega * 1000.0
    rho = np.where(w <= 40.0, (w / 40.0) ** 2, 0.0)
    rho += 0.5 * np.exp(-0.5 * ((w - 90.0) / 5.0) ** 2)
    return {"symbol": "C", "omega_ev": omega, "rho": rho, "awr": 11.898,
            "sigma_bound_b": 5.551, "sigma_inc_b": 0.001, "multiplicity": 1}


def test_direct_map_poly_default_is_geometry_aware(monkeypatch):
    """A direct map with resolution_model='poly' and no sigma_coeffs must use the
    direct 0.02*Ei width, not the VISION indirect polynomial."""
    import irma.spectra.sqe as sqe
    captured = {}
    orig = sqe.resolution_kernel

    def spy(E_out, width, shape="gaussian"):
        captured["width"] = width
        return orig(E_out, width, shape=shape)

    monkeypatch.setattr(sqe, "resolution_kernel", spy)
    compute_sqe_map(
        dos_species=[_carbon()], geometry="direct", e_fixed_meV=250.0,
        phonopy_yaml=None, temperature_k=296.0, mesh=None,
        sab_mass_ratio=11.898, sab_sigma_barn=5.551,
        multiphonon_max_order=40, auto_multiphonon_order=False,
        q_min=1.0, q_max=4.0, dQ_map=0.5, e_min=-20.0, e_max=40.0, dE=1.5,
        elastic=False, broaden=True, resolution_model="poly", sigma_coeffs=None,
        progress=lambda *a, **k: None)
    assert captured["width"] == (0.02 * 250.0, 0.0, 0.0), (
        f"direct poly default should be 0.02*Ei, got {captured['width']}")


@pytest.mark.parametrize("resolution_model", ["chopper", "poly"])
def test_run_map_coverage_follows_resolution_model(monkeypatch, resolution_model):
    """With no map_coverage_deg, a chopper config masks the map to the
    instrument's tabulated 2-theta span (like the GUI); a poly config with a
    stray chopper_spec uses angles_deg instead."""
    pytest.importorskip("yaml")
    from irma.spectra.config import SpectraConfig, run_map
    from irma.spectra.chopper_resolution import default_coverage
    import irma.spectra.forward as fwd

    captured = {}

    def fake_map(**kw):
        captured.update(kw)
        raise RuntimeError("stop-before-engine")

    monkeypatch.setattr(fwd, "compute_sqe_map", fake_map)
    cfg = SpectraConfig.from_dict({
        "material": {"phonopy_yaml": "x.yaml", "temperature_K": 296.0,
                     "scatterers": [{"symbol": "C", "sigma_bound_b": 5.55,
                                     "awr": 11.898, "b_coh_fm": 6.646}]},
        "instrument": {"geometry": "direct", "e_fixed_meV": 300.0,
                       "angles_deg": [30.0, 60.0],
                       "resolution_model": resolution_model,
                       "chopper_spec": {"instrument": "ARCS",
                                        "package": "ARCS-700-1.5-AST",
                                        "frequency": 600.0}},
        # deliberately NO instrument.map_coverage_deg
    })
    with pytest.raises(RuntimeError, match="stop-before-engine"):
        run_map(cfg, progress=lambda *a, **k: None)
    want = default_coverage("ARCS") if resolution_model == "chopper" else (30.0, 60.0)
    assert captured["angle_range_deg"] == want
