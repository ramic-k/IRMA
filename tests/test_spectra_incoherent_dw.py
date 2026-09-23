"""Directional incoherent-elastic option in the spectra elastic line.

Engine-free: the elastic_state is hand-built (same fixture shape as
test_spectra_elastic_from_engine.py). Pins the contract of
``from_engine_elastic_state(..., incoherent_elastic_mode='directional')``:
cubic tensors reproduce the isotropic path, anisotropic tensors exceed it at
high E/Q (Jensen), the coherent Bragg peaks are untouched, and the config validator
guards the new knob.
"""
import numpy as np
import pytest

from irma.spectra.config import SpectraConfig, SpectraConfigError, validate
from irma.spectra.elastic import from_engine_elastic_state


GRAPHITE_U = np.diag([0.00226, 0.00226, 0.0149])


def _state(U, a=3.567, T=296.0):
    return {
        "thermal_displacement_matrices_ang2": np.array([U], float),
        "primitive_lattice_ang": np.diag([a, a, a]).astype(float),
        "primitive_scaled_positions": np.zeros((1, 3)),
        "primitive_symbols": ["C"],
        "primitive_masses_amu": np.array([12.0]),
        "temperature_k": T,
    }


def _model(U, mode, kind="incoherent"):
    return from_engine_elastic_state(
        _state(U), b_coh_fm=6.646, sigma_inc_b=80.0, awr=11.898,
        elastic_kind=kind, emax_eV=0.3, incoherent_elastic_mode=mode)


def test_cubic_directional_matches_isotropic():
    U = 0.004 * np.eye(3)
    iso = _model(U, "isotropic")
    dirm = _model(U, "directional")
    E = np.array([0.5, 5.0, 50.0, 500.0, 5000.0])   # meV
    Q = np.array([0.5, 2.0, 10.0, 30.0])            # 1/A
    assert np.allclose(dirm.sigma_incoherent(E), iso.sigma_incoherent(E),
                       rtol=1e-6)
    assert np.allclose(dirm.incoherent_dsigma_dOmega(Q),
                       iso.incoherent_dsigma_dOmega(Q), rtol=1e-12)


def test_anisotropic_directional_exceeds_isotropic():
    iso = _model(GRAPHITE_U, "isotropic")
    dirm = _model(GRAPHITE_U, "directional")
    E_hi = np.array([2000.0, 5000.0])               # meV
    assert np.all(dirm.sigma_incoherent(E_hi) > iso.sigma_incoherent(E_hi))
    Q_hi = np.array([15.0, 30.0])
    assert np.all(dirm.incoherent_dsigma_dOmega(Q_hi)
                  > iso.incoherent_dsigma_dOmega(Q_hi))
    # and they agree where the DW factor is still ~1
    E_lo = np.array([0.1, 1.0])
    assert np.allclose(dirm.sigma_incoherent(E_lo), iso.sigma_incoherent(E_lo),
                       rtol=1e-3)


def test_coherent_peaks_unaffected_by_mode():
    iso = _model(GRAPHITE_U, "isotropic", kind="both")
    dirm = _model(GRAPHITE_U, "directional", kind="both")
    assert np.array_equal(iso.Q_bragg, dirm.Q_bragg)
    assert np.array_equal(iso.f_bragg, dirm.f_bragg)


def _cfg_dict(**physics_over):
    physics = {"inelastic_mode": 2, "max_phonon_order": "auto"}
    physics.update(physics_over)
    return {
        "material": {
            "phonopy_yaml": "graphite.yaml", "mesh": [40, 40, 40],
            "temperature_K": 296.0,
            "scatterers": [{"symbol": "C", "sigma_bound_b": 5.551,
                            "awr": 11.898, "b_coh_fm": 6.646,
                            "sigma_inc_b": 0.001}],
        },
        "physics": physics,
        "grid": {"e_min_meV": 0.0, "e_max_meV": 250.0, "de_meV": 0.5,
                 "dq_max_invA": 0.05},
        "instrument": {"geometry": "vision", "e_fixed_meV": 3.5,
                       "angles_deg": [45.0, 135.0],
                       "sigma_coeffs": [0.31, 0.005, 8.07e-7]},
    }


def test_config_accepts_and_defaults_incoherent_elastic_mode():
    cfg = validate(SpectraConfig.from_dict(_cfg_dict()))
    assert cfg.physics.incoherent_elastic_mode == "isotropic"
    cfg = validate(SpectraConfig.from_dict(
        _cfg_dict(incoherent_elastic_mode="directional")))
    assert cfg.physics.incoherent_elastic_mode == "directional"


def test_config_rejects_bad_incoherent_elastic_mode():
    with pytest.raises(SpectraConfigError, match="incoherent_elastic_mode"):
        validate(SpectraConfig.from_dict(
            _cfg_dict(incoherent_elastic_mode="banana")))


def test_config_rejects_directional_with_mode0():
    d = _cfg_dict(incoherent_elastic_mode="directional", inelastic_mode=0,
                  dos_source="file")
    d["material"]["phonopy_yaml"] = None
    d["material"]["scatterers"][0]["dos_file"] = "c.dat"
    with pytest.raises(SpectraConfigError, match="directional"):
        validate(SpectraConfig.from_dict(d))
