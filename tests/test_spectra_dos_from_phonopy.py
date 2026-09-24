"""phonopy.yaml -> per-species partial DOS bridge for mode 0
(irma.spectra.dos_from_phonopy + the dos_source='phonopy' config path).

Uses the committed graphite fixture and a coarse mesh; skipped without phonopy.
"""
import os

import numpy as np
import pytest

pytest.importorskip("phonopy")

_YAML = os.path.join(os.path.dirname(__file__),
                     "mode2_euphonic_n1_validation", "graphite", "phonopy.yaml")

MESH = (6, 6, 6)


def test_partial_dos_normalized_per_atom_and_grouped_by_symbol():
    from irma.spectra.dos_from_phonopy import partial_dos_from_phonopy
    sp = partial_dos_from_phonopy(_YAML, MESH)
    assert len(sp) == 1                         # graphite: one species (C)
    d = sp[0]
    assert d["symbol"] == "C" and d["multiplicity"] == 4
    omega, rho = d["omega_ev"], d["rho"]
    # uniform grid from 0, non-negative DOS, ~1 phonon state per atom (int rho dE_eV)
    assert omega[0] == 0.0 and np.allclose(np.diff(omega), np.diff(omega)[0])
    assert np.all(rho >= 0.0) and rho[0] == 0.0
    assert np.trapezoid(rho, omega) == pytest.approx(1.0, abs=0.05)


def test_run_spectra_dos_source_phonopy_end_to_end():
    from irma.spectra.config import SpectraConfig, run_spectra
    cfg = SpectraConfig.from_dict({
        "material": {"temperature_K": 300.0, "phonopy_yaml": _YAML, "mesh": list(MESH),
                     "scatterers": [{"symbol": "C", "awr": 11.898,
                                     "sigma_bound_b": 5.551}]},
        "physics": {"inelastic_mode": 0, "dos_source": "phonopy",
                    "elastic": False, "max_phonon_order": 100},
        "grid": {"e_max_meV": 250.0, "de_meV": 1.0, "dq_max_invA": 0.1},
        "instrument": {"geometry": "vision", "e_fixed_meV": 3.5,
                       "angles_deg": [45.0, 135.0]},
    })
    res = run_spectra(cfg, progress=lambda *a, **k: None)
    assert res.metadata["sqe_key"] == "mode0_dos"
    assert np.all(np.isfinite(res.I_inelastic)) and res.I_inelastic.max() > 0


def _phonopy_cfg(scatterers):
    from irma.spectra.config import SpectraConfig
    return SpectraConfig.from_dict({
        "material": {"temperature_K": 300.0, "phonopy_yaml": _YAML, "mesh": list(MESH),
                     "scatterers": scatterers},
        "physics": {"inelastic_mode": 0, "dos_source": "phonopy",
                    "elastic": False, "max_phonon_order": 100},
        "grid": {"e_max_meV": 250.0, "de_meV": 1.0, "dq_max_invA": 0.1},
        "instrument": {"geometry": "vision", "e_fixed_meV": 3.5},
    })


def test_unknown_symbol_rejected():
    from irma.spectra.config import run_spectra, SpectraConfigError
    cfg = _phonopy_cfg([{"symbol": "H", "awr": 0.999, "sigma_bound_b": 80.0}])
    with pytest.raises(SpectraConfigError):                  # H absent from the C cell
        run_spectra(cfg, progress=lambda *a, **k: None)


def test_duplicate_symbol_rejected():
    """Two C scatterers would both map to the one phonopy C DOS (double count)."""
    from irma.spectra.config import run_spectra, SpectraConfigError
    cfg = _phonopy_cfg([{"symbol": "C", "awr": 11.898, "sigma_bound_b": 5.551},
                        {"symbol": "C", "awr": 11.898, "sigma_bound_b": 5.551}])
    with pytest.raises(SpectraConfigError):
        run_spectra(cfg, progress=lambda *a, **k: None)
