"""Regression coverage for the optional minimum phonon-energy cutoff."""

import numpy as np
import pytest

from irma.core.phonopy_io import (
    PhonopyMeshData,
    compute_atom_dos,
    mode_floor_mask,
    compute_thermal_displacement_matrices,
)


def test_positive_cutoff_is_strict():
    qpoints = np.array([[0.25, 0.0, 0.0]])
    energies = np.array([-1.0, 0.0, 0.5, 0.5000001])
    assert mode_floor_mask(energies, qpoints, 4, 0.5).tolist() == [
        False, False, False, True]


def _synthetic_mesh(cutoff):
    qpoints = np.array([[0.25, 0.0, 0.0], [0.25, 0.25, 0.0]])
    frequencies_ev = np.array([
        [0.0001, 0.0005, 0.0010],
        [0.0001, 0.0005, 0.0010],
    ])
    eig = np.zeros((2, 3, 1, 3), dtype=complex)
    eig[:, 0, 0, 0] = 1.0
    eig[:, 1, 0, 1] = 1.0
    eig[:, 2, 0, 2] = 1.0
    return PhonopyMeshData(
        qpoints=qpoints, frequencies_ev=frequencies_ev,
        eigenvectors=eig, weights=np.ones(2), masses_amu=np.array([12.0]),
        atom_symbols=["C"], atom_positions=np.zeros((1, 3)),
        min_phonon_energy_mev=cutoff,
    )


def test_dos_and_debye_waller_use_the_same_cutoff(capsys):
    mesh = _synthetic_mesh(0.5)
    dos, _ = compute_atom_dos(mesh, 0.002, 101)
    log = capsys.readouterr().out
    assert "2 valid modes" in log  # only the two 1-meV modes survive
    assert np.flatnonzero(dos[0]).tolist() == [50]    # the 1 meV bin

    u = compute_thermal_displacement_matrices(mesh, 296.0)
    assert u[0, 0, 0] == 0.0
    assert u[0, 1, 1] == 0.0
    assert u[0, 2, 2] > 0.0


def test_coherent_mask_keeps_every_positive_mode_without_a_cutoff_and_shares_the_rule_with_one():
    from irma.core.phonopy_io import coherent_mode_mask, mode_floor_mask
    q_folded = np.array([[0.0, 0.0, 0.0], [0.25, 0.0, 0.0]])
    energies = np.array([[-0.5, 0.05, 0.3, 7.0], [0.0005, 0.05, 0.3, 7.0]])
    # no cutoff: the established coherent rule, strictly positive
    assert coherent_mode_mask(energies, q_folded).tolist() == [
        [False, True, True, True], [True, True, True, True]]
    # with a cutoff: the mesh consumers' rule (floors, Gamma-aware) plus the cutoff
    expected = mode_floor_mask(energies.reshape(-1), q_folded, 4, 0.2).reshape(2, 4)
    assert coherent_mode_mask(energies, q_folded, 0.2).tolist() == expected.tolist()
    assert expected.tolist() == [[False, False, True, True], [False, False, True, True]]


def test_cutoff_summary_reports_removed_weight_and_displacement_change():
    from irma.core.phonopy_io import (
        PHONON_CUTOFF_WARN_FRACTION, format_phonon_cutoff_summary, phonon_cutoff_summary)
    mesh = _synthetic_mesh(0.5)
    s = phonon_cutoff_summary(mesh, 296.0)
    assert s["min_phonon_energy_meV"] == 0.5 and s["mode_count"] == 6
    assert s["imaginary_modes"] == 0 and s["baseline_floor_excluded_modes"] == 0
    # the 0.1 and 0.5 meV modes of both q points go (<= cutoff), the 1 meV ones stay
    assert s["cutoff_removed_modes"] == 4
    assert s["cutoff_removed_weight_fraction"] == pytest.approx(4.0 / 6.0)
    assert s["dos_trace_deficit_per_atom"] == pytest.approx([2.0])
    assert s["trace_u_before_A2"][0] > s["trace_u_after_A2"][0] > 0.0
    assert s["mean_trace_u_change"] < -PHONON_CUTOFF_WARN_FRACTION
    assert s["warn"] is True
    lines = format_phonon_cutoff_summary(s)
    assert lines[0].startswith("Phonon-energy cutoff 0.5 meV at 296 K")
    assert "removes 4 more of 6 modes" in lines[0]
    assert any(line.startswith("WARNING") and "1%" in line for line in lines)
    # a cutoff below every mode changes nothing and does not warn
    quiet = phonon_cutoff_summary(_synthetic_mesh(0.01), 296.0)
    assert quiet["cutoff_removed_modes"] == 0 and quiet["warn"] is False
    assert quiet["trace_u_before_A2"] == pytest.approx(quiet["trace_u_after_A2"])


def test_spectra_and_ncrystal_configs_carry_and_validate_the_cutoff(tmp_path):
    from irma.spectra.config import SpectraConfigError
    from irma.ncrystal.config import NCrystalExportConfig
    from irma.ncrystal.provenance import collect_provenance
    import irma.spectra.config as sc
    base = {"material": {"phonopy_yaml": "model/phonopy.yaml", "mesh": [4, 4, 4],
                         "temperature_K": 296.0,
                         "scatterers": [{"symbol": "C", "sigma_bound_b": 5.551, "awr": 11.898,
                                         "b_coh_fm": 6.646, "sigma_inc_b": 0.001}]},
            "export": {"material_id": "graphite", "inelastic_mode": 2,
                       "min_phonon_energy_meV": 0.5}}
    cfg = NCrystalExportConfig.from_dict(base)
    assert cfg.min_phonon_energy_meV == 0.5
    default = NCrystalExportConfig.from_dict({**base, "export": {"material_id": "g"}})
    assert default.min_phonon_energy_meV == 0.0
    for bad in (-1.0, float("nan")):
        with pytest.raises(SpectraConfigError, match="min_phonon_energy_meV"):
            NCrystalExportConfig.from_dict({**base, "export": {"material_id": "g",
                                                               "min_phonon_energy_meV": bad}})
    yaml_path = tmp_path / "phonopy.yaml"
    yaml_path.write_text("phonopy: data\n")
    meta = collect_provenance(phonopy_yaml=yaml_path, mesh=(4, 4, 4), temperature_K=296.0,
                              num_directions=10, multiphonon_num_directions=5,
                              multiphonon_max_order="auto", min_phonon_energy_meV=0.5)
    assert meta["min_phonon_energy_meV"] == "0.5"
    physics = sc.PhysicsConfig()
    assert physics.min_phonon_energy_meV == 0.0


def test_spectra_mode1_cutoff_removes_the_low_energy_one_phonon_intensity():
    # the engine context must be built with the same cutoff as the run; with
    # the default context the intensity below the cutoff stayed at ~20%
    pytest.importorskip("phonopy")
    from test_spectra_engine_integration import _LIVE
    from irma.spectra.forward import compute_spectrum
    r = compute_spectrum(**dict(_LIVE, elastic=False), min_phonon_energy_mev=30.0)
    E, intensity = r.E, r.I_inelastic
    low = E < 30.0
    assert (np.trapezoid(intensity[low], E[low])
            < 1e-2 * np.trapezoid(intensity, E))
