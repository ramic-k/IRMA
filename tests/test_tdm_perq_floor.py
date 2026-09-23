"""Per-q Debye-Waller TDM floor (audit §B 3b / b-ii).

The thermal-displacement tensor U_ij feeds the Debye-Waller factor. phonopy's
``ThermalDisplacementMatrices`` accepts only a single GLOBAL freq_min, so on a
Gamma-containing mesh it applies the Gamma-tier floor (0.1 meV) everywhere and
drops off-Gamma modes in (1 ueV, 0.1 meV] that the one-phonon sum keeps. The
per-q sum (``compute_thermal_displacement_matrices``) applies the SAME two-tier
``mode_floor_mask`` as the DOS-tensor / one-phonon paths, keeping the TDM
consistent with the one-phonon mode set.

Correctness reference: on a Gamma-FREE mesh the two floors coincide, so the per-q
sum must reproduce phonopy's TDM (to ~machine precision). The synthetic test
exercises the case the two floors actually disagree (a soft off-Gamma mode).
"""
import os

import numpy as np
import pytest

from irma.core.constants import GAMMA_ACOUSTIC_FLOOR_MEV, MODE_ENERGY_FLOOR_MEV, THZ_TO_EV
from irma.core.phonopy_io import (
    PhonopyMeshData,
    compute_thermal_displacement_matrices,
    mode_floor_mask,
)

_BE = "tests/mode2_euphonic_n1_validation/beryllium/phonopy.yaml"


def _load(mesh):
    pytest.importorskip("phonopy")
    if not os.path.exists(_BE):
        pytest.skip("Be phonopy fixture not present (self-contained validation dir)")
    from irma.core.phonopy_io import load_phonopy_mesh
    return load_phonopy_mesh(_BE, mesh)


def _has_gamma(q):
    return bool(np.any(np.all(np.abs(q) < 1.0e-9, axis=1)))


def test_perq_reproduces_phonopy_on_gamma_free_mesh():
    """The reference: Gamma-free mesh -> per-q floor == global floor -> must match
    phonopy's ThermalDisplacementMatrices (units + formula correctness)."""
    md = _load([4, 4, 4])
    assert not _has_gamma(md.qpoints)
    from phonopy.phonon.thermal_displacement import ThermalDisplacementMatrices
    tdm = ThermalDisplacementMatrices(
        md.phonopy_mesh_object, freq_min=MODE_ENERGY_FLOOR_MEV * 1e-3 / THZ_TO_EV)
    tdm.temperatures = [296.0]
    tdm.run()
    u_phonopy = np.asarray(tdm.thermal_displacement_matrices[0], dtype=float)
    u_perq = compute_thermal_displacement_matrices(md, 296.0)
    # ~1e-10 in practice; 1e-8 leaves headroom for BLAS/summation-order noise.
    assert np.allclose(u_phonopy, u_perq, atol=1e-8, rtol=0.0)




def test_perq_keeps_off_gamma_soft_mode_a_global_floor_drops():
    """Synthetic mesh where the two floors DISAGREE: one atom, a Gamma point and
    an off-Gamma q with a 0.05 meV mode polarised along x. That mode sits in
    (1 ueV, 0.1 meV] -- the per-q off-Gamma floor (1 ueV) keeps it; a global
    Gamma-tier floor (0.1 meV) would drop it. Being soft, it dominates U_xx."""
    qpoints = np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]])
    # eV: Gamma triple near 0 (dropped either way); off-Gamma = [0.05 meV, 20, 20]
    freqs = np.array([[1.0e-6, 1.0e-6, 1.0e-6],
                      [5.0e-5, 0.02, 0.02]])
    eig = np.zeros((2, 3, 1, 3), dtype=complex)
    for iq in range(2):                              # each branch a cartesian unit vec
        eig[iq, 0, 0] = [1, 0, 0]
        eig[iq, 1, 0] = [0, 1, 0]
        eig[iq, 2, 0] = [0, 0, 1]
    md = PhonopyMeshData(
        qpoints=qpoints, frequencies_ev=freqs, eigenvectors=eig,
        weights=np.array([1, 1]), masses_amu=np.array([12.0]),
        atom_symbols=["C"], atom_positions=np.zeros((1, 3)))

    mask = mode_floor_mask(freqs.reshape(-1) * 1.0e3, qpoints, 3)
    soft = 3                                         # flat index (q1, branch0)
    assert mask[soft]                                # per-q KEEPS the off-Gamma soft mode
    assert MODE_ENERGY_FLOOR_MEV < freqs.reshape(-1)[soft] * 1.0e3 <= GAMMA_ACOUSTIC_FLOOR_MEV

    u = compute_thermal_displacement_matrices(md, 296.0)
    # coth(E/2kT)/E ~ 2kT/E^2 -> the 0.05 meV x-mode dwarfs the 20 meV y/z modes
    assert u[0, 0, 0] > 10.0 * u[0, 1, 1]
    assert u[0, 0, 0] > 10.0 * u[0, 2, 2]
