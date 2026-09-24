"""Displacement-engine tests on the EMT test backend: supercell rule,
FC production + symmetrization metrics, spawn-pool parallelism, and the
fingerprinted resume contract. No torch, no downloads."""
import os

import numpy as np
import pytest

ase = pytest.importorskip("ase")
pytest.importorskip("phonopy")

from ase.build import bulk                          # noqa: E402
from ase.calculators.emt import EMT                 # noqa: E402

from irma.mlip.calculators import CalculatorSpec    # noqa: E402
from irma.mlip.phonons import (                     # noqa: E402
    compute_force_constants, supercell_matrix)
from irma.mlip.relax import relax                   # noqa: E402

SPEC = CalculatorSpec("emt")


def _relaxed_rattled_al():
    """A relaxed-but-low-symmetry 4-atom Al cell: several displacements."""
    atoms = bulk("Al", "fcc", a=4.05, cubic=True)
    atoms.rattle(stdev=0.02, seed=7)
    return relax(atoms, EMT(), fmax=0.01, nmax=200).atoms


def test_supercell_rule():
    assert supercell_matrix([4.05, 4.05, 4.05], 12.0) == [3, 3, 3]
    assert supercell_matrix([2.46, 2.46, 6.71], 12.0) == [5, 5, 2]
    assert supercell_matrix([30.0, 25.0, 28.0], 12.0) == [1, 1, 1]
    assert supercell_matrix([4.05, 4.05, 4.05], (2, 2, 1)) == [2, 2, 1]
    with pytest.raises(ValueError):
        supercell_matrix([4.0, 4.0, 4.0], (2, 2))
    with pytest.raises(ValueError):
        supercell_matrix([4.0, 4.0, 4.0], (0, 2, 2))


def _gamma_freqs(phonon):
    phonon.run_qpoints([[0.0, 0.0, 0.0]])
    return np.asarray(phonon.qpoints.frequencies[0])   # get_qpoints_dict is deprecated


def test_ideal_crystal_fc_and_gamma_modes(tmp_path):
    atoms = bulk("Al", "fcc", a=4.05, cubic=True)
    res = compute_force_constants(
        atoms, SPEC, supercell=(2, 2, 2), delta=0.03, jobs=1,
        scratch_dir=str(tmp_path / "scratch"), progress=lambda *_: None)

    assert res.n_displacements >= 1 and res.n_from_cache == 0
    fc = res.phonon.force_constants
    assert fc.shape == (32, 32, 3, 3)                  # full FC, 4 atoms x 2x2x2
    assert np.isfinite(fc).all()
    # symmetrization enforced the translational ASR on the corrected FC
    from phonopy.harmonic.force_constants import get_drift_force_constants
    d1, d2, _, _ = get_drift_force_constants(fc, primitive=res.phonon.primitive)
    assert max(abs(d1), abs(d2)) < 1e-8

    # Gamma: three acoustic modes at ~0, none imaginary beyond noise
    freqs = _gamma_freqs(res.phonon)
    assert np.isfinite(freqs).all()
    assert np.abs(np.sort(freqs)[:3]).max() < 0.1     # THz
    assert res.asr_drift_before >= 0.0
    assert res.symmetrization_delta >= 0.0
    assert res.wall_s > 0.0


def test_custom_masses_reach_the_phonon_model(tmp_path):
    # Doubling every mass must scale frequencies by 1/sqrt(2); a model that
    # silently reinstalls periodic-table defaults would leave them unchanged.
    atoms = bulk("Al", "fcc", a=4.05, cubic=True)
    heavy = atoms.copy()
    heavy.set_masses(atoms.get_masses() * 2.0)

    kw = dict(supercell=(2, 2, 2), delta=0.03, jobs=1,
              progress=lambda *_: None)
    r_std = compute_force_constants(atoms, SPEC,
                                    scratch_dir=str(tmp_path / "a"), **kw)
    r_heavy = compute_force_constants(heavy, SPEC,
                                      scratch_dir=str(tmp_path / "b"), **kw)
    assert r_std.fingerprint != r_heavy.fingerprint
    f_std = np.sort(_gamma_freqs(r_std.phonon))[3:]    # the 9 nonzero branches
    f_heavy = np.sort(_gamma_freqs(r_heavy.phonon))[3:]  # of the 4-atom cell
    np.testing.assert_allclose(f_heavy, f_std / np.sqrt(2.0), rtol=1e-6)


def test_worker_failure_names_the_displacement_and_returns_promptly(tmp_path):
    # EMT has no Fe parametrization: the worker raises, and the driver must
    # surface it wrapped with the displacement index instead of hanging on
    # the queued backlog.
    atoms = bulk("Fe", "bcc", a=2.87, cubic=True)
    with pytest.raises(RuntimeError, match="displacement"):
        compute_force_constants(
            atoms, SPEC, supercell=(2, 2, 2), delta=0.03, jobs=2,
            scratch_dir=str(tmp_path / "s"), progress=lambda *_: None)


def test_parallel_matches_serial(tmp_path):
    atoms = _relaxed_rattled_al()
    kw = dict(supercell=(2, 2, 2), delta=0.03, progress=lambda *_: None)
    r1 = compute_force_constants(
        atoms, SPEC, jobs=1, scratch_dir=str(tmp_path / "s1"), **kw)
    r2 = compute_force_constants(
        atoms, SPEC, jobs=2, worker_threads=2,
        scratch_dir=str(tmp_path / "s2"), **kw)

    assert r1.n_displacements == r2.n_displacements > 1
    # EMT is deterministic and spawn workers use the same libraries, so the
    # tolerance is tight; it is a tolerance, not a bit-identity contract.
    np.testing.assert_allclose(r1.phonon.force_constants,
                               r2.phonon.force_constants,
                               rtol=0, atol=1e-12)
    assert r1.fingerprint == r2.fingerprint


def test_resume_reuses_and_invalidates(tmp_path):
    atoms = _relaxed_rattled_al()
    scratch = str(tmp_path / "scratch")
    kw = dict(supercell=(2, 2, 2), jobs=1, scratch_dir=scratch,
              progress=lambda *_: None)

    r1 = compute_force_constants(atoms, SPEC, delta=0.03, **kw)
    n = r1.n_displacements
    assert r1.n_from_cache == 0

    # delete one cached force set -> rerun recomputes exactly the hole
    victims = sorted(f for f in os.listdir(scratch) if f.startswith("forces_"))
    os.remove(os.path.join(scratch, victims[0]))
    r2 = compute_force_constants(atoms, SPEC, delta=0.03, **kw)
    assert r2.n_from_cache == n - 1
    np.testing.assert_allclose(r1.phonon.force_constants,
                               r2.phonon.force_constants, rtol=0, atol=1e-12)

    # any input change (here: delta) invalidates the whole cache
    r3 = compute_force_constants(atoms, SPEC, delta=0.02, **kw)
    assert r3.n_from_cache == 0
    assert r3.fingerprint != r1.fingerprint
