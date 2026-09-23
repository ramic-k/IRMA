"""Relaxation-machinery tests on ASE's built-in EMT potential (Al): no torch,
no downloads, seconds of runtime. The pipeline is calculator-agnostic, so
these exercise the same code paths a real MLIP uses."""
import pytest

ase = pytest.importorskip("ase")

from ase.build import bulk                     # noqa: E402
from ase.calculators.emt import EMT            # noqa: E402

from irma.mlip.relax import relax              # noqa: E402


def _rattled_al(seed=42, stdev=0.05):
    atoms = bulk("Al", "fcc", a=4.05, cubic=True)   # 4-atom cubic cell
    atoms.rattle(stdev=stdev, seed=seed)
    return atoms


def test_ideal_crystal_converges_immediately_and_keeps_symmetry():
    atoms = bulk("Al", "fcc", a=4.05, cubic=True)
    res = relax(atoms, EMT(), fmax=0.02, nmax=50)
    assert res.converged
    assert res.fmax_achieved <= 0.02
    assert res.spacegroup_before == res.spacegroup_after
    assert not res.symmetry_changed
    assert "225" in res.spacegroup_before      # Fm-3m


def test_rattled_crystal_relaxes_to_target():
    res = relax(_rattled_al(), EMT(), fmax=0.01, nmax=200)
    assert res.converged
    assert res.fmax_achieved <= 0.01
    assert res.fmax_initial > 0.01             # rattle produced real forces
    assert res.steps_taken > 0
    assert res.atoms.calc is None              # detached result


def test_unconverged_relaxation_is_reported_not_hidden():
    res = relax(_rattled_al(stdev=0.15), EMT(), fmax=1e-6, nmax=2)
    assert not res.converged
    assert res.fmax_achieved > 1e-6
    assert res.steps_taken <= 2


def test_snap_to_symmetry_restores_exact_wyckoffs():
    # rattle Al fcc below the detection tolerance: spglib still sees
    # Fm-3m at 1e-3, but the positions are off the exact orbits; the
    # snap must land them back to machine precision
    import numpy as np

    from irma.mlip.relax import snap_to_symmetry

    atoms = bulk("Al", "fcc", a=4.05, cubic=True)
    ideal = atoms.get_scaled_positions().copy()
    rng = np.random.default_rng(7)
    atoms.set_scaled_positions(
        ideal + rng.uniform(-5e-5, 5e-5, ideal.shape))

    shift = snap_to_symmetry(atoms, symprec=1e-3)
    assert 0 < shift < 1e-3
    snapped = atoms.get_scaled_positions()
    # symmetrization is exact only up to a rigid translation (the group
    # does not pin the origin); the RELATIVE geometry must be restored
    # to machine precision...
    delta = snapped - ideal
    delta -= np.rint(delta)
    delta -= delta.mean(axis=0)
    assert np.abs(delta).max() < 1e-12
    # ...and the snapped structure must be exactly symmetric, i.e. the
    # full spacegroup is detected even at a near-machine tolerance
    import spglib
    cell = (atoms.get_cell().array, snapped, atoms.get_atomic_numbers())
    assert spglib.get_spacegroup(cell, symprec=1e-8).endswith("(225)")

    # a heavily distorted structure is simply P1 at a tight tolerance:
    # the snap degrades to an exact no-op (identity averaging), never a
    # corruption (MlipRelaxError guards the pathological op-mapping case)
    atoms.set_scaled_positions(ideal)
    atoms.rattle(stdev=0.3, seed=1)
    before = atoms.get_scaled_positions().copy()
    assert snap_to_symmetry(atoms, symprec=1e-6) < 1e-10
    assert np.allclose(atoms.get_scaled_positions(), before, atol=1e-12)


def test_relax_with_snap_symmetry_records_the_shift():
    atoms = bulk("Al", "fcc", a=4.05, cubic=True)
    atoms.rattle(stdev=1e-5, seed=3)      # sub-tolerance drift
    rr = relax(atoms, EMT(), fmax=0.05, nmax=50, snap_symmetry=1e-2)
    assert rr.snapped is True
    assert rr.snap_max_shift_A >= 0.0
    assert rr.spacegroup_after.endswith("(225)")   # Fm-3m restored

    rr2 = relax(bulk("Al", "fcc", a=4.05, cubic=True), EMT(),
                fmax=0.05, nmax=50)
    assert rr2.snapped is False and rr2.snap_max_shift_A == 0.0


def test_snap_that_breaks_convergence_is_reported(monkeypatch):
    # review finding: if the snap moves the structure off the optimizer's
    # stationary point, the pre-snap converged=True must not survive
    import irma.mlip.relax as rel

    def violent_snap(atoms, symprec=1e-2):
        pos = atoms.get_positions()
        pos[0] += 0.15                     # well off the minimum
        atoms.set_positions(pos)
        return 0.15

    monkeypatch.setattr(rel, "snap_to_symmetry", violent_snap)
    atoms = bulk("Al", "fcc", a=4.05, cubic=True)
    rr = rel.relax(atoms, EMT(), fmax=0.01, nmax=80, snap_symmetry=1e-2)
    assert rr.snapped is True
    assert rr.fmax_achieved > 0.01
    assert rr.converged is False           # re-gated on the post-snap state


def test_cell_relaxation_moves_a_strained_cell_toward_equilibrium():
    atoms = bulk("Al", "fcc", a=4.05, cubic=True)
    strained = atoms.copy()
    strained.set_cell(atoms.get_cell() * 1.04, scale_atoms=True)
    a_start = strained.get_cell().lengths()[0]

    res = relax(strained, EMT(), fmax=0.01, nmax=200, relax_cell=True)
    a_end = res.atoms.get_cell().lengths()[0]
    assert res.converged and res.cell_relaxed
    assert a_end < a_start                     # compressed back toward a0
    assert abs(a_end - 4.05) < abs(a_start - 4.05)


def test_cell_residual_is_the_filter_residual_not_the_atomic_one():
    # Symmetric strain: atomic forces are ~0 by symmetry, but the cell
    # gradient is large. With nmax too small to converge, the reported
    # residual must reflect the optimization target (finding: a
    # stress-unconverged cell must not report "1e-15 converged").
    atoms = bulk("Al", "fcc", a=4.05, cubic=True)
    atoms.set_cell(atoms.get_cell() * 1.06, scale_atoms=True)
    res = relax(atoms, EMT(), fmax=1e-4, nmax=1, relax_cell=True)
    assert not res.converged
    assert res.fmax_achieved > 1e-3            # cell gradient still large
    assert res.fmax_atoms < 1e-6               # atomic forces ~0 by symmetry
    assert res.fmax_initial > 1e-3             # target residual, not atomic


def test_fixed_cell_relaxation_does_not_touch_the_cell():
    atoms = _rattled_al()
    cell_before = atoms.get_cell().array.copy()
    res = relax(atoms, EMT(), fmax=0.01, nmax=200, relax_cell=False)
    assert (res.atoms.get_cell().array == cell_before).all()


# ---- --jitter-cycles ---------------------------------------------------------
class _NoiseFloorCalc:
    """Harmonic well + a constant spurious force on atom 0: reported forces
    can never drop below `floor` there, reproducing the measured MLIP
    force/energy-inconsistency stall class (PMMA glass kinks)."""
    def __init__(self, x0, k=2.0, floor=0.05):
        import numpy as np
        self.x0 = np.array(x0)
        self.k, self.floor = k, floor
        self.results = {}

    def get_forces(self, atoms=None):
        f = -self.k * (atoms.positions - self.x0)
        # discontinuous restoring kink on atom 0's x: the total force
        # magnitude is k|d| + floor >= floor on both sides of the
        # crossing, so no geometry ever reports below `floor` (a constant
        # offset would merely SHIFT the minimum and converge)
        d0 = atoms.positions[0, 0] - self.x0[0, 0]
        f[0, 0] += -self.floor if d0 >= 0 else self.floor
        return f

    def get_potential_energy(self, atoms=None, force_consistent=False):
        d = atoms.positions - self.x0
        return float(0.5 * self.k * (d ** 2).sum())

    def get_stress(self, atoms=None):
        raise NotImplementedError

    def check_state(self, atoms, tol=1e-15):
        return ["positions"]

    def calculation_required(self, atoms, quantities):
        return True


def _floor_system():
    import numpy as np
    atoms = bulk("Al", "fcc", a=4.05, cubic=True)
    x0 = atoms.positions.copy()
    rng = np.random.default_rng(3)
    atoms.positions += rng.normal(scale=0.03, size=atoms.positions.shape)
    return atoms, _NoiseFloorCalc(x0)


def test_jitter_never_fires_when_converged():
    atoms = _rattled_al()
    res = relax(atoms, EMT(), fmax=0.02, nmax=200, jitter_cycles=3)
    assert res.converged
    assert res.jitter_cycles_used == 0


def test_jitter_cycles_run_on_a_noise_floor_stall_and_keep_best():
    atoms, calc = _floor_system()
    res = relax(atoms, calc, fmax=0.01, nmax=150, jitter_cycles=2)
    # the spurious 0.05 eV/A force on atom 0 is unbeatable: never converged,
    # both cycles spent, and the returned frame is the best one visited
    assert not res.converged
    assert res.jitter_cycles_used == 2
    assert res.fmax_achieved == pytest.approx(0.05, abs=0.02)
    assert res.steps_taken > 150            # accumulated across cycles
    import numpy as np
    final = float(np.linalg.norm(calc.get_forces(res.atoms), axis=1).max())
    assert final == pytest.approx(res.fmax_achieved, abs=1e-6)


def test_jitter_default_zero_is_single_pass():
    atoms, calc = _floor_system()
    res = relax(atoms, calc, fmax=0.01, nmax=150)
    assert not res.converged
    assert res.jitter_cycles_used == 0
    assert res.steps_taken <= 150
