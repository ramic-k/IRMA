"""FIRE relaxation with a convergence gate and a symmetry-drift check.

The structure must be a stationary point of the same potential that later
computes the displaced forces; otherwise linear force terms contaminate the
harmonic expansion and appear as spurious imaginary acoustic modes. The
caller (CLI) therefore treats an unconverged relaxation as a hard error
unless the user explicitly forces past it; this module only reports.

All heavy imports are function-level (core-clean module).
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RelaxResult:
    atoms: object                 # relaxed ase.Atoms (calculator detached)
    converged: bool
    fmax_target: float
    fmax_initial: float           # optimizer-target residual before, eV/A
    fmax_achieved: float          # optimizer-target residual after, eV/A
    fmax_atoms: float             # max ATOMIC force after relaxation, eV/A
    steps_taken: int
    nmax: int
    cell_relaxed: bool
    spacegroup_before: str
    spacegroup_after: str
    symprec: float
    snapped: bool = False         # positions snapped to the detected
    snap_max_shift_A: float = 0.0  # symmetry orbits (--snap-symmetry)
    jitter_cycles_used: int = 0   # kick+re-relax cycles taken (--jitter-cycles)

    @property
    def symmetry_changed(self) -> bool:
        return (self.spacegroup_before != "unknown"
                and self.spacegroup_after != "unknown"
                and self.spacegroup_before != self.spacegroup_after)


def _spacegroup(atoms, symprec: float) -> str:
    try:
        import spglib
    except ImportError:
        return "unknown"
    cell = (atoms.get_cell().array, atoms.get_scaled_positions(),
            atoms.get_atomic_numbers())
    sg = spglib.get_spacegroup(cell, symprec=symprec)
    return sg if sg else "unknown"


def _max_force(atoms) -> float:
    import numpy as np
    forces = atoms.get_forces()
    return float(np.sqrt((forces ** 2).sum(axis=1)).max())


class MlipRelaxError(RuntimeError):
    """Relaxation post-processing (e.g. symmetry snapping) failed."""


def snap_to_symmetry(atoms, symprec: float = 1e-3) -> float:
    """Snap positions onto the exact orbits of the detected spacegroup.

    Float32 relaxations routinely land within `symprec` of the ideal
    Wyckoff positions but not exactly on them; phonopy then sees P1 at
    its tighter tolerance and generates the full displacement set (24
    instead of 8 on wurtzite BeO, 72 instead of 18 on baddeleyite), and
    a symmetry-reduced BORN file stops matching. This averages each
    atom's images over ALL detected symmetry operations in the ORIGINAL
    cell setting (no reorientation, unlike spglib standardization), which
    projects the positions onto exact invariance under the detected
    group. Positions are applied through the constraint-aware setter, so
    ASE constraints (FixAtoms etc.) are honored; the returned value is
    the largest UNCONSTRAINED shift the symmetrization asked for, in
    Angstrom. Raises MlipRelaxError when the operations cannot be mapped
    (symprec too loose for the actual distortion).
    """
    import numpy as np
    import spglib

    cell = atoms.get_cell().array
    scaled = atoms.get_scaled_positions()
    numbers = atoms.get_atomic_numbers()
    sym = spglib.get_symmetry((cell, scaled, numbers), symprec=symprec)
    if sym is None:
        raise MlipRelaxError(
            f"spglib found no symmetry at symprec={symprec}")

    n = len(atoms)
    accum = np.zeros((n, 3))
    for rot, trans in zip(sym["rotations"], sym["translations"]):
        images = scaled @ rot.T + trans
        # map each image to the atom it lands on (minimum image)
        delta = images[:, None, :] - scaled[None, :, :]
        delta -= np.rint(delta)
        cart = np.einsum("ijk,kl->ijl", delta, cell)
        dist = np.linalg.norm(cart, axis=2)
        j = dist.argmin(axis=1)
        if (np.sort(j) != np.arange(n)).any() \
                or (numbers[j] != numbers).any():
            raise MlipRelaxError(
                f"could not map the structure onto itself under the "
                f"symmetry detected at symprec={symprec}; the distortion "
                f"is too large to snap")
        # accumulate this operation's version of atom j's position,
        # unwrapped to the current position's branch
        target = scaled[j] + delta[np.arange(n), j]
        accum[j] += target
    snapped = accum / len(sym["rotations"])
    shift = snapped - scaled
    shift -= np.rint(shift)
    max_shift = float(np.linalg.norm(shift @ cell, axis=1).max())
    # set_positions, NOT set_scaled_positions: only the former runs
    # constraint.adjust_positions (FixAtoms etc.), exactly like the
    # jitter path — a structure read with constraints must keep them
    atoms.set_positions((snapped % 1.0) @ cell)
    return max_shift


def _cell_filter():
    """The supported cell-relaxation filter (ase>=3.23: both live in
    ase.filters; FrechetCellFilter is the current recommendation)."""
    from ase.filters import FrechetCellFilter
    return FrechetCellFilter


def relax(atoms, calculator, *, fmax: float = 0.01, nmax: int = 100,
          relax_cell: bool = False, symprec: float = 1e-3,
          snap_symmetry=False, jitter_cycles: int = 0,
          logfile=None) -> RelaxResult:
    """Relax `atoms` in place with FIRE; return the full report.

    Convergence comes from Optimizer.run()'s return value (INSPIRED ignores
    it; we do not). fmax is the ASE convention: the largest per-atom force
    norm, eV/A. With snap_symmetry, the relaxed positions are projected
    onto the exact orbits of the spacegroup detected at `symprec` (see
    snap_to_symmetry); the reported spacegroup_after and force residuals
    describe the SNAPPED structure.

    jitter_cycles (default 0 = exact single-pass behavior): when the
    plain relaxation ends UNconverged, kick the min(3, natoms)
    highest-force atoms (Gaussian, 0.05 A std per Cartesian component,
    constraint-aware, fixed per-call seed) and re-relax, up to this many
    extra cycles, each with its own `nmax` budget (steps_taken reports
    the cumulative total). A per-step observer snapshots the
    lowest-residual frame anywhere along every pass, and that frame is
    what the call returns; --snap-symmetry applies afterwards to it.
    This escapes the measured MLIP force-noise stall class, where
    reported forces stay above the target at the energy minimum (PMMA
    glass: FIRE stalls at 0.087 eV/A; three cycles reach 0.005-0.010).
    It cannot fire at a true zero-gradient saddle: FIRE reports those
    converged.
    """
    import numpy as np
    from ase.optimize import FIRE

    if jitter_cycles < 0:
        raise ValueError(f"jitter_cycles must be >= 0, got {jitter_cycles}")
    atoms.calc = calculator
    sg_before = _spacegroup(atoms, symprec)

    # With a cell filter, ASE's convergence criterion is the max row norm of
    # the FILTER forces (atomic forces plus cell-gradient rows). Reporting
    # only the atomic fmax would call a stress-unconverged cell "1e-15
    # converged" (review finding 6): the reported residuals are those of the
    # actual optimization target, with the atomic-only number kept alongside.
    target = _cell_filter()(atoms) if relax_cell else atoms
    fmax_initial = _max_force(target)

    # the best-frame tracker is a per-step observer, not an endpoint
    # check: on a noisy surface a pass can dip through its lowest
    # residual mid-trajectory and end higher (review finding 6). The
    # forces at observation time are ASE-cached from the step itself,
    # so observing costs no extra force evaluations.
    best = {"fmax": np.inf, "positions": None, "cell": None}

    def _observe():
        f_now = _max_force(target)
        if f_now < best["fmax"]:
            best.update(fmax=f_now, positions=atoms.get_positions(),
                        cell=atoms.cell.array.copy())

    dyn = FIRE(target, logfile=logfile)
    if jitter_cycles > 0:
        dyn.attach(_observe, interval=1)
    converged = bool(dyn.run(fmax=fmax, steps=nmax))
    steps_total = int(dyn.nsteps)

    jitter_used = 0
    if not converged and jitter_cycles > 0:
        rng = np.random.default_rng(20260718)
        _observe()
        for _ in range(int(jitter_cycles)):
            jitter_used += 1
            fmags = np.linalg.norm(atoms.get_forces(), axis=1)
            kick = atoms.get_positions()
            # stable descending order with index tie-break: symmetry-
            # equivalent sites carry identical forces and plain argsort
            # would pick implementation-defined atoms (review finding 2)
            for i in np.argsort(-fmags, kind="stable")[:3]:
                kick[i] += rng.normal(scale=0.05, size=3)
            # set_positions applies ASE constraints (FixAtoms etc.);
            # writing atoms.positions directly would move fixed atoms
            # (review finding 1)
            atoms.set_positions(kick)
            dyn = FIRE(target, logfile=logfile)
            dyn.attach(_observe, interval=1)
            converged = bool(dyn.run(fmax=fmax, steps=nmax))
            steps_total += int(dyn.nsteps)
            _observe()
            if converged:
                break
        # always end on the best frame observed (a final kick that landed
        # worse must never become the reported structure)
        if best["positions"] is not None and _max_force(target) > best["fmax"]:
            atoms.set_cell(best["cell"], scale_atoms=False)
            atoms.set_positions(best["positions"])
            converged = best["fmax"] <= fmax

    snap_shift = 0.0
    if snap_symmetry:
        # the snap DETECTION tolerance is looser than the reporting
        # symprec by design: the drift it exists to repair (float32
        # relaxations) can exceed 1e-3; True selects the 1e-2 default,
        # a float selects it explicitly
        tol = 1e-2 if snap_symmetry is True else float(snap_symmetry)
        snap_shift = snap_to_symmetry(atoms, symprec=tol)

    fmax_achieved = _max_force(target)
    fmax_atoms = _max_force(atoms)
    if snap_symmetry and converged and fmax_achieved > fmax:
        # the snap moved the structure off the optimizer's stationary
        # point: the pre-snap convergence no longer describes the atoms
        # the force constants will see (review finding -- a snapped
        # bundle must never be falsely recorded as converged)
        converged = False
    sg_after = _spacegroup(atoms, symprec)

    relaxed = atoms.copy()        # detach the calculator from the result
    return RelaxResult(
        atoms=relaxed,
        converged=converged,
        fmax_target=float(fmax),
        fmax_initial=fmax_initial,
        fmax_achieved=fmax_achieved,
        fmax_atoms=fmax_atoms,
        steps_taken=steps_total,
        nmax=int(nmax),
        cell_relaxed=bool(relax_cell),
        spacegroup_before=sg_before,
        spacegroup_after=sg_after,
        symprec=float(symprec),
        snapped=bool(snap_symmetry),
        snap_max_shift_A=float(snap_shift),
        jitter_cycles_used=jitter_used,
    )
