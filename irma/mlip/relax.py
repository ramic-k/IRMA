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
    fmax_atoms: float             # max atomic force after relaxation, eV/A
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
        return self.spacegroup_before != self.spacegroup_after


def _spacegroup(atoms, symprec: float) -> str:
    import spglib
    cell = (atoms.get_cell().array, atoms.get_scaled_positions(),
            atoms.get_atomic_numbers())
    return spglib.get_spacegroup(cell, symprec=symprec) or "unknown"


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
    atom's images over all detected symmetry operations in the original
    cell setting (no reorientation, unlike spglib standardization), which
    projects the positions onto exact invariance under the detected
    group. Positions are applied through the constraint-aware setter, so
    ASE constraints (FixAtoms etc.) are honored; the returned value is
    the largest shift the symmetrization asked for before constraints,
    in Angstrom. Raises MlipRelaxError when the operations cannot be
    mapped (symprec too loose for the actual distortion).
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
    # set_positions, not set_scaled_positions: only the former runs
    # constraint.adjust_positions (FixAtoms etc.), as in the jitter path
    atoms.set_positions((snapped % 1.0) @ cell)
    return max_shift


def relax(atoms, calculator, *, fmax: float = 0.01, nmax: int = 100,
          relax_cell: bool = False, symprec: float = 1e-3,
          snap_symmetry=None, jitter_cycles: int = 0,
          logfile=None) -> RelaxResult:
    """Relax `atoms` in place with FIRE; return the full report.

    Convergence comes from Optimizer.run()'s return value. fmax is the
    ASE convention: the largest per-atom force norm, eV/A. With
    snap_symmetry, the relaxed positions are projected onto the exact
    orbits of the spacegroup detected at that tolerance (see
    snap_to_symmetry); spacegroup_after and the force residuals then
    describe the snapped structure.

    jitter_cycles (default 0, a single pass): when the relaxation ends
    unconverged, kick the min(3, natoms) highest-force atoms (Gaussian,
    0.05 A per Cartesian component, fixed seed) and re-relax, up to this
    many extra cycles of `nmax` steps each (steps_taken is the total).
    The lowest-residual frame seen during any pass is returned. This
    helps when MLIP force noise keeps the forces above the target at
    the energy minimum (a PMMA glass stalls at 0.087 eV/A; three cycles
    reach 0.005-0.010). It does not act at a zero-gradient saddle, which
    FIRE reports as converged.
    """
    import numpy as np
    from ase.filters import FrechetCellFilter
    from ase.optimize import FIRE

    atoms.calc = calculator
    sg_before = _spacegroup(atoms, symprec)

    # With a cell filter, ASE's convergence criterion is the max row norm of
    # the filter forces (atomic forces plus cell-gradient rows). The reported
    # residuals are those of this target, so a cell whose stress is not
    # converged is not reported as converged; the atomic-only number is kept
    # alongside.
    target = FrechetCellFilter(atoms) if relax_cell else atoms
    fmax_initial = _max_force(target)

    # the best frame is tracked at every step, not only at the end of a
    # pass: on a noisy surface a pass can reach its lowest residual
    # mid-trajectory and end higher. The forces are ASE-cached from the
    # step itself, so observing costs no extra force evaluations.
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
            # would pick implementation-defined atoms
            for i in np.argsort(-fmags, kind="stable")[:3]:
                kick[i] += rng.normal(scale=0.05, size=3)
            # set_positions applies ASE constraints (FixAtoms etc.);
            # writing atoms.positions directly would move fixed atoms
            atoms.set_positions(kick)
            dyn = FIRE(target, logfile=logfile)
            dyn.attach(_observe, interval=1)
            converged = bool(dyn.run(fmax=fmax, steps=nmax))
            steps_total += int(dyn.nsteps)
            _observe()
            if converged:
                break
        # end on the best frame observed, not on a final kick that landed
        # worse
        if best["positions"] is not None and _max_force(target) > best["fmax"]:
            atoms.set_cell(best["cell"], scale_atoms=False)
            atoms.set_positions(best["positions"])
            converged = best["fmax"] <= fmax

    snap_shift = 0.0
    if snap_symmetry:
        # the snap tolerance is looser than the reporting symprec by
        # design: float32 relaxation drift can exceed 1e-3
        snap_shift = snap_to_symmetry(atoms, symprec=float(snap_symmetry))

    fmax_achieved = _max_force(target)
    fmax_atoms = _max_force(atoms)
    if snap_symmetry and converged and fmax_achieved > fmax:
        # the snap moved the structure off the optimizer's stationary
        # point: the pre-snap convergence no longer describes the atoms
        # the force constants will see, so the bundle is not recorded as
        # converged
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
