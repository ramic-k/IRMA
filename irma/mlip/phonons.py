"""Finite-displacement force constants with a fingerprinted, resumable,
optionally parallel force loop.

The phonopy model is constructed directly with an identity primitive matrix
(the INSPIRED convention; irma.core.phonopy_io's pinned kwargs are a
phonopy.load policy and apply only when RELOADING a bundle). The displacement
force loop is the entire cost center: it runs serially in-process by default,
or across a spawn-context process pool where each worker builds its own
calculator from the pickled CalculatorSpec with ONE native thread (the
measured-oversubscription convention shared with irma.core.noncubic_workers).

Per-displacement forces are persisted to a scratch directory as they
complete, keyed by a versioned fingerprint over everything that affects them
(structure, displacement dataset, supercell/primitive matrices, delta,
potential identity). A rerun with a matching fingerprint reuses cached
forces; ANY mismatch wipes the scratch with a printed reason -- stale-force
reuse is the exact artifact-hazard class this front end exists to eliminate.

All heavy imports are function-level (core-clean module).
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import time
from dataclasses import dataclass

from irma.mlip.calculators import CalculatorSpec, make_calculator

FINGERPRINT_VERSION = 1


@dataclass
class PhononResult:
    phonon: object                # phonopy.Phonopy with produced+symmetrized FC
    n_displacements: int
    n_from_cache: int
    delta: float
    jobs: int
    supercell: tuple
    wall_s: float
    asr_drift_before: float       # max |sum_j Phi_ij| per component, pre-symmetrization
    symmetrization_delta: float   # max |Phi' - Phi| applied by symmetrization
    fingerprint: str


def supercell_matrix(cellpar_abc, lmin_or_dims):
    """The Lmin rule: ceil(lmin/a_i) per axis, or three explicit integers."""
    if isinstance(lmin_or_dims, bool):
        raise ValueError(f"supercell got a boolean: {lmin_or_dims!r}")
    if isinstance(lmin_or_dims, (int, float)):
        lmin = float(lmin_or_dims)
        if not math.isfinite(lmin) or lmin <= 0:
            raise ValueError(f"Lmin must be positive and finite, got {lmin!r}")
        return [math.ceil(lmin / float(a)) for a in cellpar_abc]
    dims = list(lmin_or_dims)
    if (len(dims) != 3
            or any(isinstance(n, bool) or not isinstance(n, int) for n in dims)
            or any(n < 1 for n in dims)):
        raise ValueError(
            f"supercell must be a positive Lmin or three positive integers, "
            f"got {lmin_or_dims!r}")
    return dims


def _build_phonopy(atoms, supercell):
    import numpy as np
    from phonopy import Phonopy
    from phonopy.structure.atoms import PhonopyAtoms
    # masses passed explicitly: PhonopyAtoms otherwise installs periodic-table
    # defaults, silently discarding isotopic/custom masses the fingerprint
    # already accounts for (review finding 3).
    unitcell = PhonopyAtoms(
        symbols=list(atoms.get_chemical_symbols()),
        cell=atoms.get_cell().array,
        scaled_positions=atoms.get_scaled_positions(),
        masses=atoms.get_masses())
    return Phonopy(unitcell,
                   supercell_matrix=np.diag(supercell),
                   primitive_matrix=np.eye(3))


def _fingerprint(atoms, phonon, delta, supercell, spec: CalculatorSpec) -> str:
    """sha256 over everything that affects the displaced forces.

    Canonical binary hashing throughout (no repr of floats); the model is the
    RESOLVED checkpoint identity, never the bare token "default" (a changed
    upstream default must invalidate the cache); local checkpoint files are
    hashed streaming. The phonopy version is included because the
    displacement generator's conventions live there. jobs/threads are
    deliberately excluded: the cache contract is physical, not bitwise.
    """
    import numpy as np
    import phonopy as _phonopy
    from irma.mlip.calculators import (
        effective_package_version, resolved_checkpoint_identity)

    h = hashlib.sha256()
    h.update(f"fp-v{FINGERPRINT_VERSION}".encode())
    h.update(np.ascontiguousarray(atoms.get_cell().array, dtype="<f8").tobytes())
    h.update(np.ascontiguousarray(atoms.get_scaled_positions(), dtype="<f8").tobytes())
    h.update(np.ascontiguousarray(atoms.get_atomic_numbers(), dtype="<i8").tobytes())
    h.update(np.ascontiguousarray(atoms.get_masses(), dtype="<f8").tobytes())
    h.update(np.ascontiguousarray(np.diag(supercell), dtype="<i8").tobytes())
    h.update(np.eye(3).astype("<f8").tobytes())          # primitive matrix
    h.update(np.asarray([float(delta)], dtype="<f8").tobytes())
    disp = np.ascontiguousarray(
        [[float(d[0]), float(d[1]), float(d[2]), float(d[3])]
         for d in phonon.displacements], dtype="<f8")
    h.update(disp.tobytes())
    ckpt = resolved_checkpoint_identity(spec)
    h.update(f"{spec.potential}|{ckpt}|{effective_package_version(spec)}"
             f"|phonopy={_phonopy.__version__}".encode())
    return h.hexdigest()


# --- worker side (top-level: picklable by reference under spawn) -----------

_WORKER_CALC = None


def _worker_init(spec: CalculatorSpec):
    global _WORKER_CALC
    _WORKER_CALC = make_calculator(spec)[0]


def _worker_forces(payload):
    index, symbols, cell, spos = payload
    from ase import Atoms
    sc = Atoms(symbols=symbols, scaled_positions=spos, cell=cell, pbc=True)
    sc.calc = _WORKER_CALC
    return index, sc.get_forces()


# --- scratch persistence -----------------------------------------------------

def _force_path(scratch, index):
    return os.path.join(scratch, f"forces_{index:04d}.npy")


def _save_force(scratch, index, forces):
    import numpy as np
    path = _force_path(scratch, index)
    tmp = path + ".tmp"
    with open(tmp, "wb") as fh:           # file handle: np.save appends no
        np.save(fh, forces)               # extension, so the rename is exact
    os.replace(tmp, path)


def _purge_scratch(scratch, progress, reason):
    progress(f"  {reason}: discarding cached forces")
    for name in os.listdir(scratch):
        if name.startswith("forces_") or name.startswith("fingerprint.json"):
            os.remove(os.path.join(scratch, name))


def _load_cached(scratch, fingerprint, n_disp, natoms_sc, progress):
    """Return {index: forces} for valid cached files, or wipe on mismatch.

    ANY untrusted metadata state (absent, corrupt, wrong version, mismatched
    fingerprint) purges every cached force file before the new fingerprint is
    installed -- otherwise metadata loss followed by an interrupted rerun
    could bless stale forces on shape alone (review finding 2).
    """
    import numpy as np
    fp_path = os.path.join(scratch, "fingerprint.json")
    os.makedirs(scratch, exist_ok=True)
    stored = None
    if os.path.isfile(fp_path):
        try:
            stored = json.load(open(fp_path))
        except (json.JSONDecodeError, OSError):
            stored = None
        if stored is not None and (
                stored.get("version") != FINGERPRINT_VERSION
                or stored.get("fingerprint") != fingerprint):
            _purge_scratch(scratch, progress,
                           "scratch fingerprint mismatch (structure/model/"
                           "settings changed)")
            stored = None
        elif stored is None:
            _purge_scratch(scratch, progress, "corrupt scratch metadata")
    elif any(n.startswith("forces_") for n in os.listdir(scratch)):
        _purge_scratch(scratch, progress,
                       "cached forces without fingerprint metadata")
    if stored is None:
        tmp = fp_path + ".tmp"
        with open(tmp, "w") as fh:
            json.dump({"version": FINGERPRINT_VERSION,
                       "fingerprint": fingerprint,
                       "n_displacements": n_disp}, fh)
        os.replace(tmp, fp_path)
        return {}
    cached = {}
    for index in range(n_disp):
        path = _force_path(scratch, index)
        if not os.path.isfile(path):
            continue
        try:
            arr = np.load(path)
        except (ValueError, OSError, EOFError):
            progress(f"  corrupt cached force file {os.path.basename(path)}: "
                     f"recomputing that displacement")
            os.remove(path)
            continue
        if arr.shape == (natoms_sc, 3) and np.isfinite(arr).all():
            cached[index] = arr
    return cached


# --- main entry ---------------------------------------------------------------

def compute_force_constants(atoms_relaxed, spec: CalculatorSpec, *,
                            supercell, delta=0.03, jobs=1,
                            worker_threads=1, scratch_dir,
                            progress=print) -> PhononResult:
    """Finite-displacement FC for the relaxed structure; see module docstring."""
    import numpy as np
    from irma.mlip.calculators import (_clamp_native_threads,
                                       canonicalize_spec)

    # clamp before canonicalize can import torch (see cli._cmd_build)
    _clamp_native_threads(spec.threads)

    # enforce pinned model identities HERE, not just in the CLI: a direct
    # API caller with a floating spec (e.g. pet-mad without @version) must
    # not fingerprint one upstream release while workers load another
    # (idempotent for already-canonical and non-floating specs)
    spec = canonicalize_spec(spec)
    # likewise freeze the dispatch interpreter: a registry edit during the
    # run must not split the fingerprint and the workers across envs (the
    # pin is an env var, inherited by spawn workers)
    from irma.mlip import envs
    envs.pin_interpreter_env(spec.potential)

    delta = float(delta)
    if not math.isfinite(delta) or delta <= 0:
        raise ValueError(f"delta must be positive and finite, got {delta!r}")
    if isinstance(jobs, bool) or not isinstance(jobs, int):
        raise ValueError(f"jobs must be an integer, got {jobs!r}")
    if jobs < 1:
        raise ValueError(f"jobs must be >= 1, got {jobs}")
    if isinstance(worker_threads, bool) or not isinstance(worker_threads,
                                                          int):
        raise ValueError(
            f"worker_threads must be an integer, got {worker_threads!r}")
    if worker_threads < 1:
        raise ValueError(
            f"worker_threads must be >= 1, got {worker_threads}")
    supercell = supercell_matrix([1.0, 1.0, 1.0], supercell) \
        if not isinstance(supercell, (int, float)) else supercell
    # (explicit dims re-validated; an Lmin scalar here is a caller bug)
    if isinstance(supercell, (int, float)):
        raise ValueError("compute_force_constants needs explicit supercell "
                         "dimensions; apply the Lmin rule via "
                         "supercell_matrix() first")

    t0 = time.perf_counter()
    phonon = _build_phonopy(atoms_relaxed, supercell)

    phonon.generate_displacements(distance=delta)

    supercells = phonon.supercells_with_displacements
    n_disp = len(supercells)
    natoms_sc = len(supercells[0].symbols)
    fingerprint = _fingerprint(atoms_relaxed, phonon, delta, supercell, spec)
    progress(f"  {n_disp} displacement(s), {natoms_sc} atoms/supercell, "
             f"jobs={jobs}")

    cached = _load_cached(scratch_dir, fingerprint, n_disp, natoms_sc, progress)
    todo = [i for i in range(n_disp) if i not in cached]
    if cached:
        progress(f"  reusing {len(cached)} cached force set(s), "
                 f"{len(todo)} to compute")

    def payload(i):
        sc = supercells[i]
        return (i, list(sc.symbols), np.asarray(sc.cell),
                np.asarray(sc.scaled_positions))

    forces = dict(cached)
    if todo and jobs <= 1:
        calc = make_calculator(spec)[0]
        from ase import Atoms
        for k, i in enumerate(todo, 1):
            _, symbols, cell, spos = payload(i)
            sc = Atoms(symbols=symbols, scaled_positions=spos, cell=cell,
                       pbc=True)
            sc.calc = calc
            forces[i] = sc.get_forces()
            _save_force(scratch_dir, i, forces[i])
            progress(f"  displacement {k}/{len(todo)} done")
    elif todo:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        from multiprocessing import get_context
        # 1 native thread per worker by default (the measured-safe
        # convention); --worker-threads widens each worker for machines
        # where jobs x threads < cores has headroom
        worker_spec = CalculatorSpec(spec.potential, model=spec.model,
                                     threads=worker_threads)
        pool = ProcessPoolExecutor(max_workers=jobs,
                                   mp_context=get_context("spawn"),
                                   initializer=_worker_init,
                                   initargs=(worker_spec,))
        futures = {pool.submit(_worker_forces, payload(i)): i for i in todo}
        try:
            for k, fut in enumerate(as_completed(futures), 1):
                try:
                    i, f = fut.result()
                except Exception as exc:
                    raise RuntimeError(
                        f"force evaluation failed for displacement "
                        f"{futures[fut]}: {exc}") from exc
                forces[i] = f
                _save_force(scratch_dir, i, f)
                progress(f"  displacement {k}/{len(todo)} done")
        except BaseException:
            # Cancel everything still queued -- without this, a worker error
            # or Ctrl-C waits out thousands of pending displacements before
            # surfacing (review finding 4). Completed forces are already on
            # disk, so the fingerprinted resume picks up exactly here.
            pool.shutdown(wait=False, cancel_futures=True)
            raise
        else:
            pool.shutdown(wait=True)

    phonon.forces = [forces[i] for i in range(n_disp)]
    phonon.produce_force_constants()

    # Raw-FC drift via phonopy's own diagnostic (both the translational and
    # the permutation/net-force components; a hand-rolled fc.sum() picks one
    # axis convention and misses the other -- review finding 7), then explicit
    # symmetrization (produce_force_constants alone does not symmetrize).
    from phonopy.harmonic.force_constants import get_drift_force_constants
    d1, d2, _, _ = get_drift_force_constants(phonon.force_constants,
                                             primitive=phonon.primitive)
    drift = max(abs(float(d1)), abs(float(d2)))
    before = phonon.force_constants.copy()
    phonon.symmetrize_force_constants()
    sym_delta = float(np.abs(phonon.force_constants - before).max())

    return PhononResult(
        phonon=phonon, n_displacements=n_disp, n_from_cache=len(cached),
        delta=delta, jobs=jobs, supercell=tuple(int(n) for n in supercell),
        wall_s=time.perf_counter() - t0, asr_drift_before=drift,
        symmetrization_delta=sym_delta, fingerprint=fingerprint)
