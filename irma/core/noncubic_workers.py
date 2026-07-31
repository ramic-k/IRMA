"""Spawn-pool worker kernels + IPC for the noncubic inelastic engine.

Kept separate from noncubic_engine.py so that module stays navigable. This
module owns, and MUST keep co-located:
  * WORKER_STATE -- the per-process compute context -- and set_worker_state,
    which rebinds it in the PARENT before each compute phase. Workers are
    started with the SPAWN method (the one start method that exists on
    every platform, Windows included) in ONE pool reused across all
    phases: per phase the engine stages the state once (share_worker_state
    -- ndarrays as zero-copy multiprocessing.shared_memory blocks, the
    remainder pickled into one more block) and every task carries only a
    tiny state_ref; _dispatch_block attaches on the first task of a new
    generation and reuses the attachment for the rest of the phase. The
    serial ncpu=1 path reads the parent's WORKER_STATE directly.
  * limit_native_threads_to_one + the _pool_worker_init pool initializer
    (one BLAS/OMP thread per worker + warning-latch adoption),
  * the star-averaged projection + directional-multiphonon precompute helpers,
  * the sparse-block result compression / accumulation IPC, and
  * the three accumulate_*_block kernels the engine maps over the pool.

noncubic_engine re-imports these names, so its _cfa_* phases and existing
``from irma.core.noncubic_engine import ...`` callers are unaffected. Dependency
direction is one-way (workers -> constants / numerics leaf modules); there is no
import cycle with the engine.
"""
from __future__ import annotations

import math
import multiprocessing as _mp
import os
import sys

# Pin native thread pools BEFORE numpy (hence BLAS/OMP) imports -- the env-var
# clamp only binds if set first. Mirrors noncubic_engine's header and is
# idempotent (setdefault), so doing it in both modules is harmless. The pin
# target is 1 thread per process (the measured optimum for the
# process-parallel design); IRMA_WORKER_THREADS overrides it for tuning
# experiments -- spawned workers inherit the variable through the
# environment, so one setting governs parent and pool alike.


def _worker_thread_limit() -> int:
    """Per-process native-thread limit: IRMA_WORKER_THREADS, default 1."""
    try:
        return max(1, int(os.environ.get("IRMA_WORKER_THREADS", "1")))
    except ValueError:
        return 1


for _name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
):
    os.environ.setdefault(_name, str(_worker_thread_limit()))

import numpy as np

from irma.core.constants import (
    BK as _BK_EV_PER_K,
    THZ_TO_EV as THzToEv,
)
from irma.core.noncubic_numerics import (
    add_line_to_row,
    bincount_add,
    gaussian_add,
)

THz = 1000000000000.0


WORKER_STATE: dict = {}
NATIVE_THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
)
BARN_PER_M2 = 1e28

# Temperature (K) at or below which the Bose occupancy is forced to its exact
# T=0 limit (n=0, bose_plus_one=1). The expm1 branch is itself numerically safe
# at all T>0, so this guard is a convenience/labelling cutoff rather than a
# numerical-safety requirement: it short-circuits the occupancy evaluation for
# the degenerate near-0 K case. No production TSL run uses T<=1 K. Shared by the
# coherent worker and the context occupancy setup so the two stay in sync.
BOSE_T0_LIMIT_K = 1.0


def limit_native_threads_to_one() -> None:
    """Best-effort limit on native thread pools used inside worker processes.

    One thread per process is the measured optimum: the mode-1/2 paths
    parallelize across processes (Card 6f ncpu), and threaded BLAS inside
    every worker oversubscribes the machine (>11x slower in a 14-worker
    mode-2 run on a 16-core M4 Max). A jobs-vs-threads sweep on the
    graphite production pack bake on the same machine gives 14 procs x
    1 thread 27.1 s, 7 x 2 28.1 s, 1 x 14 46.6 s — BLAS threading cannot
    substitute for the block decomposition even at equal core count,
    because the kernels are many small operations rather than large
    matrix factorizations. The env variables only take effect if set
    before the pools initialize, so threadpoolctl — when available — clamps
    pools that were already spun up by an earlier numpy import; it is an
    optional dependency and silently skipped otherwise. The
    IRMA_WORKER_THREADS environment variable overrides the limit (default 1)
    for tuning experiments; the function name states the default, not the
    override.
    """
    limit = _worker_thread_limit()
    for name in NATIVE_THREAD_ENV_VARS:
        os.environ.setdefault(name, str(limit))
    try:
        from threadpoolctl import threadpool_limits

        threadpool_limits(limits=limit)
    except Exception:
        pass


def _pool_worker_init(warned_latch=None) -> None:
    """Initializer run once in each spawned worker process.

    Three jobs (the compute context itself does NOT come through here — it
    arrives per task via _dispatch_block, so the one long-lived pool can
    switch state between compute phases):

    1. Native-thread pinning. The parent pins via ``setdefault`` (preserving any
       deliberate user override in the parent), but a worker that inherited a
       large ``OMP_NUM_THREADS`` through the environment would oversubscribe
       the machine — pinning is process-local and a hard requirement for the
       process-parallel design, so here we OVERRIDE rather than setdefault
       (forcing the inherited values to the IRMA_WORKER_THREADS limit,
       default 1) and then clamp any already spun-up pools via threadpoolctl.
    2. Rebind ``sys.stdout``/``sys.stderr`` to the real interpreter streams.
       Defensive only: spawned workers start with fresh streams, but a
       programmatic embedder's sitecustomize may still have swapped
       ``sys.stdout`` for a non-picklable redirector.
    3. Adopt the parent's once-per-run warning latch. Deliberately NOT via
       set_worker_state — that would re-arm (zero) the latch, and a worker
       that initializes late could erase a warning an earlier worker already
       claimed. ``warned_latch`` is the parent's multiprocessing.Value,
       handed through the spawn reducer so once-per-run warning semantics
       survive without fork.
    """
    limit = _worker_thread_limit()
    for name in NATIVE_THREAD_ENV_VARS:
        os.environ[name] = str(limit)
    try:
        from threadpoolctl import threadpool_limits

        threadpool_limits(limits=limit)
    except Exception:
        pass
    if sys.__stdout__ is not None:
        sys.stdout = sys.__stdout__
    if sys.__stderr__ is not None:
        sys.stderr = sys.__stderr__
    global _KERNEL_AREA_WARNED
    if warned_latch is not None:
        _KERNEL_AREA_WARNED = warned_latch


def contract_real_symmetric_projection_components(
    projection_components: np.ndarray,
    direction: np.ndarray,
) -> np.ndarray:
    """Evaluate ``u^T G u`` from stored real-symmetric projection components.

    ``projection_components`` stores ``(xx, yy, zz, xy, xz, yz)`` for each
    mode/atom pair, where ``G`` is the star-averaged real-symmetric tensor
    corresponding to ``|u . e|^2``.
    """
    dx, dy, dz = np.asarray(direction, dtype=float)
    return (
        projection_components[..., 0] * (dx * dx)
        + projection_components[..., 1] * (dy * dy)
        + projection_components[..., 2] * (dz * dz)
        + 2.0 * projection_components[..., 3] * (dx * dy)
        + 2.0 * projection_components[..., 4] * (dx * dz)
        + 2.0 * projection_components[..., 5] * (dy * dz)
    )


def build_star_averaged_projection_components(
    full_mesh_eigvecs: np.ndarray,
    reduced_mesh,
    chunk_size: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    """Build exact star-averaged ``|u.e|^2`` tensors for an irreducible mesh.

    The returned tensor has shape ``(n_ir, n_branches, n_atoms, 6)`` storing the
    real-symmetric components ``(xx, yy, zz, xy, xz, yz)`` of the star-averaged
    outer product ``<e^* e^T>_star`` for each irreducible mode. Contracting that
    tensor with a real unit vector ``u`` reproduces the exact full-mesh star
    average of ``|u.e|^2`` without revisiting all symmetry-equivalent q-points.
    """
    mapping_table = getattr(reduced_mesh, "grid_mapping_table", None)
    ir_grid_points = getattr(reduced_mesh, "ir_grid_points", None)
    if mapping_table is None or ir_grid_points is None:
        raise ValueError(
            "Symmetry-reduced mesh is missing grid_mapping_table/ir_grid_points needed "
            "for star-averaged projection tensors."
        )

    mapping = np.asarray(mapping_table, dtype=np.intp)
    ir_grid_points = np.asarray(ir_grid_points, dtype=np.intp)
    if full_mesh_eigvecs.shape[0] != len(mapping):
        raise ValueError(
            "Full-mesh eigenvector count does not match reduced-mesh mapping table length."
        )

    rep_to_ir = np.full(len(mapping), -1, dtype=np.intp)
    rep_to_ir[ir_grid_points] = np.arange(len(ir_grid_points), dtype=np.intp)
    ir_index_for_full = rep_to_ir[mapping]
    if np.any(ir_index_for_full < 0):
        raise ValueError("Failed to map all full-mesh q-points onto irreducible representatives.")

    n_ir = len(ir_grid_points)
    n_branches = full_mesh_eigvecs.shape[1]
    n_atoms = full_mesh_eigvecs.shape[2]
    projection_components = np.zeros((n_ir, n_branches, n_atoms, 6), dtype=float)

    chunk_size = max(1, int(chunk_size))
    for start in range(0, full_mesh_eigvecs.shape[0], chunk_size):
        stop = min(full_mesh_eigvecs.shape[0], start + chunk_size)
        eig_block = np.asarray(full_mesh_eigvecs[start:stop], dtype=np.complex128)
        ex = eig_block[..., 0]
        ey = eig_block[..., 1]
        ez = eig_block[..., 2]
        block_components = np.empty((stop - start, n_branches, n_atoms, 6), dtype=float)
        block_components[..., 0] = np.abs(ex) ** 2
        block_components[..., 1] = np.abs(ey) ** 2
        block_components[..., 2] = np.abs(ez) ** 2
        block_components[..., 3] = np.real(np.conjugate(ex) * ey)
        block_components[..., 4] = np.real(np.conjugate(ex) * ez)
        block_components[..., 5] = np.real(np.conjugate(ey) * ez)
        np.add.at(projection_components, ir_index_for_full[start:stop], block_components)

    star_counts = np.bincount(ir_index_for_full, minlength=n_ir).astype(float)
    if np.any(star_counts <= 0.0):
        raise ValueError("Encountered an irreducible q-point with zero star multiplicity.")
    projection_components /= star_counts[:, None, None, None]
    return projection_components, star_counts


def precompute_directional_multiphonon_orders(
    directions: np.ndarray,
    base_prefactors: np.ndarray,
    mesh_mode_energies_mev: np.ndarray,
    mesh_mode_emission_prefactors: np.ndarray,
    mesh_mode_absorption_prefactors: np.ndarray,
    mesh_mode_eigvecs: np.ndarray,
    mesh_mode_projection_components: np.ndarray | None,
    e_signed_grid_mev: np.ndarray,
    e_signed_edges_mev: np.ndarray,
    e_signed_bin_widths_mev: np.ndarray,
    de_mev: float,
    sigma_mev: float,
    max_order: int,
    positive_slice: slice,
    sigma0_emission_lookup: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
    sigma0_absorption_lookup: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
    emit_gain_side: bool = False,
    q2_max: float | None = None,
) -> tuple[np.ndarray, np.ndarray, "np.ndarray | None"]:
    """Precompute per-direction, per-atom signed self orders at unit Q^2.

    The unit-Q signed ``p=1`` kernel is normalized so that integrating it over
    signed energy returns the directional mean-square displacement
    ``u_hat . U_d . u_hat``. This is the fixed-direction self object that is
    convolved before the Debye-Waller and cross-section factors are applied.

    Returns:
        order_tables: shape ``(n_dirs, n_atoms, max_order, n_e)`` unit-area
            signed self orders sliced to the positive-energy support.
        kernel_areas: shape ``(n_dirs, n_atoms)`` discrete signed area of each
            ``p=1`` kernel (= ``u_hat . U_d . u_hat`` by construction), used by
            ``_warn_kernel_area_mismatch`` to check consistency against the
            analytic directional MSD.
        order_tables_gain: ``None`` unless ``emit_gain_side``; then the same
            orders sliced to the ENERGY-GAIN window ``signed[0:start+1] =
            -e_work[::-1]`` (the negative half of the SAME signed convolution,
            for free) -- shape matches ``order_tables`` because the gain window
            length ``start+1`` equals the positive length ``n_e``.
    """
    n_dirs = len(directions)
    n_atoms = len(base_prefactors)
    n_orders = max_order
    n_e = positive_slice.stop - positive_slice.start
    # Gain window = the mirror (negative + zero) half of the signed grid:
    # signed[0:start+1] = -e_work[::-1]; same length as the positive window.
    gain_slice = slice(0, positive_slice.start + 1)
    n_e_signed = len(e_signed_grid_mev)
    order_tables = np.zeros((n_dirs, n_atoms, n_orders, n_e), dtype=float)
    order_tables_gain = (np.zeros((n_dirs, n_atoms, n_orders, n_e), dtype=float)
                         if emit_gain_side else None)
    # Discrete signed area of each p=1 kernel (= u_hat . U_d . u_hat by construction).
    # The kernels are stored *unit-area* (normalized by this), so that the per-order
    # weighting can be applied as a bounded Poisson(2W) factor downstream instead of the
    # over/underflow-prone (Q^2)^n/n! * (u^2)^n split. Returned for a consistency check
    # against the analytic directional MSD.
    kernel_areas = np.zeros((n_dirs, n_atoms), dtype=float)

    # Valid (direction, atom) kernels are collected and self-convolved in ONE
    # batched FFT recursion after the loops instead of one single-row FFT
    # call per kernel.
    pending_rows: list[np.ndarray] = []
    pending_dirs: list[int] = []
    pending_atoms: list[int] = []

    for idir, direction in enumerate(directions):
        if mesh_mode_projection_components is not None:
            mode_weights_by_atom = contract_real_symmetric_projection_components(
                mesh_mode_projection_components,
                direction,
            )
        else:
            projections = np.einsum(
                "mai,i->ma",
                mesh_mode_eigvecs,
                direction,
                optimize=True,
            )
            mode_weights_by_atom = np.abs(projections) ** 2
        mode_weights_by_atom *= base_prefactors[None, :]
        emission_weights_by_atom = mesh_mode_emission_prefactors[:, None] * mode_weights_by_atom
        absorption_weights_by_atom = mesh_mode_absorption_prefactors[:, None] * mode_weights_by_atom

        for atom_index in range(n_atoms):
            row = np.zeros(n_e_signed, dtype=float)
            emission_weights = emission_weights_by_atom[:, atom_index]
            absorption_weights = absorption_weights_by_atom[:, atom_index]

            if sigma_mev > 0.0:
                for energy_mev, weight in zip(mesh_mode_energies_mev, emission_weights):
                    if weight != 0.0:
                        add_line_to_row(
                            row,
                            float(energy_mev),
                            float(weight),
                            e_signed_grid_mev,
                            e_signed_edges_mev,
                            e_signed_bin_widths_mev,
                            sigma_mev,
                        )
                for energy_mev, weight in zip(mesh_mode_energies_mev, absorption_weights):
                    if weight != 0.0:
                        add_line_to_row(
                            row,
                            float(-energy_mev),
                            float(weight),
                            e_signed_grid_mev,
                            e_signed_edges_mev,
                            e_signed_bin_widths_mev,
                            sigma_mev,
                        )
            else:
                if sigma0_emission_lookup is None or sigma0_absorption_lookup is None:
                    raise ValueError("Missing precomputed sigma=0 histogram lookup for multiphonon kernels.")
                emission_valid, emission_bins, emission_inv_widths = sigma0_emission_lookup
                absorption_valid, absorption_bins, absorption_inv_widths = sigma0_absorption_lookup
                bincount_add(
                    row,
                    emission_bins,
                    emission_weights[emission_valid],
                    emission_inv_widths,
                )
                bincount_add(
                    row,
                    absorption_bins,
                    absorption_weights[absorption_valid],
                    absorption_inv_widths,
                )

            # Normalize the p=1 kernel to unit signed area BEFORE self-convolving, so
            # every order S_hat_n = (T1/area)^{*n} stays unit-area (never under/overflows).
            # The physical (Q^2)^n/n! * e^{-Q^2 a} * (u^2)^n weighting is reconstructed
            # downstream as the bounded Poisson(2W = Q^2 a) factor with these unit shapes.
            area = float(np.sum(row)) * de_mev
            kernel_areas[idir, atom_index] = area
            if not np.isfinite(area) or area <= 1.0e-30:
                # Stiff/zero-displacement direction for this atom: no multiphonon support.
                continue
            row_hat = row / area
            order_tables[idir, atom_index, 0] = row_hat[positive_slice]
            if order_tables_gain is not None:
                order_tables_gain[idir, atom_index, 0] = row_hat[gain_slice]
            pending_rows.append(row_hat)
            pending_dirs.append(idir)
            pending_atoms.append(atom_index)

    # Batched T1**n recursion: pocketfft transforms each row of a 2-D input
    # independently, so the per-row arithmetic — rfft(prev),
    # irfft(base_fft * prev_fft), *= de_mev, center slice — is the FFT
    # form of recursive_multiphonon_orders_no_factorial (the single-row
    # reference implementation above), with the FFT dispatch overhead
    # amortized across rows. FFT is the only viable form here: the ladder
    # runs one convolution per order for every (direction, atom) row, and
    # the auto-sized order count reaches ~1000 for soft-mode materials, so
    # direct O(n^2) convolutions would total ~10^6 kernel calls per run
    # (measured: ~85% of the phase's serial time on a 9-atom cell) where
    # the batched FFT needs a handful of pocketfft calls per order. Each
    # order is scattered straight into order_tables so no
    # (n_rows, max_order, n_e) intermediate exists.
    #
    # Per-row order horizon: max_order is auto-sized from the GLOBAL
    # 2W_max = Q_max^2 * U_max (order ~1000 for soft-mode materials like
    # beta-quartz), but each row's Poisson weights vanish above its OWN
    # horizon 2W_row = Q_max^2 * (u_hat . U_d . u_hat) — the directional
    # MSDs span ~20x, so most rows never need the global depth. Orders
    # above a row's horizon carry weights that underflow to EXACTLY 0.0 in
    # the deposit for every Q, so leaving those table entries zero changes
    # no output bit (fma(0, k, acc) == acc); the ladder therefore processes
    # rows sorted by horizon and shrinks the FFT batch as orders pass each
    # row's cutoff. Horizon per order n: the log-weight's maximum over
    # Q^2 in (0, q2_max] sits at w = min(n, 2W_row) and must clear the
    # float64 underflow line (-760 sits safely below the true ~-745.8).
    if pending_rows and max_order >= 2:
        rows_hat = np.array(pending_rows, dtype=float)        # (n_rows, n_e)
        dir_idx = np.asarray(pending_dirs, dtype=np.intp)
        atom_idx = np.asarray(pending_atoms, dtype=np.intp)
        if q2_max is not None and q2_max > 0.0:
            order_vec = np.arange(2, max_order + 1, dtype=float)
            log_fact = np.array(
                [math.lgamma(int(n) + 1) for n in order_vec], dtype=float)
            w_rows = q2_max * kernel_areas[dir_idx, atom_idx]    # 2W at Q_max
            w_best = np.minimum(order_vec[None, :], w_rows[:, None])
            with np.errstate(divide="ignore", invalid="ignore"):
                f_best = (order_vec[None, :] * np.log(w_best)
                          - log_fact[None, :] - w_best)
            alive = f_best >= -760.0                             # (rows, orders)
            row_horizon = np.where(
                alive.any(axis=1),
                2 + alive.shape[1] - 1 - np.argmax(alive[:, ::-1], axis=1),
                1,
            )
        else:
            row_horizon = np.full(len(dir_idx), max_order, dtype=int)
        horizon_order = np.argsort(-row_horizon, kind="stable")
        rows_hat = rows_hat[horizon_order]
        dir_idx = dir_idx[horizon_order]
        atom_idx = atom_idx[horizon_order]
        row_horizon = row_horizon[horizon_order]
        n_rows, n_points = rows_hat.shape
        n_full = 2 * n_points - 1
        start = (n_full - n_points) // 2
        n_fft = 1 << (n_full - 1).bit_length()
        base_fft = np.fft.rfft(rows_hat, n=n_fft, axis=-1)
        prev = rows_hat
        active = n_rows
        for order_index in range(2, max_order + 1):
            new_active = int(np.searchsorted(
                -row_horizon, -order_index, side="right"))
            if new_active == 0:
                break
            if new_active < active:
                prev = prev[:new_active]
                base_fft = base_fft[:new_active]
                active = new_active
            prev_fft = np.fft.rfft(prev, n=n_fft, axis=-1)
            full = np.fft.irfft(base_fft * prev_fft, n=n_fft, axis=-1)[:, :n_full]
            full *= de_mev
            prev = full[:, start:start + n_points]
            order_tables[dir_idx[:active], atom_idx[:active], order_index - 1] = (
                prev[:, positive_slice])
            if order_tables_gain is not None:
                order_tables_gain[dir_idx[:active], atom_idx[:active],
                                  order_index - 1] = prev[:, gain_slice]

    return order_tables, kernel_areas, order_tables_gain


# Warn-once latch for the kernel-area guard: a multiprocessing.Value
# shared with every pool worker through the spawn initializer (a plain
# module bool would be per-process), so the FIRST block that trips the
# guard warns for the whole run instead of once per worker process (every
# worker still validates the kernels it built — only the duplicate
# printing is suppressed).
# set_worker_state() re-arms it in the parent before each compute phase, so a
# second calculation in the same process gets the guard again. The Value MUST
# come from the spawn context explicitly: the default context is spawn on
# macOS but FORK on Linux, and a fork-context SemLock raises RuntimeError
# when pickled into a spawn worker (caught by Linux CI, invisible on macOS).
_KERNEL_AREA_WARNED = _mp.get_context("spawn").Value("b", 0)


def _warn_kernel_area_mismatch(
    kernel_areas: np.ndarray,
    analytic_msd: np.ndarray,
    rel_tol: float = 1.0e-5,
) -> None:
    """Warn once if the discrete multiphonon kernel area drifts from the analytic MSD.

    The p=1 self kernel is built from the same eigenvector/occupation contraction as the
    analytic mean-square displacement ``a = u_hat . U . u_hat`` (isotropic: ``trA/3``), so
    its discrete signed area equals ``a`` to ~1e-9 in normal operation. We reuse ``a`` as the
    Poisson mean ``2W = Q^2 a`` while normalizing the shape by the discrete area; because the
    two agree to machine precision, the per-order ``(a/area)^n`` mismatch is negligible even
    at order ~hundreds. A gross divergence (>> binning noise) means the signed work grid is
    too narrow to hold the one-phonon spectrum, so the kernel is losing area and the
    multiphonon zeroth moment cannot close to the free-gas limit. ``rel_tol=1e-5`` sits ~3.5
    decades above the observed floor (~4e-9): tight enough to flag real truncation, loose
    enough never to false-positive on deposition rounding. This is a guard, not a correction.

    Note on higher orders: only orders near the Poisson peak ``n ~ 2W`` carry weight, and that
    peak convolution is centered on the recoil energy ``Q^2 hbar^2/2M``. For any kinematically
    consistent (alpha, beta) grid the recoil for the largest Q lands inside the output/work
    grid, so the weight-bearing orders are contained; only the negligible super-peak tail
    spills past the grid edge (confirmed empirically: the result is invariant to a 3x larger
    work grid). Hence the p=1 area check is the meaningful guard.
    """
    if _KERNEL_AREA_WARNED.value:
        return
    analytic = np.asarray(analytic_msd, dtype=float)
    discrete = np.asarray(kernel_areas, dtype=float)
    mask = analytic > 1.0e-12
    if not np.any(mask):
        return
    rel = np.abs(discrete[mask] - analytic[mask]) / analytic[mask]
    worst = float(np.max(rel))
    if worst > rel_tol:
        with _KERNEL_AREA_WARNED.get_lock():
            if _KERNEL_AREA_WARNED.value:
                return
            _KERNEL_AREA_WARNED.value = 1
        print(
            f"WARNING: multiphonon kernel area vs analytic MSD differs by up to "
            f"{worst * 100:.2f}% in the first direction block that tripped this check "
            f"(tolerance {rel_tol:.0e} relative; other blocks may lose more): the signed "
            f"energy work grid, which is derived from the output energy grid, is too "
            f"narrow to hold the full one-phonon spectrum, so the multiphonon background "
            f"is losing area and cannot close to the free-gas limit. Extend the output "
            f"energy grid's maximum (spectra: grid.e_max_meV; decks: the beta grid) "
            f"above the highest phonon energy, or set the multiphonon order to 1 to "
            f"drop the multiphonon terms deliberately.",
            flush=True,
        )



def build_q_bin_sampling(
    q_grid_ang_inv: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Single radial sample per finite Q bin: its nominal |Q| with unit weight.

    Returns ``(mags, weights)`` shaped ``(num_q, 1)`` so the per-bin kernels keep
    their trailing radial-sample axis. (Multi-sample within-bin quadrature was
    governed by the ``incoherent_q_bin_samples`` knob; it was inert on the
    converged production grid -- N=5 vs N=1 shifted each bin integral by ~1e-6 --
    and has been removed.)
    """
    q = np.asarray(q_grid_ang_inv, dtype=float)
    q_bin_sample_mags = q.reshape(-1, 1).copy()
    q_bin_sample_weights = np.ones((len(q), 1), dtype=float)
    return q_bin_sample_mags, q_bin_sample_weights


def set_worker_state(state: dict) -> None:
    """Set global worker state for multiprocessing.

    Called in the PARENT before every compute phase. Also re-arms the
    once-per-run kernel-area warning latch, so each new calculation in the
    same process gets the guard again. Spawned pool workers receive their
    state through _pool_worker_init (shared-memory views + pickled
    remainder) together with the parent's latch object, so clearing the
    latch here is visible to them as well.
    """
    global WORKER_STATE
    WORKER_STATE = state
    _KERNEL_AREA_WARNED.value = 0


_STATE_GENERATION = 0


def share_worker_state(state: dict):
    """Stage a worker-state dict in shared memory for the spawn pool.

    Every ndarray member is copied once into a multiprocessing.shared_memory
    block (zero-copy to attach, one copy in RAM regardless of ncpu); the
    remainder (scalars, small containers, the phonopy dynamical matrix) is
    pickled into one more shared block. What travels per task is only the
    returned ``state_ref`` — (generation, array manifest, remainder block
    name, remainder length) — so ONE long-lived pool can switch state
    between compute phases without restarting workers: _dispatch_block
    re-attaches when it sees a new generation and reuses the cached
    attachment otherwise.

    Returns (state_ref, shm_handles); the caller must keep shm_handles alive
    while the phase runs and hand them to release_shared_state afterwards.
    Workers keep their own mappings until the next generation swap, so the
    parent unlinking right after the phase is safe on POSIX (mapping
    persists until close) and on Windows (unlink is a no-op).

    Zero-size and object-dtype arrays stay in the pickled remainder: the
    former cannot back a shared-memory block, the latter hold references
    that raw bytes cannot carry.
    """
    import pickle
    from multiprocessing import shared_memory

    global _STATE_GENERATION
    _STATE_GENERATION += 1
    shared_meta = {}
    small_state = {}
    shm_handles = []
    for key, value in state.items():
        if (isinstance(value, np.ndarray) and value.nbytes > 0
                and not value.dtype.hasobject):
            shm = shared_memory.SharedMemory(create=True, size=value.nbytes)
            staged = np.ndarray(value.shape, dtype=value.dtype, buffer=shm.buf)
            staged[...] = value
            shared_meta[key] = (shm.name, value.shape, value.dtype.str)
            shm_handles.append(shm)
        else:
            small_state[key] = value
    payload = pickle.dumps(small_state, protocol=pickle.HIGHEST_PROTOCOL)
    small_shm = shared_memory.SharedMemory(create=True, size=max(1, len(payload)))
    small_shm.buf[:len(payload)] = payload
    shm_handles.append(small_shm)
    state_ref = (_STATE_GENERATION, shared_meta, small_shm.name, len(payload))
    return state_ref, shm_handles


def release_shared_state(shm_handles) -> None:
    """Close and unlink the parent's shared-memory blocks after pool shutdown.

    unlink() marks the segment for removal on POSIX; on Windows it is a
    no-op and the segment disappears when the last handle closes. Errors are
    swallowed: a failed unlink leaks a name until interpreter exit, which is
    preferable to masking the computation result with a cleanup exception.
    """
    for shm in shm_handles:
        try:
            shm.close()
        except Exception:
            pass
        try:
            shm.unlink()
        except Exception:
            pass


# Attached segments in a worker process, replaced wholesale on each state
# generation swap. numpy views into a SharedMemory buffer do NOT keep the
# SharedMemory object alive; if it were garbage collected the buffer would be
# unmapped under the arrays' feet, so the handles live here for exactly as
# long as WORKER_STATE points into them.
_ATTACHED_SHM: list = []
_WORKER_GENERATION: int | None = None


def _attach_ro(name: str):
    """Attach a shared-memory block without resource tracking.

    The PARENT owns unlink. On Python >= 3.13 that is the track= keyword; on
    3.11/3.12 attaching registers with the resource tracker unconditionally,
    and since all pool workers share ONE tracker process, per-worker
    register/unregister pairs on the same name race (a KeyError inside the
    tracker) and leftovers draw spurious leak warnings — so there the
    registration call is suppressed around the attach instead, which is the
    same semantics track=False provides natively.
    """
    from multiprocessing import shared_memory

    try:
        return shared_memory.SharedMemory(name=name, track=False)
    except TypeError:
        from multiprocessing import resource_tracker

        orig_register = resource_tracker.register
        resource_tracker.register = lambda *a, **k: None
        try:
            return shared_memory.SharedMemory(name=name)
        finally:
            resource_tracker.register = orig_register


def _ensure_worker_state(state_ref) -> None:
    """Bind this worker's WORKER_STATE to the referenced state generation.

    Same generation as the last task: nothing to do (the common case — every
    task of a phase shares one generation). New generation: drop the previous
    phase's attachments, attach the new manifest as read-only views (the
    kernels only read state; a stray write would otherwise silently corrupt
    every other worker, so it raises instead), and unpickle the small
    remainder out of its shared block (pickle.loads copies, so that block is
    closed again immediately).
    """
    import pickle

    global _WORKER_GENERATION, WORKER_STATE
    generation, shared_meta, small_name, small_len = state_ref
    if generation == _WORKER_GENERATION:
        return
    WORKER_STATE = {}
    while _ATTACHED_SHM:
        try:
            _ATTACHED_SHM.pop().close()
        except Exception:
            pass
    small_shm = _attach_ro(small_name)
    try:
        state = pickle.loads(bytes(small_shm.buf[:small_len]))
    finally:
        small_shm.close()
    for key, (name, shape, dtype_str) in shared_meta.items():
        shm = _attach_ro(name)
        _ATTACHED_SHM.append(shm)
        view = np.ndarray(shape, dtype=np.dtype(dtype_str), buffer=shm.buf)
        view.flags.writeable = False
        state[key] = view
    WORKER_STATE = state
    _WORKER_GENERATION = generation


def _dispatch_block(state_ref, kernel, block):
    """Pool task entry: bind the phase's state, then run the block kernel.

    This is what the engine maps over the long-lived pool —
    ``partial(_dispatch_block, state_ref, kernel)`` pickles per task as just
    the tiny state_ref tuple plus the kernel's qualified name.
    """
    _ensure_worker_state(state_ref)
    return kernel(block)


_SPARSE_BLOCK_TAG = "sparse-block"
_SPARSE_BLOCK_DENSITY_CUTOFF = 0.10


def compress_block_result(arr: np.ndarray):
    """Ship only the nonzero cells of a worker block result.

    The pool returns block deposits to the parent by pickling; a dense
    accumulator array costs 8 bytes/cell even when the block touched a
    tiny fraction of the grid (a 2-direction coherent block on a
    10000-alpha grid deposits ~0.5% of cells but shipped 1.2 GB, leaving
    the parent the serial bottleneck while every worker idled). Shipping
    (index, value) pairs keeps the parent-side accumulation bitwise
    identical to the dense ``result += partial``: the same values are
    added to the same cells in the same block order, and skipped cells
    would have added exactly +0.0 (the accumulators never hold -0.0, so
    ``x + 0.0 == x`` bitwise). Results above the density cutoff — e.g.
    multiphonon order sums with broad energy support — ship unchanged,
    where compression would cost more than it saves.
    """
    flat = arr.reshape(-1)
    # density check with count_nonzero first (no allocation); only build the
    # nonzero-index array on the sparse path that actually ships it -- a dense
    # block (the common multiphonon case) skips the flatnonzero allocation.
    if np.count_nonzero(flat) > flat.size * _SPARSE_BLOCK_DENSITY_CUTOFF:
        return arr
    idx = np.flatnonzero(flat)
    return (_SPARSE_BLOCK_TAG, arr.shape, idx, flat[idx])


def accumulate_block_result(result: np.ndarray, partial) -> np.ndarray:
    """Add one worker block result (dense or compressed) into the sum."""
    if (isinstance(partial, tuple) and partial
            and partial[0] == _SPARSE_BLOCK_TAG):
        _, shape, idx, vals = partial
        if shape != result.shape:
            raise ValueError(
                f"sparse block shape {shape} does not match accumulator "
                f"shape {result.shape}")
        # On a non-C-contiguous accumulator, reshape(-1) returns a COPY and
        # the fancy-index add lands in that copy — every sparse-compressed
        # deposit would be silently dropped while dense blocks still
        # accumulate. All production initial accumulators are fresh
        # np.zeros, so this only fires on a caller-supplied view.
        if not result.flags['C_CONTIGUOUS']:
            raise ValueError(
                "accumulate_block_result requires a C-contiguous result "
                "array for sparse block deposits (got a non-contiguous "
                "view; pass a fresh or np.ascontiguousarray accumulator)")
        # flatnonzero indices are unique, so fancy-index add is exact
        result.reshape(-1)[idx] += vals
        return result
    result += partial
    return result


def build_q_vectors(
    q_bin_sample_mags: np.ndarray,
    q_bin_sample_weights: np.ndarray,
    directions: np.ndarray,
    rec_lat_no_2pi: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Convert sampled Q-shell points to reduced and physical coordinates.

    The sampling layout is flattened as:
    ``(shell index, radial sample index, direction index)``.
    """
    num_shells, num_radial_samples = q_bin_sample_mags.shape
    num_directions = len(directions)

    radial_mags = q_bin_sample_mags.reshape(-1)
    radial_weights = q_bin_sample_weights.reshape(-1)
    shell_index_per_radial = np.repeat(np.arange(num_shells, dtype=int), num_radial_samples)

    # Reduced-coordinate direction basis is independent of Q magnitude, so solve
    # the lattice system only once per direction instead of once per shell point.
    direction_red_basis = np.linalg.solve(rec_lat_no_2pi, directions.T).T / (2.0 * np.pi)

    q_red = (radial_mags[:, None, None] * direction_red_basis[None, :, :]).reshape(-1, 3)
    q_cart_physical = (radial_mags[:, None, None] * directions[None, :, :]).reshape(-1, 3)
    unit_dirs = np.broadcast_to(directions[None, :, :], (len(radial_mags), num_directions, 3)).reshape(-1, 3)
    q_mags_physical = np.repeat(radial_mags, num_directions)
    q_shell_index = np.repeat(shell_index_per_radial, num_directions)
    sample_weights = np.repeat(radial_weights / num_directions, num_directions)

    return q_red, q_shell_index, sample_weights, q_cart_physical, unit_dirs, q_mags_physical


def principal_weighted_coherent_partition(
    group_amplitudes: np.ndarray,
    principal_group_index: int,
    group_coherent_weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Approximate a principal-only coherent partial from group amplitudes.

    ``group_amplitudes`` may be shaped ``(n_groups, n_modes)`` or
    ``(n_qsamples, n_groups, n_modes)``. ``group_coherent_weights`` holds the
    per-group bound coherent cross-section weights (any common scale). The
    returned tuple is:

    - selected principal-group coherent contribution
    - exact principal self term ``|F_p|^2``
    - weighted principal/non-principal interference sum

    Each pairwise interference term ``2 Re(F_p F_o*)`` is shared between its
    two participants only, in proportion to their coherent weights: the
    principal keeps ``w_p / (w_p + w_o)`` of the (p, o) term. Summed over all
    principals every pair term regains coefficient exactly 1, so the
    per-species partials add back to the exact coherent total ``|sum_p F_p|^2``
    for any number of site groups. A single scalar share of the TOTAL cross
    term cannot do this beyond two groups: each pair would come back with
    coefficient ``w_p + w_o < 1`` and the partials would undercount the
    interference. For exactly two groups the pairwise factor reduces to the
    scalar ``sigma_coh`` share, bit-for-bit. In the degenerate
    all-zero-weight case the 0.5 fallback below is arbitrary by
    construction: zero coherent weight forces zero coherent amplitudes
    upstream (the prefactors scale with ``b_coh``), so the factor
    multiplies an exactly-zero term.
    """
    principal_amplitude = np.take(group_amplitudes, principal_group_index, axis=-2)
    principal_self = np.abs(principal_amplitude) ** 2
    if group_amplitudes.shape[-2] <= 1:
        return principal_self, principal_self, np.zeros_like(principal_self)

    weights = np.asarray(group_coherent_weights, dtype=float)
    principal_weight = float(weights[principal_group_index])
    weighted_cross = None
    for group_index in range(group_amplitudes.shape[-2]):
        if group_index == principal_group_index:
            continue
        other_amplitude = np.take(group_amplitudes, group_index, axis=-2)
        pair_term = 2.0 * np.real(principal_amplitude * np.conjugate(other_amplitude))
        pair_weight_sum = principal_weight + float(weights[group_index])
        # Zero-weight pairs carry zero amplitude (b_coh = 0 on both sides);
        # the 0.5 fallback only avoids a 0/0, it never contributes intensity.
        pair_factor = principal_weight / pair_weight_sum if pair_weight_sum > 0.0 else 0.5
        contribution = pair_factor * pair_term
        weighted_cross = contribution if weighted_cross is None else weighted_cross + contribution
    return principal_self + weighted_cross, principal_self, weighted_cross


def _batched_qpoints_eigh(dynamical_matrix, qpoints, factor_to_thz):
    """Frequencies (THz) + eigenvectors for a batch of q-points.

    Replicates phonopy's ``QpointsPhonon(..., with_eigenvectors=True)``
    arithmetic — the batched C dynamical-matrix solver, then eigh, then
    ``sqrt(|eigval|) * sign(eigval) * factor`` — with ONE stacked
    ``np.linalg.eigh`` over the (M, nb, nb) array instead of phonopy's
    per-q Python loop. LAPACK runs the same zheevd
    on the same matrices, so with pinned BLAS threads the results are
    bit-identical; only the per-q Python/dispatch overhead is removed.

    phonopy >= 4 removed ``run_dynamical_matrix_solver_c`` from
    ``phonopy.harmonic.dynamical_matrix`` (Rust backend); there the
    reference ``QpointsPhonon`` path is used instead — identical results,
    without the batching speedup.

    ``factor_to_thz`` is the MODEL's own frequency conversion factor
    (``Phonopy.unit_conversion_factor``), and is REQUIRED rather than
    defaulted. phonopy's dynamical matrix is built from force constants and a
    cell in the CALCULATOR's native units, so the eigenvalue -> THz factor is
    calculator-specific: 15.633302 for vasp/lammps/castep/aims/crystal/pwmat,
    108.970772 for qe, 21.490680 for abinit/siesta/abacus, 112.105157 for
    cp2k, 154.107943 for elk/fleur/TURBOMOLE/DFTB+, 3.445958 for wien2k. Both
    phonopy's own default and the constant this function used to hardcode are
    the vasp value, which silently rescaled the coherent one-phonon
    frequencies of every model built with another interface (the incoherent
    and multiphonon channels read phonopy's already-converted
    ``mesh.frequencies``, so the two halves of the kernel disagreed) — hence
    no default here.
    """
    try:
        from phonopy.harmonic.dynamical_matrix import (
            run_dynamical_matrix_solver_c,
        )
    except ImportError:
        from phonopy.phonon.qpoints import QpointsPhonon
        qp = QpointsPhonon(qpoints, dynamical_matrix, with_eigenvectors=True,
                           factor=factor_to_thz)
        return qp.frequencies, qp.eigenvectors

    dynmat = run_dynamical_matrix_solver_c(dynamical_matrix, qpoints, None)
    eigvals, eigvecs = np.linalg.eigh(dynmat)
    eigvals = eigvals.real
    frequencies = np.sqrt(np.abs(eigvals)) * np.sign(eigvals) * factor_to_thz
    return frequencies, eigvecs


def accumulate_coherent_block(indices: np.ndarray) -> "np.ndarray | tuple":
    """Accumulate coherent one-phonon S(Q,E) and its diagonal/interference split.

    This is the exact harmonic coherent ``n=1`` term: we sum atom amplitudes
    first and square afterward. The diagonal and interference pieces are stored
    separately so the user can inspect the coherent decomposition explicitly.
    """

    state = WORKER_STATE
    sqe_total = np.zeros((state["num_q"], state["num_e"]), dtype=float)
    sqe_diagonal = np.zeros_like(sqe_total)
    sqe_interference = np.zeros_like(sqe_total)

    q_red_all = state["q_red"]
    q_cart_all = state["q_cart_physical"]
    unit_dirs_all = state["unit_directions"]
    q_mags_all = state["q_mags_physical"]
    dynamical_matrix = state["dynamical_matrix"]
    frequency_factor_to_thz = state["frequency_factor_to_thz"]
    thermal_mats = state["thermal_mats"]
    positions_t = state["positions_t"]
    coherent_atom_prefactors = state["coherent_atom_prefactors"]
    unit_conversion = state["unit_conversion"]
    temperature = state["temperature"]
    q_shell_index = state["q_shell_index"]
    e_grid_mev = state["e_grid_mev"]
    e_edges_mev = state["e_edges_mev"]
    e_bin_widths_mev = state["e_bin_widths_mev"]
    sigma_mev = state["sigma_mev"]
    sample_weights = state["sample_weights"]
    mev_to_joule = state["mev_to_joule"]
    one_phonon_creation_scale = state["one_phonon_creation_scale"]
    coherent_partition_mode = state["coherent_partition_mode"]
    coherent_group_site_indices = state["coherent_group_site_indices"]
    principal_group_index = state["principal_group_index"]
    group_coherent_weights = state["group_coherent_weights"]

    q_red = q_red_all[indices]
    q_cart_block = q_cart_all[indices]
    unit_dirs_block = unit_dirs_all[indices]
    q_mags_block = q_mags_all[indices]
    # Parallelepiped fold to [-1/2, 1/2)^3 in reduced coordinates instead of
    # phonopy's Wigner-Seitz fold: any image choice
    # differs only by an integer G, and the explicit exp(2*pi*i G.r) phase
    # below compensates it exactly. The dynamical matrix is evaluated at an
    # equivalent-but-different q, so results differ at floating-point level
    # only; gated at ENDF tape precision (1e-6), like the rest of the
    # physics-metric-validated coherent path.
    q_folded = np.ascontiguousarray(q_red - np.rint(q_red))
    frequencies, eigvecs = _batched_qpoints_eigh(dynamical_matrix, q_folded,
                                                 frequency_factor_to_thz)
    assert eigvecs is not None

    g_vectors = q_red - q_folded
    projected_u2_block = np.einsum("di,aij,dj->da", unit_dirs_block, thermal_mats, unit_dirs_block)
    debye_waller_block = np.exp(-0.5 * (q_mags_block[:, None] ** 2) * projected_u2_block)
    phase_block = np.exp(2j * np.pi * (g_vectors @ positions_t))
    qpoints_num_bands = eigvecs.shape[2]
    # unit_conversion * mev_to_joule matches the Phonopy DSF-style THz-based
    # prefactor. one_phonon_creation_scale supplies the separate THz->meV
    # Jacobian and per-principal-site MT4 normalization.
    creation_prefactor_scale = (
        unit_conversion * mev_to_joule * BARN_PER_M2 * one_phonon_creation_scale
    )

    # --- Block-vectorized structure factor and line weights ----------------
    # The whole (M directions x nbands modes) grid is computed in bulk: the
    # former per-q Python loop spawned ~30 micro-numpy ops and re-found the
    # einsum contraction path on every q-point. Invalid modes (freq <= 0,
    # i.e. the Gamma Goldstone modes) and degenerate directions (|q| < 1e-12)
    # are carried through with zero weight rather than skipped, keeping every
    # array rectangular. The result equals the per-q computation to floating-
    # point roundoff (different summation order only); the coherent path is
    # validated against Euphonic by physics metrics, not byte identity.
    n_atoms_coh = eigvecs.shape[1] // 3
    valid_modes = frequencies > 0.0                                  # (M, B)
    # 1/sqrt(freq) with invalid modes -> 0 (zeros their amplitude exactly)
    inv_sqrt_freq = np.zeros_like(frequencies)
    np.divide(1.0, np.sqrt(frequencies, where=valid_modes,
                           out=np.ones_like(frequencies)),
              out=inv_sqrt_freq, where=valid_modes)

    eig = eigvecs.reshape(len(indices), n_atoms_coh, 3, qpoints_num_bands)
    # q . conj(e):  (M, atoms, bands); one path-find for the whole block
    q_dot_e = np.einsum("dc,dacb->dab", q_cart_block, np.conj(eig),
                        optimize=True)
    atom_amplitudes = (
        coherent_atom_prefactors[None, :, None]
        * debye_waller_block[:, :, None]
        * phase_block[:, :, None]
        * q_dot_e
        * inv_sqrt_freq[:, None, :]
    )                                                                # (M,A,B)

    if coherent_partition_mode == "exact-total":
        summed = atom_amplitudes.sum(axis=1)                         # (M, B)
        total_w = np.abs(summed) ** 2
        diagonal_w = (np.abs(atom_amplitudes) ** 2).sum(axis=1)
        interference_w = total_w - diagonal_w
    else:
        group_amplitudes = np.stack(
            [atom_amplitudes[:, group_indices, :].sum(axis=1)
             for group_indices in coherent_group_site_indices],
            axis=1,
        )                                                            # (M,G,B)
        total_w, diagonal_w, interference_w = (
            principal_weighted_coherent_partition(
                group_amplitudes, principal_group_index,
                group_coherent_weights,
            )
        )

    # Bose(+1) on valid modes only (invalid stay 0): evaluating the full
    # grid would divide by zero at the freq=0 Goldstone modes — a spurious
    # warning under default numpy, a hard error under seterr(divide="raise").
    bose_plus_one = np.zeros_like(frequencies)
    if temperature <= BOSE_T0_LIMIT_K:
        bose_plus_one[valid_modes] = 1.0
    else:
        exponent = np.clip(
            frequencies[valid_modes] * THzToEv / (_BK_EV_PER_K * temperature),
            0.0, 700.0)
        bose_plus_one[valid_modes] = 1.0 / (-np.expm1(-exponent))

    lw_total = total_w * bose_plus_one * creation_prefactor_scale    # (M, B)
    lw_diagonal = diagonal_w * bose_plus_one * creation_prefactor_scale
    lw_interference = interference_w * bose_plus_one * creation_prefactor_scale
    energy_mev = frequencies * THzToEv * 1000.0

    block_q_bins = q_shell_index[indices]                            # (M,)
    block_weights = sample_weights[indices]                         # (M,)
    direction_ok = q_mags_block >= 1e-12                            # (M,)

    # ENERGY-GAIN (annihilation) line weights, computed independently of the
    # loss `bose_plus_one` block so the loss weights/deposit stay byte-identical
    # (the ENDF byte-exact gate). Same structure factors total_w/diagonal_w/
    # interference_w and the same Bose-independent creation_prefactor_scale; only
    # the occupancy differs -- n(omega) (= 1/expm1(x)) for absorption vs n+1 for
    # emission -- and the lines deposit at -energy. T <= BOSE_T0_LIMIT_K: no
    # thermal phonons to absorb, bose_n = 0. Deposited AFTER the loss pass under
    # a single emit_gain guard (never interleaved with loss accumulation).
    emit_gain = state.get("emit_gain_side", False)
    if emit_gain:
        bose_n = np.zeros_like(frequencies)
        if temperature > BOSE_T0_LIMIT_K:
            exponent_gain = np.clip(
                frequencies[valid_modes] * THzToEv / (_BK_EV_PER_K * temperature),
                0.0, 700.0)
            bose_n[valid_modes] = 1.0 / np.expm1(exponent_gain)
        lw_total_gain = total_w * bose_n * creation_prefactor_scale
        lw_diagonal_gain = diagonal_w * bose_n * creation_prefactor_scale
        lw_interference_gain = interference_w * bose_n * creation_prefactor_scale
        gain_energy_mev = -energy_mev
        e_gain_grid_mev = state["e_gain_grid_mev"]
        e_gain_edges_mev = state["e_gain_edges_mev"]
        e_gain_bin_widths_mev = state["e_gain_bin_widths_mev"]
        num_e_gain = len(e_gain_grid_mev)
        sqe_total_gain = np.zeros((state["num_q"], num_e_gain), dtype=float)
        sqe_diagonal_gain = np.zeros_like(sqe_total_gain)
        sqe_interference_gain = np.zeros_like(sqe_total_gain)

    if sigma_mev > 0.0:
        # Non-histogram (Gaussian) deposition: rare, non-production. The bulk
        # weights above are reused; only the spread is per-line.
        for local_i in range(len(indices)):
            if not direction_ok[local_i]:
                continue
            modes = valid_modes[local_i]
            if not np.any(modes):
                continue
            q_bin = block_q_bins[local_i]
            sample_weight = block_weights[local_i]
            for energy, total, diagonal, interference in zip(
                energy_mev[local_i][modes],
                lw_total[local_i][modes],
                lw_diagonal[local_i][modes],
                lw_interference[local_i][modes],
                strict=True,
            ):
                gaussian_add(sqe_total, q_bin, float(energy), float(total),
                             e_grid_mev, e_edges_mev, e_bin_widths_mev,
                             sigma_mev, sample_weight)
                gaussian_add(sqe_diagonal, q_bin, float(energy), float(diagonal),
                             e_grid_mev, e_edges_mev, e_bin_widths_mev,
                             sigma_mev, sample_weight)
                gaussian_add(sqe_interference, q_bin, float(energy),
                             float(interference), e_grid_mev, e_edges_mev,
                             e_bin_widths_mev, sigma_mev, sample_weight)
        if emit_gain:
            for local_i in range(len(indices)):
                if not direction_ok[local_i]:
                    continue
                modes = valid_modes[local_i]
                if not np.any(modes):
                    continue
                q_bin = block_q_bins[local_i]
                sample_weight = block_weights[local_i]
                for energy, total, diagonal, interference in zip(
                    gain_energy_mev[local_i][modes],
                    lw_total_gain[local_i][modes],
                    lw_diagonal_gain[local_i][modes],
                    lw_interference_gain[local_i][modes],
                    strict=True,
                ):
                    gaussian_add(sqe_total_gain, q_bin, float(energy),
                                 float(total), e_gain_grid_mev, e_gain_edges_mev,
                                 e_gain_bin_widths_mev, sigma_mev, sample_weight)
                    gaussian_add(sqe_diagonal_gain, q_bin, float(energy),
                                 float(diagonal), e_gain_grid_mev, e_gain_edges_mev,
                                 e_gain_bin_widths_mev, sigma_mev, sample_weight)
                    gaussian_add(sqe_interference_gain, q_bin, float(energy),
                                 float(interference), e_gain_grid_mev,
                                 e_gain_edges_mev, e_gain_bin_widths_mev,
                                 sigma_mev, sample_weight)
            return compress_block_result(np.stack(
                (sqe_total, sqe_diagonal, sqe_interference,
                 sqe_total_gain, sqe_diagonal_gain, sqe_interference_gain), axis=0))
        return compress_block_result(
            np.stack((sqe_total, sqe_diagonal, sqe_interference), axis=0))

    # Histogram deposition (production path): one masked scatter per array
    # over the whole (M x B) grid instead of ~24000 np.add.at calls.
    bin_indices = np.searchsorted(e_edges_mev, energy_mev,
                                  side="right").astype(np.intp) - 1
    bin_indices[energy_mev == e_edges_mev[-1]] = len(e_bin_widths_mev) - 1
    good = (
        valid_modes
        & direction_ok[:, None]
        & (energy_mev >= e_edges_mev[0])
        & (energy_mev <= e_edges_mev[-1])
        & (bin_indices >= 0)
        & (bin_indices < len(e_bin_widths_mev))
    )
    if np.any(good):
        rows = np.broadcast_to(block_q_bins[:, None], energy_mev.shape)[good]
        cols = bin_indices[good]                       # in-range by `good`
        # index widths on the masked bins only — bin_indices can be
        # len(e_bin_widths) for an above-top-edge mode (dropped by `good`),
        # which would be out of bounds if indexed over the full grid.
        block_weights_b = np.broadcast_to(block_weights[:, None],
                                          energy_mev.shape)
        density = block_weights_b[good] / e_bin_widths_mev[cols]
        np.add.at(sqe_total, (rows, cols), lw_total[good] * density)
        np.add.at(sqe_diagonal, (rows, cols), lw_diagonal[good] * density)
        np.add.at(sqe_interference, (rows, cols), lw_interference[good] * density)

    if emit_gain:
        # Mirror histogram deposit at -energy onto the gain grid; structurally
        # identical to the loss scatter above (only the energies, edges and
        # arrays differ). Runs strictly after the loss accumulation.
        gbin = np.searchsorted(e_gain_edges_mev, gain_energy_mev,
                               side="right").astype(np.intp) - 1
        gbin[gain_energy_mev == e_gain_edges_mev[-1]] = len(e_gain_bin_widths_mev) - 1
        ggood = (
            valid_modes
            & direction_ok[:, None]
            & (gain_energy_mev >= e_gain_edges_mev[0])
            & (gain_energy_mev <= e_gain_edges_mev[-1])
            & (gbin >= 0)
            & (gbin < len(e_gain_bin_widths_mev))
        )
        if np.any(ggood):
            grows = np.broadcast_to(block_q_bins[:, None], gain_energy_mev.shape)[ggood]
            gcols = gbin[ggood]
            gblock_weights_b = np.broadcast_to(block_weights[:, None],
                                               gain_energy_mev.shape)
            gdensity = gblock_weights_b[ggood] / e_gain_bin_widths_mev[gcols]
            np.add.at(sqe_total_gain, (grows, gcols), lw_total_gain[ggood] * gdensity)
            np.add.at(sqe_diagonal_gain, (grows, gcols), lw_diagonal_gain[ggood] * gdensity)
            np.add.at(sqe_interference_gain, (grows, gcols),
                      lw_interference_gain[ggood] * gdensity)
        return compress_block_result(np.stack(
            (sqe_total, sqe_diagonal, sqe_interference,
             sqe_total_gain, sqe_diagonal_gain, sqe_interference_gain), axis=0))

    return compress_block_result(
        np.stack((sqe_total, sqe_diagonal, sqe_interference), axis=0))


def accumulate_incoherent_shell_block(shell_indices: np.ndarray) -> "np.ndarray | tuple":
    """Accumulate the exact incoherent one-phonon self term over powder shells."""
    state = WORKER_STATE
    sqe_incoherent = np.zeros((state["num_q"], state["num_e"]), dtype=float)

    directions = state["directions"]
    q_bin_sample_mags = state["q_bin_sample_mags"]
    q_bin_sample_weights = state["q_bin_sample_weights"]
    thermal_mats = state["thermal_mats"]
    physical_q_prefactors = state["incoherent_prefactors"]
    mesh_mode_energies_mev = state["mesh_mode_energies_mev"]
    mesh_mode_bose_prefactors = state["mesh_mode_bose_prefactors"]
    mesh_mode_eigvecs = state["mesh_mode_eigvecs"]
    e_grid_mev = state["e_grid_mev"]
    e_edges_mev = state["e_edges_mev"]
    e_bin_widths_mev = state["e_bin_widths_mev"]
    sigma_mev = state["sigma_mev"]
    hist_valid = state.get("hist_valid_indices")
    hist_bins = state.get("hist_bin_indices")
    hist_inv_widths = state.get("hist_inv_bin_widths")

    # Energy-gain (annihilation) deposition, gated and byte-isolated from loss.
    emit_gain = state.get("emit_gain_side", False)
    if emit_gain:
        mesh_mode_absorption_prefactors = state["mesh_mode_absorption_prefactors"]
        e_gain_grid_mev = state["e_gain_grid_mev"]
        e_gain_edges_mev = state["e_gain_edges_mev"]
        e_gain_bin_widths_mev = state["e_gain_bin_widths_mev"]
        hist_gain_valid = state.get("hist_gain_valid_indices")
        hist_gain_bins = state.get("hist_gain_bin_indices")
        hist_gain_inv_widths = state.get("hist_gain_inv_bin_widths")
        sqe_incoherent_gain = np.zeros((state["num_q"], len(e_gain_grid_mev)),
                                       dtype=float)

    projected_u2 = np.einsum("di,aij,dj->da", directions, thermal_mats, directions)

    for q_bin in shell_indices:
        for q_mag, radial_weight in zip(q_bin_sample_mags[q_bin], q_bin_sample_weights[q_bin]):
            if q_mag < 1e-12:
                continue

            debye_waller_sq = np.exp(-(q_mag**2) * projected_u2)
            orientation_tensors = np.einsum(
                "da,di,dj->aij",
                debye_waller_sq,
                directions,
                directions,
            ) / len(directions)

            mode_weights = np.zeros(len(mesh_mode_energies_mev), dtype=float)
            for atom_index, atom_prefactor in enumerate(physical_q_prefactors):
                weighted_tensor = (q_mag**2) * atom_prefactor * orientation_tensors[atom_index]
                mode_weights += np.einsum(
                    "mi,ij,mj->m",
                    np.conjugate(mesh_mode_eigvecs[:, atom_index, :]),
                    weighted_tensor,
                    mesh_mode_eigvecs[:, atom_index, :],
                    optimize=True,
                ).real
            line_weights = radial_weight * mesh_mode_bose_prefactors * mode_weights
            if emit_gain:
                line_weights_gain = (radial_weight
                                     * mesh_mode_absorption_prefactors * mode_weights)

            if sigma_mev > 0.0:
                for energy_mev, weight_barn in zip(mesh_mode_energies_mev, line_weights):
                    gaussian_add(
                        sqe_incoherent,
                        q_bin,
                        float(energy_mev),
                        float(weight_barn),
                        e_grid_mev,
                        e_edges_mev,
                        e_bin_widths_mev,
                        sigma_mev,
                        1.0,
                    )
                if emit_gain:
                    for energy_mev, weight_barn in zip(mesh_mode_energies_mev,
                                                       line_weights_gain):
                        gaussian_add(sqe_incoherent_gain, q_bin,
                                     float(-energy_mev), float(weight_barn),
                                     e_gain_grid_mev, e_gain_edges_mev,
                                     e_gain_bin_widths_mev, sigma_mev, 1.0)
            else:
                if hist_valid is None or hist_bins is None or hist_inv_widths is None:
                    raise ValueError("Missing precomputed sigma=0 histogram lookup for incoherent accumulation.")
                if len(hist_bins) == 0:
                    continue
                bincount_add(
                    sqe_incoherent[q_bin],
                    hist_bins,
                    line_weights[hist_valid],
                    hist_inv_widths,
                )
                if emit_gain and hist_gain_bins is not None and len(hist_gain_bins):
                    bincount_add(
                        sqe_incoherent_gain[q_bin],
                        hist_gain_bins,
                        line_weights_gain[hist_gain_valid],
                        hist_gain_inv_widths,
                    )

    if emit_gain:
        return compress_block_result(
            np.stack((sqe_incoherent, sqe_incoherent_gain), axis=0))
    return compress_block_result(sqe_incoherent)


def recursive_multiphonon_orders_no_factorial(
    one_phonon_row: np.ndarray,
    de_mev: float,
    max_order: int,
) -> list[np.ndarray]:
    """Build signed ``T1**n`` convolution rows without the harmonic ``1/n!``."""
    if max_order < 2:
        return []

    base = np.array(one_phonon_row, dtype=float, copy=False)
    prev = np.array(one_phonon_row, dtype=float, copy=True)
    n_points = len(base)
    orders: list[np.ndarray] = []
    n_full = 2 * n_points - 1
    use_fft = n_points >= 512
    if use_fft:
        n_fft = 1 << (n_full - 1).bit_length()
        base_fft = np.fft.rfft(base, n=n_fft)
    else:
        n_fft = 0
        base_fft = None

    for _order in range(2, max_order + 1):
        if use_fft:
            prev_fft = np.fft.rfft(prev, n=n_fft)
            full = np.fft.irfft(base_fft * prev_fft, n=n_fft)[:n_full]
        else:
            full = np.convolve(base, prev, mode="full")
        full *= de_mev
        start = (n_full - n_points) // 2
        current = full[start : start + n_points]
        orders.append(current)
        prev = current

    return orders


def accumulate_incoherent_multiphonon_direction_block(direction_indices: np.ndarray) -> "np.ndarray | tuple":
    """Accumulate a direction block of the sigma_total multiphonon background."""
    state = WORKER_STATE
    emit_gain = state.get("emit_gain_side", False)
    directions = state["directions"][direction_indices]
    order_tables, kernel_areas, order_tables_gain = precompute_directional_multiphonon_orders(
        directions,
        state["base_prefactors"],
        state["mesh_mode_energies_mev"],
        state["mesh_mode_emission_prefactors"],
        state["mesh_mode_absorption_prefactors"],
        state["mesh_mode_eigvecs"],
        state.get("mesh_mode_projection_components"),
        state["e_signed_grid_mev"],
        state["e_signed_edges_mev"],
        state["e_signed_bin_widths_mev"],
        state["de_mev"],
        state["sigma_mev"],
        state["max_order"],
        state["positive_slice"],
        sigma0_emission_lookup=state.get("sigma0_emission_lookup"),
        sigma0_absorption_lookup=state.get("sigma0_absorption_lookup"),
        emit_gain_side=emit_gain,
        q2_max=float(np.max(state["q_bin_sample_mags"])) ** 2,
    )
    collect_last = state.get("collect_last_order", False)
    if order_tables.shape[2] <= 1:
        # No multiphonon support (order<=1): return zeros of the same stacked
        # shape the populated path would. Stack order: approx, [last_order],
        # [gain] -- the engine unpacks by the flags it set.
        n_stack = 1 + int(collect_last) + int(emit_gain)
        if n_stack == 1:
            return compress_block_result(
                np.zeros((state["num_q"], state["num_e"]), dtype=float))
        return compress_block_result(
            np.zeros((n_stack, state["num_q"], state["num_e"]), dtype=float))

    sqe_multiphonon_approx = np.zeros((state["num_q"], state["num_e"]), dtype=float)
    sqe_last_order = np.zeros_like(sqe_multiphonon_approx) if collect_last else None
    sqe_multiphonon_gain = np.zeros_like(sqe_multiphonon_approx) if emit_gain else None

    q_bin_sample_mags = state["q_bin_sample_mags"]
    q_bin_sample_weights = state["q_bin_sample_weights"]
    thermal_mats = state["thermal_mats"]
    projected_u2 = np.einsum("di,aij,dj->da", directions, thermal_mats, directions, optimize=True)
    # Consistency check: the discrete kernel area must equal the analytic directional MSD
    # u_hat . U . u_hat (the Debye-Waller exponent base). A mismatch beyond binning noise
    # means the work grid is dropping kernel support and the zeroth moment cannot close.
    _warn_kernel_area_mismatch(kernel_areas, projected_u2)
    sigma_total_scale = state["multiphonon_sigma_total_scale"]
    total_num_dirs = state["num_total_dirs"]
    multiphonon_orders = order_tables[:, :, 1:, :]
    last_order_table = multiphonon_orders[:, :, -1, :] if sqe_last_order is not None else None
    # Per-order n>=2 weighting is the bounded Poisson term Poisson(2W; n) = (2W)^n/n! e^{-2W}
    # with mean 2W = Q^2 * (u_hat . U . u_hat), multiplying the unit-area self shapes built
    # by precompute. Because the shapes carry the (u^2)^n scaling, this is mathematically
    # identical to the exact (Q^2)^n/n! e^{-Q^2 u^2} (u^2)^n form, but every factor is now
    # representable: the Poisson log is <= 0, so it cannot overflow and needs no clip, and
    # the unit shapes cannot underflow. This lets the high-Q orders build the free-gas limit.
    n_orders = multiphonon_orders.shape[2]
    order_numbers = np.arange(2, 2 + n_orders, dtype=float)
    log_factorials = np.array(
        [math.lgamma(int(n) + 1) for n in order_numbers], dtype=float
    )

    # --- Batched deposit sweep ---------------------------------------------
    # The Poisson(2W; n) weights for a chunk of Q rows contract with the
    # per-(direction, atom, order) energy tables as one
    # (rows x d*a*o) @ (d*a*o x e) matrix product rather than one small
    # einsum per Q row: the contraction is ~10^12 MACs per block, so it
    # must run inside BLAS, not through ~num_q Python iterations. BLAS
    # chooses its own (d, a, o) summation order, so the result is
    # deterministic per run but pinned only to ~1e-13 relative — about 6
    # orders below the ENDF writer's 6-significant-figure rounding; the
    # regression gate for this path is tape-byte comparison (the fast-CI
    # baseline pins), not float64 identity. Rows with q ~ 0 or
    # non-positive quadrature weight carry zero row_scale and deposit
    # nothing.
    n_e_out = multiphonon_orders.shape[3]
    orders_c = np.ascontiguousarray(multiphonon_orders)
    kernel_matrix = orders_c.reshape(-1, n_e_out)
    last_kernel_matrix = (
        np.ascontiguousarray(last_order_table).reshape(-1, n_e_out)
        if last_order_table is not None
        else None
    )
    # Gain kernel matrix: the SAME (direction, atom, order) Poisson weighting
    # `term` (computed below) contracts with the gain-window orders, so the gain
    # side costs only a second matmul -- no recomputation of weights.
    gain_orders_c = (
        np.ascontiguousarray(order_tables_gain[:, :, 1:, :])
        if sqe_multiphonon_gain is not None
        else None
    )
    gain_kernel_matrix = (
        gain_orders_c.reshape(-1, n_e_out)
        if gain_orders_c is not None
        else None
    )
    num_q_bins, num_radial_samples = q_bin_sample_mags.shape
    flat_mags = q_bin_sample_mags.reshape(-1)
    flat_weights = q_bin_sample_weights.reshape(-1)
    row_scale = np.where(
        (flat_mags >= 1.0e-12) & (flat_weights > 0.0),
        flat_weights / total_num_dirs,
        0.0,
    )
    # Clean NaN and -inf to 0, but leave +inf as +inf: a genuine overflow must
    # reach the engine's finite-value gate (which raises with the offending
    # array named) instead of being silently clamped to finfo.max -- a huge but
    # FINITE value that passes the gate and lands as garbage in the tape.
    nan_kwargs = dict(nan=0.0, posinf=np.inf, neginf=0.0)
    dao = kernel_matrix.shape[0]
    chunk_bins = max(1, 20_000_000 // max(1, dao * num_radial_samples))
    for q_start in range(0, num_q_bins, chunk_bins):
        q_stop = min(num_q_bins, q_start + chunk_bins)
        rows = slice(q_start * num_radial_samples,
                     q_stop * num_radial_samples)
        q2 = flat_mags[rows] ** 2                                # (r,)
        two_w = q2[:, None, None] * projected_u2[None, :, :]     # (r,d,a)
        positive_2w = two_w > 0.0
        # Order window. A Poisson weight whose log falls below the float64
        # underflow line is EXACTLY 0.0 after exp(), and dropping exact
        # zeros from the matmul's sequential k-accumulation leaves every
        # partial sum bit-identical (fma(0, k, acc) == acc), so restricting
        # the contraction to the surviving order range changes nothing but
        # the work. For soft-mode materials the ladder is auto-sized by
        # 2W at the LARGEST Q (order ~1000), while low-Q chunks weight only
        # orders near their own small 2W — this window is where that
        # asymmetry stops costing dense matmul columns. Per order the log
        # term's maximum over the chunk's 2W range [w_min, w_max] sits at
        # w = clamp(n, w_min, w_max) (d/dw of n*ln(w) - w is n/w - 1), an
        # exact pointwise bound with no interval assumptions; -760 sits
        # safely below exp()'s true zero-underflow line (~-745.8).
        w_pos = two_w[positive_2w]
        if w_pos.size == 0:
            # Every weight in the chunk is exactly zero; the full matmul
            # would only add exact zeros.
            continue
        w_best = np.clip(order_numbers, float(w_pos.min()), float(w_pos.max()))
        keep = np.nonzero(
            order_numbers * np.log(w_best) - log_factorials - w_best
            >= -760.0)[0]
        if keep.size == 0:
            continue
        o_lo, o_hi = int(keep[0]), int(keep[-1])
        if o_lo == 0 and o_hi == n_orders - 1:
            osl = slice(None)
            k_mat = kernel_matrix
            k_gain = gain_kernel_matrix
        else:
            osl = slice(o_lo, o_hi + 1)
            k_mat = orders_c[:, :, osl, :].reshape(-1, n_e_out)
            k_gain = (gain_orders_c[:, :, osl, :].reshape(-1, n_e_out)
                      if gain_orders_c is not None else None)
        ln_2w = np.log(np.where(positive_2w, two_w, 1.0))
        log_term = (
            order_numbers[None, None, None, osl] * ln_2w[..., None]
            - log_factorials[None, None, None, osl]
            - two_w[..., None]
        )
        # Poisson log is bounded above by 0; no clip needed. Mask
        # stiff/zero-MSD pairs to zero (no multiphonon support, not a floor).
        term = np.where(positive_2w[..., None], np.exp(log_term), 0.0)
        term *= sigma_total_scale[None, None, :, None]
        term *= row_scale[rows][:, None, None, None]             # (r,d,a,o)
        n_rows = term.shape[0]
        contrib = np.nan_to_num(
            term.reshape(n_rows, -1) @ k_mat, **nan_kwargs
        ).reshape(q_stop - q_start, num_radial_samples, n_e_out).sum(axis=1)
        sqe_multiphonon_approx[q_start:q_stop] += contrib
        if (sqe_last_order is not None and last_kernel_matrix is not None
                and o_hi == n_orders - 1):
            # o_hi < n_orders - 1 would mean the last order's weights are
            # all exactly zero in this chunk: contribution exactly zero.
            last_contrib = np.nan_to_num(
                term[:, :, :, -1].reshape(n_rows, -1) @ last_kernel_matrix,
                **nan_kwargs,
            ).reshape(
                q_stop - q_start, num_radial_samples, n_e_out
            ).sum(axis=1)
            sqe_last_order[q_start:q_stop] += last_contrib
        if sqe_multiphonon_gain is not None and k_gain is not None:
            gain_contrib = np.nan_to_num(
                term.reshape(n_rows, -1) @ k_gain, **nan_kwargs
            ).reshape(q_stop - q_start, num_radial_samples, n_e_out).sum(axis=1)
            sqe_multiphonon_gain[q_start:q_stop] += gain_contrib

    # Stack order matches the early-return: approx, [last_order], [gain].
    stack = [sqe_multiphonon_approx]
    if sqe_last_order is not None:
        stack.append(sqe_last_order)
    if sqe_multiphonon_gain is not None:
        stack.append(sqe_multiphonon_gain)
    if len(stack) == 1:
        return compress_block_result(sqe_multiphonon_approx)
    return compress_block_result(np.stack(stack, axis=0))
