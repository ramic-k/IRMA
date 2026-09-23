"""Worker kernels and process-pool plumbing for the noncubic inelastic engine.

- WORKER_STATE and set_worker_state: the compute state of the current stage.
  Workers are spawned once per run; per stage the engine stages the state in
  shared memory (share_worker_state) and every task carries only a small
  reference, which _dispatch_block attaches on first use. The serial path
  (ncpu = 1) reads WORKER_STATE directly.
- Thread pinning (limit_native_threads_to_one, _pool_worker_init).
- The star-averaged projections and the directional multiphonon precompute.
- The three accumulate_*_block kernels the engine maps over the pool.
"""
from __future__ import annotations

import math
import os

# One BLAS/OpenMP thread per process: the engine parallelizes across worker
# processes (Card 6f ncpu), and threaded BLAS in every worker oversubscribes the
# cores. IRMA_WORKER_THREADS overrides the count for tuning. The variables only
# bind if set before numpy loads.
NATIVE_THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
)


def _worker_thread_limit() -> int:
    """Per-process native-thread limit: IRMA_WORKER_THREADS, default 1."""
    try:
        return max(1, int(os.environ.get("IRMA_WORKER_THREADS", "1")))
    except ValueError:
        return 1


for _name in NATIVE_THREAD_ENV_VARS:
    os.environ.setdefault(_name, str(_worker_thread_limit()))

import numpy as np

from irma.core.constants import (
    BK as _BK_EV_PER_K,
    THZ_TO_EV as THzToEv,
)
from irma.core.noncubic_numerics import bincount_add

THz = 1000000000000.0


WORKER_STATE: dict = {}
BARN_PER_M2 = 1e28



def limit_native_threads_to_one() -> None:
    """Pin native thread pools to IRMA_WORKER_THREADS threads (default 1).

    Sets the environment variables (setdefault, so a caller's choice wins) and,
    when threadpoolctl is installed, clamps pools an earlier numpy import
    already started.
    """
    limit = _worker_thread_limit()
    for name in NATIVE_THREAD_ENV_VARS:
        os.environ.setdefault(name, str(limit))
    try:
        from threadpoolctl import threadpool_limits

        threadpool_limits(limits=limit)
    except Exception:
        pass


def _pool_worker_init() -> None:
    """Initializer of each spawned worker: force the thread limit.

    Unlike the parent, the worker overrides inherited values, so a large
    OMP_NUM_THREADS in the environment cannot oversubscribe the machine. The
    compute state arrives per task through _dispatch_block.
    """
    limit = _worker_thread_limit()
    for name in NATIVE_THREAD_ENV_VARS:
        os.environ[name] = str(limit)
    limit_native_threads_to_one()


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
    de_mev: float,
    max_order: int,
    positive_slice: slice,
    sigma0_emission_lookup: tuple[np.ndarray, np.ndarray, np.ndarray],
    sigma0_absorption_lookup: tuple[np.ndarray, np.ndarray, np.ndarray],
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

            emission_valid, emission_bins, emission_inv_widths = sigma0_emission_lookup
            absorption_valid, absorption_bins, absorption_inv_widths = sigma0_absorption_lookup
            bincount_add(row, emission_bins, emission_weights[emission_valid],
                         emission_inv_widths)
            bincount_add(row, absorption_bins, absorption_weights[absorption_valid],
                         absorption_inv_widths)

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

    return order_tables, order_tables_gain


def multiphonon_seed_area_deficit(base_prefactors, mode_eigvecs, emission_prefactors,
                                  absorption_prefactors, emission_valid, absorption_valid,
                                  thermal_mats) -> float:
    """Largest relative difference, over atoms, between the direction-averaged
    area of the multiphonon one-phonon seed and Tr(U_a)/3.

    The two agree to rounding when every mode lands inside the signed work grid
    and the seed uses the same modes as the Debye-Waller tensors; a gap means
    the multiphonon background loses area and cannot reach the free-atom limit.
    """
    weights = np.zeros(len(emission_prefactors), dtype=float)
    weights[emission_valid] += emission_prefactors[emission_valid]
    weights[absorption_valid] += absorption_prefactors[absorption_valid]
    e2 = np.sum(np.abs(mode_eigvecs) ** 2, axis=2)              # (modes, atoms)
    area = np.asarray(base_prefactors, dtype=float) * (weights @ e2) / 3.0
    msd = np.trace(np.asarray(thermal_mats, dtype=float), axis1=1, axis2=2) / 3.0
    mask = msd > 1.0e-12
    if not np.any(mask):
        return 0.0
    return float(np.max(np.abs(area[mask] - msd[mask]) / msd[mask]))


def set_worker_state(state: dict) -> None:
    """Set the worker state in the parent before each compute stage.

    The serial path reads it directly; pool workers receive it through
    share_worker_state and _dispatch_block.
    """
    global WORKER_STATE
    WORKER_STATE = state


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
    """Frequencies (THz) and eigenvectors for a batch of q-points.

    ``factor_to_thz`` is the model's own ``Phonopy.unit_conversion_factor``.
    It is required: the eigenvalue-to-THz factor depends on the calculator the
    force constants came from (vasp and qe differ, for example).
    """
    from phonopy.phonon.qpoints import QpointsPhonon
    qp = QpointsPhonon(qpoints, dynamical_matrix, with_eigenvectors=True,
                       factor=factor_to_thz)
    return qp.frequencies, qp.eigenvectors


def accumulate_coherent_block(indices: np.ndarray) -> np.ndarray:
    """Accumulate coherent one-phonon S(Q,E) and its diagonal/interference split.

    This is the exact harmonic coherent ``n=1`` term: we sum atom amplitudes
    first and square afterward. The diagonal and interference pieces are stored
    separately so the user can inspect the coherent decomposition explicitly.
    """

    state = WORKER_STATE
    q_grid = state["q_grid_ang_inv"]
    directions = state["directions"]
    direction_red_basis = state["direction_red_basis"]
    dynamical_matrix = state["dynamical_matrix"]
    frequency_factor_to_thz = state["frequency_factor_to_thz"]
    min_phonon_energy_mev = float(state.get("min_phonon_energy_mev", 0.0))
    thermal_mats = state["thermal_mats"]
    positions_t = state["positions_t"]
    coherent_atom_prefactors = state["coherent_atom_prefactors"]
    unit_conversion = state["unit_conversion"]
    temperature = state["temperature"]
    mev_to_joule = state["mev_to_joule"]
    one_phonon_creation_scale = state["one_phonon_creation_scale"]
    coherent_partition_mode = state["coherent_partition_mode"]
    coherent_group_site_indices = state["coherent_group_site_indices"]
    principal_group_index = state["principal_group_index"]
    group_coherent_weights = state["group_coherent_weights"]

    # Sample index = Q shell * n_dirs + direction; each direction has weight
    # 1/n_dirs in the powder average.
    n_dirs = len(directions)
    block_q_bins = indices // n_dirs
    direction_index = indices % n_dirs
    q_mags_block = q_grid[block_q_bins]
    q_red = q_mags_block[:, None] * direction_red_basis[direction_index]
    q_cart_block = q_mags_block[:, None] * directions[direction_index]
    unit_dirs_block = directions[direction_index]
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
    # One mode population for every term: with a user cutoff the coherent
    # term applies the mesh consumers' rule (floors plus cutoff, Gamma-aware
    # on the folded q); without one it keeps every positive mode, as before.
    from irma.core.phonopy_io import coherent_mode_mask
    valid_modes = coherent_mode_mask(frequencies * THzToEv * 1000.0, q_folded,
                                     min_phonon_energy_mev)          # (M, B)
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
    exponent = np.clip(
        frequencies[valid_modes] * THzToEv / (_BK_EV_PER_K * temperature),
        0.0, 700.0)
    bose_plus_one[valid_modes] = 1.0 / (-np.expm1(-exponent))

    energy_mev = frequencies * THzToEv * 1000.0
    direction_ok = q_mags_block >= 1e-12                            # (M,)

    # Energy loss deposits the n+1 (emission) lines at +E; the optional gain
    # side deposits the n (absorption) lines of the same structure factors at -E.
    sides = [(energy_mev, bose_plus_one, state["e_edges_mev"], state["e_bin_widths_mev"])]
    if state.get("emit_gain_side", False):
        bose_n = np.zeros_like(frequencies)
        bose_n[valid_modes] = 1.0 / np.expm1(exponent)
        sides.append((-energy_mev, bose_n, state["e_gain_edges_mev"],
                      state["e_gain_bin_widths_mev"]))
    stack = []
    for energies, occupancy, edges, widths in sides:
        bins = np.searchsorted(edges, energies, side="right").astype(np.intp) - 1
        bins[energies == edges[-1]] = len(widths) - 1
        good = (valid_modes & direction_ok[:, None]
                & (energies >= edges[0]) & (energies <= edges[-1]))
        rows = np.broadcast_to(block_q_bins[:, None], energies.shape)[good]
        cols = bins[good]
        density = (1.0 / n_dirs) / widths[cols]
        for weight in (total_w, diagonal_w, interference_w):
            acc = np.zeros((state["num_q"], len(widths)), dtype=float)
            line_weights = weight * occupancy * creation_prefactor_scale
            np.add.at(acc, (rows, cols), line_weights[good] * density)
            stack.append(acc)
    return np.stack(stack, axis=0)


def accumulate_incoherent_shell_block(shell_indices: np.ndarray) -> np.ndarray:
    """Accumulate the exact incoherent one-phonon self term over powder shells."""
    state = WORKER_STATE
    sqe_incoherent = np.zeros((state["num_q"], state["num_e"]), dtype=float)

    directions = state["directions"]
    q_grid = state["q_grid_ang_inv"]
    thermal_mats = state["thermal_mats"]
    physical_q_prefactors = state["incoherent_prefactors"]
    mesh_mode_energies_mev = state["mesh_mode_energies_mev"]
    mesh_mode_bose_prefactors = state["mesh_mode_bose_prefactors"]
    mesh_mode_eigvecs = state["mesh_mode_eigvecs"]
    hist_valid = state["hist_valid_indices"]
    hist_bins = state["hist_bin_indices"]
    hist_inv_widths = state["hist_inv_bin_widths"]

    # Energy-gain (annihilation) deposition, gated and byte-isolated from loss.
    emit_gain = state.get("emit_gain_side", False)
    if emit_gain:
        mesh_mode_absorption_prefactors = state["mesh_mode_absorption_prefactors"]
        e_gain_grid_mev = state["e_gain_grid_mev"]
        hist_gain_valid = state["hist_gain_valid_indices"]
        hist_gain_bins = state["hist_gain_bin_indices"]
        hist_gain_inv_widths = state["hist_gain_inv_bin_widths"]
        sqe_incoherent_gain = np.zeros((state["num_q"], len(e_gain_grid_mev)),
                                       dtype=float)

    projected_u2 = np.einsum("di,aij,dj->da", directions, thermal_mats, directions)

    for q_bin in shell_indices:
        q_mag = q_grid[q_bin]
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
        line_weights = mesh_mode_bose_prefactors * mode_weights
        if emit_gain:
            line_weights_gain = mesh_mode_absorption_prefactors * mode_weights

        if len(hist_bins) == 0:
            continue
        bincount_add(sqe_incoherent[q_bin], hist_bins, line_weights[hist_valid],
                     hist_inv_widths)
        if emit_gain and len(hist_gain_bins):
            bincount_add(sqe_incoherent_gain[q_bin], hist_gain_bins,
                         line_weights_gain[hist_gain_valid], hist_gain_inv_widths)

    if emit_gain:
        return np.stack((sqe_incoherent, sqe_incoherent_gain), axis=0)
    return sqe_incoherent


def accumulate_incoherent_multiphonon_direction_block(direction_indices: np.ndarray) -> np.ndarray:
    """Accumulate a direction block of the sigma_total multiphonon background."""
    state = WORKER_STATE
    emit_gain = state.get("emit_gain_side", False)
    directions = state["directions"][direction_indices]
    order_tables, order_tables_gain = precompute_directional_multiphonon_orders(
        directions,
        state["base_prefactors"],
        state["mesh_mode_energies_mev"],
        state["mesh_mode_emission_prefactors"],
        state["mesh_mode_absorption_prefactors"],
        state["mesh_mode_eigvecs"],
        state.get("mesh_mode_projection_components"),
        state["e_signed_grid_mev"],
        state["de_mev"],
        state["max_order"],
        state["positive_slice"],
        sigma0_emission_lookup=state["sigma0_emission_lookup"],
        sigma0_absorption_lookup=state["sigma0_absorption_lookup"],
        emit_gain_side=emit_gain,
        q2_max=float(np.max(state["q_grid_ang_inv"])) ** 2,
    )
    sqe_multiphonon_approx = np.zeros((state["num_q"], state["num_e"]), dtype=float)
    sqe_multiphonon_gain = np.zeros_like(sqe_multiphonon_approx) if emit_gain else None

    q_grid = state["q_grid_ang_inv"]
    thermal_mats = state["thermal_mats"]
    projected_u2 = np.einsum("di,aij,dj->da", directions, thermal_mats, directions, optimize=True)
    sigma_total_scale = state["multiphonon_sigma_total_scale"]
    total_num_dirs = state["num_total_dirs"]
    multiphonon_orders = order_tables[:, :, 1:, :]
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
    num_q_bins = len(q_grid)
    row_scale = np.where(q_grid >= 1.0e-12, 1.0 / total_num_dirs, 0.0)
    dao = kernel_matrix.shape[0]
    chunk_bins = max(1, 20_000_000 // max(1, dao))
    for q_start in range(0, num_q_bins, chunk_bins):
        q_stop = min(num_q_bins, q_start + chunk_bins)
        rows = slice(q_start, q_stop)
        q2 = q_grid[rows] ** 2                                   # (r,)
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
        contrib = term.reshape(n_rows, -1) @ k_mat
        sqe_multiphonon_approx[q_start:q_stop] += contrib
        if sqe_multiphonon_gain is not None:
            gain_contrib = term.reshape(n_rows, -1) @ k_gain
            sqe_multiphonon_gain[q_start:q_stop] += gain_contrib

    if sqe_multiphonon_gain is not None:
        return np.stack((sqe_multiphonon_approx, sqe_multiphonon_gain), axis=0)
    return sqe_multiphonon_approx
