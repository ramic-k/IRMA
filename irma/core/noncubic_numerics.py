"""Low-level numerical primitives for the noncubic inelastic engine.

Array accumulation, sphere sampling and energy-grid construction and
rebinning, used by the setup path and the worker kernels. They do not touch
``WORKER_STATE``; ``noncubic_engine`` imports them for its own phases.
"""
from __future__ import annotations

import numpy as np

# Relative tolerance for treating a grid as uniform (grids written to 6-7
# significant figures carry round-trip noise well below it).
UNIFORM_GRID_RTOL = 2.5e-6

# A phonon region is a run of at least this many equal spacings below the
# highest phonon energy...
PHONON_REGION_MIN_REPEATS = 10
# ...equal to within this fraction of the run's first spacing.
PHONON_REGION_STEP_RTOL = 5.0e-3

# Bin cap for the work grid of a non-uniform output grid, whose minimum
# spacing (a log tail) can be far finer than the multiphonon term needs.
MAX_MULTIPHONON_WORK_BINS = 200_000
# Any work grid larger than this is an error rather than an out-of-memory kill.
HARD_WORK_BIN_LIMIT = 5_000_000


def fibonacci_sphere(num_points: int) -> np.ndarray:
    """Return approximately uniform unit vectors on a sphere."""
    if num_points < 2:
        return np.array([[0.0, 0.0, 1.0]], dtype=float)

    indices = np.arange(num_points, dtype=float)
    z = 1.0 - 2.0 * (indices + 0.5) / num_points
    phi = np.pi * (3.0 - np.sqrt(5.0)) * indices
    r_xy = np.sqrt(np.clip(1.0 - z * z, 0.0, None))
    x = r_xy * np.cos(phi)
    y = r_xy * np.sin(phi)
    return np.column_stack((x, y, z))


def precompute_histogram_lookup(
    energies_mev: np.ndarray,
    e_edges_mev: np.ndarray,
    e_bin_widths_mev: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Precompute bin indices and density factors for repeated histogram adds."""
    if len(e_bin_widths_mev) == 0 or len(energies_mev) == 0:
        return (
            np.empty(0, dtype=np.intp),
            np.empty(0, dtype=np.intp),
            np.empty(0, dtype=float),
        )

    bin_indices = np.searchsorted(e_edges_mev, energies_mev, side="right").astype(np.intp) - 1
    # A line exactly at the top edge belongs to the last bin (searchsorted
    # side="right" would otherwise index one past it and drop the line).
    bin_indices[energies_mev == e_edges_mev[-1]] = len(e_bin_widths_mev) - 1
    valid = (energies_mev >= e_edges_mev[0]) & (energies_mev <= e_edges_mev[-1])
    valid_indices = np.flatnonzero(valid).astype(np.intp, copy=False)
    valid_bins = bin_indices[valid]
    inv_bin_widths = 1.0 / e_bin_widths_mev[valid_bins]
    return valid_indices, valid_bins, inv_bin_widths


def bincount_add(
    row: np.ndarray,
    bin_indices: np.ndarray,
    weights: np.ndarray,
    inv_bin_widths: np.ndarray,
) -> None:
    """Accumulate histogram weights into a row using ``np.bincount``."""
    if len(bin_indices) == 0:
        return
    row += np.bincount(
        bin_indices,
        weights=weights * inv_bin_widths,
        minlength=len(row),
    )


def build_signed_energy_grid(e_grid_mev: np.ndarray) -> tuple[np.ndarray, slice]:
    """Mirror a non-negative energy grid to a signed grid.

    This helper is used for the *uniform* internal multiphonon work grid. The
    public output grid may be non-uniform; in that case the multiphonon stage
    works on a separate uniform grid and is rebinned afterward.
    """
    neg = -e_grid_mev[::-1]
    if abs(float(e_grid_mev[0])) < 1.0e-12:
        neg = neg[:-1]
    signed = np.concatenate((neg, e_grid_mev))
    positive_slice = slice(len(signed) - len(e_grid_mev), len(signed))
    return signed, positive_slice


def build_gain_output_grid(e_grid_mev: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The energy-GAIN output grid: the mirror of the loss grid's positive part.

    Returns ``(e_gain_mev, e_gain_edges_mev)`` where
    ``e_gain_mev = -e_grid_mev[e_grid_mev > 0][::-1]`` (strictly negative,
    ascending). This is the grid the direct energy-gain side S(Q, E<0) is
    deposited on for the NS bridge, chosen so it equals ``-E[E>0][::-1]`` of the
    loss output grid -- exactly what ``irma.spectra.sqe.signed_sqe`` expects of
    an attached direct gain side so the signed assembly meshes with the loss
    side. Excludes E=0 (the loss grid carries the zero column)."""
    e = np.asarray(e_grid_mev, dtype=float)
    pos = e[e > 0.0]
    if pos.size < 2:
        raise ValueError(
            "build_gain_output_grid needs at least two positive loss energies "
            "to mirror into a gain grid.")
    e_gain = -pos[::-1]
    return e_gain, centers_to_edges(e_gain)


def centers_to_edges(centers: np.ndarray, lower_bound: float | None = None) -> np.ndarray:
    """Construct bin edges from sorted bin centers."""
    centers = np.asarray(centers, dtype=float)
    if centers.ndim != 1 or centers.size < 2:
        raise ValueError("Need at least two bin centers to infer bin edges.")
    if np.any(np.diff(centers) <= 0.0):
        raise ValueError("Grid centers must be strictly increasing.")

    edges = np.empty(centers.size + 1, dtype=float)
    edges[1:-1] = 0.5 * (centers[:-1] + centers[1:])
    edges[0] = centers[0] - 0.5 * (centers[1] - centers[0])
    edges[-1] = centers[-1] + 0.5 * (centers[-1] - centers[-2])
    if lower_bound is not None:
        edges[0] = max(lower_bound, edges[0])
    if np.any(np.diff(edges) <= 0.0):
        raise ValueError("Inferred bin edges are not strictly increasing.")
    return edges


def infer_uniform_spacing_or_none(grid: np.ndarray) -> float | None:
    """Return the spacing for a nearly uniform grid, otherwise ``None``."""
    if len(grid) < 2:
        return None
    diffs = np.diff(grid)
    spacing = float(np.median(diffs))
    atol = max(1.0e-10, UNIFORM_GRID_RTOL * abs(spacing))
    if np.allclose(diffs, spacing, rtol=0.0, atol=atol):
        return spacing
    return None


def phonon_region_spacing(e_output_mev: np.ndarray,
                          phonon_max_energy_mev: float) -> float | None:
    """The output grid's phonon-region step, or None if it has none.

    An automatic beta grid is linear over the phonon range (``n_phonon``
    equal steps) between a log-spaced thermal tail below it and a high-beta
    tail above it. The phonon region is therefore the FIRST run of at least
    ``PHONON_REGION_MIN_REPEATS`` consecutive spacings equal to the run's
    first spacing within ``PHONON_REGION_STEP_RTOL``, below
    ``phonon_max_energy_mev``. The first run is used, not the most common
    spacing, so that a uniform high-beta tail below the phonon maximum is not
    taken for the phonon region. Runs with a step below 1e-5 of the phonon
    maximum (a nearly flat log segment) are ignored. Returns None when there
    is no such run or the phonon maximum is not positive.
    """
    if phonon_max_energy_mev is None or not float(phonon_max_energy_mev) > 0.0:
        return None
    e_max = float(phonon_max_energy_mev)
    grid = np.asarray(e_output_mev, dtype=float)
    region = grid[grid <= e_max * (1.0 + 1.0e-9)]
    diffs = np.diff(region)
    if diffs.size < PHONON_REGION_MIN_REPEATS:
        return None
    start = 0
    while start < diffs.size:
        first = diffs[start]
        stop = start + 1
        if first > 0.0:
            while (stop < diffs.size
                   and abs(diffs[stop] - first) <= PHONON_REGION_STEP_RTOL * first):
                stop += 1
        run = diffs[start:stop]
        if run.size >= PHONON_REGION_MIN_REPEATS and first >= 1.0e-5 * e_max:
            return float(np.median(run))
        start = stop
    return None


def build_uniform_positive_work_grid(
    e_output_mev: np.ndarray,
    phonon_max_energy_mev: float | None = None,
) -> tuple[np.ndarray, float]:
    """Build a uniform internal energy grid for the multiphonon convolutions.

    The first work-grid bin is centered at E=0 with its lower edge clamped
    to 0, so it is half a spacing wide and holds the positive-energy half of
    the mass around E=0. Rebinning onto the output grid conserves the
    integral.

    ``phonon_max_energy_mev`` is the highest phonon energy of the model. A
    non-uniform output grid uses its minimum spacing, unless that needs more
    than ``MAX_MULTIPHONON_WORK_BINS`` bins. Then the work spacing is the
    output grid's phonon-region step (``phonon_region_spacing``), so the
    deck's phonon subdivision sets the multiphonon resolution and tail points
    above the phonon range cannot change it; without a phonon region it is
    the median output spacing. The multiphonon term is smooth, so it needs the
    resolution of the one-phonon seed, not of the finest output bin.
    """
    spacing = infer_uniform_spacing_or_none(e_output_mev)
    e_max = float(centers_to_edges(e_output_mev)[-1])
    if spacing is None:
        diffs = np.diff(e_output_mev)
        min_spacing = float(np.min(diffs))
        if e_max / min_spacing > MAX_MULTIPHONON_WORK_BINS:
            floor = e_max / MAX_MULTIPHONON_WORK_BINS
            phonon_spacing = (phonon_region_spacing(e_output_mev, phonon_max_energy_mev)
                              if phonon_max_energy_mev is not None else None)
            if phonon_spacing is None:
                phonon_spacing = float(np.median(diffs))
            spacing = max(phonon_spacing, floor)
            print(
                f"Multiphonon work grid: spacing {spacing:.6g} meV, "
                f"{int(np.ceil(e_max / spacing))} bins (non-uniform output grid).",
                flush=True,
            )
        else:
            spacing = min_spacing
    n_bins = max(2, int(np.ceil(e_max / spacing)) + 1)
    if n_bins > HARD_WORK_BIN_LIMIT:
        raise ValueError(
            f"Multiphonon work grid would need {n_bins} bins "
            f"(e_max={e_max:.6g} meV, spacing={spacing:.6g} meV), above the "
            f"hard limit of {HARD_WORK_BIN_LIMIT}. The output energy grid is "
            f"too large or too finely spaced for the multiphonon convolution."
        )
    e_work_mev = np.arange(n_bins, dtype=float) * spacing
    return e_work_mev, spacing


def build_rebin_matrix(src_edges: np.ndarray, dst_edges: np.ndarray) -> np.ndarray:
    """Build a conservative density rebinning matrix from source to target bins."""
    n_src = len(src_edges) - 1
    n_dst = len(dst_edges) - 1
    matrix = np.zeros((n_dst, n_src), dtype=float)
    dst_widths = np.diff(dst_edges)
    for j in range(n_dst):
        dst_lo = dst_edges[j]
        dst_hi = dst_edges[j + 1]
        i_lo = max(0, np.searchsorted(src_edges, dst_lo, side="right") - 1)
        i_hi = min(n_src, np.searchsorted(src_edges, dst_hi, side="left"))
        for i in range(i_lo, i_hi):
            overlap = min(dst_hi, src_edges[i + 1]) - max(dst_lo, src_edges[i])
            if overlap > 0.0:
                matrix[j, i] = overlap / dst_widths[j]
    return matrix


def rebin_energy_axis(
    sqe_src: np.ndarray,
    src_edges_mev: np.ndarray,
    dst_edges_mev: np.ndarray,
) -> np.ndarray:
    """Rebin a density-like S(Q,E) map conservatively from one energy grid to another."""
    if np.array_equal(src_edges_mev, dst_edges_mev):
        return np.array(sqe_src, copy=True)
    matrix = build_rebin_matrix(src_edges_mev, dst_edges_mev)
    return sqe_src @ matrix.T
