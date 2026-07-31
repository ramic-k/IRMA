"""Multiphonon work-grid sizing must survive IRMA's own auto grids.

build_uniform_positive_work_grid previously fell back to the raw minimum
spacing for non-uniform output grids; an auto beta grid with log-spaced
tails (the default GUI grid converted to energy) made that fallback request
~1e10 bins (~140 GB) and the run died in the allocator. Non-uniform grids
are now bounded to MAX_MULTIPHONON_WORK_BINS (conservative rebinning keeps
the integral exact); uniform grids — every validated production grid — are
untouched.
"""
import numpy as np
import pytest

from irma.core.noncubic_engine import (
    build_uniform_positive_work_grid,
    MAX_MULTIPHONON_WORK_BINS,
)


def test_log_tailed_auto_grid_uses_median_not_min():
    # IRMA-auto-like grid: dense log tail (min spacing ~1e-5 meV) + linear
    # region up to 5 eV. The min spacing would need ~5e8 bins (above the cap),
    # so the smooth multiphonon work grid switches to the MEDIAN spacing
    # instead of the pathological log-thermal minimum.
    grid = np.unique(np.concatenate([
        np.geomspace(1.0e-5, 1.0, 300),
        np.linspace(1.0, 5000.0, 600),
    ]))
    work, spacing = build_uniform_positive_work_grid(grid)
    median_spacing = float(np.median(np.diff(grid)))
    assert spacing == pytest.approx(median_spacing)        # median, not min
    assert len(work) <= MAX_MULTIPHONON_WORK_BINS + 1
    assert len(work) < 5000                                # far below the cap
    # still a proper uniform grid from 0 covering the output range
    assert work[0] == 0.0
    assert work[-1] >= 5000.0 * 0.999
    assert np.allclose(np.diff(work), spacing)


def test_mildly_nonuniform_grid_keeps_minimum_spacing():
    # A non-uniform grid whose MINIMUM spacing already fits under the bin cap
    # keeps that minimum exactly — the median override only triggers for a
    # true log-thermal floor. This is what keeps the noncubic baselines
    # byte-identical.
    grid = np.unique(np.concatenate([
        np.linspace(0.9, 200.0, 220),       # ~0.9 meV spacing
        np.linspace(200.0, 5000.0, 200),    # coarser tail
    ]))
    min_spacing = float(np.min(np.diff(grid)))
    e_max = 5000.0
    # not pathological: min fits under the cap
    assert e_max / min_spacing <= MAX_MULTIPHONON_WORK_BINS
    work, spacing = build_uniform_positive_work_grid(grid)
    assert spacing == pytest.approx(min_spacing)           # min kept, not median


def test_uniform_grid_is_never_bounded():
    # A validated-path-style uniform grid keeps its own spacing exactly,
    # even when very fine.
    grid = np.arange(1.0, 5001.0, 1.0)          # dE = 1 meV, OCLIMAX-style
    work, spacing = build_uniform_positive_work_grid(grid)
    assert spacing == pytest.approx(1.0)
    fine = np.arange(0.5, 500.0, 0.01)          # 50k points, still uniform
    work_f, spacing_f = build_uniform_positive_work_grid(fine)
    assert spacing_f == pytest.approx(0.01)


def test_hard_backstop_raises_named_error():
    # An absurd request fails with a clear message instead of an OOM kill.
    grid = np.arange(0.0001, 10000.0, 0.0001)   # 100M-point uniform grid
    with pytest.raises(ValueError, match="hard limit"):
        build_uniform_positive_work_grid(grid)
