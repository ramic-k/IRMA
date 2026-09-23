"""Multiphonon work-grid sizing must survive IRMA's own auto grids.

For a non-uniform output grid (an auto beta grid with log-spaced tails)
build_uniform_positive_work_grid uses the phonon-region step, bounded to
MAX_MULTIPHONON_WORK_BINS; uniform grids keep their own spacing.
"""
import numpy as np
import pytest

from irma.core.noncubic_engine import (
    build_uniform_positive_work_grid,
    MAX_MULTIPHONON_WORK_BINS,
)


def _auto_like_grid(n_tail, n_phonon=300):
    # IRMA-auto-like grid in meV: dense log tail (min spacing ~1e-5 meV), a
    # linear phonon region to 200 meV in n_phonon equal steps, and n_tail
    # coarse points to 5 eV. The min spacing would need ~5e8 bins (above the
    # cap).
    return np.unique(np.concatenate([
        np.geomspace(1.0e-5, 1.0, 300),
        np.linspace(1.0, 200.0, n_phonon),
        np.linspace(200.0, 5000.0, n_tail),
    ]))


def test_log_tailed_auto_grid_spacing_is_the_phonon_region_step():
    # With the phonon maximum given, the work spacing is the step that
    # repeats across the output grid's phonon region, whatever the tail
    # carries: a tail with 20 points and one with 700 points (more points
    # than the phonon region has) get the same multiphonon resolution.
    step = 199.0 / 299.0
    work_a, spacing_a = build_uniform_positive_work_grid(
        _auto_like_grid(20), phonon_max_energy_mev=200.0)
    work_b, spacing_b = build_uniform_positive_work_grid(
        _auto_like_grid(700), phonon_max_energy_mev=200.0)
    assert spacing_a == pytest.approx(step)
    assert spacing_b == pytest.approx(step)
    assert len(work_a) <= MAX_MULTIPHONON_WORK_BINS + 1
    # still a proper uniform grid from 0 covering the output range
    assert work_a[0] == 0.0
    assert work_a[-1] >= 5000.0 * 0.999
    assert np.allclose(np.diff(work_a), spacing_a)
    # the deck's phonon subdivision sets the resolution: a coarser phonon
    # region gives a coarser work grid
    _, spacing_c = build_uniform_positive_work_grid(
        _auto_like_grid(20, n_phonon=200), phonon_max_energy_mev=200.0)
    assert spacing_c == pytest.approx(199.0 / 199.0)


def test_grid_without_a_repeated_spacing_falls_back_to_the_median():
    from irma.core.noncubic_numerics import phonon_region_spacing
    # a purely geometric grid has no phonon region: no spacing repeats
    grid = np.geomspace(1.0e-5, 5000.0, 700)
    assert phonon_region_spacing(grid, 200.0) is None
    from irma.core.noncubic_numerics import centers_to_edges
    _, spacing = build_uniform_positive_work_grid(grid, phonon_max_energy_mev=200.0)
    median = float(np.median(np.diff(grid)))
    floor = float(centers_to_edges(grid)[-1]) / MAX_MULTIPHONON_WORK_BINS
    assert spacing == pytest.approx(max(median, floor))


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


def test_hard_backstop_raises_named_error(monkeypatch):
    # An absurd request fails with a clear message instead of an OOM kill;
    # the limit is lowered so the test does not allocate a huge grid.
    from irma.core import noncubic_numerics
    monkeypatch.setattr(noncubic_numerics, "HARD_WORK_BIN_LIMIT", 1000)
    grid = np.arange(1.0, 10001.0, 1.0)         # 10k-point uniform grid
    with pytest.raises(ValueError, match="hard limit"):
        build_uniform_positive_work_grid(grid)


def test_phonon_region_is_the_first_run_not_the_most_common_spacing():
    from irma.core.noncubic_numerics import phonon_region_spacing
    # a deck whose freq_max (10 meV) is far below the model maximum (1000 meV):
    # the uniform tail below the model maximum has more points than the
    # phonon region, and must not win
    grid = np.unique(np.concatenate([
        np.geomspace(1.0e-5, 0.25, 15),
        np.linspace(0.25, 10.0, 40),          # phonon region, 0.25 meV
        np.arange(10.0, 5000.0, 12.65),       # uniform 0.5-beta tail
    ]))
    assert phonon_region_spacing(grid, 1000.0) == pytest.approx(0.25, rel=1e-6)
    # a phonon region of fewer steps than the threshold is not recognised:
    # the caller falls back to the median
    short = np.unique(np.concatenate([
        np.geomspace(1.0e-5, 25.0, 15), np.linspace(25.0, 200.0, 8),
        np.geomspace(200.0, 5000.0, 80)]))
    assert phonon_region_spacing(short, 200.0) is None
    # a nearly flat log segment whose spacings agree to six digits is not a
    # phonon region either
    flat = np.unique(np.concatenate([
        np.geomspace(1.0e-5, 10.0, 15), np.geomspace(10.0, 10.00001, 200),
        np.geomspace(10.00001, 5000.0, 80)]))
    assert phonon_region_spacing(flat, 200.0) is None


def test_phonon_region_is_found_on_a_grid_read_back_from_a_deck_file():
    """A deck carries six significant digits, so the phonon region's equal
    steps read back unequal in the last digit: the detector must still see
    one run. This is the graphite automatic grid as the engine gets it."""
    from irma.core.grids import AUTO_GRID_DEFAULTS as D, generate_beta_grid, grid_reference_temperature_K
    from irma.core.noncubic_numerics import phonon_region_spacing
    t_ref = grid_reference_temperature_K(1, 296.0)
    beta = generate_beta_grid(0.2007, t_ref, iint=1, awr=11.898,
                              n_lower=D["n_lower"], n_phonon=D["n_phonon"],
                              n_upper=D["n_upper"], beta_max_eV=5.0)
    kT_mev = 8.617333262e-5 * t_ref * 1.0e3
    written = np.array([float(f"{x:.6e}") for x in beta]) * kT_mev
    step = phonon_region_spacing(written, 200.692)
    assert step == pytest.approx(0.2007e3 / D["n_phonon"], rel=2e-3)
    # and a 0.25-beta tail above the phonon range (the cap at half the grid
    # temperature) does not change it
    fine = generate_beta_grid(0.2007, t_ref, iint=1, awr=11.898,
                              n_lower=D["n_lower"], n_phonon=D["n_phonon"],
                              n_upper=D["n_upper"], beta_max_eV=5.0,
                              evaluation_temperatures_K=[t_ref / 2.0, t_ref])
    written_fine = np.array([float(f"{x:.6e}") for x in fine]) * kT_mev
    assert phonon_region_spacing(written_fine, 200.692) == pytest.approx(step, rel=1e-6)
