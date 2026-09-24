"""irma.core.noncubic_numerics.centers_to_edges -- bin-edge inference.

Pins the lower-bound clamp: the first edge is clamped to a supplied
``lower_bound`` (a grid starting at 0 keeps a 0 first edge instead of going
negative), but a genuinely-negative grid is left alone when no bound is given.
Pure numpy -- runs in the bare-install gate too.
"""
import numpy as np
import pytest

from irma.core.noncubic_numerics import centers_to_edges


def test_uniform_centers_give_midpoint_edges():
    c = np.array([1.0, 2.0, 3.0, 4.0])
    e = centers_to_edges(c)
    assert e.size == c.size + 1
    assert np.allclose(e, [0.5, 1.5, 2.5, 3.5, 4.5])
    assert np.all(np.diff(e) > 0)


def test_nonuniform_centers_extrapolate_end_half_widths():
    c = np.array([0.0, 1.0, 3.0])            # widths 1, 2
    e = centers_to_edges(c)
    assert np.allclose(e, [-0.5, 0.5, 2.0, 4.0])


def test_lower_bound_clamps_first_edge_to_zero():
    c = np.array([0.0, 1.0, 2.0])            # first edge would be -0.5
    e = centers_to_edges(c, lower_bound=0.0)
    assert e[0] == 0.0                        # clamped, not negative
    assert np.allclose(e[1:], [0.5, 1.5, 2.5])
    assert np.all(np.diff(e) > 0)


def test_negative_grid_unclamped_without_bound():
    c = np.array([-3.0, -2.0, -1.0])
    e = centers_to_edges(c)                   # no bound -> first edge stays negative
    assert e[0] == pytest.approx(-3.5)


def test_requires_two_strictly_increasing_centers():
    with pytest.raises(ValueError):
        centers_to_edges(np.array([1.0]))
    with pytest.raises(ValueError):
        centers_to_edges(np.array([2.0, 1.0]))      # not increasing
    with pytest.raises(ValueError):
        centers_to_edges(np.array([1.0, 1.0, 2.0]))  # repeated center
