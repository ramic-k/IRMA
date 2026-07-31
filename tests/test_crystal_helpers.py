"""Fractional-position matching (phonopy-site to Card 6d atom-type pairing).

_frac_pos_matches used min(d, 1-d) without first wrapping d into [0, 1):
for unwrapped coordinates more than one cell apart the second term goes
negative, so EVERY such position pair spuriously "matched" — silently
assigning phonopy sites to the wrong Card 6d atom type.
"""
from irma.core.crystal import _frac_pos_matches


def test_identical_positions_match():
    assert _frac_pos_matches((0.25, 0.5, 0.75), (0.25, 0.5, 0.75))


def test_periodic_images_match():
    assert _frac_pos_matches((0.75, 0.0, 0.0), (-0.25, 0.0, 0.0))
    assert _frac_pos_matches((1.25, 0.0, 0.0), (0.25, 0.0, 0.0))
    assert _frac_pos_matches((2.25, 0.5, -1.5), (0.25, 0.5, 0.5))


def test_wraparound_near_cell_boundary_matches():
    assert _frac_pos_matches((0.9999, 0.0, 0.0), (0.0001, 0.0, 0.0))


def test_distinct_positions_do_not_match():
    assert not _frac_pos_matches((0.25, 0.0, 0.0), (0.75, 0.0, 0.0))
    assert not _frac_pos_matches((0.5, 0.5, 0.5), (0.5, 0.5, 0.51))


def test_far_unwrapped_positions_do_not_match():
    """The regression: delta > 1 made 1-delta negative -> always 'matched'."""
    assert not _frac_pos_matches((2.75, 0.0, 0.0), (0.25, 0.0, 0.0))
    assert not _frac_pos_matches((-1.6, 0.0, 0.0), (0.0, 0.0, 0.0))
