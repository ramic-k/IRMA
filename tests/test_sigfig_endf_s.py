"""``sigfig``: NJOY's significant-figure rounding, applied to every value
IRMA writes into a tape (the S storage transform is in
test_endf_s_flag_matrix.py).
"""
import pytest

from irma.core.kernels import sigfig


# ---------------------------------------------------------------------------
# sigfig — NJOY util.f90 semantics
# ---------------------------------------------------------------------------

def test_sigfig_zero():
    assert sigfig(0.0, 7, 0) == 0.0


@pytest.mark.parametrize("x", [1.2345678e-8, 4.73918, 197.6285, 6.6e4, -2.53e-2])
def test_sigfig_7digits_is_near_identity(x):
    """7-significant-figure rounding must stay within 5e-7 relative."""
    assert sigfig(x, 7, 0) == pytest.approx(x, rel=5e-7)


def test_sigfig_rounds_to_ndig():
    assert sigfig(1.2345678, 3, 0) == pytest.approx(1.23, rel=1e-9)
    assert sigfig(1.2345678, 4, 0) == pytest.approx(1.235, rel=1e-9)
    assert sigfig(-1.2345678, 3, 0) == pytest.approx(-1.23, rel=1e-9)


def test_sigfig_idig_increments_last_digit():
    """idig adds units in the last kept digit (used for jittering grids)."""
    assert sigfig(1.23, 3, 1) == pytest.approx(1.24, rel=1e-9)
    assert sigfig(1.23, 3, -1) == pytest.approx(1.22, rel=1e-9)


def test_sigfig_rollover_carries_into_next_decade():
    """9.996 at 3 digits rounds to 10.0, not 1.00e+1-with-4-digits."""
    assert sigfig(9.996, 3, 0) == pytest.approx(10.0, rel=1e-9)


def test_sigfig_extreme_magnitude_guards():
    # |x| so small that the scaling power overflows -> 0.0
    assert sigfig(1e-310, 7, 0) == 0.0
    # |x| so large that the scaling power underflows -> x (bias only)
    assert sigfig(1e308, 7, 0) == pytest.approx(1e308, rel=1e-9)
