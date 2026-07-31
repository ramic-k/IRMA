"""Number-formatting layer: ``sigfig`` and ``_compute_endf_s``.

Every value IRMA writes into a tape passes through ``sigfig`` (NJOY's
Fortran-bit-compatible significant-figure rounding) and the S-values
additionally through ``_compute_endf_s`` (the isym/ilog storage transform).
A regression here corrupts every output file, so the contracts are pinned
explicitly.
"""
import numpy as np
import pytest

from irma.core.engine import sigfig, _compute_endf_s


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


# ---------------------------------------------------------------------------
# _compute_endf_s — the isym/ilog storage transform
# ---------------------------------------------------------------------------

@pytest.fixture
def sab_arrays():
    """Small synthetic ssm/ssp blocks [nbeta=4, nalpha=2, nt=1]."""
    ssm = np.array([[[0.31], [0.011]],
                    [[0.22], [0.012]],
                    [[0.13], [0.013]],
                    [[0.04], [0.014]]])
    ssp = ssm * 0.5
    return ssm, ssp


SMIN, SMALL = 1e-75, 1e-9


def test_symmetric_storage_applies_detailed_balance_factor(sab_arrays):
    """isym=0, ilog=0: stored value is S * exp(-beta/2) (downscatter side)."""
    ssm, ssp = sab_arrays
    be = 1.7
    got = _compute_endf_s(ssm, ssp, 0, 0, 2, 4, 0, 0, be, SMIN, SMALL)
    assert got == pytest.approx(ssm[1, 0, 0] * np.exp(-be / 2.0), rel=5e-7)


def test_symmetric_log_storage(sab_arrays):
    """isym=0, ilog=1: stored value is ln(S) - beta/2; S=0 -> -999 sentinel."""
    ssm, ssp = sab_arrays
    be = 1.7
    got = _compute_endf_s(ssm, ssp, 0, 1, 3, 4, 1, 0, be, SMIN, SMALL)
    assert got == pytest.approx(np.log(ssm[2, 1, 0]) - be / 2.0, rel=5e-7)

    zeros = np.zeros_like(ssm)
    got0 = _compute_endf_s(zeros, ssp, 0, 1, 1, 4, 0, 0, be, SMIN, SMALL)
    assert got0 == -999.0


def test_asymmetric_negative_beta_side_mirrors_ssm(sab_arrays):
    """isym=1, ii<nbeta: upscatter side read from mirrored ssm * exp(+beta/2)."""
    ssm, ssp = sab_arrays
    be = 0.9
    ii = 2                      # < nbeta -> i_beta = nbeta - ii = 2
    got = _compute_endf_s(ssm, ssp, 1, 0, ii, 4, 0, 0, be, SMIN, SMALL)
    assert got == pytest.approx(ssm[2, 0, 0] * np.exp(be / 2.0), rel=5e-7)


def test_asymmetric_positive_beta_side_reads_ssp(sab_arrays):
    """isym=1, ii>=nbeta: positive-beta side comes from ssp (cold hydrogen)."""
    ssm, ssp = sab_arrays
    be = 0.9
    ii = 5                      # >= nbeta -> i_beta = ii - nbeta = 1
    got = _compute_endf_s(ssm, ssp, 1, 0, ii, 4, 0, 0, be, SMIN, SMALL)
    assert got == pytest.approx(ssp[1, 0, 0] * np.exp(be / 2.0), rel=5e-7)


def test_isabt_storage_is_raw(sab_arrays):
    """isym=2 (isabt=1): SS is stored without any detailed-balance factor."""
    ssm, ssp = sab_arrays
    got = _compute_endf_s(ssm, ssp, 2, 0, 1, 4, 1, 0, 2.3, SMIN, SMALL)
    assert got == pytest.approx(ssm[0, 1, 0], rel=5e-7)
