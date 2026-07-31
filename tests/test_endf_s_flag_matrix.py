"""Full (isym, ilog) flag matrix of the ENDF S storage transform.

``_compute_endf_s`` (irma/core/endf_writer.py) selects among eight storage
forms: symmetric S*exp(-be/2) / ln(S)-be/2 (isym=0), the cold-hydrogen
asymmetric S*exp(+be/2) / ln(S)+be/2 forms reading mirrored ssm for
ii<nbeta and ssp otherwise (isym=1), and the raw-SS forms with no
detailed-balance factor at all (isym=2 for -beta only, isym=3 for both
sides). Before this file only corners were pinned — no test ever executed
a '+ be/2' log branch, so a sign flip there (the headline regression risk)
would have shipped silently. Every cell below asserts a hand-computed
expectation (math.exp/math.log arithmetic on the literal array values, not
numbers re-derived from the implementation), plus the zero-S policy
(linear: 0.0 passthrough; log: the uniform -999 sentinel at every
temperature), the smin floor (linear-only), and the <small
6-significant-figure fallback on the ssp side.
"""
import math

import numpy as np
import pytest

from irma.core.endf_writer import _compute_endf_s

NBETA = 4
SMIN, SMALL = 1e-75, 1e-9
BE = 0.9


def _arrays():
    """ssm/ssp blocks [nbeta=4, nalpha=2, nt=1] with distinct entries."""
    ssm = np.array([[[0.31], [0.011]],
                    [[0.22], [0.012]],
                    [[0.13], [0.013]],
                    [[0.04], [0.014]]])
    return ssm, ssm * 0.5


def _call(ssm, ssp, isym, ilog, ii, j=0, be=BE):
    return _compute_endf_s(ssm, ssp, isym, ilog, ii, NBETA, j, 0,
                           be, SMIN, SMALL)


# ---------------------------------------------------------------------------
# isym=0 — symmetric S(a,b), ssm[ii-1]
# ---------------------------------------------------------------------------

def test_isym0_ilog0_stores_s_times_exp_minus_be_half():
    ssm, ssp = _arrays()
    got = _call(ssm, ssp, 0, 0, ii=2)           # ssm[1,0,0] = 0.22
    assert got == pytest.approx(0.22 * math.exp(-BE / 2.0), rel=5e-7)


def test_isym0_ilog0_below_smin_floors_to_zero():
    ssm, ssp = _arrays()
    ssm[0, 0, 0] = 1e-80                        # stored value < smin=1e-75
    assert _call(ssm, ssp, 0, 0, ii=1) == 0.0


def test_isym0_ilog1_stores_log_s_minus_be_half():
    ssm, ssp = _arrays()
    got = _call(ssm, ssp, 0, 1, ii=3)           # ssm[2,0,0] = 0.13
    assert got == pytest.approx(math.log(0.13) - BE / 2.0, rel=5e-7)


def test_isym0_ilog1_tiny_s_is_not_floored():
    # The smin floor is linear-storage-only: a very negative ln(S) must
    # pass through, not be zeroed (and not hit the zero-S sentinel).
    ssm, ssp = _arrays()
    ssm[0, 0, 0] = 1e-80
    got = _call(ssm, ssp, 0, 1, ii=1)
    assert got == pytest.approx(math.log(1e-80) - BE / 2.0, rel=5e-7)


# ---------------------------------------------------------------------------
# isym=1 — asymmetric S, cold hydrogen: the only forms with '+ be/2'
# ---------------------------------------------------------------------------

def test_isym1_ilog0_negative_beta_mirrors_ssm_with_exp_plus_be_half():
    ssm, ssp = _arrays()
    got = _call(ssm, ssp, 1, 0, ii=1)           # i_beta = nbeta-1 -> ssm[3]
    assert got == pytest.approx(0.04 * math.exp(BE / 2.0), rel=5e-7)


def test_isym1_ilog0_positive_beta_reads_ssp_with_exp_plus_be_half():
    ssm, ssp = _arrays()
    got = _call(ssm, ssp, 1, 0, ii=5)           # i_beta = ii-nbeta=1 -> ssp[1]
    assert got == pytest.approx(0.11 * math.exp(BE / 2.0), rel=5e-7)


def test_isym1_ilog1_negative_beta_is_log_s_PLUS_be_half():
    # Headline pin: the asymmetric log form adds be/2. A '-' here (copied
    # from the isym=0 form) would shift the result by be and ship a wrong
    # cold-hydrogen tape.
    ssm, ssp = _arrays()
    got = _call(ssm, ssp, 1, 1, ii=2)           # i_beta = nbeta-2 -> ssm[2]
    assert got == pytest.approx(math.log(0.13) + BE / 2.0, rel=5e-7)


def test_isym1_ilog1_positive_beta_is_log_ssp_PLUS_be_half():
    ssm, ssp = _arrays()
    got = _call(ssm, ssp, 1, 1, ii=6)           # i_beta = 2 -> ssp[2] = 0.065
    assert got == pytest.approx(math.log(0.065) + BE / 2.0, rel=5e-7)


def test_isym1_ilog0_small_ssp_takes_6_sigfig_fallback():
    # Stored value < small triggers sigfig(.,6,0): 1.2345678e-12 rounds to
    # 1.23457e-12; the 7-digit path would give 1.234568e-12 (rel ~1.6e-6
    # away), so rel=1e-7 discriminates the two branches. be=0 keeps the
    # exp factor at exactly 1 so the expectation stays hand-computable.
    ssm, ssp = _arrays()
    ssp[0, 0, 0] = 1.2345678e-12
    got = _call(ssm, ssp, 1, 0, ii=NBETA, be=0.0)
    assert got == pytest.approx(1.23457e-12, rel=1e-7)


# ---------------------------------------------------------------------------
# isym=2 — asymmetric SS, -beta only (isabt=1): raw, no be factor
# ---------------------------------------------------------------------------

def test_isym2_ilog0_stores_raw_ssm():
    ssm, ssp = _arrays()
    got = _call(ssm, ssp, 2, 0, ii=3, j=1)      # ssm[2,1,0] = 0.013
    assert got == pytest.approx(0.013, rel=5e-7)


def test_isym2_ilog1_stores_log_with_no_be_term():
    # be=0.9 in the call: any +/- be/2 contamination shifts by 0.45.
    ssm, ssp = _arrays()
    got = _call(ssm, ssp, 2, 1, ii=2)           # ssm[1,0,0] = 0.22
    assert got == pytest.approx(math.log(0.22), rel=5e-7)


# ---------------------------------------------------------------------------
# isym=3 — asymmetric SS, both sides: raw, no be factor
# ---------------------------------------------------------------------------

def test_isym3_ilog0_negative_beta_stores_raw_mirrored_ssm():
    ssm, ssp = _arrays()
    got = _call(ssm, ssp, 3, 0, ii=1)           # i_beta = nbeta-1 -> ssm[3]
    assert got == pytest.approx(0.04, rel=5e-7)


def test_isym3_ilog0_positive_beta_stores_raw_ssp():
    ssm, ssp = _arrays()
    got = _call(ssm, ssp, 3, 0, ii=6)           # i_beta = 2 -> ssp[2] = 0.065
    assert got == pytest.approx(0.065, rel=5e-7)


def test_isym3_ilog1_negative_beta_is_log_with_no_be_term():
    ssm, ssp = _arrays()
    got = _call(ssm, ssp, 3, 1, ii=3)           # i_beta = 1 -> ssm[1] = 0.22
    assert got == pytest.approx(math.log(0.22), rel=5e-7)


def test_isym3_ilog1_positive_beta_is_log_ssp_with_no_be_term():
    ssm, ssp = _arrays()
    got = _call(ssm, ssp, 3, 1, ii=5)           # i_beta = 1 -> ssp[1] = 0.11
    assert got == pytest.approx(math.log(0.11), rel=5e-7)


def test_isym3_ilog0_small_ssp_takes_6_sigfig_fallback():
    ssm, ssp = _arrays()
    ssp[0, 0, 0] = 1.2345678e-12                # raw, no exp factor at all
    got = _call(ssm, ssp, 3, 0, ii=NBETA)
    assert got == pytest.approx(1.23457e-12, rel=1e-7)


# ---------------------------------------------------------------------------
# Zero-S policy across the whole matrix (both sides of the asymmetric forms)
# ---------------------------------------------------------------------------

ZERO_CELLS = [
    (0, 1),            # isym=0
    (1, 1),            # isym=1, ssm side (ii < nbeta)
    (1, NBETA),        # isym=1, ssp side (ii >= nbeta)
    (2, 1),            # isym=2
    (3, 1),            # isym=3, ssm side
    (3, NBETA),        # isym=3, ssp side
]


@pytest.mark.parametrize("isym,ii", ZERO_CELLS)
def test_linear_zero_passes_through_as_zero(isym, ii):
    z = np.zeros((NBETA, 2, 1))
    assert _call(z, z, isym, 0, ii=ii) == 0.0


@pytest.mark.parametrize("isym,ii", ZERO_CELLS)
def test_log_zero_maps_to_minus999_sentinel(isym, ii):
    z = np.zeros((NBETA, 2, 1))
    assert _call(z, z, isym, 1, ii=ii) == -999.0
