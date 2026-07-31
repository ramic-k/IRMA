"""F10: pin the isym=2/3 (asymmetric SS) zero-S passthrough in the ENDF
writer's _compute_endf_s.

For the mode-2 exact-zero case the law genuinely contains hard zeros. In
linear-S output (ilog=0) those must pass through as 0.0 (not the -999
ln-S sentinel, and not resurrected to exp(0)=1 as in the NJOY isym=0
additional-temperature bug the writer comments call out). In ln-S output
(ilog!=0) a zero maps to the -999 sentinel (below THERMR's sabflg floor).
"""
import numpy as np

from irma.core.endf_writer import _compute_endf_s

NBETA, NALPHA, NTEMP = 3, 2, 1
SMALL = 1.0e-30


def _ssm():
    a = np.zeros((NBETA, NALPHA, NTEMP))
    a[1, 0, 0] = 0.5            # one nonzero reference value
    return a


def test_isym2_linear_zero_passthrough():
    ssm = _ssm()
    # ii=1 -> i_beta=0 -> ssm[0,0,0] = 0.0
    s = _compute_endf_s(ssm, ssm, isym=2, ilog=0, ii=1, nbeta=NBETA,
                        j=0, nt=0, be=1.0, smin=SMALL, small=SMALL)
    assert s == 0.0


def test_isym2_linear_nonzero_passthrough():
    ssm = _ssm()
    s = _compute_endf_s(ssm, ssm, isym=2, ilog=0, ii=2, nbeta=NBETA,
                        j=0, nt=0, be=1.0, smin=SMALL, small=SMALL)
    assert abs(s - 0.5) < 1e-6


def test_isym2_log_zero_is_sentinel():
    ssm = _ssm()
    s = _compute_endf_s(ssm, ssm, isym=2, ilog=1, ii=1, nbeta=NBETA,
                        j=0, nt=0, be=1.0, smin=SMALL, small=SMALL)
    assert s == -999.0


def test_isym3_linear_zero_passthrough_both_sides():
    ssm = _ssm()
    ssp = np.zeros((NBETA, NALPHA, NTEMP))
    # -beta side: ii<nbeta -> i_beta=nbeta-ii from ssm; ii=1 -> ssm[2,0,0]=0
    s_minus = _compute_endf_s(ssm, ssp, isym=3, ilog=0, ii=1, nbeta=NBETA,
                              j=0, nt=0, be=1.0, smin=SMALL, small=SMALL)
    # +beta side: ii>=nbeta -> i_beta=ii-nbeta from ssp; ii=nbeta -> ssp[0,0,0]=0
    s_plus = _compute_endf_s(ssm, ssp, isym=3, ilog=0, ii=NBETA, nbeta=NBETA,
                             j=0, nt=0, be=1.0, smin=SMALL, small=SMALL)
    assert s_minus == 0.0 and s_plus == 0.0
