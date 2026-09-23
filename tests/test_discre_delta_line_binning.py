"""discre delta-line binning: a discrete oscillator's surviving delta line
lands at the nearest beta grid point, with the histogram density
dwf * 2*w/(betan[t+1] - betan[t-1]). Line weights come from the standard
oscillator formulas in the tiny-x limit (I1 -> x/2, I2 -> x**2/8).
"""
from math import exp

import numpy as np
import pytest

from irma.core.kernels import discre
from irma.core.constants import BK


def _osc_terms(al, adel, bdeln):
    """Tiny-x discrete-oscillator factors from the standard formulas:
    weight of the n-th energy-loss line is exp(-dwc + n*bdeln/2) * In(x),
    with I1(x) -> x/2 and I2(x) -> x**2/8 as x -> 0, and the zero-phonon
    factor is exp(-dwc)."""
    eb = exp(bdeln / 2.0)
    sn = (eb - 1.0 / eb) / 2.0
    cn = (eb + 1.0 / eb) / 2.0
    ar = adel / (sn * bdeln)
    x = al * ar
    dwc = al * ar * cn
    bzero = exp(-dwc)
    w1 = exp(-dwc + bdeln / 2.0) * (x / 2.0)
    w2 = exp(-dwc + bdeln) * (x * x / 8.0)
    return bzero, w1, w2


def _nearest_deposit(betan, be, w):
    """Intended nearest-grid-point deposit: bin index and density value."""
    t = int(np.argmin(np.abs(betan - be)))
    if t == 0:
        # jj <= 1 branch with jj = t + 1 = 1
        return t, w / betan[1]
    return t, 2.0 * w / (betan[t + 1] - betan[t - 1])


def test_ordinary_deck_delta_binning_unchanged():
    """An ordinary 296 K deck with two oscillators: the trim drops the
    lighter be = 9.5 line and the be = 8 line lands on the nearest point."""
    tev = BK * 296.0
    bdeln = [8.0, 9.5]
    al = 1.0e-5
    adel = [0.1, 0.1]
    beta = np.array([0.0, 2.0, 4.0, 6.0, 7.5, 8.2, 9.0, 10.0, 12.0, 15.0,
                     20.0, 25.0])
    nbeta = len(beta)
    ssm = np.zeros((nbeta, 1))
    dwpix, tempf = discre(ssm, np.array([al]), beta, 1, nbeta, 0, 1.0, tev,
                          0.0, 0.8, 2, np.array([b * tev for b in bdeln]),
                          np.array(adel), 0.0, 296.0, 296.0)

    bz2, _, _ = _osc_terms(al, adel[1], bdeln[1])
    _, w1_first, _ = _osc_terms(al, adel[0], bdeln[0])
    w_line = w1_first * bz2  # be = 8 line through oscillator 2's zero phonon
    t, expect = _nearest_deposit(beta, bdeln[0], w_line)
    assert t == 5  # betan[5] = 8.2 is the nearest point to 8
    assert ssm[t, 0] == pytest.approx(expect, rel=1e-10)
    rest = np.delete(ssm[:, 0], t)
    assert np.all(rest == 0.0)
    assert np.isfinite(dwpix) and np.isfinite(tempf)
