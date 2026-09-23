"""Regression pins: discre delta-line nearest-grid-point binning (PHY-5).

The nearest-grid-point search for a discrete oscillator's delta line seeded
its running-minimum distance with db = 1000.0 as a stand-in for infinity
(NJOY2016 leapr.f90:1562 does the same). On a cryogenic lat=0 deck the beta
grid exceeds 1000 (beta_max = 5 eV/kT is ~2900 at 20 K), so a delta line at
be > 1000 ended the search on the very first comparison with jj = 0: the
numpy division wts[m]/betan[0] then produced inf (betan[0] == 0) deposited
into sexpb[-1] -- the WRONG bin at the far end of the grid. The fix seeds a
true infinity so the search always records the first distance; grids with
beta_max <= 1000 are bit-identical either way.

Reachability note: with the shipped cutoffs (the 1e-30 Bessel floor and
recursion overflow in bfact, the 1e-8 line-weight gate and maxdd=500 in
discre) the genuine oscillator machinery cannot place a surviving line above
beta ~730, so the first test injects the cryogenic line spectrum through a
bfact stub to exercise the search loop itself in the regime the sentinel
breaks. The other tests use the real machinery.

Expected values are computed from the intended semantics, not from the code:
the nearest grid point is argmin |be - betan[j]|, and the deposit is the
histogram density dwf * 2*w/(betan[t+1] - betan[t-1]) around that point
(dwf * w/betan[1] when the nearest point is betan[0] or betan[1]). Line
weights for the real-machinery tests come from the standard discrete-
oscillator formulas in the tiny-x limit, where the code's I1/I2 evaluations
reduce exactly to x/2 and x**2/8.
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


def test_cryo_delta_line_beyond_1000_lands_in_nearest_bin(monkeypatch):
    """A delta line at beta = 1056 on a 20 K lat=0 grid reaching 2900.

    Pre-fix, |1056 - betan[0]| > 1000 ended the search at j=0 and the line
    went to sexpb[-1] as inf (wts/betan[0]). The line spectrum is injected
    via a bfact stub because the genuine cutoffs cannot populate this
    regime (see module docstring); the search loop under test is exercised
    exactly as a real deck would."""
    w_line = 0.4

    def fake_bfact(x, dwc, betai):
        bminus = np.zeros(50)
        bminus[0] = w_line   # the be = bdeln line
        bminus[1] = 0.25     # sacrificial: the trim always drops one line
        return 0.5, np.zeros(50), bminus

    monkeypatch.setattr("irma.core.kernels.bfact", fake_bfact)

    tev = BK * 20.0
    bdeln = 1056.0
    beta = np.array([0.0, 150.0, 300.0, 450.0, 600.0, 750.0, 850.0, 950.0,
                     990.0, 1020.0, 1050.0, 1070.0, 1300.0, 1700.0, 2200.0,
                     2900.0])
    nbeta = len(beta)
    ssm = np.zeros((nbeta, 1))
    with np.errstate(divide="raise"):  # promote the pre-fix inf to an error
        dwpix, tempf = discre(ssm, np.array([2000.0]), beta, 1, nbeta, 0,
                              1.0, tev, 0.0, 0.5, 1, np.array([bdeln * tev]),
                              np.array([0.1]), 0.0, 20.0, 20.0)

    t, expect = _nearest_deposit(beta, bdeln, w_line)
    assert t == 10  # betan[10] = 1050 is the nearest point to 1056
    assert ssm[t, 0] == pytest.approx(expect, rel=1e-12)
    # nothing anywhere else: input law was zero, so the only deposit is the
    # delta line -- in particular NOT in the last bin (the pre-fix sink)
    rest = np.delete(ssm[:, 0], t)
    assert np.all(rest == 0.0)
    assert np.isfinite(dwpix) and np.isfinite(tempf)


def test_cryo_real_oscillator_on_grid_beyond_1000():
    """Real bfact machinery, cryogenic 20 K lat=0 deck, beta grid to 2900.

    A 0.0517 eV oscillator (bdeln = 60) at al = 2000 has line weights
    w2 > w1, so after the descending sort the trim drops the be = 60 line
    and the sole survivor is the two-phonon line at be = 120. It must land
    at the grid point nearest 120 with the intended density normalization,
    unaffected by the grid extending past 1000."""
    tev = BK * 20.0
    bdeln = 60.0
    al = 2000.0
    adel = 0.1
    beta = np.array([0.0, 50.0, 100.0, 115.0, 130.0, 300.0, 600.0, 900.0,
                     1200.0, 1800.0, 2400.0, 2900.0])
    nbeta = len(beta)
    ssm = np.zeros((nbeta, 1))
    dwpix, tempf = discre(ssm, np.array([al]), beta, 1, nbeta, 0, 1.0, tev,
                          0.0, 0.5, 1, np.array([bdeln * tev]),
                          np.array([adel]), 0.0, 20.0, 20.0)

    _, w1, w2 = _osc_terms(al, adel, bdeln)
    assert w2 > w1  # the trim keeps the two-phonon line
    t, expect = _nearest_deposit(beta, 2.0 * bdeln, w2)
    assert t == 3  # betan[3] = 115 is the nearest point to 120
    assert ssm[t, 0] == pytest.approx(expect, rel=1e-10)
    rest = np.delete(ssm[:, 0], t)
    assert np.all(rest == 0.0)
    assert np.isfinite(dwpix) and np.isfinite(tempf)


def test_ordinary_deck_delta_binning_unchanged():
    """Near-miss pin: an ordinary 296 K deck (beta_max = 25 << 1000) bins
    its delta line exactly as always -- every search distance is below the
    old 1000.0 seed, so the fix changes nothing here. Two oscillators; the
    trim drops the lighter be = 9.5 line and the be = 8 line survives."""
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
