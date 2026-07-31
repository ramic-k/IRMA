"""Regression pins: cold-temperature exp overflow.

NJOY's Fortran exp() returns +Inf and continues; IRMA's math.exp raised
OverflowError ("math range error") and aborted. The fix:
  * discrete oscillator (discre): clamp the argument so the cold limit
    coth(bdeln/2) -> 1 is reproduced (the term has a clean finite limit);
  * continuous spectrum (start/fsum): a clamp there cascades to NaN, so raise a
    clear error up front instead.
Both only act above the float64 exp ceiling (~709), i.e. never for a physical
deck -- pinned byte-identical by the native-LEAPR / minitape reference tapes.
"""
from math import exp

import numpy as np
import pytest

from irma.core.kernels import _safe_exp, _EXP_MAX_ARG, start, discre
from irma.core.constants import BK


def test_safe_exp_byte_identical_below_threshold():
    for x in (-50.0, -1.0, 0.0, 1.0, 100.0, 500.0, 709.0):
        assert _safe_exp(x) == exp(x)  # exact no-op below the ceiling
    # above the ceiling: clamped to a finite value, never OverflowError
    assert np.isfinite(_safe_exp(1000.0))
    assert _safe_exp(1000.0) == exp(_EXP_MAX_ARG)


def test_discre_cold_high_energy_oscillator_stays_finite():
    """A high-energy oscillator (0.5 eV) at 4 K gives bdeln/2 ~ 725 > 709.
    Pre-fix: OverflowError. Now: finite (cold coth limit)."""
    nalpha, nbeta = 4, 3
    alpha = np.array([0.1, 0.5, 1.0, 2.0])
    beta = np.array([0.0, 1.0, 2.0])
    ssm = np.ones((nbeta, nalpha))
    tev = BK * 4.0
    dwpix, tempf = discre(ssm, alpha, beta, nalpha, nbeta, 1, 1.0, tev,
                          0.5, 1.0, 1, np.array([0.5]), np.array([0.1]),
                          0.0, 0.0, 4.0, 0)
    assert np.all(np.isfinite(ssm)), "discre wrote a non-finite law at cryogenic T"
    assert np.isfinite(dwpix) and np.isfinite(tempf)


def test_start_rejects_overflowing_cold_spectrum():
    """delta1=0.005 at 2 K -> deltab*(npt-1)/2 ~ 1435 >> 709: a clear error,
    not OverflowError or a silent NaN law."""
    p1 = np.ones(100)
    with pytest.raises(ValueError, match="overflow"):
        start(p1, 100, 0.005, BK * 2.0, 1.0)


def test_start_normal_spectrum_unaffected():
    """A warm spectrum stays well below the ceiling and runs normally."""
    p1 = np.ones(100)
    out = start(p1, 100, 0.005, BK * 296.0, 1.0)
    assert len(out) == 4
    p, f0, tbar, deltab = out
    assert np.isfinite(f0) and np.all(np.isfinite(p))
