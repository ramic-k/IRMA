"""Cold-temperature exp overflow (arguments above the float64 ceiling, ~709):
discre clamps to the cold coth limit, start() refuses with a clear error.
"""
import numpy as np
import pytest

from irma.core.kernels import start, discre
from irma.core.constants import BK


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
                          0.0, 0.0, 4.0)
    assert np.all(np.isfinite(ssm)), "discre wrote a non-finite law at cryogenic T"
    assert np.isfinite(dwpix) and np.isfinite(tempf)


def test_start_rejects_overflowing_cold_spectrum():
    """delta1=0.005 at 2 K -> deltab*(npt-1)/2 ~ 1435 >> 709: a clear error,
    not OverflowError or a silent NaN law."""
    p1 = np.ones(100)
    with pytest.raises(ValueError, match="overflow"):
        start(p1, 100, 0.005, BK * 2.0, 1.0)
