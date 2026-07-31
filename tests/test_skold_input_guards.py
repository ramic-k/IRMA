"""Skold S(kappa) zero/negative input guards.

irma/core/kernels.py:skold_approx + irma/core/deck.py Card 18: a zero
static structure factor (a physical between-Bragg null) must not divide
alpha to inf and write NaN into the law; a negative one must be rejected
at deck read.
"""
import os
import tempfile

import numpy as np
import pytest

from irma.core.deck import DeckError
from irma.core.engine import run_leapr


def test_skold_zero_structure_factor_stays_finite(monkeypatch):
    """S(kappa) == 0 (a physical between-Bragg null) must yield a finite law,
    not inf/NaN from alpha / S(kappa). Force terpk -> 0 so the division
    guard is hit deterministically."""
    import irma.core.kernels as kern
    monkeypatch.setattr(kern, "terpk", lambda *a, **k: 0.0)
    nalpha, nbeta, ntempr = 3, 2, 1
    alpha = np.array([0.1, 0.5, 1.0])
    beta = np.array([0.0, 1.0])
    ssm = np.ones((nbeta, nalpha, ntempr))
    ska = np.zeros(5)
    kern.skold_approx(ssm, alpha, beta, nalpha, nbeta, 0, ntempr, 1,
                      1.0, 12.0, 0.0253, ska, 5, 0.5, 0.5)
    assert np.all(np.isfinite(ssm)), "Skold with S(kappa)=0 produced a non-finite law"
    # coherent contribution collapses to its zero limit, so only the (1 - cfrac)
    # incoherent part of the original unit law survives
    assert np.allclose(ssm, 0.5)


# A minimal ortho-H deck that exercises Skold (ncold=1 + nsk=2), copied from
# tests/test_coldh_skold_minitape.py but with ONE S(kappa) value made negative.
_BAD_SKA_DECK = """20 /
'mini ortho-H deck with a negative S(kappa)'/
1 1 20/
3 1001. 0 0 1e-100/
.99917 20.43634 2 0 1 2/
0/
6 10/
1e-4 5e-4 0.0025 0.01 0.05 0.25/
0.0 0.5 1.0 2.0 4.0 7.0 11.0 16.0 22.0 30.0/
14.0/
0.0005 8/
0.0 0.4 0.9 1.0 0.8 0.5 0.2 0.0/
0.1104682205 1.124899936572044 0.3895317795/
1/
0.546/
0.166666666666/
12 0.05/
0.4 0.7 1.3 1.15 0.95 1.0 1.02 0.99 1.0 -1.0 1.0 1.0/
0.02144/
'mini coldh reference'/
/
"""


def test_deck_rejects_negative_skold_structure_factor():
    d = tempfile.mkdtemp()
    inp = os.path.join(d, "badska.input")
    out = os.path.join(d, "badska.endf")
    with open(inp, "w") as f:
        f.write(_BAD_SKA_DECK)
    with pytest.raises(DeckError, match=r"S\(kappa\).*>= 0"):
        run_leapr(inp, out)
