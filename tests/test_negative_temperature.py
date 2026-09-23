"""Negative-temperature (spectrum-reuse) convention in the LEAPR temperature loop.

In an NJOY LEAPR deck a NEGATIVE temperature card means "reuse the previous
temperature's scattering-law inputs (continuous phonon spectrum, translational
and oscillator data, Skold/cold-hydrogen data) unchanged, and only recompute
the law at the new |T|." The deck omits the detail block for those temperatures.

A deck using the negative-temperature shorthand must produce the same
inelastic law as a deck that spells the repeated detail block out with
positive temperatures, recomputed at each new |T|.

This is the exact convention used throughout the reference tsl-*.leapr decks
(graphite, Fe, Al, CH2), so it underpins the deck-level validation in
tests/native_LEAPR_NJOY_ENDF_validation/.
"""
import os
import tempfile

import numpy as np

from irma.core.engine import run_leapr

# A tiny classic (iel=0) deck. Two temperatures: 300 K then 400 K. The two
# variants differ ONLY in how the 400 K detail block is supplied.
_HEAD = """20 /
'negtemp unit deck'/
2 1 4/
1 1./
1.0 20.0 1 0 0/
0/
3 4 1/
0.05 1.0 8.0/
0.0 0.6 2.0 6.0/
300/
0.005 6/
0.0 0.20 0.45 0.55 0.30 0.0/
0. 0. 1./
0/
"""

# Variant A: 400 K given as a negative card (reuse the 300 K block).
_DECK_NEG = _HEAD + """-400/
' reuse-via-negative-temperature deck'/
/
"""

# Variant B: 400 K given as a positive card with the SAME detail block repeated.
_DECK_EXPLICIT = _HEAD + """400/
0.005 6/
0.0 0.20 0.45 0.55 0.30 0.0/
0. 0. 1./
0/
' explicit repeated-block deck'/
/
"""


def _run(deck_text):
    d = tempfile.mkdtemp()
    inp = os.path.join(d, "deck.input")
    out = os.path.join(d, "deck.endf")
    with open(inp, "w") as f:
        f.write(deck_text)
    run_leapr(inp, out)
    return out


def _mt4_S_all_temps(path):
    """Return list of (nbeta, nalpha) S arrays, one per temperature."""
    from endf_parserpy import EndfParserPy
    p = EndfParserPy(ignore_number_mismatch=True, ignore_zero_mismatch=True,
                     ignore_varspec_mismatch=True)
    mt4 = p.parsefile(path)[7][4]
    st = mt4["S_table"]
    nbeta = len(st)
    nalpha = len(st[1]["alpha"])
    out = []
    T0 = np.zeros((nbeta, nalpha))
    for j in range(1, nbeta + 1):
        T0[j - 1] = st[j]["S"]
    out.append(T0)
    extra = mt4.get("T", {})
    if extra:
        S = mt4["S"]                       # S[alpha][beta][temp]
        for it in sorted(extra.keys()):
            arr = np.zeros((nbeta, nalpha))
            for i in range(1, nalpha + 1):
                for j in range(1, nbeta + 1):
                    arr[j - 1, i - 1] = S[i][j][it]
            out.append(arr)
    return out


def test_negative_temperature_matches_explicit_block():
    """Reuse-via-negative-T must equal an explicit repeated positive-T block,
    and the reused spectrum is evaluated at the new |T| (the laws differ)."""
    neg = _mt4_S_all_temps(_run(_DECK_NEG))
    exp = _mt4_S_all_temps(_run(_DECK_EXPLICIT))
    assert len(neg) == len(exp) == 2
    for it, (a, b) in enumerate(zip(neg, exp)):
        assert a.shape == b.shape
        np.testing.assert_allclose(
            a, b, rtol=0, atol=0,
            err_msg=f"temperature index {it}: negative-T reuse != explicit block")
    assert not np.allclose(neg[0], neg[1]), \
        "300 K and 400 K laws are identical — temperature was not reapplied"
