"""The isym=1 (cold-hydrogen) asymmetric law with linear (ilog=0) storage.

At lat=1 and cryogenic temperature exp(be/2) overflows float64 although the
stored product S*exp(be/2) usually does not. The writer evaluates it in log
space and raises a DeckError naming Card 4 ilog=1 where no evaluation order
can represent the point. The reference ``tests/data/coldh_20K_nearmiss.endf.gz``
pins the MF7 bytes of a deck just below the overflow.
"""
import gzip
import math
import os
import tempfile

import pytest

from irma.core.engine import run_leapr, DeckError
from irma.core.endf_writer import _endf_s, _LN_FLOAT_MAX
from irma.core.kernels import sigfig

_REF = os.path.join(os.path.dirname(__file__), "data",
                    "coldh_20K_nearmiss.endf.gz")

# Miniature ortho-H deck at 20 K, lat=1, ilog=0. beta_max=100 card units
# -> be = 100*THERM/(k*20) = 1468, be/2 = 734 > ln(DBL_MAX) = 709.78, so a
# direct exp(be/2) in the writer's +beta half would overflow. The 20 K kernel
# stores exact zeros on the beta=100 row, so the writer must fail with the
# layer-2 DeckError, not a bare OverflowError.
_DECK_OVERFLOW = """20 /
'mini ortho-H 20 K overflow deck'/
1 1 20/
3 1001. 0 0 1e-100/
.99917 20.43634 2 0 1 0/
0/
6 10 1/
1e-4 5e-4 0.0025 0.01 0.05 0.25/
0.0 0.5 1.0 2.0 4.0 7.0 11.0 16.0 30.0 100.0/
20.0/
0.0005 8/
0.0 0.4 0.9 1.0 0.8 0.5 0.2 0.0/
0.1104682205 1.124899936572044 0.3895317795/
1/
0.546/
0.166666666666/
12 0.05/
0.4 0.7 1.3 1.15 0.95 1.0 1.02 0.99 1.0 1.0 1.0 1.0/
'mini cold overflow deck'/
/
"""

# Near-miss variant: beta_max=95 -> be = 1394.6, be/2 = 697.3, ~12 below
# the ceiling. Every point takes the direct exp(be/2) evaluation; the tape
# must be byte-identical to the vendored reference made with that
# evaluation, and the parse-time warning must stay quiet.
_DECK_NEARMISS = _DECK_OVERFLOW.replace("30.0 100.0/", "30.0 95.0/").replace(
    "overflow deck", "near-miss deck")

SMIN, SMALL = 1e-75, 1e-9


def _run(deck_text, out):
    d = tempfile.mkdtemp()
    inp = os.path.join(d, "deck.input")
    with open(inp, "w") as f:
        f.write(deck_text)
    run_leapr(inp, out)
    return out


def _call(s, be, smin=SMIN):
    return _endf_s(s, be, 1, 0, smin, beta_card=8.0, temp_k=20.0)


def _original_isym1_value(s_stored, be, smin=SMIN, small=SMALL):
    """The direct exp(be/2) evaluation for the isym=1 ilog=0 storage form."""
    v = s_stored * math.exp(be / 2.0)
    v = sigfig(v, 7, 0) if v >= small else sigfig(v, 6, 0)
    if v < smin:
        v = 0.0
    return v


def _mf7_lines(text):
    return [ln[:75] for ln in text.splitlines()
            if len(ln) >= 75 and ln[70:72] == " 7"]


# ---------------------------------------------------------------------------
# The end-to-end 20 K lat=1 repro: never a bare OverflowError
# ---------------------------------------------------------------------------

def test_repro_20K_lat1_deck_raises_deckerror_not_overflowerror():
    d = tempfile.mkdtemp()
    with pytest.raises(DeckError) as exc:   # OverflowError would NOT match
        _run(_DECK_OVERFLOW, os.path.join(d, "out.endf"))
    msg = str(exc.value)
    assert "ilog=1" in msg and "Card 4" in msg        # the remedy
    assert "beta=100" in msg and "Card 9" in msg      # the offending point
    assert "20" in msg                                # the temperature


def test_recovered_value_is_analytically_correct():
    # S = 2^-1000 and be = 2200*ln2 give an exactly known product 2^100:
    # correctness, not just non-crashing. be/2 = 1100*ln2 = 762 > 709.78.
    s, be = 2.0 ** -1000, 2200.0 * math.log(2.0)
    got = _call(s, be)
    # rel=1e-6 admits the writer's 7-significant-figure ENDF rounding
    # (<= 5e-7) but not a log-space evaluation error.
    assert got == pytest.approx(2.0 ** 100, rel=1e-6)


# ---------------------------------------------------------------------------
# Layer 1/2 boundary: byte-identity of every non-overflowing value
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("be", [-1468.0, -0.9, 0.0, 0.9, 697.0, 1394.6])
@pytest.mark.parametrize("s", [0.0, 1e-80, 1e-12, 0.31, 3.7])
def test_non_overflowing_values_take_the_identical_path(be, s):
    # For every (be, S) where exp(be/2) does not raise, the writer must
    # produce bit-for-bit the direct evaluation -- including the
    # exact-zero rows next to the ceiling (be=1394.6 is the near-miss
    # regime). An over-eager gate would reroute these through log space
    # (or DeckError on the zeros) and fail here.
    assert be / 2.0 <= _LN_FLOAT_MAX                  # sweep sanity
    assert _call(s, be) == _original_isym1_value(s, be)


def test_nearmiss_20K_tape_mf7_byte_identical_to_prefix_reference():
    d = tempfile.mkdtemp()
    out = _run(_DECK_NEARMISS, os.path.join(d, "nearmiss.endf"))
    with gzip.open(_REF, "rt") as f:
        ref = _mf7_lines(f.read())
    with open(out) as f:
        now = _mf7_lines(f.read())
    assert len(ref) > 50
    assert now == ref


# ---------------------------------------------------------------------------
# Layer 2: the unrecoverable points raise a DeckError naming the remedy
# ---------------------------------------------------------------------------

def test_underflowed_zero_raises_deckerror_naming_ilog1():
    with pytest.raises(DeckError) as exc:
        _call(0.0, 1468.0)
    msg = str(exc.value)
    assert "underflowed to 0" in msg
    assert "ilog=1" in msg and "Card 4" in msg
    assert "beta=8" in msg and "Card 9" in msg


def test_true_product_overflow_raises_deckerror():
    # stored O(1) at be=1468: log(S)+be/2 = 734 > ln(DBL_MAX); the true
    # product really exceeds float64 and no evaluation order helps.
    with pytest.raises(DeckError, match="largest float64"):
        _call(1.0, 1468.0)


def test_underflowed_zero_below_smin_ceiling_is_a_true_zero():
    # be=1420: the product of the largest value that stores as 0.0 with
    # exp(be/2) is bounded by exp(710 - 745.1) ~ 5.5e-16 < smin=1e-9, so
    # the smin floor would zero the point regardless -- 0 is provably
    # correct and no DeckError fires.
    assert _call(0.0, 1420.0, smin=1e-9) == 0.0
