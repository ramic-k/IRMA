"""ENG-4: the isym=1 (cold-hydrogen) asymmetric law at cryogenic temperature.

Writing the LASYM=1 law with linear (ilog=0) storage evaluates
``S * exp(be/2)`` with ``be = beta*THERM/(k*T)`` at lat=1. At cold-moderator
temperatures be/2 exceeds the float64 exp ceiling (ln(DBL_MAX) ~ 709.78,
i.e. be > ~1419.6) and ``math.exp`` used to raise a bare
"OverflowError: math range error" AFTER the full kernel run -- an
evaluation-ORDER artifact: by detailed balance the stored value is
~exp(-be/2), so the PRODUCT is representable even when exp(be/2) alone is
not (the ilog=1 branch writes the same quantity as log(S)+be/2 and never
overflows). Three layers are pinned here:

 1. cause fix -- the overflow-gated log-space evaluation
    (``_asym_overflow_s``), byte-identical for every non-overflowing value
    (the gate is the OverflowError itself, so no non-raising value can
    change path);
 2. diagnostic -- a DeckError naming the card, the beta value, and the
    Card 4 ilog=1 (LLN=1) remedy where the point is genuinely
    unrecoverable (stored S underflowed to 0, or the true product exceeds
    float64) -- never a bare OverflowError;
 3. early warning -- a Card 10 parse-time warning in the driver, emitted
    BEFORE the expensive kernel run, when the beta grid and temperature
    will enter the regime (threshold derived from the same
    be/2 > ln(DBL_MAX) arithmetic, not a hard-coded temperature).

The byte-identity reference ``tests/data/coldh_20K_nearmiss.endf.gz`` is
IRMA's own output for the near-miss deck below, generated at the
pre-ENG-4-fix tree (see SHA256SUMS.txt): the fixed writer must reproduce
it byte-for-byte in MF7.
"""
import gzip
import math
import os
import tempfile

import numpy as np
import pytest

from irma.core.engine import run_leapr, DeckError
from irma.core.endf_writer import (_compute_endf_s, _asym_overflow_s,
                                   _LN_FLOAT_MAX)
from irma.core.kernels import sigfig

_REF = os.path.join(os.path.dirname(__file__), "data",
                    "coldh_20K_nearmiss.endf.gz")

# Miniature ortho-H deck at 20 K, lat=1, ilog=0. beta_max=100 card units
# -> be = 100*THERM/(k*20) = 1468, be/2 = 734 > ln(DBL_MAX) = 709.78: the
# writer's +beta half used to raise a bare OverflowError after the kernel
# run. The 20 K kernel stores exact zeros on the beta=100 row, so the fixed
# writer must fail with the layer-2 DeckError instead.
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
# the ceiling. Every point takes the pre-fix direct path; the tape must be
# byte-identical to the vendored pre-fix reference and the parse-time
# warning must stay quiet.
_DECK_NEARMISS = _DECK_OVERFLOW.replace("30.0 100.0/", "30.0 95.0/").replace(
    "overflow deck", "near-miss deck")

NBETA = 4
SMIN, SMALL = 1e-75, 1e-9


def _run(deck_text, out):
    d = tempfile.mkdtemp()
    inp = os.path.join(d, "deck.input")
    with open(inp, "w") as f:
        f.write(deck_text)
    run_leapr(inp, out)
    return out


def _call(ssm, ssp, ii, be, smin=SMIN):
    return _compute_endf_s(ssm, ssp, 1, 0, ii, NBETA, 0, 0,
                           be, smin, SMALL, beta_card=8.0, temp_k=20.0)


def _blocks(fill):
    """ssm/ssp blocks [nbeta=4, nalpha=1, nt=1] with a uniform value."""
    a = np.full((NBETA, 1, 1), fill, dtype=float)
    return a, a.copy()


def _original_isym1_value(s_stored, be, smin=SMIN, small=SMALL):
    """The exact pre-ENG-4 arithmetic for the isym=1 ilog=0 storage form."""
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


# ---------------------------------------------------------------------------
# Layer 1: overflow-gated log-space evaluation, value correctness
# ---------------------------------------------------------------------------

def test_recovered_ssp_value_matches_log_space_arithmetic():
    # The live ENG-4 numbers: stored ~1.7e-319 (subnormal) at be=1468.
    # exp(be/2) alone overflows; the product is O(1) and must round-trip
    # through the identical sigfig formatting the direct path applies.
    s, be = 1.7e-319, 1468.0
    ssm, ssp = _blocks(0.5)
    ssp[0, 0, 0] = s
    got = _call(ssm, ssp, ii=NBETA, be=be)            # +beta half -> ssp[0]
    expected = math.exp(math.log(s) + be / 2.0)
    assert expected >= SMALL                          # 7-sigfig branch
    assert got == sigfig(expected, 7, 0)


def test_recovered_value_is_analytically_correct():
    # S = 2^-1000 and be = 2200*ln2 give an exactly known product 2^100:
    # correctness, not just non-crashing. be/2 = 1100*ln2 = 762 > 709.78.
    s, be = 2.0 ** -1000, 2200.0 * math.log(2.0)
    ssm, ssp = _blocks(0.5)
    ssp[0, 0, 0] = s
    got = _call(ssm, ssp, ii=NBETA, be=be)
    # rel=1e-6 admits the writer's 7-significant-figure ENDF rounding
    # (<= 5e-7) but not a log-space evaluation error.
    assert got == pytest.approx(2.0 ** 100, rel=1e-6)


def test_ssm_half_is_gated_too():
    # The caller only reaches the -beta (ssm) half with be <= 0, but the
    # storage form is the same expression; both halves carry the gate.
    s, be = 1.0e-310, 1450.0
    ssm, ssp = _blocks(0.5)
    ssm[NBETA - 1, 0, 0] = s
    got = _call(ssm, ssp, ii=1, be=be)                # ii<nbeta -> ssm[3]
    assert got == sigfig(math.exp(math.log(s) + be / 2.0), 7, 0)


# ---------------------------------------------------------------------------
# Layer 1/2 boundary: byte-identity of every non-overflowing value
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("be", [-1468.0, -0.9, 0.0, 0.9, 697.0, 1394.6])
@pytest.mark.parametrize("s", [0.0, 1e-80, 1e-12, 0.31, 3.7])
def test_non_overflowing_values_take_the_identical_path(be, s):
    # For every (be, S) where exp(be/2) does not raise, the fixed writer
    # must produce bit-for-bit the pre-fix arithmetic -- including the
    # exact-zero rows next to the ceiling (be=1394.6 is the near-miss
    # regime). An over-eager gate would reroute these through log space
    # (or DeckError on the zeros) and fail here.
    assert be / 2.0 <= _LN_FLOAT_MAX                  # sweep sanity
    ssm, ssp = _blocks(0.5)
    ssp[0, 0, 0] = s
    assert _call(ssm, ssp, ii=NBETA, be=be) == _original_isym1_value(s, be)


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
    ssm, ssp = _blocks(0.0)
    with pytest.raises(DeckError) as exc:
        _call(ssm, ssp, ii=NBETA, be=1468.0)
    msg = str(exc.value)
    assert "underflowed to 0" in msg
    assert "ilog=1" in msg and "Card 4" in msg
    assert "beta=8" in msg and "Card 9" in msg


def test_true_product_overflow_raises_deckerror():
    # stored O(1) at be=1468: log(S)+be/2 = 734 > ln(DBL_MAX); the true
    # product really exceeds float64 and no evaluation order helps.
    ssm, ssp = _blocks(1.0)
    with pytest.raises(DeckError, match="largest float64"):
        _call(ssm, ssp, ii=NBETA, be=1468.0)


def test_underflowed_zero_below_smin_ceiling_is_a_true_zero():
    # be=1420: the product of the largest value that stores as 0.0 with
    # exp(be/2) is bounded by exp(710 - 745.1) ~ 5.5e-16 < smin=1e-9, so
    # the smin floor would zero the point regardless -- 0 is provably
    # correct and no DeckError fires.
    ssm, ssp = _blocks(0.0)
    assert _call(ssm, ssp, ii=NBETA, be=1420.0, smin=1e-9) == 0.0


def test_helper_never_lets_overflowerror_escape():
    # Directly at the helper: for a spread of unrepresentable points the
    # escape hatch is always DeckError, never OverflowError.
    for s in (0.0, 1.0, 1e300):
        with pytest.raises(DeckError):
            _asym_overflow_s(s, 1468.0, 8.0, 20.0, SMIN)


# ---------------------------------------------------------------------------
# Layer 3: the Card 10 parse-time early warning
# ---------------------------------------------------------------------------

def test_parse_warning_fires_before_the_kernel_run(capsys):
    d = tempfile.mkdtemp()
    with pytest.raises(DeckError):
        _run(_DECK_OVERFLOW, os.path.join(d, "out.endf"))
    out = capsys.readouterr().out
    assert "asymmetric-law overflow regime" in out
    assert "ilog=1" in out
    # emitted at Card 10 parse, before the first kernel output
    assert out.index("asymmetric-law overflow regime") < out.index("DW lambda")


def test_parse_warning_silent_for_nearmiss_deck(capsys):
    d = tempfile.mkdtemp()
    _run(_DECK_NEARMISS, os.path.join(d, "out.endf"))
    out = capsys.readouterr().out
    assert "asymmetric-law overflow regime" not in out
