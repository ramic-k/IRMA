"""Card 4 output options (isabt, ilog) vs vendored NJOY2016 mini tapes.

The isabt=1 (asymmetric S-tilde, LASYM+2) and ilog=1 (ln-S storage, LLN=1)
writer branches had no coverage of any kind. Each test runs a tiny deck and
requires IRMA's MF7 output to be BYTE-IDENTICAL to an NJOY2016.78 reference
tape generated from the same deck (vendored gzipped under
tests/njoy_minitape_references/). MF1 is excluded: IRMA deliberately writes
a consistent NWD and exact directory counts where NJOY does not.
"""
import gzip
import os
import tempfile

import pytest

from irma.core.engine import run_leapr

_REF_DIR = os.path.join(os.path.dirname(__file__), "njoy_minitape_references")

# The writer-roundtrip toy deck (3 alpha x 4 beta, 300 K + reused 400 K) with
# Card 4 = "mat za isabt ilog": flag under test set per case.
_DECK = """20 /
'tiny {name} flag deck'/
2 1 4/
{card4}
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
-400/
'{name} reference deck'/
/
"""

_CASES = {
    "isabt": "1 1. 1 0/",
    "ilog": "1 1. 0 1/",
}


def _mf7_lines(text):
    """All MF7 data records of a tape (74-char content + MAT/MF/MT columns,
    excluding the sequence number so MF1's length cannot shift them)."""
    out = []
    for ln in text.splitlines():
        if len(ln) >= 75 and ln[70:72] == " 7":
            out.append(ln[:75])
    return out


@pytest.mark.parametrize("name", sorted(_CASES))
def test_flag_tape_mf7_byte_identical_to_njoy(name):
    d = tempfile.mkdtemp()
    inp = os.path.join(d, f"{name}.input")
    out = os.path.join(d, f"{name}.endf")
    with open(inp, "w") as f:
        f.write(_DECK.format(name=name, card4=_CASES[name]))
    run_leapr(inp, out)

    with gzip.open(os.path.join(_REF_DIR, f"{name}.endf.gz"), "rt") as f:
        ref = _mf7_lines(f.read())
    with open(out) as f:
        thw = _mf7_lines(f.read())

    assert len(ref) > 20
    assert thw == ref          # byte-identical MF7 (flags, grids, S values)


def test_ilog_zero_s_sentinel_is_minus_999_at_every_temperature():
    """DELIBERATE NJOY DIVERGENCE: zero-S points on an ilog=1 tape store the
    ln-S sentinel -999 at EVERY temperature. NJOY's isym=0
    additional-temperature branch writes 0 instead (leapr.f90:3482), which
    THERMR — storing ilog values verbatim as ln(S) — resurrects as
    S = exp(0) = 1 downstream. The beta grid here reaches 30, guaranteeing
    underflowed-to-zero S points in both temperature blocks."""
    deck = _DECK.format(name="ilogz", card4="1 1. 0 1/").replace(
        "3 4 1/", "3 6 1/").replace(
        "0.0 0.6 2.0 6.0/", "0.0 0.6 2.0 6.0 15.0 30.0/")
    d = tempfile.mkdtemp()
    inp, out = os.path.join(d, "z.input"), os.path.join(d, "z.endf")
    with open(inp, "w") as f:
        f.write(deck)
    run_leapr(inp, out)
    with open(out) as f:
        mt4 = [ln for ln in f.read().splitlines()
               if len(ln) >= 75 and ln[70:75] == " 7  4"]
    sentinels = sum(ln.count("-9.990000+2") for ln in mt4)
    assert sentinels == 4          # 2 zero-S points x 2 temperatures
    assert not any(" 0.000000+0-" in ln for ln in mt4[8:])  # no 0-sentinels
