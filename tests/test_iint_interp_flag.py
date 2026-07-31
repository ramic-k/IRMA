"""Card 4 ``iint`` flag — MF7/MT4 S(alpha,beta) interpolation form.

iint selects the ENDF INT flag written on BOTH the beta (TAB2) and per-beta
alpha (TAB1) interpolation tables of the inelastic law:
  iint=0 (or absent) -> INT=4 (log-lin, classic/NJOY-faithful, default)
  iint=1             -> INT=2 (lin-lin), which preserves the structural dips of
                        coherent one-phonon laws that log interpolation floors.

These tests pin: the flag flips both tables, the default is byte-identical to
pre-feature output (absent == explicit 0), and out-of-range is rejected.
"""
import os
import tempfile

import pytest

from irma.core.deck import DeckError
from irma.core.engine import run_leapr

_DECK = """20 /
'iint flag deck'/
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
'iint reference deck'/
/
"""


def _run(card4):
    d = tempfile.mkdtemp()
    inp = os.path.join(d, "deck.input")
    out = os.path.join(d, "deck.endf")
    with open(inp, "w") as f:
        f.write(_DECK.format(card4=card4))
    run_leapr(inp, out)
    return out


def _mf7_lines(out):
    """MF7 data records (74-char content + MAT/MF/MT, no sequence number)."""
    with open(out) as f:
        return [ln[:75] for ln in f.read().splitlines()
                if len(ln) >= 75 and ln[70:72] == " 7"]


def _mt4_ints(out):
    """(beta_interp INT, set of per-beta alpha-table INTs) from MF7/MT4."""
    from endf_parserpy import EndfParserPy
    p = EndfParserPy(ignore_number_mismatch=True, ignore_zero_mismatch=True,
                     ignore_varspec_mismatch=True)
    m = p.parsefile(out)[7][4]
    beta_int = list(m['beta_interp']['INT'])
    alpha_ints = sorted({tuple(m['S_table'][ii]['INT']) for ii in m['S_table']})
    return beta_int, alpha_ints


def test_iint1_writes_linlin_both_axes():
    beta_int, alpha_ints = _mt4_ints(_run("1 1. 0 0 1e-75 1/"))
    assert beta_int == [2]
    assert alpha_ints == [(2,)]


def test_iint0_writes_loglin_both_axes():
    beta_int, alpha_ints = _mt4_ints(_run("1 1. 0 0 1e-75 0/"))
    assert beta_int == [4]
    assert alpha_ints == [(4,)]


def test_iint_absent_defaults_to_loglin_and_byte_identical():
    absent = _run("1 1./")            # 5-field classic Card 4
    explicit0 = _run("1 1. 0 0 1e-75 0/")
    # default is log-lin (INT=4) ...
    assert _mt4_ints(absent) == ([4], [(4,)])
    # ... and the whole MF7 section is byte-identical to the explicit-0 deck,
    # so the feature changes nothing unless iint=1 is set.
    assert _mf7_lines(absent) == _mf7_lines(explicit0)


def test_iint1_changes_mf7_vs_iint0():
    assert _mf7_lines(_run("1 1. 0 0 1e-75 1/")) != _mf7_lines(_run("1 1. 0 0 1e-75 0/"))


@pytest.mark.parametrize("bad", ["2", "-1", "3"])
def test_iint_out_of_range_rejected(bad):
    with pytest.raises(DeckError, match="iint must be 0 or 1"):
        _run(f"1 1. 0 0 1e-75 {bad}/")
