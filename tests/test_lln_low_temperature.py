"""Low-temperature LLN underflow guard.

With ilog=0 the symmetric law is stored linearly as S*exp(-beta/2); at low T
(large beta) the high-energy values underflow the ENDF field and are written as
0, silently dropping the high-energy phonon structure. The writer must WARN when
this happens, naming the cold temperature, and ilog=1 (LLN log storage) must
suppress the warning by keeping the values. Warm tables must never warn.
"""
import os
import tempfile

import pytest

from irma.core.engine import run_leapr

# Two-temperature deck: the second temperature is NEGATIVE (LEAPR "reuse the
# previous phonon spectrum at |T|"), so only the first carries a spectrum block.
_DECK_2T = """20 /
'lln low-temperature guard deck, two temperatures'/
2 1 4/
1 1 0 {ilog} 1e-75/
1.0 20.0 1 0 0/
0/
3 4 1/
0.05 1.0 8.0/
0.0 0.6 2.0 6.0/
{temp1}/
0.005 6/
0.0 0.20 0.45 0.55 0.30 0.0/
0. 0. 1./
0/
{temp2}/
' lln test '/
/
"""


@pytest.mark.parametrize("temp1, temp2, ilog, warn_T", [
    (300.0, -5.0, 0, "T=5 K"),      # cold SECOND table
    (5.0, -300.0, 0, "T=5 K"),      # cold first table
    (300.0, -350.0, 0, None),       # two warm tables
    (300.0, -5.0, 1, None),         # ilog=1 keeps the values
])
def test_underflow_warning(temp1, temp2, ilog, warn_T, capsys):
    d = tempfile.mkdtemp()
    inp, out = os.path.join(d, "deck.input"), os.path.join(d, "deck.endf")
    with open(inp, "w") as f:
        f.write(_DECK_2T.format(temp1=temp1, temp2=temp2, ilog=ilog))
    run_leapr(inp, out)
    log = capsys.readouterr().out
    assert ("underflow" in log.lower()) == (warn_T is not None)
    if warn_T:
        assert "WARNING" in log and "ilog=1" in log
        assert warn_T in log and "T=300 K" not in log
