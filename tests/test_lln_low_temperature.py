"""Low-temperature LLN underflow guard.

With ilog=0 the symmetric law is stored linearly as S*exp(-beta/2); at low T
(large beta) the high-energy values underflow the ENDF field and are written as
0, silently dropping the high-energy phonon structure. The writer must WARN when
this happens, and ilog=1 (LLN log storage) must suppress the warning by keeping
the values. A room-temperature deck must never warn (no underflow there).
"""
import os
import tempfile


from irma.core.engine import run_leapr

# Tiny single-temperature, single-oscillator deck. Card 4 carries
# `mat za isabt ilog smin`; the temperature and ilog are templated.
_DECK = """20 /
'lln low-temperature guard deck'/
1 1 4/
1 1 0 {ilog} 1e-75/
1.0 20.0 1 0 0/
0/
3 4 1/
0.05 1.0 8.0/
0.0 0.6 2.0 6.0/
{temp}/
0.005 6/
0.0 0.20 0.45 0.55 0.30 0.0/
0. 0. 1./
0/
' lln test '/
/
"""


def _run(temp, ilog, capsys):
    d = tempfile.mkdtemp()
    inp, out = os.path.join(d, "deck.input"), os.path.join(d, "deck.endf")
    with open(inp, "w") as f:
        f.write(_DECK.format(temp=temp, ilog=ilog))
    run_leapr(inp, out)
    return capsys.readouterr().out


def test_warns_at_low_temperature_ilog0(capsys):
    out = _run(temp=5.0, ilog=0, capsys=capsys)
    assert "WARNING" in out and "ilog=1" in out
    assert "underflow" in out.lower()


def test_no_warning_with_ilog1(capsys):
    out = _run(temp=5.0, ilog=1, capsys=capsys)
    assert "underflow" not in out.lower()


def test_no_warning_at_room_temperature(capsys):
    out = _run(temp=300.0, ilog=0, capsys=capsys)
    assert "underflow" not in out.lower()


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


def _run_2t(temp1, temp2, ilog, capsys):
    d = tempfile.mkdtemp()
    inp, out = os.path.join(d, "deck.input"), os.path.join(d, "deck.endf")
    with open(inp, "w") as f:
        f.write(_DECK_2T.format(temp1=temp1, temp2=temp2, ilog=ilog))
    run_leapr(inp, out)
    return capsys.readouterr().out


def test_warns_for_cold_second_temperature(capsys):
    """Warm T0 + cryogenic T1: the underflow lives in the SECOND table, which
    the original detector never checked (pre-release review P5)."""
    out = _run_2t(temp1=300.0, temp2=-5.0, ilog=0, capsys=capsys)
    assert "underflow" in out.lower()
    assert "T=5 K" in out
    assert "T=300 K" not in out                   # the warm table is fine


def test_warns_for_cold_first_of_two(capsys):
    out = _run_2t(temp1=5.0, temp2=-300.0, ilog=0, capsys=capsys)
    assert "underflow" in out.lower()
    assert "T=5 K" in out
    assert "T=300 K" not in out


def test_no_warning_for_two_warm_temperatures(capsys):
    out = _run_2t(temp1=300.0, temp2=-350.0, ilog=0, capsys=capsys)
    assert "underflow" not in out.lower()


def test_no_warning_multitemp_with_ilog1(capsys):
    out = _run_2t(temp1=300.0, temp2=-5.0, ilog=1, capsys=capsys)
    assert "underflow" not in out.lower()
