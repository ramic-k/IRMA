"""End-to-end run_leapr API surface on a tiny two-temperature deck.

The toy deck (3x4 alpha/beta grid, 300 K plus a -400 K duplicate) runs in
well under a second and produces a real ENDF tape, which makes it cheap to
pin behavior that needs a full engine pass:

* the LeaprResult summary object returned by run_leapr;
* the nsk=1 Vineyard no-op: Cards 17-19 are read but not applied (NJOY2016
  parity), which must be announced loudly and leave the tape identical to
  an nsk=0 run;
* the MF7/MT2 reader's temperature handling on the produced tapes (clamp
  warning on LTHR=2 tapes, nearest-temperature warning on LTHR=1 tapes).
"""
import os
import tempfile

import pytest

_TOY_DECK = """20 /
'toy two-temperature deck'/
2 1 4/
1 1. 0 0/
1.0 20.0 1 {iel} 0 {nsk}/
0/
3 4 1/
0.05 1.0 8.0/
0.0 0.6 2.0 6.0/
300/
0.005 6/
0.0 0.20 0.45 0.55 0.30 0.0/
0. 0. 1./
0/
{skold_cards}-400/
'toy two-temperature deck'/
/
"""


def _run_toy_deck(iel=0, nsk=0, skold_cards=""):
    from irma.core.engine import run_leapr
    d = tempfile.mkdtemp()
    inp = os.path.join(d, "toy.input")
    out = os.path.join(d, "toy.endf")
    with open(inp, "w") as f:
        f.write(_TOY_DECK.format(iel=iel, nsk=nsk, skold_cards=skold_cards))
    result = run_leapr(inp, out)
    return result, out


@pytest.fixture(scope="module")
def toy_run():
    return _run_toy_deck()


@pytest.fixture(scope="module")
def toy_run_iel1():
    return _run_toy_deck(iel=1)


def test_nsk1_is_announced_and_changes_nothing(toy_run, capsys):
    # nsk=1 reads Cards 17-19 and applies nothing (NJOY2016 parity); the run
    # must say so loudly, and its MF7 equals the nsk=0 tape's
    _, out0 = toy_run
    _, out1 = _run_toy_deck(nsk=1, skold_cards="1 1.0/\n1.0/\n0.5/\n")
    text = capsys.readouterr().out
    assert "Vineyard" in text and "not implemented" in text

    def mf7(path):
        with open(path) as f:
            return [ln[:66] for ln in f if len(ln) >= 75 and ln[70:72] == " 7"]

    assert mf7(out0) == mf7(out1)


def test_run_leapr_returns_result_summary(toy_run):
    from irma import LeaprResult
    result, out = toy_run
    assert isinstance(result, LeaprResult)
    assert result.output_file == out
    assert result.ntempr == 2
    assert result.temperatures_K == pytest.approx((300.0, 400.0))
    assert len(result.teff_K) == 2 and result.teff_K[0] > 300.0
    assert len(result.dw_lambda) == 2 and result.dw_lambda[0] > 0.0
    assert result.iel == -1          # iel=0 + twt=0 -> incoherent elastic
    assert result.isym == 0
    assert result.nedge == 0


def test_mf7mt2_incoherent_temperature_clamp_warns(toy_run):
    from irma.spectra.elastic import from_endf_mf7mt2
    _, out = toy_run                       # LTHR=2 tape at 300/400 K
    with pytest.warns(UserWarning, match="outside the tabulated range"):
        from_endf_mf7mt2(out, T_K=1000.0)
    # inside the tabulated range: no warning
    import warnings as _w
    with _w.catch_warnings():
        _w.simplefilter("error")
        from_endf_mf7mt2(out, T_K=350.0)


def test_mf7mt2_coherent_nearest_temperature_warns(toy_run_iel1):
    from irma.spectra.elastic import from_endf_mf7mt2
    result, out = toy_run_iel1             # LTHR=1 graphite tape at 300/400 K
    assert result.nedge > 0
    with pytest.warns(UserWarning, match="nearest tabulated temperature"):
        model = from_endf_mf7mt2(out, T_K=1000.0)
    assert model.T_K == pytest.approx(400.0)
