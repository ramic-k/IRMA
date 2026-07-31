"""ENDF writer backend selection (IRMA_ENDF_WRITER) pins.

The tape serializer defaults to the compiled endf-parserpy backend
(EndfParserCpp) — verified byte-identical to the pure-Python writer on the
full reference evidence set (native NJOY expected decks, the NJOY minitape byte
pins, the noncubic fast-CI/lat0 tapes and the iel=10 Bragg-grouping path)
and several times faster. These tests pin the selection contract:

  * default / 'auto' / 'cpp'  -> EndfParserCpp,
  * 'py'                      -> the legacy pure-Python EndfParserPy,
  * compiled backend missing  -> loud fallback to EndfParserPy ('auto')
                                 or a hard ImportError ('cpp'),
  * anything else             -> ValueError (no silent misconfiguration),

plus an end-to-end check that both backends write a byte-identical tape.
"""
import os
import tempfile

import pytest

import endf_parserpy
from endf_parserpy import EndfParserPy

from irma.core.endf_writer import _select_writer_backend
from irma.core.engine import run_leapr


def _cpp_available():
    try:
        from endf_parserpy import EndfParserCpp
        EndfParserCpp()
        return True
    except Exception:
        return False


CPP_AVAILABLE = _cpp_available()
needs_cpp = pytest.mark.skipif(
    not CPP_AVAILABLE,
    reason="compiled endf-parserpy backend not available in this install")


@needs_cpp
def test_default_backend_is_cpp(monkeypatch):
    """Unset env -> the compiled writer backend is the DEFAULT."""
    from endf_parserpy import EndfParserCpp
    monkeypatch.delenv("IRMA_ENDF_WRITER", raising=False)
    assert isinstance(_select_writer_backend(), EndfParserCpp)


@needs_cpp
@pytest.mark.parametrize("value", ["auto", "cpp", " CPP ", ""])
def test_auto_and_cpp_select_compiled(monkeypatch, value):
    from endf_parserpy import EndfParserCpp
    monkeypatch.setenv("IRMA_ENDF_WRITER", value)
    assert isinstance(_select_writer_backend(), EndfParserCpp)


@pytest.mark.parametrize("value", ["py", " Py "])
def test_py_selects_pure_python(monkeypatch, value):
    """IRMA_ENDF_WRITER=py keeps the legacy pure-Python writer selectable."""
    monkeypatch.setenv("IRMA_ENDF_WRITER", value)
    parser = _select_writer_backend()
    assert isinstance(parser, EndfParserPy)


def test_invalid_value_raises(monkeypatch):
    monkeypatch.setenv("IRMA_ENDF_WRITER", "fortran")
    with pytest.raises(ValueError, match="IRMA_ENDF_WRITER"):
        _select_writer_backend()


class _UnavailableCpp:
    """Stand-in for EndfParserCpp on a source-only endf-parserpy build:
    the name imports fine but instantiation raises (the compiled module
    fails to load inside __init__)."""

    def __init__(self, *a, **k):
        raise ImportError("simulated: cpp module not built")


def test_auto_falls_back_to_pure_python_loudly(monkeypatch, capsys):
    monkeypatch.delenv("IRMA_ENDF_WRITER", raising=False)
    monkeypatch.setattr(endf_parserpy, "EndfParserCpp", _UnavailableCpp)
    parser = _select_writer_backend()
    assert isinstance(parser, EndfParserPy)
    out = capsys.readouterr().out
    assert "WARNING" in out and "EndfParserCpp" in out


def test_forced_cpp_is_a_hard_error_when_unavailable(monkeypatch):
    monkeypatch.setenv("IRMA_ENDF_WRITER", "cpp")
    monkeypatch.setattr(endf_parserpy, "EndfParserCpp", _UnavailableCpp)
    with pytest.raises(ImportError):
        _select_writer_backend()


# The writer-flag-tapes toy deck (3 alpha x 4 beta, 300 K + reused 400 K):
# runs in ~0.1 s and exercises MF1 + MF7/MT4 end to end.
_TINY_DECK = """20 /
'tiny backend parity deck'/
2 1 4/
1 1. 0 0/
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
'backend parity deck'/
/
"""


@needs_cpp
def test_backends_write_byte_identical_tape(monkeypatch):
    """Whole-file byte parity between the compiled and pure-Python writers."""
    d = tempfile.mkdtemp()
    inp = os.path.join(d, "parity.input")
    with open(inp, "w") as f:
        f.write(_TINY_DECK)

    tapes = {}
    for backend in ("py", "cpp"):
        out = os.path.join(d, f"parity_{backend}.endf")
        monkeypatch.setenv("IRMA_ENDF_WRITER", backend)
        run_leapr(inp, out)
        with open(out, "rb") as f:
            tapes[backend] = f.read()

    assert len(tapes["py"]) > 1000
    assert tapes["py"] == tapes["cpp"]


@needs_cpp
def test_backends_byte_identical_on_a_real_expected_deck(monkeypatch):
    """Whole-file py/cpp parity on a REAL expected deck (Al, iel=4 built-in
    coherent elastic, full production grids) -- the tiny-deck parity above
    cannot stand in for the one-off release audit; this keeps a production-
    shaped deck under both backends in CI (~3 s)."""
    deck = os.path.join(os.path.dirname(__file__),
                        "native_LEAPR_NJOY_ENDF_validation", "leapr_decks",
                        "tsl-013_Al_027.input")
    d = tempfile.mkdtemp()
    tapes = {}
    for backend in ("py", "cpp"):
        out = os.path.join(d, f"al_{backend}.endf")
        monkeypatch.setenv("IRMA_ENDF_WRITER", backend)
        run_leapr(deck, out)
        with open(out, "rb") as f:
            tapes[backend] = f.read()
    assert len(tapes["py"]) > 100_000
    assert tapes["py"] == tapes["cpp"]
