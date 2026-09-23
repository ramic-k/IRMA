"""The compiled and pure-Python endf-parserpy writers produce identical tapes."""
import os
import tempfile

import pytest

import endf_parserpy

from irma.core.engine import run_leapr


def _cpp_available():
    try:
        from endf_parserpy import EndfParserCpp
        EndfParserCpp()
        return True
    except Exception:
        return False


class _UnavailableCpp:
    """EndfParserCpp on a source-only endf-parserpy build: instantiation fails."""

    def __init__(self, *a, **k):
        raise ImportError("simulated: cpp module not built")


@pytest.mark.skipif(not _cpp_available(),
                    reason="compiled endf-parserpy backend not available")
def test_backends_byte_identical_on_a_real_expected_deck(monkeypatch):
    """Whole-file parity on the Al deck (iel=4, production grids, ~3 s)."""
    deck = os.path.join(os.path.dirname(__file__),
                        "native_LEAPR_NJOY_ENDF_validation", "leapr_decks",
                        "tsl-013_Al_027.input")
    d = tempfile.mkdtemp()
    tapes = {}
    for backend in ("cpp", "py"):
        if backend == "py":
            monkeypatch.setattr(endf_parserpy, "EndfParserCpp", _UnavailableCpp)
        out = os.path.join(d, f"al_{backend}.endf")
        run_leapr(deck, out)
        with open(out, "rb") as f:
            tapes[backend] = f.read()
    assert len(tapes["py"]) > 100_000
    assert tapes["py"] == tapes["cpp"]
