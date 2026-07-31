"""Deck validation guards: unquoted paths and cold-law/mixing conflicts.

An unquoted phonopy/BORN path containing '/' would be silently truncated
at the first slash (the card terminator) and accepted as a bad path, so
it must be rejected with a quoting hint. The two-pass mixed-moderator
merge (nss>0) cannot represent the cold-H asymmetric law (ssp, ncold>0),
so that combination must be rejected rather than silently dropping ssp.
"""
import os
import tempfile

import pytest

from irma.core.deck import DeckError
from irma.core.engine import run_leapr


def _run(deck_text):
    d = tempfile.mkdtemp()
    inp = os.path.join(d, "deck.input")
    with open(inp, "w") as f:
        f.write(deck_text)
    run_leapr(inp, os.path.join(d, "out.endf"))


# iel=10 inelastic_mode=2 head reaching the Card 6f-1 path read, with the path
# written UNQUOTED so the '/' truncates `nonexistent/phonopy.yaml` -> `nonexistent`.
_UNQUOTED_PATH_DECK = """20 /
'unquoted path deck'/
1 1 4/
1 6012./
11.9 4.74 1 10 0 0/
0/
1 1 0 2/
2.46 2.46 6.7 90. 90. 120./
6 12 11.9 6.646 0.001 1/
0.0 0.0 0.0/
nonexistent/phonopy.yaml /
"""


def test_unquoted_path_with_slash_rejected():
    with pytest.raises(DeckError, match="truncated at"):
        _run(_UNQUOTED_PATH_DECK)


# Same ortho-H coldh deck as the byte-exact minitape (ncold=1), but Card 6 now
# requests a secondary scatterer (nss=1) -> the dropped-ssp combination.
_COLDH_PLUS_SECONDARY = """20 /
'coldh + secondary scatterer (must reject)'/
1 1 20/
3 1001. 0 0 1e-100/
.99917 20.43634 2 0 1 2/
1 0 2.0 20.0 1/
6 10/
1e-4 5e-4 0.0025 0.01 0.05 0.25/
0.0 0.5 1.0 2.0 4.0 7.0 11.0 16.0 22.0 30.0/
14.0/
0.0005 8/
0.0 0.4 0.9 1.0 0.8 0.5 0.2 0.0/
0.1104682205 1.124899936572044 0.3895317795/
1/
0.546/
0.166666666666/
12 0.05/
0.4 0.7 1.3 1.15 0.95 1.0 1.02 0.99 1.0 1.0 1.0 1.0/
0.02144/
'mini coldh reference'/
/
"""


def test_secondary_scatterer_with_cold_law_rejected():
    with pytest.raises(DeckError, match="two-pass"):
        _run(_COLDH_PLUS_SECONDARY)


# -- Codex-review refinements: the guards must not over-fire ------------------

_BARE_FILENAME_PATH_DECK = """20 /
'qa wave3 bare filename + glued terminator'/
1 1 4/
1 6012./
11.9 4.74 1 10 0 0/
0/
1 1 0 2/
2.46 2.46 6.7 90. 90. 120./
6 12 11.9 6.646 0.001 1/
0.0 0.0 0.0/
phonopy.yaml/
"""


def test_bare_filename_with_glued_terminator_not_rejected():
    # `phonopy.yaml/` is a valid bare filename + the legal glued '/' terminator,
    # NOT a truncated path -- it must reach the mesh loader (and fail there on
    # the missing file), not be rejected by the truncation guard.
    try:
        _run(_BARE_FILENAME_PATH_DECK)
    except Exception as e:  # noqa: BLE001 - any non-truncation failure is fine
        assert "truncated at" not in str(e), f"bare filename wrongly rejected: {e}"


# b7=1 (free gas) is single-pass analytic -- it does NOT enter the lossy two-pass
# ssp merge, so cold-H + an analytic secondary must NOT trip the guard. The deck
# ends after Card 6, so it fails downstream (missing cards), not at the guard.
_COLDH_FREEGAS_SECONDARY_INCOMPLETE = """20 /
'b7=1 free-gas secondary + cold (guard must NOT fire)'/
1 1 20/
3 1001. 0 0 1e-100/
.99917 20.43634 2 0 1 2/
1 1 2.0 20.0 1/
"""


def test_analytic_freegas_secondary_with_cold_allowed():
    with pytest.raises(DeckError) as exc:
        _run(_COLDH_FREEGAS_SECONDARY_INCOMPLETE)
    msg = str(exc.value)
    assert "two-pass" not in msg and "asymmetric ssp" not in msg, (
        f"analytic b7=1 secondary wrongly rejected by the two-pass guard: {msg}")
