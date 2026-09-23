"""MF7 byte identity of the cold-deuterium kernels (ncold=3/4) against
unmodified NJOY2016.78 tapes, on two miniature decks (6 alpha x 10 beta,
19 K, diffusion translation, a 12-point S(kappa) table, nsk=0). The Card 4/5
constants and the twt/c/tbeta weights follow the ENDF/B tsl-ortho-D /
tsl-para-D LEAPR evaluations (LA-12639-MS). MF1 is excluded: IRMA writes a
consistent NWD and exact directory counts where NJOY does not.
"""
import gzip
import os
import tempfile

import pytest

from irma.core.engine import run_leapr

_REF_DIR = os.path.join(os.path.dirname(__file__), "njoy_minitape_references")

_DECK_ORTHO_D = """20 /
'mini ortho-D coldd deck'/
1 1 20/
13 1002. 0 0 1e-100/
1.9968 3.395 2 0 3/
0/
6 10/
1e-4 5e-4 0.0025 0.01 0.05 0.25/
0.0 0.5 1.0 2.0 4.0 7.0 11.0 16.0 22.0 30.0/
19.0/
0.0005 8/
0.0 0.4 0.9 1.0 0.8 0.5 0.2 0.0/
0.025 40.0 0.475/
0/
12 0.05/
0.4 0.7 1.3 1.15 0.95 1.0 1.02 0.99 1.0 1.0 1.0 1.0/
'mini coldd ortho-D reference'/
/
"""

_DECK_PARA_D = (_DECK_ORTHO_D.replace("13 1002.", "12 1002.")
                .replace(" 2 0 3/", " 2 0 4/").replace("ortho", "para"))

_CASES = {
    "ortho_d": (_DECK_ORTHO_D, "coldd_ortho.endf.gz"),
    "para_d": (_DECK_PARA_D, "coldd_para.endf.gz"),
}


def _mf7_lines(text):
    return [ln[:75] for ln in text.splitlines()
            if len(ln) >= 75 and ln[70:72] == " 7"]


def _run_case(name):
    deck, refname = _CASES[name]
    d = tempfile.mkdtemp()
    inp = os.path.join(d, f"{name}.input")
    out = os.path.join(d, f"{name}.endf")
    with open(inp, "w") as f:
        f.write(deck)
    run_leapr(inp, out)

    with gzip.open(os.path.join(_REF_DIR, refname), "rt") as f:
        ref = _mf7_lines(f.read())
    with open(out) as f:
        thw = _mf7_lines(f.read())
    return ref, thw


@pytest.mark.parametrize("name", sorted(_CASES))
def test_coldd_mf7_byte_identical_to_njoy(name):
    ref, thw = _run_case(name)
    assert len(ref) > 50
    assert thw == ref
