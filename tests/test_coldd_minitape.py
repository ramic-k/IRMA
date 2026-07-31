"""Fast CI byte-identity for the cold-deuterium kernels (ncold=3/4).

Cold ortho/para-DEUTERIUM was the last classic kernel with zero test
execution: the coldh law>3 branch (deuteron mass, sampcd/sampid spin
amplitudes, ded rotational energy) and the law=4/5 even/odd-J
spin-correlation factors were live and user-reachable but unvalidated.
These two miniature decks (6 alpha x 10 beta, 19 K, diffusion
translation + a 12-point S(kappa) table, ncold=3 and ncold=4) must
reproduce the vendored unmodified-NJOY2016.78 tapes BYTE-IDENTICALLY in
MF7. Card 4/5 constants (mat 13/12, za 1002, awr 1.9968, spr 3.395,
npr 2) and the translational/continuous weights (twt=.025, c=40.,
tbeta=.475, 19 K, no discrete oscillators, nsk=0) follow the ENDF/B
tsl-ortho-D / tsl-para-D LEAPR evaluations (LANL eval-apr93,
LA-12639-MS); the alpha/beta grid, toy rho and toy S(kappa) table are
the coldh_skold minitape shapes. nsk=0 with ncold>0 also pins the
Card 17/18-without-Card 19 deck-reading path. MF1 is excluded: IRMA
deliberately writes a consistent NWD and exact directory counts where
NJOY does not.
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

_DECK_PARA_D = """20 /
'mini para-D coldd deck'/
1 1 20/
12 1002. 0 0 1e-100/
1.9968 3.395 2 0 4/
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
'mini coldd para-D reference'/
/
"""

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


def test_ortho_and_para_d_laws_differ():
    """The two spin-correlation laws must produce different S(a,b).

    Guards against a regression that collapses the law=4/5 branches
    (e.g. a swapped even/odd-J factor that makes ortho == para). Only
    the 66 data columns are compared: the MAT column (13 vs 12) would
    otherwise make the tapes differ trivially.
    """
    with gzip.open(os.path.join(_REF_DIR, "coldd_ortho.endf.gz"), "rt") as f:
        ortho = [ln[:66] for ln in _mf7_lines(f.read())]
    with gzip.open(os.path.join(_REF_DIR, "coldd_para.endf.gz"), "rt") as f:
        para = [ln[:66] for ln in _mf7_lines(f.read())]
    assert ortho != para
