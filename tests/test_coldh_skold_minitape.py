"""Fast CI byte-identity for the cold-hydrogen and Skold kernels.

The 8.8x performance rewrite of coldh (hoisted rotational sums, batched
sint) and the Skold correction previously had byte-identity coverage only
in the slow, manually-run expected validation (7-temperature ortho/para-H).
This pins them in CI: a miniature ortho-H deck (6 alpha x 10 beta, 14 K,
diffusion translation + one discrete oscillator + a 12-point S(kappa)
table, nsk=2) must reproduce the vendored NJOY tape BYTE-IDENTICALLY in
MF7. With ncold=1 the deck exercises contin, trans (diffusion:
stable/besk1/terps), discre and coldh (ortho rotational sums + SCT); NJOY
and IRMA run the Skold step only for ncold=0, so the same deck with
ncold=0 pins skold. MF1 is excluded: IRMA deliberately writes a consistent
NWD and exact directory counts where NJOY does not.
"""
import gzip
import os
import tempfile

import pytest

from irma.core.engine import run_leapr

_REFS = os.path.join(os.path.dirname(__file__), "njoy_minitape_references")

_DECK = """20 /
'mini ortho-H coldh+skold deck'/
1 1 20/
3 1001. 0 0 1e-100/
.99917 20.43634 2 0 1 2/
0/
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


def _mf7_lines(text):
    return [ln[:75] for ln in text.splitlines()
            if len(ln) >= 75 and ln[70:72] == " 7"]


@pytest.mark.parametrize("ncold, ref", [
    (1, "coldh_skold.endf.gz"),    # cold-hydrogen kernel
    (0, "skold.endf.gz"),          # Skold correction (runs only for ncold=0)
])
def test_coldh_skold_mf7_byte_identical_to_njoy(ncold, ref):
    d = tempfile.mkdtemp()
    inp = os.path.join(d, "coldh.input")
    out = os.path.join(d, "coldh.endf")
    with open(inp, "w") as f:
        f.write(_DECK.replace(".99917 20.43634 2 0 1 2/",
                              f".99917 20.43634 2 0 {ncold} 2/"))
    run_leapr(inp, out)

    with gzip.open(os.path.join(_REFS, ref), "rt") as f:
        ref = _mf7_lines(f.read())
    with open(out) as f:
        thw = _mf7_lines(f.read())

    assert len(ref) > 40
    assert thw == ref
