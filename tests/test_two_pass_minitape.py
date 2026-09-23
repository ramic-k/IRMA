"""Fast CI byte-identity for the two-pass bound-secondary path (b7=0).

The mixed-moderator merge (principal law combined with a bound secondary
computed in a second full temperature pass, BeO-style) was previously
pinned only by the slow, manually-run BeO expected validation. A miniature
two-pass deck (4 alpha x 6 beta, 296 K, principal + secondary spectrum
blocks, nss=1 b7=0 mss=1) must reproduce the vendored NJOY2016.78 tape
in MF7: byte-identical everywhere except the incoherent-elastic SB head
record, which NJOY stores raw and renders with its adaptive 9-digit
no-exponent form while IRMA uses the uniform 7-significant-figure
convention (deliberate format divergence, documented in endf_writer; the
record is compared numerically here instead). MF1 is excluded as usual.
"""
import gzip
import os
import tempfile

from irma.core.engine import run_leapr

_REF = os.path.join(os.path.dirname(__file__), "njoy_minitape_references",
                    "two_pass.endf.gz")

_DECK = """20 /
'mini two-pass secondary deck'/
1 1 12/
1 127. 0 0 1e-100/
8.93478 6.15 1 0/
1 0. 15.858 3.7481 1/
4 6 1/
0.05 0.2 0.8 3.0/
0.0 0.5 1.5 3.0 5.5 9.0/
296./
0.002 6/
0.0 0.3 0.8 1.0 0.6 0.0/
0. 0. 1./
0/
296./
0.002 6/
0.0 0.5 1.0 0.7 0.3 0.0/
0. 0. 1./
0/
'mini two-pass reference'/
/
"""


def _endf_float(s):
    s = s.strip().replace(' ', '')
    for k in range(len(s) - 1, 0, -1):
        if s[k] in '+-' and s[k - 1] not in 'eE':
            return float(s[:k] + 'e' + s[k:])
    return float(s)


def _mf7_lines(text):
    return [ln[:75] for ln in text.splitlines()
            if len(ln) >= 75 and ln[70:72] == " 7"]


def test_two_pass_mf7_byte_identical_to_njoy():
    d = tempfile.mkdtemp()
    inp = os.path.join(d, "twopass.input")
    out = os.path.join(d, "twopass.endf")
    with open(inp, "w") as f:
        f.write(_DECK)
    run_leapr(inp, out)

    with gzip.open(_REF, "rt") as f:
        ref = _mf7_lines(f.read())
    with open(out) as f:
        thw = _mf7_lines(f.read())

    assert len(ref) > 30
    assert len(thw) == len(ref)
    mismatches = []
    for r, t in zip(ref, thw):
        if r == t:
            continue
        # The only tolerated difference: the LTHR=2 SB head record's
        # rendering (NJOY adaptive 9-digit vs IRMA uniform 7-sig).
        assert r[11:] == t[11:], f"non-SB difference:\nREF {r}\nIRM {t}"
        sb_ref, sb_irm = _endf_float(r[:11]), _endf_float(t[:11])
        assert abs(sb_irm / sb_ref - 1.0) < 1.0e-6
        mismatches.append((r, t))
    assert len(mismatches) <= 1          # at most the SB record
