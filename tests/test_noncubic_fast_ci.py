"""Fast CI coverage for the in-process noncubic MT4 assembly (modes 1/2).

Drives run_leapr end-to-end (not stubbed) on the vendored graphite phonopy
model with a tiny configuration (mesh 4^3, ndir=40, mpdir=20, nphon=6,
6 alpha x 8 beta) and pins:

  * the alpha-summed S integral for mode 1 (incoherent-approx n=1 +
    multiphonon) and mode 2 (exact n=1 + multiphonon) to frozen values,
  * Teff0 bookkeeping,
  * physicality (finite, non-negative S),
  * determinism (a 2-worker run reproduces the serial tape byte for byte).

The pins guard the assembly end-to-end: phonopy load, FC resolution, site
matching, DOS tensor, one-phonon and multiphonon accumulation, SAB
conversion, and the writer. Physics-level validation (vs Euphonic and
OCLIMAX) lives in the dedicated harnesses.
"""
import os
import tempfile

import pytest

pytest.importorskip("phonopy")

from irma.core.engine import run_leapr  # noqa: E402

_YAML = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "mode2_euphonic_n1_validation", "graphite",
    "phonopy.yaml"))

_DECK = """20 /
'fast noncubic CI deck'/
1 1 6/
31 6012. 0 0 1e-100/
11.898 4.7392 1 10 0 0/
0/
1 1 0 {mode}/
2.467 2.467 6.701 90.0 90.0 120.0/
6 12 11.898 6.6484 0.001 4/
0.0 0.0 0.25  0.0 0.0 0.75  0.333333333333 0.666666666667 0.25  0.666666666667 0.333333333333 0.75/
'{yaml}'/
4 4 4 1 0/
40 20/
6 8 1/
0.1 0.4 1.0 2.5 6.0 15.0/
0.0 0.4 1.0 2.0 3.5 5.5 8.0 12.0/
296.0/
'fast nc ci'/
/
"""

# Frozen regression pins; re-bless only for an intended physics change.
_PINS = {1: 9.098600568286e-01, 2: 2.633836729729e+00}
_TEFF0 = 707.2952


def _run(mode, tag):
    d = tempfile.mkdtemp()
    inp = os.path.join(d, f"nc{tag}.input")
    out = os.path.join(d, f"nc{tag}.endf")
    with open(inp, "w") as f:
        f.write(_DECK.format(mode=mode, yaml=_YAML))
    run_leapr(inp, out)
    return out


def _mt4_stats(path):
    from endf_parserpy import EndfParserPy
    mt4 = EndfParserPy().parsefile(path)[7][4]
    s_sum, s_min = 0.0, float("inf")
    import math
    for b, tab in mt4["S_table"].items():
        if not isinstance(b, int):
            continue
        vals = tab["S"].values() if isinstance(tab["S"], dict) else tab["S"]
        for v in vals:
            assert math.isfinite(v)
            s_sum += v
            s_min = min(s_min, v)
    teff = mt4["teff0_table"]["Teff0"]
    teff0 = list(teff.values())[0] if isinstance(teff, dict) else teff[0]
    return s_sum, s_min, teff0


@pytest.fixture(scope="module")
def tapes():
    return {1: _run(1, "m1"), 2: _run(2, "m2")}


@pytest.mark.parametrize("mode", [1, 2])
def test_law_integral_pinned_and_physical(tapes, mode):
    s_sum, s_min, teff0 = _mt4_stats(tapes[mode])
    assert s_sum == pytest.approx(_PINS[mode], rel=1.0e-6)
    assert s_min >= 0.0
    assert teff0 == pytest.approx(_TEFF0, abs=0.01)
    assert teff0 > 296.0                      # bookkeeping sanity


def test_user_cutoff_reaches_complete_mode2_calculation(tapes, tmp_path, capsys):
    """The optional card must affect the integrated mode-2 calculation,
    which combines DOS/Debye-Waller setup, exact incoherent one-phonon,
    coherent one-phonon, and multiphonon scattering."""
    deck = _DECK.format(mode=2, yaml=_YAML).replace(
        "4 4 4 1 0/\n40 20/", "4 4 4 1 0/\n5.0/\n40 20/")
    inp = tmp_path / "cutoff.input"
    out = tmp_path / "cutoff.endf"
    inp.write_text(deck)
    run_leapr(str(inp), str(out))

    log = capsys.readouterr().out
    # the engine reports what the cutoff did to the displacements, once per
    # temperature, and the metadata carries the same numbers
    assert "Phonon-energy cutoff 5 meV at 296 K" in log
    assert "Mean-square displacement trace per atom" in log
    cutoff_sum, cutoff_min, _ = _mt4_stats(out)
    default_sum, _, _ = _mt4_stats(tapes[2])
    assert cutoff_min >= 0.0
    assert cutoff_sum != pytest.approx(default_sum, rel=1.0e-6)


def test_parallel_pool_matches_serial_byte_for_byte(tapes):
    """Cross-ncpu byte identity is structural: the block partitions are
    jobs-independent (fixed multiphonon direction chunk,
    noncubic_inelastic_context._MULTIPHONON_DIR_CHUNK; incoherent shell
    blocks own disjoint Q rows) and the ordered pool accumulates them in
    fixed order, so a 2-worker run reproduces the serial float64 sums bit
    for bit. Also exercises the ProcessPoolExecutor path."""
    d = tempfile.mkdtemp()
    inp = os.path.join(d, "ncpu2.input")
    out = os.path.join(d, "ncpu2.endf")
    with open(inp, "w") as f:
        f.write(_DECK.format(mode=1, yaml=_YAML).replace(
            "4 4 4 1 0/", "4 4 4 2 0/"))
    run_leapr(inp, out)
    assert open(out).read() == open(tapes[1]).read()


def test_split_principal_tape_matches_merged_single_type(tapes, tmp_path):
    """The same graphite cell spelled as TWO Card 6d carbon entries (2 + 2
    positions) must produce a byte-identical mode-1 tape to the single entry
    with 4 positions: the parse-time principal merge rebuilds that deck."""
    base = _DECK.format(mode=1, yaml=_YAML)
    split = base.replace("1 1 0 1/", "1 2 0 1/").replace(
        "6 12 11.898 6.6484 0.001 4/\n"
        "0.0 0.0 0.25  0.0 0.0 0.75  0.333333333333 0.666666666667 0.25  "
        "0.666666666667 0.333333333333 0.75/",
        "6 12 11.898 6.6484 0.001 2/\n"
        "0.0 0.0 0.25  0.0 0.0 0.75/\n"
        "6 12 11.898 6.6484 0.001 2/\n"
        "0.333333333333 0.666666666667 0.25  0.666666666667 0.333333333333 0.75/")
    assert split != base                       # the replace really happened
    inp = tmp_path / "two_types.input"
    out = tmp_path / "two_types.endf"
    inp.write_text(split)
    run_leapr(str(inp), str(out))
    # tapes[1] was built from `base`; file names are not written to the tape
    assert out.read_bytes() == open(tapes[1], "rb").read()
