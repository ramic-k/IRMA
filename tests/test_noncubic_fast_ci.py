"""Fast CI coverage for the in-process noncubic MT4 assembly (modes 1/2).

The core noncubic physics previously had no fast test: only pure helpers
ran in CI, while the full harnesses are slow and manual. This drives
run_leapr end-to-end (NOT stubbed) on the vendored graphite phonopy model
with a deliberately tiny configuration (mesh 4^3, ndir=40, mpdir=20,
nphon=6, 6 alpha x 8 beta) and pins:

  * the alpha-summed S integral for mode 1 (incoherent-approx n=1 +
    multiphonon) and mode 2 (exact n=1 + multiphonon) to frozen values,
  * Teff0 bookkeeping,
  * physicality (finite, non-negative S),
  * determinism (two identical runs produce byte-identical tapes; the
    ordered worker pool and pinned BLAS threads exist precisely for this).

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

# Frozen regression pins (deterministic ordered pool; see module docstring).
# Re-blessed for QA4 F16 (first-energy-bin clamp): with the energy grid
# starting at 0, the first bin's width halved from [-de/2, de/2] to
# [0, de/2], DOUBLING the tabulated density at beta=0 (deposited weight and
# integrals are unchanged; every other row is bit-identical — verified at
# the 40^3 production gauge). The raw sum-of-S pins below include that row,
# hence the one-time shift (previously 8.394240797286e-01 / 1.728225100629).
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


def test_mode2_exact_n1_differs_from_mode1(tapes):
    s1, _, _ = _mt4_stats(tapes[1])
    s2, _, _ = _mt4_stats(tapes[2])
    # graphite's coherent n=1 roughly doubles the tiny-grid integral
    assert s2 > 1.5 * s1


def test_noncubic_run_is_deterministic(tapes):
    rerun = _run(1, "m1_again")
    a, b = open(tapes[1]).read(), open(rerun).read()
    assert a == b                              # byte-identical tapes


def test_parallel_pool_matches_serial_byte_for_byte(tapes):
    """Cross-ncpu byte identity is STRUCTURAL: the block partitions are
    jobs-independent (fixed multiphonon direction chunk, see
    noncubic_inelastic_context._MULTIPHONON_DIR_CHUNK; incoherent shell
    blocks own disjoint Q rows) and the ordered pool accumulates identical
    block lists in fixed order, so a 2-worker run reproduces the serial
    float64 sums bit-for-bit — not merely to within the ENDF writer's
    6-significant-figure rounding, as the old ceil(mpdir/num_jobs)
    partition did. Also exercises the ProcessPoolExecutor path (incl. its
    dead-worker detection wiring)."""
    d = tempfile.mkdtemp()
    inp = os.path.join(d, "ncpu2.input")
    out = os.path.join(d, "ncpu2.endf")
    with open(inp, "w") as f:
        f.write(_DECK.format(mode=1, yaml=_YAML).replace(
            "4 4 4 1 0/", "4 4 4 2 0/"))
    run_leapr(inp, out)
    assert open(out).read() == open(tapes[1]).read()


def test_batched_qpoints_eigh_matches_fallback(monkeypatch):
    """The stacked-eigh fast path and the QpointsPhonon fallback (used on
    phonopy >= 4, which removed run_dynamical_matrix_solver_c) must agree —
    this is the import break the phonopy-4.2.0 CI venv caught after perf
    round 2."""
    import numpy as np
    import phonopy
    import phonopy.harmonic.dynamical_matrix as dmmod
    from irma.core.noncubic_engine import _batched_qpoints_eigh
    from irma.core.phonopy_io import (
        isolated_phonopy_cwd, resolve_force_constants_source)

    fc_kwargs = resolve_force_constants_source(_YAML)
    with isolated_phonopy_cwd():
        ph = phonopy.load(phonopy_yaml=_YAML, is_nac=False, **fc_kwargs)
    ph.run_mesh([2, 2, 2], with_eigenvectors=False)   # builds the dyn matrix
    qpts = np.array([[0.1, 0.2, 0.3], [0.0, 0.0, 0.25], [0.4, -0.3, 0.1]])

    f_fast, e_fast = _batched_qpoints_eigh(ph.dynamical_matrix, qpts)
    if hasattr(dmmod, "run_dynamical_matrix_solver_c"):
        monkeypatch.delattr(dmmod, "run_dynamical_matrix_solver_c")
    f_ref, e_ref = _batched_qpoints_eigh(ph.dynamical_matrix, qpts)

    assert np.allclose(f_fast, f_ref, rtol=0, atol=1e-10)
    # eigenvector phases are LAPACK-arbitrary per column; compare physical
    # overlap |<e1|e2>| = 1 per (q, band) instead of raw components
    for iq in range(qpts.shape[0]):
        for ib in range(f_ref.shape[1]):
            v1, v2 = e_fast[iq][:, ib], e_ref[iq][:, ib]
            assert abs(abs(np.vdot(v1, v2)) - 1.0) < 1e-10


def test_split_principal_tape_matches_merged_single_type(tmp_path):
    """QA4 F3 equivalence proof: the same graphite cell spelled as TWO Card 6d
    carbon entries (2 + 2 positions) must produce a BYTE-IDENTICAL mode-1 tape
    to the canonical single entry with 4 positions — the parse-time principal
    merge reconstructs exactly that deck."""
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
    tapes_bytes = []
    for tag, deck in (("one_type", base), ("two_types", split)):
        inp = tmp_path / f"{tag}.input"
        out = tmp_path / f"{tag}.endf"
        inp.write_text(deck)
        run_leapr(str(inp), str(out))
        tapes_bytes.append(out.read_bytes())
    assert tapes_bytes[0] == tapes_bytes[1]
