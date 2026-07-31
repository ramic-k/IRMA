"""Pin the tape-gauge --diff comparison so it fails CLOSED.

tools/tape_gauge.py is the re-blessing instrument for modes-1/2 tape changes;
its 1e-6 relative criterion is only meaningful if the walk over the parsed
tapes cannot silently skip data. Two silent-skip failure modes are pinned
here: (a) a walk that parses only MF7/MT4 would print GAUGE: PASS through a
regression confined to MT2 coherent elastic (~98% of MF7 on the iel=10
gauge decks), and (b) a fail-open walk — dropped dict keys skipped, added
keys never visited, zip() truncating unequal lists — gives max_rel=0 when a
key is deleted or an array is cut from 12 to 2 entries. These tests drive the comparison helpers directly (no engine run,
the gauge's deck runs are minutes-expensive) and pin: structural mismatch =
inf, MT2 gauged alongside MT4, MT2-absent-from-both legitimate, and the
printed GAUGE verdict failing with exit code 1 when the metric exceeds tol.
"""
import importlib.util
import json
import math
import os

import pytest

# tools/ is not a package; load the gauge straight from its file path.
_TG_PATH = os.path.join(os.path.dirname(__file__), os.pardir,
                        "tools", "tape_gauge.py")
_spec = importlib.util.spec_from_file_location("tape_gauge", _TG_PATH)
tg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tg)


def _mt4():
    """A minimal parsed-MT4-like nested structure (dicts + lists + scalars)."""
    return {"ZA": 631.0, "AWR": 11.898, "LAT": 1,
            "table": {"S": [1.0, 2.0, 3.0, 4.0], "beta": [0.0, 0.5, 1.0, 1.5]}}


def _mt2():
    return {"ZA": 631.0, "LTHR": 1,
            "bragg": {"E": [1e-3, 2e-3, 3e-3], "S": [0.1, 0.2, 0.3]}}


# -----------------------------------------------------------------------------
# _walk_rel_diff: the recursive comparison itself
# -----------------------------------------------------------------------------
def test_identical_structures_diff_zero():
    assert tg._walk_rel_diff(_mt4(), _mt4()) == 0.0


def test_small_numeric_drift_reports_max_rel():
    a, b = _mt4(), _mt4()
    b["table"]["S"][2] = 3.0 * (1.0 + 2.5e-7)       # one perturbed value
    rel = tg._walk_rel_diff(a, b)
    # denom = max(|x|,|y|) -> rel is the perturbation divided by the new value
    assert rel == pytest.approx(2.5e-7 / (1.0 + 2.5e-7), rel=1e-12)
    assert rel < 1e-6                               # still within the criterion


def test_missing_dict_key_is_inf():
    a, b = _mt4(), _mt4()
    del b["table"]["beta"]                          # candidate dropped a key
    assert math.isinf(tg._walk_rel_diff(a, b))


def test_extra_dict_key_on_candidate_is_inf():
    a, b = _mt4(), _mt4()
    b["table"]["extra"] = [9.0]                     # candidate grew a key
    assert math.isinf(tg._walk_rel_diff(a, b))


def test_truncated_list_is_inf():
    a, b = _mt4(), _mt4()
    b["table"]["S"] = b["table"]["S"][:2]           # 4 entries cut to 2
    assert math.isinf(tg._walk_rel_diff(a, b))


def test_type_mismatch_leaf_is_inf():
    # numeric leaf replaced by a sub-structure must not be skipped silently
    a, b = _mt4(), _mt4()
    b["LAT"] = [1]
    assert math.isinf(tg._walk_rel_diff(a, b))


def test_bool_vs_int_leaf_is_inf():
    """bool is an int subclass: True vs 1 must be a structural failure, not a
    zero numeric diff."""
    a, b = _mt4(), _mt4()
    b["LAT"] = True                                 # was int 1
    assert math.isinf(tg._walk_rel_diff(a, b))
    a["LAT"], b["LAT"] = True, True                 # matching bools are fine
    assert tg._walk_rel_diff(a, b) == 0.0
    b["LAT"] = False                                # flipped flag fails
    assert math.isinf(tg._walk_rel_diff(a, b))


def test_numpy_scalars_get_relative_diff_treatment():
    """endf-parserpy currently yields builtin float/int, but numpy scalars
    must gauge as numbers (rel diff), not fall to exact-equality leaves."""
    import numpy as np
    a, b = _mt4(), _mt4()
    a["table"]["S"][1] = np.float64(2.0)
    b["table"]["S"][1] = np.float64(2.0 * (1.0 + 3e-7))
    assert tg._walk_rel_diff(a, b) == pytest.approx(3e-7 / (1.0 + 3e-7), rel=1e-9)
    a["ZA"], b["ZA"] = np.int64(631), 631.0         # cross-type numerics compare
    a["table"]["S"][1] = b["table"]["S"][1]
    assert tg._walk_rel_diff(a, b) == 0.0


# -----------------------------------------------------------------------------
# _mf7_rel_diff: MT2 coverage and presence rules
# -----------------------------------------------------------------------------
def test_mt2_only_change_reaches_the_metric():
    a = {2: _mt2(), 4: _mt4()}
    b = {2: _mt2(), 4: _mt4()}
    b[2]["bragg"]["S"][0] = 0.1 * (1.0 + 5e-5)      # MT2-only drift, MT4 identical
    rel = tg._mf7_rel_diff(a, b)
    assert rel == pytest.approx(5e-5 / (1.0 + 5e-5), rel=1e-12)
    assert rel > 1e-6                               # would flip the verdict to EXCEEDS


def test_mt2_absent_from_both_tapes_is_fine():
    # not every deck emits coherent elastic: MT4-only tapes must still gauge
    assert tg._mf7_rel_diff({4: _mt4()}, {4: _mt4()}) == 0.0


def test_mt2_present_on_one_side_only_is_inf():
    assert math.isinf(tg._mf7_rel_diff({2: _mt2(), 4: _mt4()}, {4: _mt4()}))
    assert math.isinf(tg._mf7_rel_diff({4: _mt4()}, {2: _mt2(), 4: _mt4()}))


# -----------------------------------------------------------------------------
# diff(): verdict format and fail-closed exit code
# -----------------------------------------------------------------------------
def _stage_gauge_dir(tmp_path, monkeypatch, rel):
    """Two labels with differing sha256 + dummy tapes; stub the parse-based
    metric so no endf_parserpy / engine run is needed."""
    monkeypatch.setattr(tg, "GAUGE_DIR", str(tmp_path))
    for label, digest in (("before", "aaa"), ("after", "bbb")):
        os.makedirs(tmp_path / label)
        (tmp_path / label / "mode2_2T.endf").write_text("dummy\n")
        (tmp_path / f"{label}.json").write_text(
            json.dumps({"mode2_2T": {"sha256": digest, "seconds": 1.0}}))
    monkeypatch.setattr(tg, "_tape_rel_diff", lambda a, b: rel)


def test_diff_structural_failure_exits_nonzero(tmp_path, monkeypatch, capsys):
    _stage_gauge_dir(tmp_path, monkeypatch, float("inf"))
    with pytest.raises(SystemExit) as exc:
        tg.diff("before", "after")
    assert exc.value.code == 1
    out = capsys.readouterr().out
    assert "*** EXCEEDS ***" in out
    assert "GAUGE: FAIL" in out


def test_diff_within_tolerance_passes(tmp_path, monkeypatch, capsys):
    _stage_gauge_dir(tmp_path, monkeypatch, 3.0e-7)
    with pytest.raises(SystemExit) as exc:
        tg.diff("before", "after")
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "DIFFER max_rel=3.000e-07 (WITHIN 1e-06)" in out
    assert "GAUGE: PASS" in out


def test_diff_extra_artifact_label_fails(tmp_path, monkeypatch, capsys):
    """A label present in only one run's JSON (an artifact appeared or
    vanished) must fail the gauge even when all shared labels agree —
    iterating only before-keys would miss an added artifact."""
    _stage_gauge_dir(tmp_path, monkeypatch, 0.0)
    after = json.loads((tmp_path / "after.json").read_text())
    after["mode9_new"] = {"sha256": "ccc", "seconds": 1.0}
    (tmp_path / "after.json").write_text(json.dumps(after))
    # shared label byte-identical: without the set check this would PASS
    before = json.loads((tmp_path / "before.json").read_text())
    before["mode2_2T"]["sha256"] = after["mode2_2T"]["sha256"]
    (tmp_path / "before.json").write_text(json.dumps(before))
    with pytest.raises(SystemExit) as exc:
        tg.diff("before", "after")
    assert exc.value.code == 1
    out = capsys.readouterr().out
    assert "ARTIFACT SET MISMATCH" in out and "mode9_new" in out
    assert "GAUGE: FAIL" in out
