"""The expected-tape validator must FAIL cleanly, never crash, on mismatches.

The grid-mismatch branch of compare_mt4 used to return a bare scalar where
the caller unpacks (worst_sig, worst_int), so the validator's PRIMARY
failure path died with a TypeError instead of printing FAIL — a validator
that crashes on disagreement cannot gate anything. This pins the repaired
behavior end-to-end by comparing two vendored reference tapes that were
made on different grids (liquid CH4: 89x81 vs ortho-H: 193x298).
"""
import os
import sys

import pytest

_VDIR = os.path.join(os.path.dirname(__file__), "native_LEAPR_NJOY_ENDF_validation")
sys.path.insert(0, _VDIR)

from validate_native_leapr_endf import main  # noqa: E402

_DECKS = os.path.join(_VDIR, "leapr_decks")


@pytest.mark.skipif(
    not os.path.exists(os.path.join(_DECKS, "tsl-l-CH4.endf.gz")),
    reason="vendored reference tapes not present",
)
def test_grid_mismatch_reports_fail_without_raising(capsys):
    rc = main([
        "validate", "l-CH4",
        "--tape", os.path.join(_DECKS, "tsl-ortho-H.endf.gz"),
    ])
    out = capsys.readouterr().out
    assert rc == 1                       # FAIL, not a crash
    assert "GRID MISMATCH" in out
    assert "RESULT: FAIL" in out
