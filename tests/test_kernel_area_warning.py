"""The multiphonon kernel-area guard must warn once per run, readably.

A "%.2f"-style percent format would render the guard's 1e-5 tolerance as
"tol 0.00%"
(1e-5 through a :.2f percent format) and its warn-once latch was a plain
module bool — per-PROCESS, so a fork pool repeated the warning once per
worker (~16 copies on the committed q-cuts example). The latch is now a
multiprocessing.Value handed to every pool worker through the spawn
initializer: the first tripping block warns for the whole pool, and
set_worker_state() re-arms it before each compute phase so a second
calculation in the same process still gets the guard.
"""
import multiprocessing as mp

import numpy as np

from irma.core import noncubic_workers as ncw
from irma.core.noncubic_workers import (
    _KERNEL_AREA_WARNED,
    _pool_worker_init,
    _warn_kernel_area_mismatch,
    set_worker_state,
)

# A 43% area deficit against an analytic MSD of 1.0 — the committed q-cuts
# example's failure mode before its grid was fixed.
_ANALYTIC = np.array([[1.0]])
_TRUNCATED = np.array([[0.57]])


def _rearm():
    """Re-arm the latch the same way the engine does before each phase."""
    set_worker_state({})


def test_matching_areas_do_not_warn(capsys):
    _rearm()
    _warn_kernel_area_mismatch(_ANALYTIC.copy(), _ANALYTIC.copy())
    assert capsys.readouterr().out == ""
    assert not _KERNEL_AREA_WARNED.value


def test_warning_names_tolerance_value_and_remedy(capsys):
    _rearm()
    _warn_kernel_area_mismatch(_TRUNCATED, _ANALYTIC)
    out = capsys.readouterr().out
    assert out.startswith("WARNING: multiphonon kernel area")
    assert "43.00%" in out
    # The printed number is the first tripping block's worst, not the run's.
    assert "first direction block" in out
    # The tolerance must render as its actual value, not "tol 0.00%".
    assert "tolerance 1e-05 relative" in out
    assert "0.00%" not in out
    # The message must say what to do about it.
    assert "e_max_meV" in out and "beta" in out


def test_warns_once_until_rearmed_then_again(capsys):
    _rearm()
    _warn_kernel_area_mismatch(_TRUNCATED, _ANALYTIC)
    _warn_kernel_area_mismatch(_TRUNCATED, _ANALYTIC)
    assert capsys.readouterr().out.count("WARNING") == 1
    _rearm()  # next compute phase: the guard must be live again
    _warn_kernel_area_mismatch(_TRUNCATED, _ANALYTIC)
    assert capsys.readouterr().out.count("WARNING") == 1


def _child_warn(_index):
    _warn_kernel_area_mismatch(_TRUNCATED, _ANALYTIC)
    # Read through the module: _pool_worker_init rebinds the module global to
    # the parent's latch, and a from-import taken at child import time would
    # still point at the child's own (unshared) Value.
    return ncw._KERNEL_AREA_WARNED.value


def test_latch_is_shared_across_spawn_workers(capfd):
    """Two sequential spawn children both trip the guard; exactly ONE warns,
    and the parent then stays quiet too. Spawn children do NOT inherit the
    latch — they get their own on import — so this exercises the production
    wiring: _pool_worker_init receives the parent's Value via initargs and
    rebinds the module global."""
    _rearm()
    ctx = mp.get_context("spawn")
    for i in range(2):
        with ctx.Pool(1, initializer=_pool_worker_init,
                      initargs=(_KERNEL_AREA_WARNED,)) as pool:
            assert pool.map(_child_warn, [i]) == [1]
    _warn_kernel_area_mismatch(_TRUNCATED, _ANALYTIC)  # parent, same latch
    out = capfd.readouterr().out
    assert out.count("WARNING") == 1
    _rearm()  # leave the module latch re-armed for other tests
