"""irma.gui.runner (P7) -- generic subprocess runner + cancel-kills-workers.

The runner runs the calculation out-of-process in its own process group so a
single killpg tears down the engine's ProcessPoolExecutor workers. These tests
exercise the generic command path, exit-code -> message mapping, the LEAPR /
spectra argv adapters, and -- the P7 gate -- that cancel() actually terminates
forked worker processes (not just the parent).
"""
import os
import sys
import threading
import time

import pytest

from irma.gui.runner import ComputationRunner


def _wait(pred, timeout=30.0, interval=0.02):
    end = time.time() + timeout
    while time.time() < end:
        if pred():
            return True
        time.sleep(interval)
    return False


def _alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _make_runner():
    logs, done = [], {}
    ev = threading.Event()

    def on_done(ok, msg):
        done["ok"], done["msg"] = ok, msg
        ev.set()

    r = ComputationRunner(log_callback=logs.append, done_callback=on_done)
    return r, logs, done, ev


# ---- generic command + exit-code mapping ------------------------------------
def test_runs_command_and_streams_output():
    r, logs, done, ev = _make_runner()
    r.run_command([sys.executable, "-u", "-c", "print('hello world')"],
                  success_msg="ok")
    assert ev.wait(20)
    assert done["ok"] is True and done["msg"] == "ok"
    assert any("hello world" in ln for ln in logs)
    assert r.is_running is False


def test_exit_code_2_default_label_is_generic_input_error():
    # run_command itself knows nothing about decks: the generic default is
    # "Input error"; the run()/run_config() adapters pass their own labels.
    r, logs, done, ev = _make_runner()
    r.run_command([sys.executable, "-u", "-c",
                   "import sys; print('  bad card 6'); sys.exit(2)"])
    assert ev.wait(20)
    assert done["ok"] is False
    assert done["msg"].startswith("Input error")
    assert "bad card 6" in done["msg"]


def test_exit_code_2_uses_caller_error_label():
    """A spectra child exiting 2 (SpectraConfigError /
    output-path preflight) must not be reported as an 'Input deck error' --
    a LEAPR concept the spectra workflow does not have."""
    r, logs, done, ev = _make_runner()
    r.run_command(
        [sys.executable, "-u", "-c",
         "import sys; print('phonopy_yaml: no such file'); sys.exit(2)"],
        error_label="Spectra config error")
    assert ev.wait(20)
    assert done["ok"] is False
    assert done["msg"].startswith("Spectra config error:")
    assert "phonopy_yaml" in done["msg"]


def test_exit_code_3_is_failure_with_tail():
    r, logs, done, ev = _make_runner()
    r.run_command([sys.executable, "-u", "-c",
                   "import sys; print('boom detail'); sys.exit(3)"])
    assert ev.wait(20)
    assert done["ok"] is False
    assert "exit 3" in done["msg"] and "boom detail" in done["msg"]


def test_non_utf8_output_does_not_kill_reader():
    """QA4 F32: a stray non-UTF-8 byte in the child's output must not crash the
    reader thread -- bad bytes are replaced and on_done still fires."""
    r, logs, done, ev = _make_runner()
    r.run_command(
        [sys.executable, "-u", "-c",
         "import sys; sys.stdout.buffer.write(b'ok line\\n\\xff\\xfe bad bytes\\n')"],
        success_msg="ok")
    assert ev.wait(20)
    assert done["ok"] is True and done["msg"] == "ok"
    assert any("ok line" in ln for ln in logs)
    assert r.is_running is False


def test_second_run_ignored_while_running():
    r, logs, done, ev = _make_runner()
    r.run_command([sys.executable, "-u", "-c", "import time; time.sleep(2)"],
                  success_msg="first")
    time.sleep(0.3)
    assert r.is_running
    r.run_command([sys.executable, "-u", "-c", "print('second')"],
                  success_msg="second")   # must be ignored
    assert ev.wait(20)
    assert done["msg"] == "first"


# ---- argv adapters ----------------------------------------------------------
def test_run_builds_leapr_argv(monkeypatch):
    r = ComputationRunner()
    seen = {}
    monkeypatch.setattr(r, "run_command",
                        lambda argv, success_msg="", on_log=None, on_done=None,
                        output_path=None, error_label="Input error":
                        seen.update(argv=argv, msg=success_msg, out=output_path,
                                    label=error_label))
    r.run("deck.input", "out.endf")
    assert seen["argv"][1:] == ["-u", "-m", "irma", "deck.input", "out.endf"]
    assert "completed successfully" in seen["msg"]
    assert seen["out"] == "out.endf"        # output verified on exit 0
    assert seen["label"] == "Input deck error"   # LEAPR runs ARE deck errors


def test_run_config_builds_spectra_argv(monkeypatch):
    r = ComputationRunner()
    seen = {}
    monkeypatch.setattr(r, "run_command",
                        lambda argv, success_msg="", on_log=None, on_done=None,
                        output_path=None, error_label="Input error":
                        seen.update(argv=argv, out=output_path,
                                    label=error_label))
    r.run_config("cfg.yaml", "spec.csv")
    assert seen["argv"][1:] == ["-u", "-m", "irma.spectra", "run", "cfg.yaml",
                                "-o", "spec.csv"]
    assert seen["out"] == "spec.csv"
    assert seen["label"] == "Spectra config error"   # not "Input deck error"


# ---- THE GATE: cancel terminates the forked workers -------------------------
@pytest.mark.skipif(os.name != "posix", reason="process-group kill is POSIX")
def test_cancel_terminates_forked_workers():
    prog = (
        "import multiprocessing as mp, time, os\n"
        "ctx = mp.get_context('fork')\n"
        "def w():\n"
        "    time.sleep(120)\n"
        "ps = [ctx.Process(target=w) for _ in range(3)]\n"
        "[p.start() for p in ps]\n"
        "print('PIDS', os.getpid(), ','.join(str(p.pid) for p in ps), flush=True)\n"
        "time.sleep(120)\n"
    )
    r, logs, done, ev = _make_runner()
    r.run_command([sys.executable, "-u", "-c", prog])

    # wait for the child to report its + the workers' PIDs
    assert _wait(lambda: any("PIDS" in ln for ln in logs), timeout=30), "no PIDS line"
    line = next(ln for ln in logs if "PIDS" in ln)
    _, parent_pid, worker_csv = line.split()
    worker_pids = [int(x) for x in worker_csv.strip().split(",")]
    parent_pid = int(parent_pid)
    assert all(_alive(p) for p in worker_pids), "workers should be alive pre-cancel"

    r.cancel()
    assert ev.wait(30), "done_callback never fired after cancel"
    assert done["ok"] is False and "cancelled" in done["msg"].lower()

    # parent AND every forked worker must be gone (killpg tore down the group)
    assert _wait(lambda: not _alive(parent_pid), timeout=15), "parent survived"
    for p in worker_pids:
        assert _wait(lambda p=p: not _alive(p), timeout=15), f"worker {p} survived cancel"
    assert r.is_running is False


def test_cancel_when_idle_is_noop():
    r, logs, done, ev = _make_runner()
    r.cancel()   # nothing running -> must not raise
    assert r.is_running is False


def test_shutdown_when_idle_is_noop():
    r, logs, done, ev = _make_runner()
    r.shutdown()   # nothing running -> must return immediately, not raise
    assert r.is_running is False


# ---- shutdown reaping -------------------------------------------------------
@pytest.mark.skipif(os.name != "posix", reason="process-group kill is POSIX")
def test_shutdown_reaps_child_before_returning(monkeypatch):
    """S8 gap 3: at GUI exit a plain cancel() delegates the SIGTERM->SIGKILL
    escalation to a daemon thread that dies with the interpreter, stranding
    the detached session-leader child. shutdown() must run the escalation on
    the calling thread and only return once the child is gone -- even when the
    child ignores SIGTERM (forcing the SIGKILL escalation)."""
    import irma.gui.runner as runner_mod
    monkeypatch.setattr(runner_mod, "_KILL_GRACE_SECONDS", 0.3)  # fast escalation
    prog = (
        "import os, signal, time\n"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"   # force SIGKILL path
        "print('PID', os.getpid(), flush=True)\n"
        "time.sleep(120)\n"
    )
    r, logs, done, ev = _make_runner()
    r.run_command([sys.executable, "-u", "-c", prog])
    assert _wait(lambda: any("PID" in ln for ln in logs), timeout=20), "no PID line"
    pid = int(next(ln for ln in logs if "PID" in ln).split()[1])
    assert _alive(pid)

    r.shutdown()
    # No polling grace here: by the time shutdown() returns, the SIGTERM ->
    # SIGKILL escalation must have completed and the child been reaped.
    assert not _alive(pid), "child survived shutdown()"
    assert ev.wait(10)
    assert done["ok"] is False and "cancelled" in done["msg"].lower()
    assert _wait(lambda: not r.is_running, 10)


# ---- #8: a "success" that wrote no (or an empty) output file is a failure ----
def test_output_problem_none_when_present(tmp_path):
    out = tmp_path / "tape.endf"
    out.write_text("data")
    assert ComputationRunner._output_problem(str(out)) is None


def test_output_problem_none_when_unchecked():
    assert ComputationRunner._output_problem(None) is None   # no path -> no check


def test_output_problem_when_missing(tmp_path):
    msg = ComputationRunner._output_problem(str(tmp_path / "missing.endf"))
    assert msg and "no output file" in msg


def test_output_problem_when_empty(tmp_path):
    out = tmp_path / "empty.endf"
    out.write_text("")
    msg = ComputationRunner._output_problem(str(out))
    assert msg and "empty" in msg


def test_run_command_success_with_missing_output_is_failure(tmp_path):
    """End-to-end: a child that exits 0 but writes nothing to output_path is
    reported as a failure, not a silent success (covers all three run paths)."""
    r, logs, done, ev = _make_runner()
    out = tmp_path / "never_written.endf"
    r.run_command([sys.executable, "-u", "-c", "print('done, honest')"],
                  success_msg="ok", output_path=str(out))
    assert ev.wait(20)
    assert done["ok"] is False and "no output file" in done["msg"]


def test_run_command_success_with_written_output_passes(tmp_path):
    out = tmp_path / "written.endf"
    r, logs, done, ev = _make_runner()
    prog = f"open({str(out)!r}, 'w').write('tape')"
    r.run_command([sys.executable, "-u", "-c", prog],
                  success_msg="ok", output_path=str(out))
    assert ev.wait(20)
    assert done["ok"] is True and done["msg"] == "ok"


# ---- #12: phase markers extracted from streamed child log lines --------------
def test_phase_from_log_line_extracts_markers():
    pytest.importorskip("tkinter")
    from irma.gui.app import _phase_from_log_line as phase
    assert (phase("Accumulating coherent one-phonon contribution...")
            == "Accumulating coherent one-phonon contribution...")
    assert phase("  Writing ENDF output...") == "Writing ENDF output..."
    # the longest real engine marker (~85 chars) must not be dropped by the cap
    longest = ("Evaluating coherent one-phonon dynamic structure factor "
               "for 123456 Q-vectors...")
    assert phase(longest) == longest


def test_phase_from_log_line_ignores_non_markers():
    pytest.importorskip("tkinter")
    from irma.gui.app import _phase_from_log_line as phase
    assert phase("=== IRMA Calculation ===") is None
    assert phase("0.123  4.56  7.89") is None
    assert phase("") is None
    assert phase("...") is None
    assert phase("x" * 200 + "...") is None   # too long to be a real phase label
    assert phase("Loading...") is None        # bare single-word progress dots
    assert phase("Retrying...") is None       # are not engine phase markers


# ---------- re-entrancy: _running is held through on_done ----------

def test_running_flag_held_through_on_done():
    """If the worker cleared `_running` the instant the child exited — before
    the completion handler ran — a second run_command in that window would
    pass the `if self._running` guard and launch an overlapping run. The
    flag must be held until on_done has been dispatched."""
    seen = {}
    ev = threading.Event()
    r = ComputationRunner(log_callback=lambda *_: None,
                          done_callback=lambda ok, msg: None)

    def on_done(ok, msg):
        # _running must still be True while the completion handler runs, so a
        # re-entrant run_command in this window is blocked by the guard.
        seen["running_at_done"] = r.is_running
        ev.set()

    r.run_command([sys.executable, "-c", "pass"], success_msg="ok", on_done=on_done)
    assert ev.wait(20)
    assert seen["running_at_done"] is True, (
        "_running was cleared before on_done -- the re-entrancy window is open")
    # and it must be cleared once the worker fully returns
    assert _wait(lambda: r.is_running is False, 5)
