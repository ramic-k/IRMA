"""irma.gui.runner -- generic subprocess runner and cancel.

The runner runs the calculation out-of-process in its own process group so a
single killpg tears down the engine's ProcessPoolExecutor workers. These tests
exercise the generic command path, exit-code -> message mapping, the LEAPR
argv adapter, and that cancel() terminates the forked worker processes, not
just the parent.
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


def test_second_run_ignored_while_running():
    r, logs, done, ev = _make_runner()
    r.run_command([sys.executable, "-u", "-c", "import time; time.sleep(1)"],
                  success_msg="first")
    time.sleep(0.3)
    assert r.is_running
    r.run_command([sys.executable, "-u", "-c", "print('second')"],
                  success_msg="second")   # must be ignored
    assert ev.wait(20)
    assert done["msg"] == "first"
    time.sleep(0.3)                             # a queued run would print now
    assert "second" not in [ln.strip() for ln in logs]


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


# ---- shutdown reaping -------------------------------------------------------
@pytest.mark.skipif(os.name != "posix", reason="process-group kill is POSIX")
def test_shutdown_reaps_child_before_returning(monkeypatch):
    """At GUI exit a plain cancel() would leave the SIGTERM->SIGKILL
    escalation to a daemon thread that dies with the interpreter, stranding
    the detached session-leader child. shutdown() runs the escalation on the
    calling thread and returns only once the child is gone, even when the
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


# ---- a "success" that wrote no (or an empty) output file is a failure --------
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


# ---- phase markers extracted from streamed child log lines -------------------
def test_phase_from_log_line_extracts_markers():
    pytest.importorskip("tkinter")
    from irma.gui.endf_form import _phase_from_log_line as phase
    assert (phase("Accumulating coherent one-phonon contribution...")
            == "Accumulating coherent one-phonon contribution...")
    assert phase("  Writing ENDF output...") == "Writing ENDF output..."
    # the longest real engine marker (~85 chars) must not be dropped by the cap
    longest = ("Evaluating coherent one-phonon dynamic structure factor "
               "for 123456 Q-vectors...")
    assert phase(longest) == longest


def test_phase_from_log_line_ignores_non_markers():
    pytest.importorskip("tkinter")
    from irma.gui.endf_form import _phase_from_log_line as phase
    assert phase("=== IRMA Calculation ===") is None
    assert phase("x" * 200 + "...") is None   # too long to be a real phase label
    assert phase("Loading...") is None        # bare single-word progress dots
    assert phase("Retrying...") is None       # are not engine phase markers
