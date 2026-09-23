"""Background computation runner for the IRMA GUI.

Runs a calculation as a child process (the ``irma`` CLI) in its own process
group and streams its combined stdout/stderr to a log callback. Cancel kills
the whole group, which includes the engine's worker pool; the pool is created
in the child (spawn start method), never forked from a GUI thread.
"""

import collections
import os
import signal
import subprocess
import sys
import threading

# Suppresses the macOS Objective-C fork abort for third-party code that forks
# in-process (IRMA itself never forks); setdefault keeps an explicit override.
os.environ.setdefault("OBJC_DISABLE_INITIALIZE_FORK_SAFETY", "YES")

_KILL_GRACE_SECONDS = 2.5


class ComputationRunner:
    """Runs an IRMA calculation in a cancellable background subprocess."""

    def __init__(self, log_callback=None, done_callback=None):
        self.log_callback = log_callback or (lambda x: None)
        self.done_callback = done_callback or (lambda ok, msg: None)
        self._proc = None
        self._running = False
        self._cancelled = False
        self._tail = collections.deque(maxlen=60)
        # Pin children to the launch directory: the ENDF form's phonopy
        # freq_max detection briefly chdir()s the process into a scratch
        # directory, and a relative output path must not resolve there.
        self._spawn_cwd = os.getcwd()

    @property
    def is_running(self):
        """True while a computation subprocess is active."""
        return self._running

    # -- command adapters -----------------------------------------------------
    def run(self, input_file, output_file, on_log=None, on_done=None):
        """LEAPR / ENDF-TSL evaluation."""
        self.run_command(
            [sys.executable, "-u", "-m", "irma", input_file, output_file],
            success_msg="Calculation completed successfully.",
            on_log=on_log, on_done=on_done, output_path=output_file,
            error_label="Input deck error")

    @staticmethod
    def _output_problem(output_path):
        """Error message if a zero-exit run left no (or an empty) output file."""
        if not output_path:
            return None
        try:
            size = os.path.getsize(output_path)
        except OSError:
            return ("Calculation reported success but no output file was "
                    f"written:\n  {output_path}")
        if size == 0:
            return ("Calculation reported success but the output file is "
                    f"empty:\n  {output_path}")
        return None

    # -- generic subprocess core ---------------------------------------------
    def run_command(self, argv, success_msg="Done.", on_log=None, on_done=None,
                    output_path=None, error_label="Input error"):
        """Start ``argv``; stream output; report ``on_done(ok, msg)`` on exit.

        ``on_log`` / ``on_done`` override the constructor callbacks for this
        run. A call while another run is active is ignored. ``output_path``,
        when given, must exist and be non-empty after a zero exit.
        ``error_label`` heads the message for exit code 2 (an input error).
        """
        if self._running:
            return
        on_log = on_log or self.log_callback
        on_done = on_done or self.done_callback
        self._cancelled = False
        self._tail.clear()
        # errors="replace": a non-UTF-8 byte must not kill the reader thread
        kw = dict(stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                  bufsize=1, universal_newlines=True, encoding="utf-8",
                  errors="replace", cwd=self._spawn_cwd)
        if os.name == "posix":
            kw["start_new_session"] = True            # own process group
        else:  # pragma: no cover - Windows
            kw["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        try:
            proc = subprocess.Popen(list(argv), **kw)
        except OSError as exc:
            on_done(False, f"Failed to start calculation: {exc}")
            return
        self._proc, self._running = proc, True
        threading.Thread(
            target=self._worker,
            args=(proc, success_msg, on_log, on_done, output_path, error_label),
            daemon=True).start()

    def _worker(self, proc, success_msg, on_log, on_done, output_path,
                error_label):
        """Reader thread: stream child output, then report completion."""
        try:
            try:
                for line in proc.stdout:
                    self._tail.append(line)
                    on_log(line)
            except Exception as exc:
                # A dead reader would leave the child blocked on a full pipe
                # and the panel waiting: kill the child and report.
                self._tail.append(f"[output reader failed: {exc}]\n")
                self._kill_tree(proc)
            rc = proc.wait()
            if self._cancelled:
                on_done(False, "Calculation cancelled.")
            elif rc == 0:
                problem = self._output_problem(output_path)
                on_done(False, problem) if problem else on_done(True, success_msg)
            elif rc == 2:
                on_done(False, f"{error_label}:\n\n" + self._error_tail())
            else:
                on_done(False, f"Calculation failed (exit {rc}):\n\n"
                        + self._error_tail())
        finally:
            # cleared only after on_done, so a new run cannot start in between
            self._proc = None
            self._running = False

    def _error_tail(self, n=12):
        """Last non-empty output lines, for error messages."""
        lines = [ln.rstrip("\n") for ln in self._tail if ln.strip()]
        return "\n".join(lines[-n:]) if lines else "(no output captured)"

    # -- cancellation ---------------------------------------------------------
    def cancel(self):
        """Terminate the calculation and its workers (non-blocking)."""
        proc = self._proc
        if proc is None:
            return
        self._cancelled = True
        threading.Thread(target=self._kill_tree, args=(proc,),
                         daemon=True).start()

    def shutdown(self):
        """Cancel and block until the process tree is gone. For GUI exit,
        where a daemon kill thread would die with the interpreter."""
        proc = self._proc
        if proc is None:
            return
        self._cancelled = True
        self._kill_tree(proc)

    @staticmethod
    def _kill_tree(proc):
        """SIGTERM the child's process group; SIGKILL it after a grace period."""
        if proc.poll() is not None:
            return
        if os.name != "posix":  # pragma: no cover - Windows
            proc.terminate()
            return
        try:
            pgid = os.getpgid(proc.pid)
            os.killpg(pgid, signal.SIGTERM)
            try:
                proc.wait(timeout=_KILL_GRACE_SECONDS)
            except subprocess.TimeoutExpired:
                os.killpg(pgid, signal.SIGKILL)
                proc.wait(timeout=1.0)
        except (OSError, subprocess.TimeoutExpired):
            pass
