"""Background computation runner for the IRMA GUI.

Runs a calculation as a CHILD PROCESS (the ``irma`` CLI) in its own process
group, streaming the child's combined stdout/stderr to a log callback. Running
out-of-process means:

  * **Cancel actually terminates workers.** The heavy mode-1/2 path spawns a
    ``ProcessPoolExecutor``; those workers inherit the child's process group, so
    one ``killpg`` tears the whole tree down -- impossible to do cleanly with the
    old in-thread model (a Python thread cannot be killed).
  * **No fork-from-Cocoa-thread hazard.** The pool is created inside the child
    process (spawn start method, never fork), not from a GUI thread while the
    tkinter/Cocoa loop is live, so the macOS fork-safety abort cannot happen in
    the GUI process.

The runner is generic over the command: ``run`` drives the LEAPR/ENDF path and
``run_config`` drives a neutron-scattering ``SpectraConfig`` -- both go through
``run_command`` so the GUI has one code path with one Cancel.
"""

import collections
import os
import signal
import subprocess
import sys
import threading
import time

# macOS fork-safety belt-and-suspenders. IRMA itself never forks (the
# noncubic pool uses the spawn start method), so this is purely defensive:
# it suppresses the macOS Objective-C fork abort for any third-party code
# that forks in-process, and setdefault preserves an explicit operator
# override.
os.environ.setdefault("OBJC_DISABLE_INITIALIZE_FORK_SAFETY", "YES")

_KILL_GRACE_SECONDS = 2.5

# Upper bound on how long shutdown() / cancel(wait=True) may block while
# reaping the process tree: covers the SIGTERM grace period, the SIGKILL
# escalation, and the final wait, with margin.
_SHUTDOWN_TIMEOUT_SECONDS = 4.0


class ComputationRunner:
    """Runs an IRMA calculation in a cancellable background subprocess."""

    def __init__(self, log_callback=None, done_callback=None):
        self.log_callback = log_callback or (lambda x: None)
        self.done_callback = done_callback or (lambda ok, msg: None)
        self._proc = None
        self._thread = None
        self._running = False
        self._cancelled = False
        # Serializes the run/cancel handoff. Invariant: _proc is non-None only
        # while it refers to the CURRENT run's live child -- run_command clears
        # it before starting a run and the worker clears it when the run
        # finishes, both under this lock, so cancel() can never signal a
        # previous run's (possibly recycled) process.
        self._lock = threading.Lock()
        self._tail = collections.deque(maxlen=60)
        # Pin spawned children to the launch directory so a relative output path
        # always resolves there, regardless of the live process cwd. The GUI's
        # phonopy freq_max detection briefly os.chdir()s the whole process into a
        # scratch dir (isolated_phonopy_cwd); without this pin a calculation
        # launched during that window would inherit the scratch cwd and write its
        # tape into a directory about to be deleted.
        self._spawn_cwd = os.getcwd()

    @property
    def is_running(self):
        """True while a computation subprocess is active."""
        return self._running

    # -- command adapters -----------------------------------------------------
    def run(self, input_file, output_file, on_log=None, on_done=None):
        """LEAPR / ENDF-TSL evaluation (the legacy GUI run)."""
        self.run_command(
            [sys.executable, "-u", "-m", "irma", input_file, output_file],
            success_msg="Calculation completed successfully.",
            on_log=on_log, on_done=on_done, output_path=output_file,
            error_label="Input deck error")

    def run_config(self, config_path, output_file, on_log=None, on_done=None):
        """Neutron-scattering spectrum from a SpectraConfig file."""
        self.run_command(
            [sys.executable, "-u", "-m", "irma.spectra", "run",
             config_path, "-o", output_file],
            success_msg=f"Spectrum written to {output_file}.",
            on_log=on_log, on_done=on_done, output_path=output_file,
            error_label="Spectra config error")

    @staticmethod
    def _output_problem(output_path):
        """Return an error message if a rc==0 run left no usable output, else None.

        A zero exit code is necessary but not sufficient: an interrupted or
        misconfigured run can exit 0 yet leave no file on disk -- or, when the
        path was pre-created (e.g. a NamedTemporaryFile target), an empty one.
        Reporting "completed successfully" then would be a silent lie. A single
        getsize() doubles as the existence probe (it raises OSError when absent),
        so this never needs a separate exists() and cannot TOCTOU between them.
        """
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
        """Start ``argv`` as a child process; stream output; report on exit.

        ``on_log`` / ``on_done`` override the constructor callbacks FOR THIS RUN
        only, so a single serialized runner can route output to whichever GUI
        panel (ENDF or neutron-scattering) launched it. The ``is_running`` guard
        keeps the two panels from colliding -- a second run is ignored while one
        is active. ``output_path``, when given, is verified on a zero exit (it
        must exist and be non-empty) so every run path -- ENDF, spectrum, and the
        2-D map -- shares one "exit 0 must mean a file was written" guarantee.

        ``error_label`` heads the exit-2 (user-input) message. Exit 2 means
        different things per command -- a LEAPR deck error for ``run``, a
        spectra config / output-path problem for ``run_config`` -- so each
        adapter passes its own label instead of every failure reading
        "Input deck error" (a concept the spectra workflow does not have).
        """
        with self._lock:
            if self._running:
                return
            self._running = True
            self._cancelled = False
            self._proc = None    # never let a stale proc leak into this run
        self._tail.clear()
        self._active_log = on_log or self.log_callback
        self._active_done = on_done or self.done_callback
        self._thread = threading.Thread(
            target=self._worker,
            args=(list(argv), success_msg, output_path, error_label),
            daemon=True)
        self._thread.start()

    def _popen_kwargs(self):
        # errors="replace": a stray non-UTF-8 byte in the child's output must
        # not raise in the reader thread (which would wedge the panel mid-run).
        # cwd: pin to the launch directory (see __init__) so a relative output
        # path is never resolved against a transient detection scratch cwd.
        """Popen keyword set shared by every launch method."""
        kw = dict(stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                  bufsize=1, universal_newlines=True,
                  encoding="utf-8", errors="replace", cwd=self._spawn_cwd)
        if os.name == "posix":
            kw["start_new_session"] = True            # own process group -> killpg
        else:  # pragma: no cover - Windows
            kw["creationflags"] = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        return kw

    def _worker(self, argv, success_msg, output_path=None,
                error_label="Input error"):
        """Reader thread: stream child output to the log, then report completion."""
        on_log, on_done = self._active_log, self._active_done
        with self._lock:
            cancelled_early = self._cancelled
        if cancelled_early:
            # Cancelled before the child was even spawned: skip the spawn
            # entirely instead of forking a process just to kill it.
            try:
                on_done(False, "Calculation cancelled.")
            finally:
                with self._lock:
                    self._running = False
            return
        try:
            proc = subprocess.Popen(argv, **self._popen_kwargs())
        except Exception as exc:  # pragma: no cover - spawn failure
            with self._lock:
                self._running = False
            on_done(False, f"Failed to start calculation: {exc}")
            return
        with self._lock:
            self._proc = proc
            pending_cancel = self._cancelled
        if pending_cancel:
            # cancel() arrived in the spawn window, after run_command but
            # before _proc existed; it could not signal anything then, so
            # honor the recorded request now, on this thread.
            self._kill_tree(proc)
        try:
            try:
                for line in proc.stdout:
                    self._tail.append(line)
                    on_log(line)
            except Exception as exc:
                # The reader must never die silently: a wedged stdout would
                # leave Run disabled and the child blocked on a full pipe.
                # Tear the child's tree down and report through on_done.
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
                on_done(
                    False, f"Calculation failed (exit {rc}):\n\n" + self._error_tail())
        finally:
            # Clear _running only AFTER the completion handler has been dispatched,
            # not the instant the child exits. Otherwise a second run_command in
            # the window between proc.wait() and on_done could pass the
            # `if self._running` guard and launch an overlapping run that clobbers
            # the shared _cfg_tmp / state. _proc is cleared here too, under the
            # lock, so a late cancel() cannot pick up this (finished) run's
            # process and signal it during the NEXT run.
            with self._lock:
                self._proc = None
                self._running = False

    def _error_tail(self, n=12):
        """Last non-empty output lines, for error messages."""
        lines = [ln.rstrip("\n") for ln in self._tail if ln.strip()]
        return "\n".join(lines[-n:]) if lines else "(no output captured)"

    # -- cancellation ---------------------------------------------------------
    def cancel(self, wait=False, timeout=_SHUTDOWN_TIMEOUT_SECONDS):
        """Terminate the calculation and ALL its worker processes.

        Non-blocking by default: the SIGTERM -> SIGKILL escalation runs on a
        daemon thread so the GUI stays responsive. With ``wait=True`` the
        escalation runs on THIS thread and cancel only returns once the child
        has been reaped or ``timeout`` elapsed -- required at interpreter
        shutdown, where daemon threads die mid-escalation (see ``shutdown``).

        A cancel issued in the spawn window (run started, child not forked
        yet, ``_proc`` still None) is recorded via ``_cancelled``: the worker
        re-checks the flag right after assigning ``_proc`` and kills the fresh
        child immediately, so the request cannot be silently dropped. A cancel
        after the run has finished is a no-op.
        """
        with self._lock:
            if not self._running:
                return
            self._cancelled = True
            proc = self._proc
        if wait:
            self._reap(proc, timeout)
        elif proc is not None:
            threading.Thread(target=self._kill_tree, args=(proc,),
                             daemon=True).start()
        # proc is None (spawn window): the worker kills the child itself.

    def shutdown(self, timeout=_SHUTDOWN_TIMEOUT_SECONDS):
        """Cancel any active run and wait (bounded) until its tree is reaped.

        For GUI exit. A plain cancel() delegates the SIGTERM -> SIGKILL
        escalation to a daemon thread; if the interpreter exits right after
        (root.destroy() + mainloop return), that thread dies before it can
        escalate, and the child -- a detached session leader -- survives at
        full CPU. shutdown() runs the escalation on the calling thread, so
        when it returns the process tree is gone (or ``timeout`` elapsed).
        """
        self.cancel(wait=True, timeout=timeout)

    def _reap(self, proc, timeout):
        """Synchronously kill the current run's tree and wait for the child."""
        deadline = time.monotonic() + timeout
        while proc is None:
            # Spawn window: wait for the worker to assign _proc (or for the
            # run to end without one, e.g. a pre-spawn cancel / failed spawn).
            time.sleep(0.01)
            with self._lock:
                if not self._running:
                    return
                proc = self._proc
            if proc is None and time.monotonic() >= deadline:
                return   # pragma: no cover - spawn stuck longer than timeout
        self._kill_tree(proc)
        try:
            proc.wait(timeout=max(0.1, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:  # pragma: no cover - unkillable child
            pass

    def _kill_tree(self, proc):
        """Terminate ``proc`` and its whole process group/tree."""
        if proc is None or proc.poll() is not None:
            return
        if os.name == "posix":
            try:
                pgid = os.getpgid(proc.pid)
            except ProcessLookupError:
                return
            self._signal_group(pgid, signal.SIGTERM)
            try:
                proc.wait(timeout=_KILL_GRACE_SECONDS)
            except subprocess.TimeoutExpired:
                self._signal_group(pgid, signal.SIGKILL)   # escalate: kill the pool
        else:  # pragma: no cover - Windows
            try:
                proc.terminate()
            except OSError:
                pass

    @staticmethod
    def _signal_group(pgid, sig):
        """Send ``sig`` to the process group, ignoring races."""
        try:
            os.killpg(pgid, sig)
        except (ProcessLookupError, PermissionError, OSError):
            pass
