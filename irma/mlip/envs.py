"""Per-potential environment isolation for the MLIP front end.

The potential packages have mutually unsatisfiable pins (mace-torch needs
e3nn==0.4.4 while sevenn/mattersim need >=0.5; deepmd-kit's binaries are
ABI-locked to one torch), so one Python environment cannot host every
backend. This module lets each potential run under its OWN interpreter:

- a registry (envs.json in the irma-mlip cache, or the
  IRMA_MLIP_PYTHON_<POTENTIAL> environment variable) maps a potential to
  a Python interpreter;
- RemoteCalculator proxies ASE energy/forces/stress calls to a
  force_server.py subprocess running under that interpreter (the foreign
  env needs only ase + the potential package, never irma);
- create_env() provisions such an environment automatically from a
  curated (but UNPINNED) requirement set and registers it.

Dispatch is wired inside irma.mlip.calculators: when a registered
interpreter differs from the running one, make_calculator/
canonicalize_spec/resolved_checkpoint_identity transparently round-trip
through the server. Module-level imports here are stdlib-only (the
bare-install CI job imports irma.mlip).
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys

_SERVER = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "force_server.py")
_CALCULATORS = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "calculators.py")

# Curated pip requirement sets for auto-provisioned environments. These
# are deliberately UNPINNED package names, not a vetted lockfile: the
# resolver installs whatever the configured index currently serves, so
# what lands is whatever is current, not a vetted set. (Pinning would be
# false assurance — these stacks move fast, the right pins differ per
# platform/Python, and each backend package pins its own critical deps,
# e.g. deepmd-kit[torch] pins the torch its binaries were built against,
# which is exactly what a dedicated env is for.) The import check below
# gates registration on the env actually working.
ENV_REQUIREMENTS = {
    "mattersim": ("mattersim", "ase>=3.23"),
    "orb": ("orb-models", "ase>=3.23"),
    "sevennet": ("sevenn", "ase>=3.23"),
    "mace": ("mace-torch", "ase>=3.23"),
    "mace-off": ("mace-torch", "ase>=3.23"),
    "pet-mad": ("upet", "ase>=3.23"),
    "dpa3": ("deepmd-kit[torch]", "mpich", "huggingface_hub", "ase>=3.23"),
    "nequip": ("nequip", "ase>=3.23"),
    "grace": ("tensorpotential", "ase>=3.23"),
}

# potentials that share one package land in one shared env
_SHARED_ENVS = {"mace": ("mace", "mace-off"), "mace-off": ("mace", "mace-off")}

# provisioning verification imports the CALCULATOR entry point, not just
# the top-level package: deepmd's mpich/torch-ABI failures and friends
# fire on the calculator import, and a broken env must not be registered
_IMPORT_CHECKS = {
    "mattersim": "from mattersim.forcefield.potential import "
                 "MatterSimCalculator",
    "orb": "from orb_models.forcefield import pretrained",
    "sevennet": "from sevenn.calculator import SevenNetCalculator",
    "mace": "from mace.calculators import mace_mp",
    "mace-off": "from mace.calculators import mace_off",
    "pet-mad": "from upet.calculator import UPETCalculator",
    "dpa3": "from deepmd.calculator import DP",
    "nequip": "import nequip.ase",
    "grace": "from tensorpotential.calculator import grace_fm",
}

# Python series for auto-provisioned potential envs (uv path only: the
# stdlib venv module can only clone the running interpreter). Potential
# packages chronically lag new interpreters, and a host on the newest
# Python forces the env onto whatever bleeding-edge torch has wheels
# there, which is where the breakage lives (a 3.14 host resolved the
# newest torch, whose export path died compiling nequip's zoo model).
# The env's interpreter is invisible to the user; the force-server
# protocol is JSON over stdio, so the outer process and the env may run
# different Python versions.
_ENV_PYTHON = "3.12"

# Functional probe run in the fresh env AFTER the import check. The
# import check alone is too weak: a torch wheel built against NumPy 1.x
# imports fine next to NumPy 2.x (the failure is a warning, not an
# exception) and then dies at force time. The probe turns that warning
# into an error and exercises the tensor<->array round-trip that the
# force server actually needs. Envs without torch (grace) pass
# trivially.
_RUNTIME_PROBE = """\
import warnings
warnings.filterwarnings("error", message="Failed to initialize NumPy")
import numpy as np
try:
    import torch
except ImportError:
    torch = None
if torch is not None:
    # explicit raises, not assert: PYTHONOPTIMIZE strips asserts and
    # would let a one-directional failure through the gate
    t = torch.from_numpy(np.ones(3))
    if float(t.sum()) != 3.0:
        raise SystemExit("numpy->tensor round trip produced wrong data")
    if t.numpy().shape != (3,):
        raise SystemExit("tensor->numpy conversion failed")
"""

# Signatures of a torch wheel built against NumPy 1.x running next to
# NumPy 2.x. Seen in the wild on Intel macs, where torch wheels stopped
# at 2.2.2 (NumPy-1 ABI) while the resolver installs a current NumPy.
_NUMPY_ABI_SIGNATURES = (
    "_ARRAY_API not found",
    "Failed to initialize NumPy",
    "compiled using NumPy 1.x",
    "Numpy is not available",
)


def _numpy_abi_break(text: str) -> bool:
    return any(sig in text for sig in _NUMPY_ABI_SIGNATURES)


def _error_hint(text: str) -> str | None:
    """One actionable sentence for known failure signatures, or None."""
    if _numpy_abi_break(text):
        return ("the potential env's torch was built against NumPy 1.x "
                "but NumPy >= 2 sits next to it; recreate the env "
                "(irma mlip env remove <potential>; irma mlip env create "
                "<potential>) — creation detects this breakage and pins "
                "numpy<2 automatically")
    if "TF32" in text and "cuDNN" in text:
        return ("known torch.export defect with mixed cuDNN TF32 flags "
                "on bleeding-edge torch; recreate the env with uv "
                "installed so it is provisioned on the pinned Python "
                f"{_ENV_PYTHON} series with a mature torch")
    return None


def _install_failure_detail(result) -> str:
    """Failure text for a provisioning step, keeping pip's diagnosis.

    pip writes its final ERROR lines to stderr but the useful part, the
    "The conflict is caused by:" section naming the impossible pins, to
    stdout. Taking stderr alone (the old behavior) discards exactly the
    lines that say which requirement cannot be satisfied on this
    platform.
    """
    err = (result.stderr or "").strip()
    out = (result.stdout or "").strip()
    marker = "conflict is caused by"
    pos = out.lower().find(marker)
    if pos >= 0:
        # The section's HEAD names the irreconcilable requirements; a
        # long candidate walk follows. Budget the section from its start
        # (a plain tail slice could cut the marker and the named pins,
        # keeping only candidate noise) and append the stderr tail.
        start = out.rfind("\n", 0, pos) + 1
        section = out[start:].strip()
        if len(section) > 1600:
            section = section[:1600] + "\n[... resolver output truncated]"
        return (section + "\n" + err[-400:]).strip()
    return (err or out)[-2000:]


def _intel_mac() -> bool:
    import platform
    return platform.system() == "Darwin" and platform.machine() == "x86_64"


_INTEL_MAC_NOTE = (
    "  note: this is an Intel (x86_64) Mac. torch stopped shipping\n"
    "  Intel-mac wheels at 2.2.2 (April 2024), so most potentials can\n"
    "  no longer install or run here at current versions; nequip may\n"
    "  work via the automatic numpy<2 fallback. For the MLIP front end\n"
    "  prefer Linux or an Apple-Silicon Mac. Everything else in IRMA\n"
    "  works normally on this machine.")


class MlipEnvError(RuntimeError):
    """Environment provisioning or dispatch failed."""


def _cache_dir() -> str:
    return os.environ.get(
        "IRMA_MLIP_CACHE", os.path.expanduser("~/.cache/irma-mlip"))


def _registry_path() -> str:
    return os.path.join(_cache_dir(), "envs.json")


def _load_registry() -> dict:
    try:
        with open(_registry_path()) as fh:
            data = json.load(fh)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}
    if not isinstance(data, dict):
        return {}
    table = data.get("interpreters", {})
    return table if isinstance(table, dict) else {}


def _save_registry(table: dict):
    import tempfile
    os.makedirs(_cache_dir(), exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=_cache_dir(), suffix=".envs.tmp")
    with os.fdopen(fd, "w") as fh:
        json.dump({"schema": 1, "interpreters": table}, fh, indent=2)
    os.replace(tmp, _registry_path())


def _env_var(potential: str) -> str:
    return "IRMA_MLIP_PYTHON_" + potential.upper().replace("-", "_")


def _checked_interpreter(interpreter: str, source: str) -> str:
    """Reject a registry/override value that is not an executable file.

    Applied on WRITE (register_interpreter) and on every READ
    (registered_interpreter): both the envs.json registry and the
    IRMA_MLIP_PYTHON_* variables are plain editable state, and whatever
    string they yield is handed to subprocess.Popen — so an entry that is
    not an executable file must fail here, with its source named, rather
    than be executed.
    """
    if not (os.path.isfile(interpreter) and os.access(interpreter, os.X_OK)):
        raise MlipEnvError(
            f"{interpreter!r} ({source}) is not an executable interpreter; "
            f"fix or unset it (`irma mlip env create <potential>` rebuilds "
            f"a dedicated env, `irma mlip env remove <potential>` drops a "
            f"stale registration)")
    return interpreter


def registered_interpreter(potential: str) -> str | None:
    """The interpreter registered for a potential, or None.

    The IRMA_MLIP_PYTHON_<POTENTIAL> environment variable overrides the
    registry file; an empty value explicitly disables dispatch.

    The returned string is what the dispatch client passes to
    subprocess.Popen, so it is validated ON READ exactly as
    register_interpreter validates on write: a value (from either
    source) that is not an executable file raises MlipEnvError instead
    of being returned for execution.
    """
    env_var = _env_var(potential)
    override = os.environ.get(env_var)
    if override is not None:
        if not override:
            return None
        return _checked_interpreter(override, env_var)
    interp = _load_registry().get(potential)
    if interp is None:
        return None
    return _checked_interpreter(str(interp), f"registered in {_registry_path()}")


def register_interpreter(potential: str, interpreter: str):
    _checked_interpreter(interpreter, "being registered")
    table = _load_registry()
    table[potential] = interpreter
    _save_registry(table)
    _invalidate_meta(potential)


def unregister_interpreter(potential: str) -> bool:
    table = _load_registry()
    _invalidate_meta(potential)
    if potential not in table:
        return False
    del table[potential]
    _save_registry(table)
    return True


def is_dispatched(potential: str) -> bool:
    """True when this potential runs under a DIFFERENT interpreter.

    Compared by LAUNCHER PATH, deliberately not realpath: venv pythons
    are symlinks to their base interpreter, so realpath would collapse a
    provisioned env onto the running one and silently disable dispatch
    (review finding). Two aliases of the same environment therefore
    compare as different; the cost is a redundant but correct subprocess.
    """
    interp = registered_interpreter(potential)
    if not interp:
        return False
    return os.path.normcase(os.path.abspath(interp)) != \
        os.path.normcase(os.path.abspath(sys.executable))


def pin_interpreter_env(potential: str):
    """Freeze the dispatch decision for this process AND its children.

    Writing the resolved interpreter into IRMA_MLIP_PYTHON_<POTENTIAL>
    means a registry edit mid-build cannot split relaxation, the cache
    fingerprint, and spawned pool workers across different environments
    (the env var both overrides the registry file and is inherited
    through multiprocessing spawn).
    """
    interp = registered_interpreter(potential)
    if interp and is_dispatched(potential):
        os.environ[_env_var(potential)] = interp


# --- protocol client ---------------------------------------------------------

# init meta per (potential, interpreter), so the fingerprint can use the
# EXECUTING environment's package version without a second server spawn
_META_CACHE: dict = {}


def _invalidate_meta(potential: str):
    for key in [k for k in _META_CACHE if k[0] == potential]:
        del _META_CACHE[key]


class _ServerHandle:
    # Liveness monitoring, NOT a request timeout (review finding CDX-3):
    # force-call latency is legitimately unbounded — a big supercell on a
    # slow potential can take arbitrarily long — so a slow-but-healthy
    # server is waited on forever. What is bounded is how long a DEAD or
    # WEDGED child may keep the caller blocked: while waiting for a reply
    # the child process is polled every poll_interval_s, and a child that
    # EXITED without replying raises after drain_grace_s (grace for
    # already-buffered output) instead of blocking on a pipe that a stray
    # grandchild may hold open indefinitely. shutdown_wait_s bounds only
    # the shutdown handshake in close(): unlike force calls, that control
    # message is answered immediately by any healthy server, so timing it
    # out and escalating to terminate/kill is safe.
    poll_interval_s = 1.0
    drain_grace_s = 5.0
    shutdown_wait_s = 5.0

    def __init__(self, interpreter: str):
        import queue
        import threading

        self.interpreter = interpreter
        self.proc = subprocess.Popen(
            [interpreter, _SERVER, _CALCULATORS],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=None, text=True, bufsize=1)
        # ONE reader thread owns the protocol stream for the handle's
        # whole life. It is the only object that ever touches
        # proc.stdout, which is what lets close() walk away from a stream
        # that some rogue grandchild is still holding open (closing a
        # buffered reader another thread is blocked on waits for that
        # read to finish).
        self._replies: queue.SimpleQueue = queue.SimpleQueue()
        self._reader = threading.Thread(
            target=self._read_loop, daemon=True,
            name="irma-mlip-protocol-reader")
        self._reader.start()

    def _read_loop(self):
        try:
            for line in self.proc.stdout:
                self._replies.put(("line", line))
        except Exception as exc:            # stream died under the reader
            self._replies.put(("exc", exc))
        finally:
            self._replies.put(("eof", ""))

    def _await_reply(self, what: str, deadline=None):
        """One protocol line, liveness-monitored (see class comment).

        Blocks as long as the child process is ALIVE; raises MlipEnvError
        once the child has exited without producing a reply. A
        ``deadline`` (time.monotonic value) additionally bounds the wait
        and returns None on expiry — used only for the shutdown
        handshake, never for force calls.
        """
        import queue
        import time

        died_at = None
        while True:
            timeout = self.poll_interval_s
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                timeout = min(timeout, remaining)
            try:
                kind, value = self._replies.get(timeout=timeout)
            except queue.Empty:
                pass
            else:
                if kind == "exc":
                    raise MlipEnvError(
                        f"force server ({self.interpreter}) protocol read "
                        f"failed during {what}: {value}") from value
                return value                 # a line, or "" at EOF
            if self.proc.poll() is not None:
                if died_at is None:
                    died_at = time.monotonic()
                elif time.monotonic() - died_at >= self.drain_grace_s:
                    raise MlipEnvError(
                        f"force server ({self.interpreter}) exited with "
                        f"code {self.proc.returncode} during {what} "
                        f"without replying; see the console for its output")

    def request(self, payload: dict, what: str) -> dict:
        proc = self.proc
        if proc.poll() is not None:
            raise MlipEnvError(
                f"force server ({self.interpreter}) exited with code "
                f"{proc.returncode} before {what}")
        try:
            proc.stdin.write(json.dumps(payload) + "\n")
            proc.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            raise MlipEnvError(
                f"force server ({self.interpreter}) pipe broke during "
                f"{what}: {exc}") from exc
        line = self._await_reply(what)
        if not line:
            raise MlipEnvError(
                f"force server ({self.interpreter}) closed the protocol "
                f"stream during {what} (exit "
                f"{proc.poll()}); see the console for its output")
        reply = json.loads(line)
        if not reply.get("ok"):
            self._raise(reply, what)
        return reply

    def _raise(self, reply: dict, what: str):
        from irma.mlip.calculators import MlipDependencyError
        kind = reply.get("kind", "runtime")
        message = (f"[{self.interpreter}] {what}: "
                   f"{reply.get('error', 'unknown error')}")
        # Known failure signatures get one actionable sentence on top of
        # the raw error, which is otherwise a bare torch traceback tail.
        hint = _error_hint(message)
        if hint:
            message += f"\n  hint: {hint}"
        if kind == "dependency":
            raise MlipDependencyError(message)
        if kind == "value":
            raise ValueError(message)
        raise RuntimeError(message)

    def close(self):
        import time
        if self.proc.poll() is None:
            # Graceful shutdown with a BOUNDED wait: close() must be able
            # to escape a wedged server, so it never re-enters the
            # unbounded request() path (whose blocking reply wait made the
            # terminate/kill escalation below unreachable — CDX-3).
            try:
                self.proc.stdin.write(json.dumps({"cmd": "shutdown"}) + "\n")
                self.proc.stdin.flush()
                self._await_reply(
                    "shutdown",
                    deadline=time.monotonic() + self.shutdown_wait_s)
            except Exception:
                pass
            try:
                self.proc.terminate()
            except OSError:
                pass
        try:
            self.proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait(timeout=10)          # reap after the kill too
        streams = [self.proc.stdin]
        if not self._reader.is_alive():
            # stdout is closed only when the reader has finished with it:
            # closing a buffered stream another thread is blocked reading
            # BLOCKS until that read returns, which would hand a rogue
            # grandchild holding the protocol pipe the power to wedge
            # close() all over again. Abandoning the daemon reader (and
            # its fd) costs one descriptor until the pipe's last writer
            # goes away; interpreter exit reclaims it either way.
            streams.append(self.proc.stdout)
        for stream in streams:
            try:
                if stream:
                    stream.close()
            except OSError:
                pass


def _oneshot(interpreter: str, payload: dict, what: str) -> dict:
    handle = _ServerHandle(interpreter)
    try:
        return handle.request(payload, what)
    finally:
        handle.close()


def remote_canonicalize(spec, interpreter: str) -> str:
    """Pin a spec's model string inside the foreign environment."""
    reply = _oneshot(
        interpreter,
        {"cmd": "canonicalize",
         "spec": {"potential": spec.potential, "model": spec.model,
                  "threads": spec.threads}},
        f"canonicalize {spec.potential}")
    return reply["model"]


def remote_identity(spec, interpreter: str) -> tuple[str, str]:
    """(checkpoint identity, package version) from the foreign env."""
    reply = _oneshot(
        interpreter,
        {"cmd": "identity",
         "spec": {"potential": spec.potential, "model": spec.model,
                  "threads": spec.threads}},
        f"resolve identity of {spec.potential}")
    return reply["identity"], reply["version"]


def cached_package_version(potential: str) -> str | None:
    """Package version reported by the last server init, if any."""
    interp = registered_interpreter(potential)
    meta = _META_CACHE.get((potential, interp))
    return meta.get("package_version") if meta else None


def remote_calculator(spec, interpreter: str):
    """(RemoteCalculator, meta) proxying to the foreign environment."""
    from ase.calculators.calculator import (
        Calculator, PropertyNotImplementedError, all_changes)

    handle = _ServerHandle(interpreter)
    try:
        reply = handle.request(
            {"cmd": "init",
             "spec": {"potential": spec.potential, "model": spec.model,
                      "threads": spec.threads}},
            f"initialize {spec.potential}")
    except BaseException:
        handle.close()          # no orphan on a failed init
        raise
    meta = dict(reply["meta"])
    meta["dispatch_interpreter"] = interpreter
    _META_CACHE[(spec.potential, interpreter)] = meta

    class RemoteCalculator(Calculator):
        implemented_properties = ["energy", "free_energy", "forces",
                                  "stress"]

        def __init__(self):
            super().__init__()
            self._handle = handle

        def calculate(self, atoms=None, properties=("energy",),
                      system_changes=all_changes):
            Calculator.calculate(self, atoms)
            a = self.atoms
            rep = self._handle.request(
                {"cmd": "calc",
                 "numbers": a.get_atomic_numbers().tolist(),
                 "positions": a.get_positions().tolist(),
                 "cell": a.get_cell().tolist(),
                 "pbc": a.get_pbc().tolist()},
                "force call")
            import numpy as np
            self.results = {"energy": rep["energy"],
                            "free_energy": rep["energy"],
                            "forces": np.asarray(rep["forces"])}
            if "stress" in rep:
                self.results["stress"] = np.asarray(rep["stress"])
            elif "stress" in properties:
                raise PropertyNotImplementedError(
                    f"{spec.potential} checkpoint provides no stress "
                    f"(cell relaxation is unavailable with it)")

        def close(self):
            self._handle.close()

        def __del__(self):
            try:
                self.close()
            except Exception:
                pass

    return RemoteCalculator(), meta


# --- provisioning ------------------------------------------------------------

def _find_uv() -> str | None:
    return shutil.which("uv")


def env_root(potential: str) -> str:
    name = _SHARED_ENVS.get(potential, (potential,))[0]
    return os.path.join(_cache_dir(), "envs", name)


def _env_python(root: str) -> str:
    # Windows venvs create Scripts\python.exe; the extensionless name
    # exists only on POSIX (registration validates with os.path.isfile,
    # which no CreateProcess .exe magic rescues).
    if os.name == "nt":
        return os.path.join(root, "Scripts", "python.exe")
    return os.path.join(root, "bin", "python")


def create_env(potential: str, *, progress=print,
               dry_run=False) -> str | None:
    """Provision and register a dedicated environment for a potential.

    Uses `uv venv` when uv is on PATH (fast, and pins the env to the
    Python series in _ENV_PYTHON — potential packages lag new
    interpreters), else `python -m venv` seeded from the running
    interpreter (the venv module cannot do otherwise). Installs the
    curated requirement set with pip — UNPINNED names resolved against
    the configured package index at install time, so the resulting env
    tracks current releases and is not a reproducible, vetted set (see
    ENV_REQUIREMENTS) — sanity-imports the backend package, runs a
    torch/NumPy interop probe (retrying once with numpy<2 when the
    NumPy-1-ABI breakage is detected), and registers the interpreter for
    the potential (and its shared siblings, e.g. mace and mace-off share
    one mace-torch env). Returns the interpreter path, or None for
    dry_run.
    """
    from irma.mlip.calculators import _PACKAGES

    requirements = ENV_REQUIREMENTS.get(potential)
    if requirements is None:
        raise MlipEnvError(
            f"no known requirement set for {potential!r}; choose from "
            f"{', '.join(sorted(ENV_REQUIREMENTS))}")
    root = env_root(potential)
    python = _env_python(root)
    uv = _find_uv()

    if _intel_mac():
        progress(_INTEL_MAC_NOTE)

    if uv:
        steps = [([uv, "venv", "--python", _ENV_PYTHON, root],
                  "create venv (uv)"),
                 ([uv, "pip", "install", "--python", python, *requirements],
                  "install requirements (uv)")]
        numpy_fix = [uv, "pip", "install", "--python", python, "numpy<2"]
        pip_check = [uv, "pip", "check", "--python", python]
    else:
        if sys.version_info[:2] > tuple(
                int(part) for part in _ENV_PYTHON.split(".")):
            progress(
                f"  note: this env will run Python "
                f"{sys.version_info.major}.{sys.version_info.minor} (the "
                f"interpreter running IRMA); potential packages often "
                f"lack wheels there. Installing uv lets this command pin "
                f"the env to Python {_ENV_PYTHON} instead.")
        steps = [([sys.executable, "-m", "venv", root], "create venv"),
                 ([python, "-m", "pip", "install", *requirements],
                  "install requirements")]
        numpy_fix = [python, "-m", "pip", "install", "numpy<2"]
        pip_check = [python, "-m", "pip", "check"]

    if dry_run:
        for cmd, label in steps:
            progress(f"  [dry-run] {label}: {' '.join(cmd)}")
        return None

    if os.path.exists(root):
        raise MlipEnvError(
            f"{root} already exists; remove it first "
            f"(irma mlip env remove {potential})")
    for cmd, label in steps:
        progress(f"  {label} ...")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise MlipEnvError(
                f"{label} failed (exit {result.returncode}):\n"
                f"{_install_failure_detail(result)}")

    check = _IMPORT_CHECKS.get(potential,
                               f"import {_PACKAGES[potential][1]}")
    failure = _verify_env(python, check, progress)
    if failure is not None and _numpy_abi_break(failure):
        # A torch wheel built against NumPy 1.x next to NumPy 2.x: the
        # coherent pair on such platforms is that torch with numpy<2.
        progress("  torch/NumPy ABI mismatch detected; retrying with "
                 "numpy<2 ...")
        result = subprocess.run(numpy_fix, capture_output=True, text=True)
        if result.returncode != 0:
            raise MlipEnvError(
                f"numpy<2 remediation failed (exit {result.returncode}):\n"
                f"{_install_failure_detail(result)}")
        failure = _verify_env(python, check, progress)
        if failure is None:
            # The remediation installed numpy separately, which pip
            # documents may break an already-installed package's declared
            # requirement (a stack that pins numpy>=2 cannot coexist with
            # this platform's torch). Refuse to register a knowingly
            # inconsistent environment.
            progress("  pip check after remediation ...")
            result = subprocess.run(pip_check, capture_output=True,
                                    text=True)
            if result.returncode != 0:
                raise MlipEnvError(
                    "numpy<2 remediation leaves the environment "
                    "inconsistent (this platform's torch requires "
                    "NumPy 1.x, which another installed package "
                    "excludes); the environment was NOT registered:\n"
                    f"{_install_failure_detail(result)}")
    if failure is not None:
        raise MlipEnvError(failure)

    siblings = _SHARED_ENVS.get(potential, (potential,))
    for sibling in siblings:
        # A shared env serves every sibling (mace and mace-off ride one
        # mace-torch install), so each sibling's entry point must import
        # before ANY of them is registered.
        if sibling != potential:
            sib_check = _IMPORT_CHECKS.get(
                sibling, f"import {_PACKAGES[sibling][1]}")
            progress(f"  verifying `{sib_check}` ...")
            result = subprocess.run([python, "-c", sib_check],
                                    capture_output=True, text=True)
            if result.returncode != 0:
                raise MlipEnvError(
                    f"the provisioned env fails the import check of its "
                    f"shared sibling {sibling} ({sib_check}); nothing "
                    f"was registered:\n"
                    f"{(result.stderr or '').strip()[-2000:]}")
    for sibling in siblings:
        register_interpreter(sibling, python)
        progress(f"  registered {sibling} -> {python}")
    return python


def _verify_env(python: str, check: str, progress) -> str | None:
    """Import check + runtime probe in the env; failure text or None.

    The import check gates on the calculator entry point being present;
    the probe gates on torch actually working with the installed NumPy,
    which an import alone does not prove (the ABI failure is a warning
    at import time and an error only when a tensor crosses to NumPy).
    """
    progress(f"  verifying `{check}` ...")
    result = subprocess.run([python, "-c", check],
                            capture_output=True, text=True)
    if result.returncode != 0:
        return (f"the provisioned env fails its import check ({check}); "
                f"the environment was NOT registered:\n"
                f"{(result.stderr or '').strip()[-2000:]}")
    progress("  probing torch/NumPy interop ...")
    result = subprocess.run([python, "-c", _RUNTIME_PROBE],
                            capture_output=True, text=True)
    if result.returncode != 0:
        return (f"the provisioned env fails the torch/NumPy runtime "
                f"probe; the environment was NOT registered:\n"
                f"{(result.stderr or '').strip()[-2000:]}")
    return None


def remove_env(potential: str, *, progress=print) -> bool:
    """Unregister a potential and delete its auto-provisioned env dir.

    Deletion is triple-guarded (review finding: `env remove ../models`
    must never reach the checkpoint cache): the name must be a known
    potential, the directory must sit exactly under <cache>/envs/, and
    it must look like a venv we created (pyvenv.cfg marker).
    """
    if potential not in ENV_REQUIREMENTS:
        raise MlipEnvError(
            f"unknown potential {potential!r}; choose from "
            f"{', '.join(sorted(ENV_REQUIREMENTS))}")
    removed = unregister_interpreter(potential)
    root = os.path.realpath(env_root(potential))
    envs_dir = os.path.realpath(os.path.join(_cache_dir(), "envs"))
    if os.path.isdir(root) \
            and os.path.dirname(root) == envs_dir \
            and os.path.isfile(os.path.join(root, "pyvenv.cfg")):
        others = [p for p in _SHARED_ENVS.get(potential, (potential,))
                  if p != potential and _load_registry().get(p)]
        if others:
            progress(f"  keeping {root} (still registered for "
                     f"{', '.join(others)})")
        else:
            shutil.rmtree(root)
            progress(f"  deleted {root}")
            removed = True
    return removed


def list_envs() -> dict:
    """potential -> interpreter for everything currently registered.

    Rows report where builds will ACTUALLY dispatch, matching
    registered_interpreter: an IRMA_MLIP_PYTHON_<POTENTIAL> override wins
    over the registry file, and an EMPTY override disables dispatch — a
    registry row shadowed by an empty override is shown as disabled, not
    as the interpreter builds will no longer use. Display only: values
    are not validated or executed here.
    """
    table = dict(_load_registry())
    from irma.mlip.calculators import POTENTIALS
    for potential in POTENTIALS:
        override = os.environ.get(_env_var(potential))
        if override:
            table[potential] = override + "  (env var)"
        elif override is not None and potential in table:
            table[potential] = (f"(dispatch disabled: {_env_var(potential)} "
                                f"is set and empty)")
    return table
