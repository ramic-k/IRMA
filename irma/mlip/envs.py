"""Per-potential environment isolation for the MLIP front end.

The potential packages have mutually unsatisfiable pins (mace-torch needs
e3nn==0.4.4 while sevenn/mattersim need >=0.5; deepmd-kit's binaries are
ABI-locked to one torch), so one Python environment cannot host every
backend. This module lets each potential run under its own interpreter:

- a registry (envs.json in the irma-mlip cache, or the
  IRMA_MLIP_PYTHON_<POTENTIAL> environment variable) maps a potential to
  a Python interpreter;
- RemoteCalculator proxies ASE energy/forces/stress calls to a
  force_server.py subprocess running under that interpreter (the foreign
  env needs only ase + the potential package, never irma);
- create_env() provisions such an environment automatically from a
  curated, unpinned requirement set and registers it.

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

from irma.mlip.calculators import _PACKAGES, POTENTIALS

_SERVER = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "force_server.py")
_CALCULATORS = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "calculators.py")

# Curated pip requirement sets for auto-provisioned environments. The
# names are unpinned; the resolver installs the current releases. Each
# backend package pins its own critical dependencies (deepmd-kit[torch]
# pins the torch its binaries were built against), and the import check
# below registers an env only when it actually works.
ENV_REQUIREMENTS = {p: (_PACKAGES[p][0], "ase>=3.23") for p in POTENTIALS}
ENV_REQUIREMENTS["dpa3"] = ("deepmd-kit[torch]", "mpich", "huggingface_hub", "ase>=3.23")

# potentials that share one package land in one shared env
_SHARED_ENVS = {"mace": ("mace", "mace-off"), "mace-off": ("mace", "mace-off")}

# provisioning verification imports the calculator entry point, not just
# the top-level package: deepmd's mpich and torch-ABI failures appear only
# on the calculator import, and a broken env must not be registered
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

# Functional probe run in the fresh env after the import check. The
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
# NumPy 2.x (seen on Intel macs, whose torch wheels stop at 2.2.2).
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
                "but NumPy >= 2 sits next to it; install numpy<2 in that "
                "env (<env python> -m pip install 'numpy<2')")
    return None


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
            return json.load(fh).get("interpreters", {})
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


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

    Applied on write (register_interpreter) and on every read
    (registered_interpreter): the envs.json registry and the
    IRMA_MLIP_PYTHON_* variables are plain editable state, and the string
    they yield is handed to subprocess.Popen, so an entry that is not an
    executable file fails here, with its source named, instead of being
    executed.
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

    The returned string is passed to subprocess.Popen, so it is validated
    here as register_interpreter validates on write: a value from either
    source that is not an executable file raises MlipEnvError.
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


def unregister_interpreter(potential: str) -> bool:
    table = _load_registry()
    if potential not in table:
        return False
    del table[potential]
    _save_registry(table)
    return True


def is_dispatched(potential: str) -> bool:
    """True when this potential runs under a different interpreter.

    Compared by launcher path, not realpath: venv pythons are symlinks to
    their base interpreter, so realpath would collapse a provisioned env
    onto the running one and silently disable dispatch. Two aliases of the
    same environment therefore compare as different, which costs a
    redundant but correct subprocess.
    """
    interp = registered_interpreter(potential)
    if not interp:
        return False
    return os.path.normcase(os.path.abspath(interp)) != \
        os.path.normcase(os.path.abspath(sys.executable))


# --- protocol client ---------------------------------------------------------

class _ServerHandle:
    """One force-server subprocess speaking the JSON-lines protocol. A force
    call may take arbitrarily long, so replies are awaited without a timeout;
    a server that exits closes the pipe and the read returns empty."""

    def __init__(self, interpreter: str):
        self.interpreter = interpreter
        self.proc = subprocess.Popen(
            [interpreter, _SERVER, _CALCULATORS],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)

    def request(self, payload: dict, what: str) -> dict:
        try:
            self.proc.stdin.write(json.dumps(payload) + "\n")
            self.proc.stdin.flush()
            line = self.proc.stdout.readline()
        except OSError:
            line = ""
        if not line:
            raise MlipEnvError(
                f"force server ({self.interpreter}) exited during {what}; "
                "see the console for its output")
        reply = json.loads(line)
        if not reply.get("ok"):
            self._raise(reply, what)
        return reply

    def _raise(self, reply: dict, what: str):
        from irma.mlip.calculators import MlipDependencyError
        kind = reply.get("kind", "runtime")
        message = (f"[{self.interpreter}] {what}: "
                   f"{reply.get('error', 'unknown error')}")
        # known failure signatures get one actionable sentence
        hint = _error_hint(message)
        if hint:
            message += f"\n  hint: {hint}"
        if kind == "dependency":
            raise MlipDependencyError(message)
        if kind == "value":
            raise ValueError(message)
        raise RuntimeError(message)

    def close(self):
        try:
            self.proc.stdin.write(json.dumps({"cmd": "shutdown"}) + "\n")
            self.proc.stdin.close()
            self.proc.wait(timeout=5)
        except (OSError, subprocess.TimeoutExpired):
            self.proc.kill()
            self.proc.wait()
        self.proc.stdout.close()


def _oneshot(interpreter: str, payload: dict, what: str) -> dict:
    handle = _ServerHandle(interpreter)
    try:
        return handle.request(payload, what)
    finally:
        handle.close()


def spec_payload(spec) -> dict:
    """The spec as the force-server protocol carries it (every field)."""
    from dataclasses import asdict
    return asdict(spec)


def remote_canonicalize(spec, interpreter: str) -> str:
    """Pin a spec's model string inside the foreign environment."""
    reply = _oneshot(
        interpreter,
        {"cmd": "canonicalize", "spec": spec_payload(spec)},
        f"canonicalize {spec.potential}")
    return reply["model"]


def remote_identity(spec, interpreter: str) -> tuple[str, str]:
    """(checkpoint identity, package version) from the foreign env."""
    reply = _oneshot(
        interpreter,
        {"cmd": "identity", "spec": spec_payload(spec)},
        f"resolve identity of {spec.potential}")
    return reply["identity"], reply["version"]


def remote_calculator(spec, interpreter: str):
    """(RemoteCalculator, meta) proxying to the foreign environment."""
    from ase.calculators.calculator import (
        Calculator, PropertyNotImplementedError, all_changes)

    handle = _ServerHandle(interpreter)
    try:
        reply = handle.request(
            {"cmd": "init", "spec": spec_payload(spec)},
            f"initialize {spec.potential}")
    except BaseException:
        handle.close()          # no orphan on a failed init
        raise
    meta = dict(reply["meta"])
    meta["dispatch_interpreter"] = interpreter

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
                {"cmd": "calc", "stress": "stress" in properties,
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
    return os.path.join(root, "bin", "python")


def create_env(potential: str, *, progress=print,
               dry_run=False) -> str | None:
    """Provision and register a dedicated environment for a potential.

    Uses `uv venv` pinned to the Python series in _ENV_PYTHON when uv is on
    PATH, else `python -m venv` from the running interpreter. Installs the
    unpinned requirement set (see ENV_REQUIREMENTS), runs the import check
    and the torch/NumPy probe, and registers the interpreter for the
    potential and its shared siblings (mace and mace-off share one env).
    Returns the interpreter path, or None for dry_run.
    """
    requirements = ENV_REQUIREMENTS.get(potential)
    if requirements is None:
        raise MlipEnvError(
            f"no known requirement set for {potential!r}; choose from "
            f"{', '.join(sorted(ENV_REQUIREMENTS))}")
    root = env_root(potential)
    python = _env_python(root)
    uv = _find_uv()
    if uv:
        steps = [([uv, "venv", "--python", _ENV_PYTHON, root],
                  "create venv (uv)"),
                 ([uv, "pip", "install", "--python", python, *requirements],
                  "install requirements (uv)")]
    else:
        steps = [([sys.executable, "-m", "venv", root], "create venv"),
                 ([python, "-m", "pip", "install", *requirements],
                  "install requirements")]

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
                f"{(result.stdout or '')[-1500:]}{(result.stderr or '')[-500:]}")

    failure = _verify_env(python, _IMPORT_CHECKS[potential], progress)
    if failure is not None:
        if _numpy_abi_break(failure):
            failure += (f"\n  hint: run `{python} -m pip install 'numpy<2'` and "
                        f"set {_env_var(potential)}={python}")
        raise MlipEnvError(failure)
    for sibling in _SHARED_ENVS.get(potential, (potential,)):
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
    """Unregister a potential and delete its auto-provisioned env dir (kept
    while a shared sibling is still registered)."""
    if potential not in ENV_REQUIREMENTS:
        raise MlipEnvError(
            f"unknown potential {potential!r}; choose from "
            f"{', '.join(sorted(ENV_REQUIREMENTS))}")
    removed = unregister_interpreter(potential)
    root = env_root(potential)
    if os.path.isdir(root):
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
    """potential -> interpreter for everything currently registered (an
    IRMA_MLIP_PYTHON_<POTENTIAL> override wins over the registry file)."""
    table = dict(_load_registry())
    for potential in POTENTIALS:
        override = os.environ.get(_env_var(potential))
        if override:
            table[potential] = override + "  (env var)"
    return table
