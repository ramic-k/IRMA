"""Environment-isolation tests: registry, protocol client/server, dispatch
hooks, and provisioning command assembly. The force-server round trips run
a real subprocess under THIS interpreter with the EMT backend, so no
potential package or network is needed."""
import json
import os
import sys

import pytest

pytest.importorskip("ase")

from irma.mlip import envs                                    # noqa: E402
from irma.mlip.calculators import CalculatorSpec              # noqa: E402


@pytest.fixture(autouse=True)
def _isolated_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("IRMA_MLIP_CACHE", str(tmp_path))
    yield


def _fake_interpreter(tmp_path, name="python"):
    """A real executable file (SEC-4: read-path validation demands one)."""
    path = tmp_path / name
    path.write_text("#!/bin/sh\n")
    path.chmod(0o755)
    return str(path)


def test_registry_roundtrip_and_env_var_override(tmp_path, monkeypatch):
    assert envs.registered_interpreter("mace") is None
    envs.register_interpreter("mace", sys.executable)
    assert envs.registered_interpreter("mace") == sys.executable
    assert json.load(open(envs._registry_path()))["interpreters"] == \
        {"mace": sys.executable}

    elsewhere = _fake_interpreter(tmp_path, "elsewhere-python")
    monkeypatch.setenv("IRMA_MLIP_PYTHON_MACE", elsewhere)
    assert envs.registered_interpreter("mace") == elsewhere
    monkeypatch.setenv("IRMA_MLIP_PYTHON_MACE", "")   # explicit disable
    assert envs.registered_interpreter("mace") is None
    monkeypatch.delenv("IRMA_MLIP_PYTHON_MACE")

    assert envs.unregister_interpreter("mace") is True
    assert envs.unregister_interpreter("mace") is False
    assert envs.registered_interpreter("mace") is None

    with pytest.raises(envs.MlipEnvError, match="executable"):
        envs.register_interpreter("mace", str(tmp_path / "missing"))


def test_read_path_validates_like_the_write_path(tmp_path, monkeypatch):
    """SEC-4: whatever registered_interpreter returns goes to Popen, so a
    non-executable value from EITHER source must raise on read, never be
    returned for execution."""
    # env var pointing at nothing
    monkeypatch.setenv("IRMA_MLIP_PYTHON_MACE", "/nonexistent/python")
    with pytest.raises(envs.MlipEnvError,
                       match="IRMA_MLIP_PYTHON_MACE.*not an executable"):
        envs.registered_interpreter("mace")
    monkeypatch.delenv("IRMA_MLIP_PYTHON_MACE")

    # registry entry bypassing register_interpreter's write-time check
    # (hand-edited or stale envs.json)
    os.makedirs(envs._cache_dir(), exist_ok=True)
    with open(envs._registry_path(), "w") as fh:
        json.dump({"schema": 1,
                   "interpreters": {"mace": "/gone/python"}}, fh)
    with pytest.raises(envs.MlipEnvError, match="not an executable"):
        envs.registered_interpreter("mace")

    # a registration whose interpreter has since been DELETED fails the
    # same way (stale env), with the remediation named
    interp = _fake_interpreter(tmp_path)
    envs.register_interpreter("sevennet", interp)
    os.remove(interp)
    with pytest.raises(envs.MlipEnvError, match="env remove"):
        envs.registered_interpreter("sevennet")



@pytest.mark.skipif(
    sys.platform == "win32",
    reason="os.access(X_OK) is true for any readable existing file on "
           "Windows, so mode bits cannot express non-executability; the "
           "nonexistent- and deleted-interpreter rejections above still "
           "run there")
def test_plain_file_is_rejected_as_interpreter(tmp_path, monkeypatch):
    """SEC-4 companion: an existing but non-executable file is rejected."""
    plain = tmp_path / "notes.txt"
    plain.write_text("not a program")
    plain.chmod(0o644)
    monkeypatch.setenv("IRMA_MLIP_PYTHON_DPA3", str(plain))
    with pytest.raises(envs.MlipEnvError, match="not an executable"):
        envs.registered_interpreter("dpa3")


def test_is_dispatched_only_for_a_different_interpreter(tmp_path,
                                                        monkeypatch):
    assert envs.is_dispatched("sevennet") is False   # nothing registered
    envs.register_interpreter("sevennet", sys.executable)
    assert envs.is_dispatched("sevennet") is False   # same interpreter
    monkeypatch.setenv("IRMA_MLIP_PYTHON_SEVENNET",
                       _fake_interpreter(tmp_path, "other-python"))
    assert envs.is_dispatched("sevennet") is True


def test_force_server_roundtrip_with_emt():
    from ase.build import bulk
    from irma.mlip.calculators import make_calculator

    spec = CalculatorSpec("emt")
    atoms = bulk("Al", "fcc", a=4.05, cubic=True)
    atoms.rattle(stdev=0.02, seed=7)

    direct = atoms.copy()
    direct.calc = make_calculator(spec)[0]
    e_direct = direct.get_potential_energy()
    f_direct = direct.get_forces()
    s_direct = direct.get_stress()

    remote_atoms = atoms.copy()
    calc, meta = envs.remote_calculator(spec, sys.executable)
    try:
        remote_atoms.calc = calc
        import numpy as np
        assert abs(remote_atoms.get_potential_energy() - e_direct) < 1e-12
        assert np.allclose(remote_atoms.get_forces(), f_direct, atol=1e-12)
        assert np.allclose(remote_atoms.get_stress(), s_direct, atol=1e-12)
        assert meta["potential"] == "emt"
        assert meta["dispatch_interpreter"] == sys.executable
        # a position change round-trips too (fresh calculate call)
        remote_atoms.positions[0, 0] += 0.01
        direct.positions[0, 0] += 0.01
        assert abs(remote_atoms.get_potential_energy()
                   - direct.get_potential_energy()) < 1e-12
    finally:
        calc.close()
    assert calc._handle.proc.poll() is not None      # server really exited


def test_force_server_error_kinds_map_to_exceptions():
    # a bad dpa3 model string fails server-side BEFORE any deepmd import,
    # and must surface as the same ValueError the local path raises
    with pytest.raises(ValueError, match="unknown dpa3 model"):
        envs.remote_canonicalize(
            CalculatorSpec("dpa3", model="DPA-99"), sys.executable)


def test_remote_identity_matches_local_for_emt():
    from irma.mlip.calculators import (
        _package_version, resolved_checkpoint_identity)
    spec = CalculatorSpec("emt")
    identity, version = envs.remote_identity(spec, sys.executable)
    assert identity == resolved_checkpoint_identity(spec)
    assert version == _package_version("emt")


def test_make_calculator_dispatch_hook(monkeypatch):
    from irma.mlip import calculators
    sentinel = ("proxy-calc", {"potential": "sevennet"})
    monkeypatch.setattr(envs, "is_dispatched", lambda p: p == "sevennet")
    monkeypatch.setattr(envs, "registered_interpreter",
                        lambda p: "/foreign/python")
    calls = []

    def fake_remote(spec, interp):
        calls.append((spec.potential, interp))
        return sentinel

    monkeypatch.setattr(envs, "remote_calculator", fake_remote)
    assert calculators.make_calculator(CalculatorSpec("sevennet")) \
        == sentinel
    assert calls == [("sevennet", "/foreign/python")]
    # emt is NEVER dispatched, even if something registers it
    calc, _ = calculators.make_calculator(CalculatorSpec("emt"))
    assert type(calc).__name__ == "EMT"


def test_is_dispatched_sees_a_symlinked_venv_python(tmp_path):
    # venv launchers are symlinks to the base python; realpath comparison
    # would collapse them onto the running interpreter and silently
    # disable dispatch (review finding) -- launcher paths must be compared
    link = tmp_path / "bin" / "python"
    link.parent.mkdir()
    link.symlink_to(sys.executable)
    envs.register_interpreter("mace", str(link))
    assert envs.is_dispatched("mace") is True


def test_registry_tolerates_malformed_json(tmp_path):
    os.makedirs(envs._cache_dir(), exist_ok=True)
    with open(envs._registry_path(), "w") as fh:
        fh.write("not json at all")
    assert envs.registered_interpreter("mace") is None


def test_remove_env_rejects_unknown_names(tmp_path):
    # `env remove ../models` must never delete the checkpoint cache
    victim = tmp_path / "models"
    victim.mkdir()
    (victim / "checkpoint.pt").write_bytes(b"weights")
    with pytest.raises(envs.MlipEnvError, match="unknown potential"):
        envs.remove_env("../models")
    assert (victim / "checkpoint.pt").exists()


def test_standalone_calculators_never_import_irma(monkeypatch, tmp_path):
    # the force server loads calculators.py WITHOUT the irma package
    # (provisioned envs do not install irma); with the dispatch-disable
    # flag set, nothing may touch irma.* even when irma is unimportable
    import importlib.util

    import irma.mlip.calculators as real
    spec = importlib.util.spec_from_file_location(
        "_standalone_calcs", real.__file__)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "_standalone_calcs", module)
    spec.loader.exec_module(module)
    module._DISPATCH_DISABLED = True

    monkeypatch.setitem(sys.modules, "irma", None)
    monkeypatch.setitem(sys.modules, "irma.mlip", None)
    monkeypatch.setitem(sys.modules, "irma.mlip.envs", None)

    calc, meta = module.make_calculator(module.CalculatorSpec("emt"))
    assert meta["potential"] == "emt"
    pinned = module.canonicalize_spec(
        module.CalculatorSpec("mattersim", model="x.pth"))
    assert pinned.model == "x.pth"
    assert module.resolved_checkpoint_identity(
        module.CalculatorSpec("mattersim")) == "MatterSim-v1.0.0-5M.pth"


def test_every_public_potential_has_env_support():
    from irma.mlip.calculators import POTENTIALS
    assert set(envs.ENV_REQUIREMENTS) == set(POTENTIALS)
    assert set(envs._IMPORT_CHECKS) == set(POTENTIALS)


def test_create_env_dry_run_and_validation(tmp_path):
    lines = []
    result = envs.create_env("mace", progress=lines.append, dry_run=True)
    assert result is None
    assert any("venv" in ln for ln in lines)
    assert any("mace-torch" in ln for ln in lines)
    assert not os.path.exists(envs.env_root("mace"))   # nothing created

    with pytest.raises(envs.MlipEnvError, match="no known requirement"):
        envs.create_env("emt")


def test_cli_env_subcommands(tmp_path, capsys):
    from irma.mlip.cli import main
    assert main(["env", "list"]) == 0
    assert "no environments registered" in capsys.readouterr().out

    assert main(["env", "create", "quantum-espresso"]) == 2
    capsys.readouterr()

    assert main(["env", "create", "mace", "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "[dry-run]" in out and "mace-torch" in out

    envs.register_interpreter("mace", sys.executable)
    assert main(["env", "list"]) == 0
    assert sys.executable in capsys.readouterr().out
    assert main(["env", "remove", "mace"]) == 0
    assert "unregistered" in capsys.readouterr().out
    assert main(["env", "remove", "mace"]) == 0
    assert "nothing registered" in capsys.readouterr().out

# ---- protocol client ------------------------------------------------------------

def _fake_server(tmp_path, monkeypatch, body):
    """Point _ServerHandle at a stand-in server script."""
    script = tmp_path / "fake_server.py"
    script.write_text(body)
    monkeypatch.setattr(envs, "_SERVER", str(script))
    return script


def test_slow_but_healthy_force_call_is_not_killed(tmp_path, monkeypatch):
    """A slow reply (a big supercell on a slow potential) completes
    normally: force calls have no timeout."""
    import time
    _fake_server(tmp_path, monkeypatch, (
        "import json, sys, time\n"
        "sys.stdin.readline()\n"
        "time.sleep(2.5)\n"                       # >> poll interval below
        "sys.stdout.write(json.dumps({'ok': True, 'slow': True}) + '\\n')\n"
        "sys.stdout.flush()\n"
        "sys.stdin.readline()\n"                  # shutdown request
        "sys.stdout.write(json.dumps({'ok': True}) + '\\n')\n"
        "sys.stdout.flush()\n"))
    handle = envs._ServerHandle(sys.executable)
    try:
        t0 = time.monotonic()
        reply = handle.request({"cmd": "calc"}, "slow force call")
        elapsed = time.monotonic() - t0
        assert reply == {"ok": True, "slow": True}
        assert elapsed >= 2.4                     # it really was slow
    finally:
        handle.close()
    assert handle.proc.poll() is not None


def test_close_escapes_a_wedged_server(tmp_path, monkeypatch):
    """close() kills a server that does not exit after the shutdown
    request."""
    import time
    _fake_server(tmp_path, monkeypatch, (
        "import sys, time\n"
        "sys.stdin.readline()\n"                  # eats the shutdown cmd
        "time.sleep(30)\n"))                      # never replies
    handle = envs._ServerHandle(sys.executable)
    time.sleep(0.3)                               # let the server start
    t0 = time.monotonic()
    handle.close()
    elapsed = time.monotonic() - t0
    assert elapsed < 10                           # escaped the 30 s wedge
    assert handle.proc.poll() is not None         # server is gone


def test_requirement_set_is_not_advertised_as_vetted():
    """SEC-3: ENV_REQUIREMENTS are unpinned package names resolved against
    a live index, so nothing in the module may call them 'known-good'."""
    import inspect

    assert all(
        "==" not in item and "@" not in item
        for reqs in envs.ENV_REQUIREMENTS.values() for item in reqs), \
        "requirement set gained pins: update the docstrings to match"
    text = inspect.getsource(envs)
    assert "known-good" not in text.lower()
    assert "UNPINNED" in envs.create_env.__doc__ \
        or "unpinned" in envs.create_env.__doc__.lower()


# ---------------------------------------------------------------------------
# provisioning guards (2026-08: colleague-reported field failures)


def test_error_hint_covers_the_numpy_abi_signature():
    hint = envs._error_hint("RuntimeError: ... _ARRAY_API not found ...")
    assert hint and "numpy<2" in hint
    assert envs._error_hint("ValueError: unrelated") is None


def test_raise_appends_hint_for_known_signatures():
    from types import SimpleNamespace

    fake = SimpleNamespace(interpreter="/env/bin/python")
    reply = {"ok": False, "kind": "runtime",
             "error": "RuntimeError: Numpy is not available"}
    with pytest.raises(RuntimeError, match="hint: .*numpy<2"):
        envs._ServerHandle._raise(fake, reply, "canonicalize nequip")
    # unknown errors stay untouched
    reply = {"ok": False, "kind": "runtime", "error": "RuntimeError: boom"}
    with pytest.raises(RuntimeError) as exc:
        envs._ServerHandle._raise(fake, reply, "canonicalize nequip")
    assert "hint:" not in str(exc.value)


def test_uv_dry_run_pins_the_env_python(monkeypatch):
    monkeypatch.setattr(envs, "_find_uv", lambda: "uv")
    lines = []
    envs.create_env("mace", progress=lines.append, dry_run=True)
    create = next(ln for ln in lines if "create venv" in ln)
    assert f"--python {envs._ENV_PYTHON}" in create
    assert sys.executable not in create


def test_probe_and_bootstrap_sources_compile():
    compile(envs._RUNTIME_PROBE, "<probe>", "exec")
    assert "from_numpy" in envs._RUNTIME_PROBE
    from irma.mlip import calculators
    compile(calculators._NEQUIP_COMPILE_BOOTSTRAP, "<bootstrap>", "exec")
    assert "main()" in calculators._NEQUIP_COMPILE_BOOTSTRAP


def test_numpy_abi_breakage_is_not_registered_and_hints_numpy2(tmp_path,
                                                               monkeypatch):
    """A torch wheel built against NumPy 1.x (Intel mac): the probe fails
    with the ABI signature, nothing is registered and the error names the
    numpy<2 fix."""
    monkeypatch.setattr(envs, "_find_uv", lambda: None)
    root = envs.env_root("nequip")
    python = envs._env_python(root)

    def fake_run(cmd, capture_output=True, text=True):
        from types import SimpleNamespace

        if cmd[1:3] == ["-m", "venv"]:
            os.makedirs(os.path.dirname(python), exist_ok=True)
            with open(python, "w") as fh:
                fh.write("#!/bin/sh\n")
            os.chmod(python, 0o755)
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        if cmd[:2] == [python, "-c"] and cmd[2] == envs._RUNTIME_PROBE:
            return SimpleNamespace(
                returncode=1, stdout="",
                stderr="UserWarning: Failed to initialize NumPy: "
                       "_ARRAY_API not found")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(envs.subprocess, "run", fake_run)
    with pytest.raises(envs.MlipEnvError, match="numpy<2"):
        envs.create_env("nequip", progress=lambda *_: None)
    assert envs.registered_interpreter("nequip") is None


def test_probe_failure_without_the_signature_does_not_register(
        tmp_path, monkeypatch):
    monkeypatch.setattr(envs, "_find_uv", lambda: None)
    root = envs.env_root("nequip")
    python = envs._env_python(root)

    def fake_run(cmd, capture_output=True, text=True):
        from types import SimpleNamespace

        if cmd[1:3] == ["-m", "venv"]:
            os.makedirs(os.path.dirname(python), exist_ok=True)
            with open(python, "w") as fh:
                fh.write("#!/bin/sh\n")
            os.chmod(python, 0o755)
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        if cmd[:2] == [python, "-c"] and cmd[2] == envs._RUNTIME_PROBE:
            return SimpleNamespace(returncode=1, stdout="",
                                   stderr="Segmentation fault")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(envs.subprocess, "run", fake_run)
    with pytest.raises(envs.MlipEnvError, match="runtime *probe|probe"):
        envs.create_env("nequip", progress=lambda *_: None)
    assert envs.registered_interpreter("nequip") is None
