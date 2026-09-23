"""Standalone force server: run one MLIP calculator in a foreign env.

Launched as  `<foreign-python> force_server.py <path-to-calculators.py>`
by irma.mlip.envs. This file is deliberately self-contained: it imports
ONLY the standard library, ase, and (indirectly, through the calculator
factory) the potential's own package -- never the irma package -- so the
foreign environment needs nothing beyond `ase` and the potential. The
calculator factory is loaded from the given calculators.py path with
importlib, keeping a single source of truth for backend construction.

Protocol: newline-delimited JSON. Requests arrive on stdin; replies go
to a PRIVATE copy of the original stdout, after the real fd 1 has been
redirected onto stderr -- potential packages print freely (MatterSim,
SevenNet and friends all log to stdout) and none of it can corrupt the
protocol stream.

  {"cmd": "init", "spec": {"potential": .., "model": .., "threads": ..}}
      -> {"ok": true, "meta": {..}, "properties": ["energy", ..]}
  {"cmd": "canonicalize", "spec": {..}}
      -> {"ok": true, "model": <pinned model string>}
  {"cmd": "identity", "spec": {..}}
      -> {"ok": true, "identity": <fingerprint string>, "version": <pkg>}
  {"cmd": "calc", "positions": [[..]], "cell": [[..]], "numbers": [..],
   "pbc": [..]}
      -> {"ok": true, "energy": .., "forces": [[..]], "stress": [..]}
  {"cmd": "shutdown"} -> {"ok": true}  (and the process exits)

Failures reply {"ok": false, "kind": "dependency"|"value"|"runtime",
"error": "..."} and the server stays alive unless init itself failed.
"""
import importlib.util
import json
import os
import sys
import traceback


def _load_calculators(path):
    spec = importlib.util.spec_from_file_location("_irma_mlip_calculators",
                                                  path)
    module = importlib.util.module_from_spec(spec)
    # register BEFORE exec: dataclass machinery resolves the defining
    # module through sys.modules, and an unregistered module breaks it
    sys.modules["_irma_mlip_calculators"] = module
    spec.loader.exec_module(module)
    # the server IS the foreign environment: it must never re-enter
    # dispatch (recursion) nor import the irma package (not installed in
    # provisioned envs)
    module._DISPATCH_DISABLED = True
    return module


def _classify(exc, calculators, cmd):
    if isinstance(exc, calculators.MlipDependencyError):
        return "dependency"
    if isinstance(exc, ImportError) and cmd != "calc":
        # a raw import failure while setting UP the environment (init/
        # canonicalize/identity) is a dependency problem (CLI exit 4);
        # one inside a backend's forward pass at calc time is a runtime
        # bug and must not masquerade as "pip install something"
        return "dependency"
    if isinstance(exc, (ValueError, TypeError, KeyError)):
        return "value"
    return "runtime"


def _reply(out, payload):
    out.write(json.dumps(payload) + "\n")
    out.flush()


def _fail(out, exc, calculators, cmd=None):
    _reply(out, {"ok": False, "kind": _classify(exc, calculators, cmd),
                 "error": f"{type(exc).__name__}: {exc}",
                 "traceback": traceback.format_exc()})


def main() -> int:
    calculators = _load_calculators(sys.argv[1])

    # keep the protocol channel private: replies go to a dup of the
    # original stdout, while fd 1 (what the potential packages write to)
    # is pointed at stderr
    protocol_fd = os.dup(1)
    os.dup2(2, 1)
    sys.stdout = os.fdopen(1, "w")
    out = os.fdopen(protocol_fd, "w")

    calc = None
    atoms_proto = None

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError as exc:
            _fail(out, exc, calculators)
            continue
        cmd = req.get("cmd")

        try:
            if cmd == "shutdown":
                _reply(out, {"ok": True})
                return 0

            if cmd in ("init", "canonicalize", "identity"):
                spec = calculators.CalculatorSpec(**req["spec"])
                if cmd == "canonicalize":
                    pinned = calculators.canonicalize_spec(spec)
                    _reply(out, {"ok": True, "model": pinned.model})
                    continue
                if cmd == "identity":
                    _reply(out, {
                        "ok": True,
                        "identity":
                            calculators.resolved_checkpoint_identity(spec),
                        "version":
                            calculators._package_version(spec.potential)})
                    continue
                calc, meta = calculators.make_calculator(spec)
                from ase import Atoms
                atoms_proto = Atoms
                # the full property list is advertised unconditionally;
                # whether a checkpoint actually serves stress is decided
                # per calc call, where a stressless checkpoint (e.g.
                # Egret-1 via mace-off) simply omits "stress" from its
                # reply and the client raises only if stress was needed
                properties = ["energy", "forces", "stress"]
                _reply(out, {"ok": True, "meta": meta,
                             "properties": properties})
                continue

            if cmd == "calc":
                if calc is None:
                    raise RuntimeError("calc before successful init")
                import numpy as np
                atoms = atoms_proto(
                    numbers=req["numbers"],
                    positions=np.asarray(req["positions"], dtype=float),
                    cell=np.asarray(req["cell"], dtype=float),
                    pbc=req["pbc"])
                atoms.calc = calc
                result = {"ok": True,
                          "energy": float(atoms.get_potential_energy()),
                          "forces": np.asarray(
                              atoms.get_forces(), dtype=float).tolist()}
                from ase.calculators.calculator import (
                    PropertyNotImplementedError)
                try:
                    result["stress"] = np.asarray(
                        atoms.get_stress(voigt=True), dtype=float).tolist()
                except (PropertyNotImplementedError, NotImplementedError):
                    pass          # stressless checkpoint: omit stress only;
                    # any OTHER stress failure is a real error and
                    # propagates through the protocol like one
                _reply(out, result)
                continue

            raise ValueError(f"unknown command {cmd!r}")
        except SystemExit:
            raise
        except BaseException as exc:              # noqa: BLE001 -- protocol boundary
            _fail(out, exc, calculators, cmd)
            if cmd == "init":
                return 1          # a server that cannot init is useless

    return 0


if __name__ == "__main__":
    sys.exit(main())
