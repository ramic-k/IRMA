"""CLI: convert an ENDF/TSL evaluation to an NCrystal ENDFTSL pack + NCMAT.

Single species:
    python -m ncrystal_plugin_ENDFTSL tape.endf -o out/ --symbol C --mass 12.0107 ...
Polyatomic (e.g. BeO), via a YAML listing one tape per principal scatterer:
    python -m ncrystal_plugin_ENDFTSL --config beo.yaml -o out/
"""
from __future__ import annotations
import argparse
import hashlib
import re
import math
import sys
from pathlib import Path

import yaml

from .reader import read_tsl
from .convert import build_pack
from .ncmat import multi_pack_ncmat
from .pack import write_pack


def _sha(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _validate_material_id(mid) -> str:
    """material_id becomes a file stem (<id>.ncmat / <id>.endftslpack), so reject
    path separators, '..', and absolute paths that would let a CLI flag or YAML
    field write outside the requested output directory."""
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", str(mid)) or ".." in str(mid):
        raise SystemExit(
            "error: material_id must be a safe file stem matching "
            f"[A-Za-z0-9][A-Za-z0-9_.-]* (no path separators or '..'), got {mid!r}")
    return str(mid)


def _validate_symbols(symbols) -> list:
    """Species symbols flow into NCMAT element/@CUSTOM_ENDFTSL lines, pack IDs
    (<material_id>__<symbol>) and pack file stems, so restrict them to plausible
    NCMAT element tokens: 1-2 letters, first uppercase. This rejects empty/
    whitespace/control characters, path separators ('/', '\\', '..') and newline
    injection into generated files. Duplicates are rejected too: two species with
    the same symbol would silently overwrite one another's pack file. Called at
    CLI entry (both modes) BEFORE any output directory or artifact is created."""
    seen = set()
    for sym in symbols:
        if not isinstance(sym, str) or not re.fullmatch(r"[A-Z][a-z]?", sym):
            raise SystemExit(
                "error: species symbol must be a plausible element symbol "
                f"matching [A-Z][a-z]? (1-2 letters, first uppercase), got {sym!r}")
        if sym in seen:
            raise SystemExit(
                f"error: duplicate species symbol {sym!r}; symbols must be unique "
                "(each symbol names one pack file and one NCMAT element)")
        seen.add(sym)
    return list(symbols)


def _run_single(a) -> int:
    _validate_material_id(a.material_id)
    _validate_symbols([a.symbol])
    ev = read_tsl(a.tape)
    pk = build_pack(ev, a.temperature, a.material_id, a.mass)
    pk.metadata["source_endf_sha256"] = _sha(a.tape)
    a.outdir.mkdir(parents=True, exist_ok=True)
    pack_path = a.outdir / f"{a.material_id}.endftslpack"
    write_pack(pk, pack_path)
    ncmat = multi_pack_ncmat([(a.symbol, 1.0)], a.density, [str(pack_path.resolve())])
    (a.outdir / f"{a.material_id}.ncmat").write_text(ncmat, encoding="utf-8")
    print(f"wrote {pack_path}")
    print(f"wrote {a.outdir / (a.material_id + '.ncmat')}")
    return 0


def _run_config(cfg_path: Path, outdir: Path) -> int:
    from .convert import SpeciesSpec, build_packs

    cfg = yaml.safe_load(Path(cfg_path).read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise ValueError(f"config root must be a mapping, got {type(cfg).__name__}")
    mid = _validate_material_id(cfg["material_id"])
    _validate_symbols([s["symbol"] for s in cfg["species"]])
    T = float(cfg["temperature"])
    density = float(cfg["density"])
    # validated before the tape reads, so a bad value leaves no partial output
    if not (math.isfinite(density) and density > 0.0):
        raise ValueError(f"density must be finite and positive, got {cfg['density']!r}")
    if not (math.isfinite(T) and T > 0.0):
        raise ValueError(f"temperature must be finite and positive, got {cfg['temperature']!r}")
    base = Path(cfg_path).parent
    specs, elements = [], []
    for s in cfg["species"]:
        tape = Path(s["tape"])
        if not tape.is_absolute():
            tape = base / tape
        specs.append(SpeciesSpec(tape=str(tape), symbol=s["symbol"], mass=float(s["mass"]),
                                 fraction=float(s["fraction"])))
        elements.append((s["symbol"], float(s["fraction"])))
    # coherent_convention disambiguates a sole LTHR=1 carrier (per_atom vs cef_scaled);
    # multi-carrier / monatomic layouts ignore it (default 'auto').
    packs = build_packs(specs, T, mid,
                        coherent_convention=cfg.get("coherent_convention", "auto"))
    outdir.mkdir(parents=True, exist_ok=True)
    pack_paths = []
    for pk, sp in zip(packs, specs):
        pk.metadata["source_endf_sha256"] = _sha(sp.tape)
        p = outdir / f"{pk.material_id}.endftslpack"
        write_pack(pk, p)
        pack_paths.append(str(p.resolve()))
        print(f"wrote {p}  (coherent={'yes' if pk.coh_edges_ev else 'no'})")
    ncmat = multi_pack_ncmat(elements, density, pack_paths)
    ncpath = outdir / f"{mid}.ncmat"
    ncpath.write_text(ncmat, encoding="utf-8")
    print(f"wrote {ncpath}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m ncrystal_plugin_ENDFTSL",
        description="Convert an ENDF/TSL evaluation to an NCrystal ENDFTSL pack + NCMAT.")
    ap.add_argument("tape", type=Path, nargs="?", help="single-species ENDF/TSL tape")
    ap.add_argument("-o", "--outdir", type=Path, required=True)
    ap.add_argument("--config", type=Path, help="polyatomic YAML (one tape per principal scatterer)")
    ap.add_argument("--temperature", type=float, default=296.0)
    ap.add_argument("--material-id", default="material")
    ap.add_argument("--symbol", default="C")
    ap.add_argument("--mass", type=float, default=12.0107)
    ap.add_argument("--density", type=float, default=1.0)
    a = ap.parse_args(argv)

    # Input/config problems exit 2, tape/runtime/I-O failures exit 3. The
    # density is checked before the tape read, so a bad value leaves no output.
    try:
        if a.config is not None:
            return _run_config(a.config, a.outdir)
        if a.tape is None:
            ap.error("provide a single-species tape positional, or --config for polyatomic")
        if not (math.isfinite(a.density) and a.density > 0.0):
            raise ValueError(f"--density must be finite and positive, got {a.density!r}")
        return _run_single(a)
    except yaml.YAMLError as exc:
        print(f"\nENDFTSL converter config error:\n  invalid YAML: {exc}", file=sys.stderr)
        return 2
    except KeyError as exc:
        print(f"\nENDFTSL converter config error:\n  missing config key {exc}",
              file=sys.stderr)
        return 2
    except (TypeError, ValueError) as exc:
        print(f"\nENDFTSL converter config error:\n  {exc}", file=sys.stderr)
        return 2
    except (RuntimeError, OSError) as exc:
        print(f"\nENDFTSL converter failed: {exc}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    sys.exit(main())
