"""CLI: ``python -m irma.ncrystal <config.yaml> -o <outdir>``.

Exports the per-principal NCrystal scattering data (``.irmapack``) for one
temperature and writes the ``@CUSTOM_IRMA`` NCMAT snippet that wires it into a
material.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: bake the packs/NCMAT described by a YAML config."""
    parser = argparse.ArgumentParser(
        prog="python -m irma.ncrystal",
        description="Export an IRMA mode-2 NCrystal scattering-data set from a "
                    "YAML config.")
    parser.add_argument("config", type=Path, help="export config YAML")
    parser.add_argument("-o", "--outdir", type=Path, required=True,
                        help="output directory for the NCrystal data (.irmapack) files")
    args = parser.parse_args(argv)

    from .config import NCrystalExportConfig
    from .build import write_packs
    from irma.spectra.config import SpectraConfigError

    # PyYAML is an extras dependency (irma[spectra]); on a bare core install
    # the exporter must still fail with a clean message, not a
    # ModuleNotFoundError traceback.
    try:
        import yaml
    except ModuleNotFoundError:
        yaml = None

    # Error boundary matching the deck CLI's stream/exit conventions (review
    # S11): input problems get a clean field-naming message and exit 2, run
    # failures a one-line message and exit 3 -- never a raw traceback for a
    # routine mistake like a typo'd path or malformed YAML.
    if not args.config.exists():
        print(f"\nNCrystal export config error:\n  config file not found: "
              f"{args.config}", file=sys.stderr)
        return 2
    if yaml is None:
        print("\nNCrystal export config error:\n  PyYAML is required to read "
              "the export config (install the spectra extras: "
              "pip install 'irma[spectra]' or pip install pyyaml)",
              file=sys.stderr)
        return 2
    try:
        cfg = NCrystalExportConfig.from_yaml(args.config)
    except FileNotFoundError:
        print(f"\nNCrystal export config error:\n  config file not found: "
              f"{args.config}", file=sys.stderr)
        return 2
    except yaml.YAMLError as exc:
        print(f"\nNCrystal export config error:\n  {args.config} is not valid "
              f"YAML: {exc}", file=sys.stderr)
        return 2
    except (SpectraConfigError, ValueError, TypeError) as exc:
        print(f"\nNCrystal export config error:\n  {exc}", file=sys.stderr)
        return 2
    try:
        pack_paths, snippet_path = write_packs(cfg, args.outdir)
    except ImportError as exc:
        # Missing optional dependency (the exporter needs phonopy for the
        # mode-1/2 engine): an actionable one-liner, not a traceback
        # (review NC-4; PyYAML gets the same treatment above).
        print(f"\nIRMA NCrystal export failed: missing optional dependency "
              f"({exc}). The exporter needs phonopy -- install it with "
              f"pip install 'irma[phonopy]'", file=sys.stderr)
        return 3
    except (RuntimeError, OSError, ValueError) as exc:
        # User-meaningful failures (phonopy model loading, missing/unreadable
        # files, semantic problems); a traceback stays reserved for genuinely
        # unexpected errors.
        print(f"\nIRMA NCrystal export failed: {exc}", file=sys.stderr)
        return 3
    print(f"\nExported {len(pack_paths)} NCrystal data file(s) + NCMAT snippet:")
    for p in pack_paths:
        print(f"  {p}")
    print(f"  {snippet_path}")
    print("\nAppend the snippet's @CUSTOM_IRMA block to your material's "
          ".ncmat file and load it with the ncrystal_plugin_IRMA plugin.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
