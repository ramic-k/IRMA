"""CLI: ``python -m irma.ncrystal <config.yaml> -o <outdir>``.

Exports the per-principal NCrystal scattering data (``.irmapack``) for one
temperature and writes a complete, loadable material ``.ncmat`` that references
them in its ``@CUSTOM_IRMA`` section.
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

    # PyYAML is an extra (irma[spectra]); a bare install gets a clean message.
    try:
        import yaml
    except ModuleNotFoundError:
        yaml = None

    # Input problems exit 2, run failures exit 3 (the deck CLI's convention).
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
    except (ValueError, TypeError) as exc:
        print(f"\nNCrystal export config error:\n  {exc}", file=sys.stderr)
        return 2
    try:
        pack_paths, snippet_path = write_packs(cfg, args.outdir)
    except ImportError as exc:
        print(f"\nIRMA NCrystal export failed: missing optional dependency "
              f"({exc}). The exporter needs phonopy -- install it with "
              f"pip install 'irma[phonopy]'", file=sys.stderr)
        return 3
    except (RuntimeError, OSError, ValueError) as exc:
        print(f"\nIRMA NCrystal export failed: {exc}", file=sys.stderr)
        return 3
    print(f"\nExported {len(pack_paths)} NCrystal data file(s) and the material file:")
    for p in pack_paths:
        print(f"  {p}")
    print(f"  {snippet_path}")
    print(f"\nWith the ncrystal_plugin_IRMA plugin installed, load it with "
          f"NCrystal.createScatter(\"{snippet_path};temp="
          f"{cfg.material.temperature_K:g}K\").")
    return 0


if __name__ == "__main__":
    sys.exit(main())
