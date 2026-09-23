"""IRMA → NCrystal exporter.

Turns an IRMA mode-2 (coherent one-phonon + anisotropic Debye-Waller,
powder-averaged) calculation into a per-temperature baked ``.irmapack`` that the
NCrystal IRMA plugin samples at runtime. IRMA is the single producer and reference.

Public API::

    from irma.ncrystal import NCrystalExportConfig, build_packs, write_packs
    cfg = NCrystalExportConfig.from_yaml("graphite_export.yaml")
    packs, ncmat_snippet = build_packs(cfg)

or from the command line::

    python -m irma.ncrystal graphite_export.yaml -o out/
"""
from __future__ import annotations

from .config import NCrystalExportConfig
from .build import build_packs, write_packs
from .pack import IRMAPack, read_pack, write_pack

__all__ = ["NCrystalExportConfig", "build_packs", "write_packs",
           "IRMAPack", "read_pack", "write_pack"]
