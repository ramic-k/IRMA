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

from .pack import (
    IRMAPack,
    write_pack,
    read_pack,
    MAGIC,
    SCHEMA_VERSION,
)
from .convert import pack_from_irma_sab, rescale_sab_to_bound_xs
from .config import NCrystalExportConfig
from .build import build_packs, write_packs
from .ncmat import (
    assemble_material_ncmat,
    build_base_ncmat,
    custom_irma_section,
    lattice_to_cell_params,
)
from .provenance import collect_provenance

__all__ = [
    "IRMAPack",
    "write_pack",
    "read_pack",
    "MAGIC",
    "SCHEMA_VERSION",
    "pack_from_irma_sab",
    "rescale_sab_to_bound_xs",
    "NCrystalExportConfig",
    "build_packs",
    "write_packs",
    "assemble_material_ncmat",
    "build_base_ncmat",
    "custom_irma_section",
    "lattice_to_cell_params",
    "collect_provenance",
]
