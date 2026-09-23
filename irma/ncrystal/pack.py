"""``.irmapack`` text container — the per-temperature baked pack the NCrystal
IRMA plugin samples at runtime.

This is the serializer half of the IRMA→NCrystal bridge. It is deliberately
free of any IRMA-core import so the format can be read/written (and unit-tested)
without phonopy or a C++ build. The physics conversion that fills an
:class:`IRMAPack` lives in :mod:`irma.ncrystal.convert`.

Format: a UTF-8 text file whose first line is the magic ``IRMAPACK_TEXT_V1``,
followed by ``key = value`` lines. ``meta.<name> = <value>`` lines carry
provenance/diagnostic metadata that the C++ loader ignores. ``schema_version``
is ``2`` -- the current and only supported schema. Both the writer and
the readers (this module and the C++ loader) require it and reject
anything else, so a pack and its loader can never silently disagree.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path


MAGIC = "IRMAPACK_TEXT_V1"
SCHEMA_VERSION = 2
VALID_UNITS = "angstrom_meV_barn_K"


@dataclass
class IRMAPack:
    """One per-principal scattering-data pack: the S(alpha,beta) table, the
    elastic (Debye-Waller / coherent) data, and the neutron constants the
    C++ NCrystal plugin needs to sample the material at one temperature."""

    material_id: str
    backend: str
    units: str = VALID_UNITS
    sab_representation: str = "sab"
    temperature_K: float | None = None
    bound_xs_barn: float | None = None
    element_mass_amu: float | None = None
    elastic_u_tensors_a2: list[float] = field(default_factory=list)
    elastic_u_symbols: list[str] = field(default_factory=list)
    elastic_u_frac_positions: list[float] = field(default_factory=list)
    # Per-tensor-site neutron data the C++ coherent F(hkl) + incoherent DW read
    # INSTEAD of NCrystal's atom DB, so the plugin honors exactly what the IRMA
    # config specified (b_coh_fm / sigma_inc_b). Coherent scattering length is in
    # sqrt(barn) to match NCrystal's coherentScatLen() (= b_coh_fm / 10); the
    # incoherent xs is in barn. Required whenever elastic_u_tensors_a2 is present.
    elastic_u_coherent_scatlen_sqrtbarn: list[float] = field(default_factory=list)
    elastic_u_incoherent_xs_barn: list[float] = field(default_factory=list)
    # 'isotropic' (default): the C++ collapses each site tensor to trace/3 and
    # uses NCrystal's stock ElIncScatter. 'directional': the C++ samples the
    # orientation-averaged <exp(-Q^2 uhat.U.uhat)> per site (requires
    # elastic_u_tensors_a2). See irma.core.incoherent_dw.
    incoherent_elastic_mode: str = "isotropic"
    alpha_grid: list[float] = field(default_factory=list)
    beta_grid: list[float] = field(default_factory=list)
    sab_values: list[float] = field(default_factory=list)
    metadata: dict[str, str] = field(default_factory=dict)


def _format_float(value: float) -> str:
    """Format a float with 17 significant digits (round-trips exactly)."""
    return f"{float(value):.17g}"


def _format_float_list(values: list[float]) -> str:
    """Space-joined float list in the exact round-trip format."""
    return " ".join(_format_float(v) for v in values)


def _parse_float_list(value: str) -> list[float]:
    """Parse a space- or comma-separated float list ("" -> [])."""
    if not value.strip():
        return []
    return [float(item) for item in value.replace(",", " ").split()]


def _parse_string_list(value: str) -> list[str]:
    """Parse a space- or comma-separated string list ("" -> [])."""
    if not value.strip():
        return []
    return value.replace(",", " ").split()


def _validate(pack: IRMAPack) -> None:
    """Reject a pack the exporter could get wrong (scalars, grids, kernel).
    The C++ loader checks the full format, including the elastic block."""
    if not pack.material_id:
        raise ValueError("material_id is required")
    for name in ("temperature_K", "bound_xs_barn", "element_mass_amu"):
        value = getattr(pack, name)
        if value is None or not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"pack {name} must be finite and positive, got {value!r}")
    for name, grid in (("alpha_grid", pack.alpha_grid), ("beta_grid", pack.beta_grid)):
        if len(grid) < 2:
            raise ValueError(f"{name} must contain at least two points")
        if not all(math.isfinite(v) for v in grid):
            raise ValueError(f"{name} must contain only finite values")
        # the C++ bilinear interpolator assumes sorted grids
        if any(b <= a for a, b in zip(grid, grid[1:])):
            raise ValueError(f"{name} must be strictly increasing")
    if pack.sab_representation == "scaled_sym_sab" and pack.beta_grid[0] != 0.0:
        raise ValueError("scaled_sym_sab beta_grid must start at zero")
    expected = len(pack.alpha_grid) * len(pack.beta_grid)
    if len(pack.sab_values) != expected:
        raise ValueError(
            "sab_values length must equal alpha_grid x beta_grid "
            f"({expected}), got {len(pack.sab_values)}")
    if not all(math.isfinite(v) and v >= 0.0 for v in pack.sab_values):
        raise ValueError("sab_values must be finite and non-negative")
    if max(pack.sab_values) == 0.0:
        # the signature of zero neutron constants reaching the engine
        raise ValueError(
            "sab_values are identically zero: the pack would describe a "
            "material with no inelastic scattering at all (typically "
            "caused by missing b_coh_fm/sigma_inc_b neutron constants)")


def write_pack(pack: IRMAPack, path: str | Path) -> None:
    """Validate ``pack`` and write it to ``path`` in the schema-2 text format."""
    _validate(pack)
    path = Path(path)
    lines = [
        MAGIC,
        f"schema_version = {SCHEMA_VERSION}",
        f"material_id = {pack.material_id}",
        f"backend = {pack.backend}",
        f"units = {pack.units}",
        f"sab_representation = {pack.sab_representation}",
    ]
    if pack.temperature_K is not None:
        lines.append(f"temperature_K = {_format_float(pack.temperature_K)}")
    if pack.bound_xs_barn is not None:
        lines.append(f"bound_xs_barn = {_format_float(pack.bound_xs_barn)}")
    if pack.element_mass_amu is not None:
        lines.append(f"element_mass_amu = {_format_float(pack.element_mass_amu)}")
    if pack.elastic_u_tensors_a2:
        lines.append(
            f"elastic_u_tensors_a2 = {_format_float_list(pack.elastic_u_tensors_a2)}"
        )
        lines.append(f"elastic_u_symbols = {' '.join(pack.elastic_u_symbols)}")
        lines.append(
            "elastic_u_frac_positions = "
            f"{_format_float_list(pack.elastic_u_frac_positions)}"
        )
        lines.append(
            "elastic_u_coherent_scatlen_sqrtbarn = "
            f"{_format_float_list(pack.elastic_u_coherent_scatlen_sqrtbarn)}"
        )
        lines.append(
            "elastic_u_incoherent_xs_barn = "
            f"{_format_float_list(pack.elastic_u_incoherent_xs_barn)}"
        )
    if pack.incoherent_elastic_mode == "directional":
        lines.append("incoherent_elastic_mode = directional")
    if pack.alpha_grid:
        lines.append(f"alpha_grid = {_format_float_list(pack.alpha_grid)}")
    if pack.beta_grid:
        lines.append(f"beta_grid = {_format_float_list(pack.beta_grid)}")
    if pack.sab_values:
        lines.append(f"sab_values = {_format_float_list(pack.sab_values)}")
    for key in sorted(pack.metadata):
        lines.append(f"meta.{key} = {pack.metadata[key]}")
    # LF-only on every platform: the round-trip and reference comparisons
    # depend on byte-identical files
    path.write_text("\n".join(lines) + "\n", encoding="utf-8",
                    newline="\n")


def read_pack(path: str | Path) -> IRMAPack:
    """Read and validate a schema-2 pack file, returning an IRMAPack."""
    path = Path(path)
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines or lines[0].strip() != MAGIC:
        raise ValueError("invalid IRMA pack magic")

    fields: dict[str, str] = {}
    metadata: dict[str, str] = {}
    for lineno, raw in enumerate(lines[1:], start=2):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise ValueError(f"line {lineno}: expected key = value")
        key, value = [part.strip() for part in line.split("=", 1)]
        if key.startswith("meta."):
            metadata[key[5:]] = value
        else:
            fields[key] = value

    if fields.get("schema_version") != str(SCHEMA_VERSION):
        raise ValueError(f"schema_version must be {SCHEMA_VERSION}, "
                         f"got {fields.get('schema_version')!r}")

    pack = IRMAPack(
        material_id=fields.get("material_id", ""),
        backend=fields.get("backend", ""),
        units=fields.get("units", ""),
        sab_representation=fields.get("sab_representation", "sab"),
        temperature_K=float(fields["temperature_K"]) if "temperature_K" in fields else None,
        bound_xs_barn=float(fields["bound_xs_barn"]) if "bound_xs_barn" in fields else None,
        element_mass_amu=float(fields["element_mass_amu"]) if "element_mass_amu" in fields else None,
        elastic_u_tensors_a2=_parse_float_list(fields.get("elastic_u_tensors_a2", "")),
        elastic_u_symbols=_parse_string_list(fields.get("elastic_u_symbols", "")),
        elastic_u_frac_positions=_parse_float_list(
            fields.get("elastic_u_frac_positions", "")
        ),
        elastic_u_coherent_scatlen_sqrtbarn=_parse_float_list(
            fields.get("elastic_u_coherent_scatlen_sqrtbarn", "")
        ),
        elastic_u_incoherent_xs_barn=_parse_float_list(
            fields.get("elastic_u_incoherent_xs_barn", "")
        ),
        incoherent_elastic_mode=fields.get("incoherent_elastic_mode",
                                           "isotropic"),
        alpha_grid=_parse_float_list(fields.get("alpha_grid", "")),
        beta_grid=_parse_float_list(fields.get("beta_grid", "")),
        sab_values=_parse_float_list(fields.get("sab_values", "")),
        metadata=metadata,
    )
    _validate(pack)
    return pack
