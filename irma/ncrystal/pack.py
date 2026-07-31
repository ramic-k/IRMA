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
SUPPORTED_SCHEMA_VERSIONS = {SCHEMA_VERSION}   # only the current schema is accepted on read
VALID_INCOHERENT_ELASTIC_MODES = {"isotropic", "directional"}
VALID_BACKENDS = {"precomputed_sab"}
VALID_SAB_REPRESENTATIONS = {"sab", "scaled_sym_sab"}
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
    elastic_msd_a2: float | None = None
    elastic_incoherent_xs_barn: float | None = None
    elastic_scale: float | None = None
    elastic_coherent: bool = True
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


def _parse_bool(value: str) -> bool:
    """Parse 1/true/yes/on and 0/false/no/off (case-insensitive)."""
    lower = value.strip().lower()
    if lower in {"1", "true", "yes", "on"}:
        return True
    if lower in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"invalid boolean value {value!r}")


def _tensor_is_positive_semidefinite(t: list[float]) -> bool:
    """Check a 3x3 tensor (row-major, 9 floats) is finite, symmetric within
    tolerance, and positive semi-definite — required of U tensors."""
    tol = 1.0e-14
    if not all(math.isfinite(value) for value in t):
        return False
    if abs(t[1] - t[3]) > tol or abs(t[2] - t[6]) > tol or abs(t[5] - t[7]) > tol:
        return False
    a, b, c = t[0], t[1], t[2]
    d, e = t[4], t[5]
    f = t[8]
    minors = [
        a,
        d,
        f,
        a * d - b * b,
        a * f - c * c,
        d * f - e * e,
        a * d * f + 2.0 * b * c * e - a * e * e - d * c * c - f * b * b,
    ]
    return all(value >= -tol for value in minors)


def _validate(pack: IRMAPack) -> None:
    """Reject inconsistent packs before writing (grids, tensors, units)."""
    if not pack.material_id:
        raise ValueError("material_id is required")
    if pack.backend not in VALID_BACKENDS:
        raise ValueError(f"backend must be one of {sorted(VALID_BACKENDS)}")
    if pack.units != VALID_UNITS:
        raise ValueError(f"units must be {VALID_UNITS!r}")
    if pack.sab_representation not in VALID_SAB_REPRESENTATIONS:
        raise ValueError(
            "sab_representation must be one of "
            f"{sorted(VALID_SAB_REPRESENTATIONS)}"
        )
    elastic_values = {
        "elastic_msd_a2": pack.elastic_msd_a2,
        "elastic_incoherent_xs_barn": pack.elastic_incoherent_xs_barn,
    }
    if any(value is not None for value in elastic_values.values()):
        missing = [name for name, value in elastic_values.items() if value is None]
        if missing:
            raise ValueError(f"pack-owned elastic metadata missing {', '.join(missing)}")
        if pack.elastic_msd_a2 is not None and pack.elastic_msd_a2 <= 0.0:
            raise ValueError("elastic_msd_a2 must be positive")
        if (
            pack.elastic_incoherent_xs_barn is not None
            and pack.elastic_incoherent_xs_barn < 0.0
        ):
            raise ValueError("elastic_incoherent_xs_barn must be non-negative")
    if pack.elastic_scale is not None and pack.elastic_scale <= 0.0:
        raise ValueError("elastic_scale must be positive")
    if pack.incoherent_elastic_mode not in VALID_INCOHERENT_ELASTIC_MODES:
        raise ValueError(
            "incoherent_elastic_mode must be one of "
            f"{sorted(VALID_INCOHERENT_ELASTIC_MODES)}, "
            f"got {pack.incoherent_elastic_mode!r}"
        )
    if (pack.incoherent_elastic_mode == "directional"
            and not pack.elastic_u_tensors_a2):
        raise ValueError(
            "incoherent_elastic_mode = directional requires the per-site "
            "elastic_u_tensors_a2 (structure mode); a scalar-MSD pack carries "
            "no directional information"
        )
    if pack.elastic_u_tensors_a2:
        if len(pack.elastic_u_tensors_a2) % 9 != 0:
            raise ValueError("elastic_u_tensors_a2 length must be a multiple of 9")
        nsite = len(pack.elastic_u_tensors_a2) // 9
        if len(pack.elastic_u_symbols) != nsite:
            raise ValueError("elastic_u_symbols length must match tensor site count")
        if len(pack.elastic_u_frac_positions) != 3 * nsite:
            raise ValueError(
                "elastic_u_frac_positions length must equal 3 x tensor site count"
            )
        if not all(math.isfinite(value) for value in pack.elastic_u_frac_positions):
            raise ValueError("elastic_u_frac_positions values must be finite")
        if not all(symbol for symbol in pack.elastic_u_symbols):
            raise ValueError("elastic_u_symbols entries must be non-empty")
        # Per-site neutron data is mandatory alongside the tensors: the C++ uses it
        # in place of NCrystal's atom DB (no silent fallback). b_coh may be negative
        # (e.g. H, Ti, Li) -> finiteness only; sigma_inc must be non-negative.
        if len(pack.elastic_u_coherent_scatlen_sqrtbarn) != nsite:
            raise ValueError(
                "elastic_u_coherent_scatlen_sqrtbarn length must match tensor site count"
            )
        if len(pack.elastic_u_incoherent_xs_barn) != nsite:
            raise ValueError(
                "elastic_u_incoherent_xs_barn length must match tensor site count"
            )
        if not all(math.isfinite(v) for v in pack.elastic_u_coherent_scatlen_sqrtbarn):
            raise ValueError(
                "elastic_u_coherent_scatlen_sqrtbarn values must be finite"
            )
        if not all(math.isfinite(v) and v >= 0.0
                   for v in pack.elastic_u_incoherent_xs_barn):
            raise ValueError(
                "elastic_u_incoherent_xs_barn values must be finite and non-negative"
            )
        for site in range(nsite):
            tensor = pack.elastic_u_tensors_a2[9 * site : 9 * (site + 1)]
            if not _tensor_is_positive_semidefinite(tensor):
                raise ValueError(
                    "elastic_u_tensors_a2 tensors must be finite symmetric "
                    "positive semidefinite matrices"
                )
    if pack.backend == "precomputed_sab":
        required = {
            "temperature_K": pack.temperature_K,
            "bound_xs_barn": pack.bound_xs_barn,
            "element_mass_amu": pack.element_mass_amu,
        }
        missing = [name for name, value in required.items() if value is None]
        if missing:
            raise ValueError(f"precomputed_sab pack missing {', '.join(missing)}")
        for name, value in required.items():
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(
                    f"precomputed_sab pack {name} must be finite and positive, "
                    f"got {value!r}")
        if len(pack.alpha_grid) < 2:
            raise ValueError("alpha_grid must contain at least two points")
        if len(pack.beta_grid) < 2:
            raise ValueError("beta_grid must contain at least two points")
        for name, grid in (("alpha_grid", pack.alpha_grid),
                           ("beta_grid", pack.beta_grid)):
            if not all(math.isfinite(v) for v in grid):
                raise ValueError(f"{name} must contain only finite values")
        # The C++ loader's bilinear interpolator assumes sorted grids; a
        # hand-assembled or round-tripped pack with an out-of-order axis would
        # otherwise be a silent physics error. Guard it here at write/read time.
        for name, grid in (("alpha_grid", pack.alpha_grid),
                           ("beta_grid", pack.beta_grid)):
            if any(b <= a for a, b in zip(grid, grid[1:])):
                raise ValueError(f"{name} must be strictly increasing")
        if pack.sab_representation == "scaled_sym_sab":
            if pack.beta_grid[0] != 0.0:
                raise ValueError(
                    "scaled_sym_sab beta_grid must start at zero and contain "
                    "only non-negative values"
                )
            if any(beta < 0.0 for beta in pack.beta_grid):
                raise ValueError(
                    "scaled_sym_sab beta_grid must start at zero and contain "
                    "only non-negative values"
                )
        expected = len(pack.alpha_grid) * len(pack.beta_grid)
        if len(pack.sab_values) != expected:
            raise ValueError(
                "sab_values length must equal alpha_grid x beta_grid "
                f"({expected}), got {len(pack.sab_values)}"
            )
        if not all(math.isfinite(v) and v >= 0.0 for v in pack.sab_values):
            raise ValueError("sab_values must be finite and non-negative")
        if max(pack.sab_values) == 0.0:
            # An identically zero kernel is never physics: it is the signature
            # of zero-substituted neutron constants reaching the engine
            # (review NC-1). Refuse to write/read a silently inert material.
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
    if pack.elastic_msd_a2 is not None:
        lines.append(f"elastic_msd_a2 = {_format_float(pack.elastic_msd_a2)}")
    if pack.elastic_incoherent_xs_barn is not None:
        lines.append(
            "elastic_incoherent_xs_barn = "
            f"{_format_float(pack.elastic_incoherent_xs_barn)}"
        )
    if pack.elastic_scale is not None:
        lines.append(f"elastic_scale = {_format_float(pack.elastic_scale)}")
    if not pack.elastic_coherent:
        lines.append("elastic_coherent = false")
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
        if "\n" in key or "=" in key:
            raise ValueError("metadata keys must not contain newlines or '='")
        value = str(pack.metadata[key])
        if "\n" in value:
            raise ValueError("metadata values must not contain newlines")
        lines.append(f"meta.{key} = {value}")
    # newline="\n" keeps material-data files byte-identical across
    # platforms (no CRLF on Windows); the pack round-trip and the
    # plugin reference comparisons depend on LF-only. (QA finding CI-1)
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

    schema_raw = fields.get("schema_version")
    try:
        schema_version = int(schema_raw) if schema_raw is not None else None
    except ValueError:
        schema_version = None
    if schema_version not in SUPPORTED_SCHEMA_VERSIONS:
        raise ValueError(
            f"schema_version must be one of {sorted(SUPPORTED_SCHEMA_VERSIONS)}, "
            f"got {schema_raw!r}"
        )

    pack = IRMAPack(
        material_id=fields.get("material_id", ""),
        backend=fields.get("backend", ""),
        units=fields.get("units", ""),
        sab_representation=fields.get("sab_representation", "sab"),
        temperature_K=float(fields["temperature_K"]) if "temperature_K" in fields else None,
        bound_xs_barn=float(fields["bound_xs_barn"]) if "bound_xs_barn" in fields else None,
        element_mass_amu=float(fields["element_mass_amu"]) if "element_mass_amu" in fields else None,
        elastic_msd_a2=float(fields["elastic_msd_a2"]) if "elastic_msd_a2" in fields else None,
        elastic_incoherent_xs_barn=(
            float(fields["elastic_incoherent_xs_barn"])
            if "elastic_incoherent_xs_barn" in fields
            else None
        ),
        elastic_scale=float(fields["elastic_scale"]) if "elastic_scale" in fields else None,
        elastic_coherent=_parse_bool(fields.get("elastic_coherent", "true")),
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
