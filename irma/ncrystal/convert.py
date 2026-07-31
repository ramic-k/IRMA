"""Convention bridge: IRMA mode-2 downscatter S(α,β) → NCrystal pack convention.

IRMA owns its S-convention, so this bridge lives in core and stays in lockstep
with the engine. The engine emits S in IRMA's per-represented-atom convention on
a downscatter (β ≥ 0) grid in IRMA's natural ``(alpha, beta)`` orientation; this
module rescales it to the pack's bound-XS convention, maps α to NCrystal's
mass-scaled units, and stores the scaled-symmetric half-table the C++ loader
reconstructs the full S from by detailed balance.

No IRMA-core import here either — the input is plain arrays, so this is pure and
unit-testable without phonopy.
"""
from __future__ import annotations

import math
from collections.abc import Sequence

from .pack import IRMAPack, VALID_UNITS


def _as_float_list(values: Sequence[float], name: str) -> list[float]:
    """Coerce to a non-empty float list, naming the field on failure."""
    out = [float(value) for value in values]
    if not out:
        raise ValueError(f"{name} must not be empty")
    return out


def _validate_strictly_increasing(values: Sequence[float], name: str) -> None:
    """Require a strictly increasing sequence, naming the field on failure."""
    for left, right in zip(values, values[1:]):
        if not right > left:
            raise ValueError(f"{name} must be strictly increasing")


def rescale_sab_to_bound_xs(
    sab_asym_downscatter: Sequence[Sequence[float]],
    *,
    source_sigma_barn: float,
    bound_xs_barn: float,
) -> tuple[list[list[float]], float]:
    """Convert a SAB table from ``source_sigma_barn`` to pack-bound convention.

    IRMA's ``S_asym = (4π·kT/σ_b)·S`` divides the bound cross section OUT. When
    the engine ran with ``sab_sigma_barn = source_sigma_barn`` but the pack
    advertises ``bound_xs_barn`` to NCrystal, the table must be rescaled by
    ``source_sigma / bound_xs`` so the SABScatter kernel reproduces the intended
    absolute cross section. Returns ``(rescaled_table, scale)``.
    """

    source_sigma = float(source_sigma_barn)
    bound_xs = float(bound_xs_barn)
    if source_sigma <= 0.0:
        raise ValueError("source_sigma_barn must be positive")
    if bound_xs <= 0.0:
        raise ValueError("bound_xs_barn must be positive")
    scale = source_sigma / bound_xs
    return (
        [[float(value) * scale for value in row] for row in sab_asym_downscatter],
        scale,
    )


def pack_from_irma_sab(
    *,
    material_id: str,
    temperature_K: float,
    bound_xs_barn: float,
    element_mass_amu: float,
    alpha_mass_ratio: float | None = None,
    alpha_grid: Sequence[float],
    beta_downscatter_abs: Sequence[float],
    sab_asym_downscatter: Sequence[Sequence[float]],
    metadata: dict[str, str] | None = None,
) -> IRMAPack:
    """Build a precomputed NCrystal pack from an IRMA mode-2 downscatter SAB.

    ``sab_asym_downscatter`` is expected in IRMA's natural orientation:
    ``sab_asym_downscatter[ialpha][ibeta]`` with ``beta_downscatter_abs`` starting
    at zero and containing non-negative downscatter beta magnitudes. NCrystal's
    beta sign convention is final-minus-initial neutron energy, so IRMA
    downscatter belongs on NCrystal's negative-beta side. The scaled-symmetric
    representation stores an even ``S_scaled`` table, later unscaled by
    ``S(beta) = S_scaled(|beta|) * exp(-beta/2)``. To reconstruct the IRMA
    downscatter value at ``beta=-|beta|``, this exporter stores
    ``S_scaled = S_downscatter * exp(-|beta|/2)``.

    ``alpha_grid`` is mapped to NCrystal units by ``alpha_ncrystal =
    alpha_irma * alpha_mass_ratio`` (the principal-scatterer AWR; defaults to
    ``element_mass_amu`` when not given separately).
    """

    alpha = _as_float_list(alpha_grid, "alpha_grid")
    beta = _as_float_list(beta_downscatter_abs, "beta_downscatter_abs")
    _validate_strictly_increasing(alpha, "alpha_grid")
    _validate_strictly_increasing(beta, "beta_downscatter_abs")
    if alpha[0] <= 0.0:
        raise ValueError("alpha_grid values must be positive")
    alpha_scale = float(element_mass_amu if alpha_mass_ratio is None else alpha_mass_ratio)
    if alpha_scale <= 0.0:
        raise ValueError("alpha_mass_ratio must be positive")
    if beta[0] != 0.0 or any(value < 0.0 for value in beta):
        raise ValueError(
            "beta_downscatter_abs must start at zero and contain only "
            "non-negative values"
        )

    rows = [[float(value) for value in row] for row in sab_asym_downscatter]
    if len(rows) != len(alpha):
        raise ValueError("sab_asym_downscatter row count must equal alpha_grid length")
    max_sab = max((value for row in rows for value in row), default=0.0)
    # Mode-2 coherent interference histograms can leave negative cancellation
    # bins in principal-partitioned laws. IRMA's ENDF writer projects those
    # non-positive linear-S entries to zero before NJOY sees them. Mirror that
    # behavior, but keep a table-scale guard so genuinely negative laws fail.
    negative_tolerance = max(1.0e-15, 1.0e-2 * max_sab)
    clipped_count = 0
    clipped_min = 0.0
    for row in rows:
        if len(row) != len(beta):
            raise ValueError(
                "each sab_asym_downscatter row must equal "
                "beta_downscatter_abs length"
            )
        if any(value < -negative_tolerance for value in row):
            raise ValueError("sab_asym_downscatter values must be non-negative")
        for index, value in enumerate(row):
            if value < 0.0:
                clipped_count += 1
                clipped_min = min(clipped_min, value)
                row[index] = 0.0

    sab_values: list[float] = []
    for ibeta, beta_value in enumerate(beta):
        scale = math.exp(-0.5 * beta_value)
        for ialpha in range(len(alpha)):
            sab_values.append(rows[ialpha][ibeta] * scale)

    metadata_out = {
        **dict(metadata or {}),
        "alpha_convention": "ncrystal_alpha=irma_alpha*alpha_mass_ratio",
        "alpha_mass_ratio": f"{alpha_scale:.17g}",
    }
    if clipped_count:
        metadata_out["negative_sab_clipped_count"] = str(clipped_count)
        metadata_out["negative_sab_clipped_min"] = f"{clipped_min:.17g}"
        metadata_out["negative_sab_clip_tolerance"] = f"{negative_tolerance:.17g}"

    return IRMAPack(
        material_id=material_id,
        backend="precomputed_sab",
        units=VALID_UNITS,
        sab_representation="scaled_sym_sab",
        temperature_K=float(temperature_K),
        bound_xs_barn=float(bound_xs_barn),
        element_mass_amu=float(element_mass_amu),
        alpha_grid=[alpha_scale * value for value in alpha],
        beta_grid=beta,
        sab_values=sab_values,
        metadata=metadata_out,
    )
