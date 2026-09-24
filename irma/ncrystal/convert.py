"""Convention bridge: IRMA mode-2 downscatter S(α,β) → NCrystal pack convention.

IRMA owns its S-convention, so this bridge is part of IRMA (irma.ncrystal) and
stays in lockstep with the engine. The engine emits S in IRMA's per-represented-atom convention on
a downscatter (β ≥ 0) grid in IRMA's natural ``(alpha, beta)`` orientation; this
module maps α to NCrystal's mass-scaled units and stores the scaled-symmetric
half-table the C++ loader reconstructs the full S from by detailed balance.

No IRMA-core import here either — the input is plain arrays, so this is pure and
unit-testable without phonopy.
"""
from __future__ import annotations

import math
from collections.abc import Sequence

from .pack import IRMAPack, VALID_UNITS


def pack_from_irma_sab(
    *,
    material_id: str,
    temperature_K: float,
    bound_xs_barn: float,
    element_mass_amu: float,
    alpha_mass_ratio: float,
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
    alpha_irma * alpha_mass_ratio`` (the principal-scatterer AWR). The grids
    and the table shape are checked by the engine and by ``write_pack``.
    """
    alpha = [float(a) for a in alpha_grid]
    beta = [float(b) for b in beta_downscatter_abs]
    alpha_scale = float(alpha_mass_ratio)
    rows = [[float(value) for value in row] for row in sab_asym_downscatter]
    max_sab = max((value for row in rows for value in row), default=0.0)
    # Rounding leaves some cells a little below zero (about -2e-17 on the
    # graphite reference export). NCrystal tables cannot hold negative
    # values, so those cells are set to zero, as the ENDF writer does. A cell
    # below -1% of the table maximum is not rounding, so the export stops.
    negative_tolerance = max(1.0e-15, 1.0e-2 * max_sab)
    clipped_count = 0
    clipped_min = 0.0
    for i, row in enumerate(rows):
        worst = min(row, default=0.0)
        if worst < -negative_tolerance:
            j = row.index(worst)
            raise ValueError(
                f"the S(alpha, beta) table has a cell at {worst:.3g} "
                f"(alpha={alpha[i]:g}, beta={beta[j]:g}), below -1% of the table "
                f"maximum {max_sab:.3g}; S values must be non-negative")
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
