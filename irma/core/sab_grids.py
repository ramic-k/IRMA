"""Shared S(α,β) ⇄ physical (Q, E) grid convention.

The single authoritative home for the α↔Q / β↔E map, used by BOTH the ENDF /
standalone SAB path (``irma.core.standalone_sab``) and the NCrystal exporter
(``irma.ncrystal``) so the convention can never drift between them.

``α`` is the dimensionless momentum-transfer variable (ENDF, ``lat=1`` →
0.0253 eV reference) and depends on the scatterer mass ratio ``A``; ``β`` is the
dimensionless energy-transfer variable and is mass-independent.
"""
from __future__ import annotations

import numpy as np

from irma.core.constants import (
    BK as _BK_EV_PER_K,
    THERM as _THERM_EV,
    HBAR2_OVER_2MN_MEV_A2 as _HBAR2,
)


def irma_grid_to_physical_qe(
    alpha: np.ndarray,
    beta: np.ndarray,
    lat: int,
    temperature_k: float,
    mass_ratio: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(α, β) → ``(q_grid_ang_inv, e_grid_mev, alpha_abs, beta_downscatter_abs)``.

    ``lat=1`` rescales the input by ``THERM/kT`` (the 0.0253 eV reference); else
    the input is already absolute. The body is byte-for-byte the original
    ``standalone_sab._irma_grid_to_physical_qe`` so the ENDF path is unchanged.
    """
    kT_ev = _BK_EV_PER_K * temperature_k
    kT_mev = 1000.0 * kT_ev
    sc = _THERM_EV / kT_ev if lat == 1 else 1.0

    alpha_abs = np.asarray(alpha, dtype=float) * sc
    beta_downscatter_abs = np.asarray(beta, dtype=float) * sc
    q_grid_ang_inv = np.sqrt(np.maximum(alpha_abs, 0.0) * mass_ratio * kT_mev / _HBAR2)
    e_grid_mev = beta_downscatter_abs * kT_mev
    return q_grid_ang_inv, e_grid_mev, alpha_abs, beta_downscatter_abs


def physical_qe_to_irma_grid(
    q_grid_ang_inv: np.ndarray,
    e_grid_mev: np.ndarray,
    *,
    lat: int,
    temperature_k: float,
    mass_ratio: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Exact inverse of :func:`irma_grid_to_physical_qe`: physical (Q, E) → (α, β).

    ``α = Q²·(ħ²/2mₙ) / (A·kT) / sc`` and ``β = E / kT / sc`` with
    ``sc = THERM/kT`` for ``lat=1`` else 1.
    """
    kT_ev = _BK_EV_PER_K * float(temperature_k)
    kT_mev = 1000.0 * kT_ev
    sc = _THERM_EV / kT_ev if int(lat) == 1 else 1.0
    q = np.asarray(q_grid_ang_inv, float)
    e = np.asarray(e_grid_mev, float)
    alpha_abs = q * q * _HBAR2 / (float(mass_ratio) * kT_mev)
    beta_abs = e / kT_mev
    return alpha_abs / sc, beta_abs / sc


def make_uniform_qe_grid(q_min, q_max, dq, e_min, e_max, de):
    """Uniform physical grids ``(q_grid_ang_inv, e_grid_mev)`` — the same
    ``arange`` convention the noncubic engine uses for its Q/E grids."""
    if dq <= 0.0 or de <= 0.0:
        raise ValueError("grid requires dq > 0 and de > 0")
    if q_max <= q_min or e_max < e_min:
        raise ValueError("grid requires q_max > q_min and e_max >= e_min")
    q = np.arange(float(q_min), float(q_max) + 0.5 * float(dq), float(dq))
    e = np.arange(float(e_min), float(e_max) + 0.5 * float(de), float(de))
    return q, e


def auto_sab_grid(*, q_min, q_max, dq, e_min, e_max, de, lat, temperature_k,
                  mass_ratio):
    """Automatic (α, β) grid from a uniform physical Q/E range.

    A grid uniform in the physical variables (Q [1/Å], E [meV]) — i.e. α is
    "linear in Q" (α ∝ Q²) — which avoids the low-Q thermal bias of a uniform-α
    grid. α is forced strictly positive (drops Q=0) and β is forced to start at 0
    (the downscatter origin); both are required by the SAB / pack format.
    """
    q, e = make_uniform_qe_grid(q_min, q_max, dq, e_min, e_max, de)
    q = q[q > 0.0]                                   # α must be strictly positive
    e = e[e >= 0.0]
    if e.size == 0 or e[0] != 0.0:                   # β must start at 0
        e = np.concatenate([[0.0], e])
    if q.size < 2 or e.size < 2:
        raise ValueError("auto grid produced fewer than two α or β points")
    return physical_qe_to_irma_grid(
        q, e, lat=lat, temperature_k=temperature_k, mass_ratio=mass_ratio)
