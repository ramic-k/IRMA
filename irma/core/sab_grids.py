"""The (α, β) to physical (Q, E) map used by ``irma.core.standalone_sab``.

``lat=1`` grids are in 0.0253 eV units and are rescaled by THERM/kT.
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

    ``lat=1`` rescales the input by ``THERM/kT``; else it is already absolute.
    """
    kT_ev = _BK_EV_PER_K * temperature_k
    kT_mev = 1000.0 * kT_ev
    sc = _THERM_EV / kT_ev if lat == 1 else 1.0

    alpha_abs = np.asarray(alpha, dtype=float) * sc
    beta_downscatter_abs = np.asarray(beta, dtype=float) * sc
    q_grid_ang_inv = np.sqrt(np.maximum(alpha_abs, 0.0) * mass_ratio * kT_mev / _HBAR2)
    e_grid_mev = beta_downscatter_abs * kT_mev
    return q_grid_ang_inv, e_grid_mev, alpha_abs, beta_downscatter_abs
