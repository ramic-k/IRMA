"""Pure setup-path helpers for the noncubic inelastic engine.

These are dependency-light utilities — grid loading, deck-value parsing,
unit/sigma inference, the S(Q,E) -> asymmetric-SAB conversion, and the
multiphonon-order sizing rule — factored out of ``noncubic_engine`` so that
module can stay focused on the multiprocessing kernels and the compute-phase
orchestration. None of these touch the ``WORKER_STATE`` global or the
fork-worker pool, so they live here and ``noncubic_engine`` re-exports them
for backward compatibility.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from irma.core.constants import (
    BK as _BK_EV_PER_K,
    HBAR2_OVER_2MN_MEV_A2,
    AMASSN as NEUTRON_MASS_AMU,
)

ANG2_TO_BARN = 1e8
KB_MEV_PER_K = _BK_EV_PER_K * 1.0e3

# Single-element fallback tables for the STANDALONE CLI driver only. The
# production engine path (mode 1/2 ENDF generation) always passes
# site_scattering_lengths_angstrom / incoherent overrides explicitly.
_FALLBACK_C_SCATTERING_LENGTHS_ANGSTROM = {
    "C": 6.646e-5,
}
_FALLBACK_C_INCOHERENT_CROSS_SECTIONS_BARN = {
    "C": 0.001,
}


def load_grid_from_text(path: Path) -> np.ndarray:
    """Load a 1D grid of bin centers from a text file."""
    text = path.read_text(encoding="ascii")
    cleaned = []
    for line in text.splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            cleaned.append(line.replace(",", " "))
    arr = np.fromstring("\n".join(cleaned), sep=" ")
    if arr.size == 0:
        raise ValueError(f"Grid file is empty: {path}")
    if np.any(~np.isfinite(arr)):
        raise ValueError(f"Grid file contains non-finite values: {path}")
    arr = np.unique(arr)
    if arr.size < 2:
        raise ValueError(f"Grid file must contain at least two distinct centers: {path}")
    return arr


def load_grid_from_oclimax_csv(csv_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load exact Q and E grid centers from an OCLIMAX CSV export."""
    arr = np.genfromtxt(csv_path, comments="#", delimiter=",", autostrip=True)
    arr = arr[~np.isnan(arr).all(axis=1)]
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise ValueError(f"Unexpected OCLIMAX CSV format: {csv_path}")
    q_grid = np.unique(arr[:, 0])
    e_grid = np.unique(arr[:, 1])
    return q_grid.astype(float), e_grid.astype(float)


def infer_mass_ratio(masses_amu: np.ndarray, mass_ratio_override: float | None) -> float:
    """Infer A = M / m_n for SAB conversion."""
    if mass_ratio_override is not None:
        if not float(mass_ratio_override) > 0.0:
            raise ValueError(
                "Asymmetric SAB conversion requires a positive mass ratio "
                f"(got {mass_ratio_override!r})."
            )
        return float(mass_ratio_override)
    unique_masses = np.unique(np.round(masses_amu, decimals=10))
    if len(unique_masses) != 1:
        raise ValueError(
            "Asymmetric SAB conversion requires a single atomic mass or --sab-mass-ratio."
        )
    return float(unique_masses[0] / NEUTRON_MASS_AMU)


def infer_sigma_barn(
    symbols: list[str],
    scattering_lengths_angstrom: dict[str, float],
    incoherent_cross_sections_barn: dict[str, float],
) -> tuple[float, float, float]:
    """Return per-species coherent, incoherent, and total sigma in barn."""
    unique_symbols = sorted(set(symbols))
    if len(unique_symbols) != 1:
        raise ValueError("infer_sigma_barn assumes a single scatterer species for SAB conversion; supply --sab-sigma-barn for mixed materials.")
    symbol = unique_symbols[0]
    sigma_coh = 4.0 * np.pi * scattering_lengths_angstrom[symbol] ** 2 * ANG2_TO_BARN
    sigma_inc = float(incoherent_cross_sections_barn[symbol])
    return float(sigma_coh), sigma_inc, float(sigma_coh + sigma_inc)


def normalize_site_groups(
    site_groups: object | None,
    num_sites: int,
) -> tuple[np.ndarray, ...]:
    """Normalize exported site groups to validated primitive-site index arrays."""
    if site_groups is None:
        return (np.arange(num_sites, dtype=np.intp),)

    normalized: list[np.ndarray] = []
    seen = np.zeros(num_sites, dtype=bool)
    for group_index, group in enumerate(site_groups):
        group_arr = np.asarray(group, dtype=np.intp)
        if group_arr.ndim != 1:
            raise ValueError(f"site_groups[{group_index}] must be a 1D index list.")
        if len(group_arr) == 0:
            raise ValueError(f"site_groups[{group_index}] must not be empty.")
        if np.any(group_arr < 0) or np.any(group_arr >= num_sites):
            raise ValueError(f"site_groups[{group_index}] contains out-of-range site indices.")
        if len(np.unique(group_arr)) != len(group_arr):
            raise ValueError(f"site_groups[{group_index}] contains duplicate site indices.")
        if np.any(seen[group_arr]):
            raise ValueError("site_groups must be disjoint in primitive-site space.")
        seen[group_arr] = True
        normalized.append(np.array(group_arr, dtype=np.intp, copy=True))

    return tuple(normalized)


def convert_sqe_to_asym_downscatter_sab(
    sqe_barn_per_mev: np.ndarray,
    q_grid_ang_inv: np.ndarray,
    e_grid_mev: np.ndarray,
    temperature_k: float,
    sigma_barn: float,
    mass_ratio: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert positive-energy-loss S(Q,E) to downscatter-side asymmetric SAB.

    This is the object we want for downstream thermal scattering-law generation:

    ``S_asym_downscatter(alpha, beta_downscatter_abs) = (4*pi*kT/sigma_b) * S(Q, E_tr)``

    Here ``sqe_barn_per_mev`` is the powder-averaged differential quantity
    ``d^2 sigma / (dOmega dE')``. The ``4*pi`` converts from the internal
    per-steradian convention to the angle-integrated ``S(alpha,beta)``
    normalization.
    """
    if not temperature_k > 0.0:
        raise ValueError(
            "convert_sqe_to_asym_downscatter_sab requires temperature_k > 0 K "
            f"(got {temperature_k!r}); beta = E / kT and alpha = .../kT are "
            "undefined at T <= 0."
        )
    kT_meV = KB_MEV_PER_K * temperature_k
    beta_downscatter_abs = e_grid_mev / kT_meV
    alpha = HBAR2_OVER_2MN_MEV_A2 * q_grid_ang_inv**2 / (mass_ratio * kT_meV)
    if sigma_barn <= 0.0:
        if np.any(np.abs(sqe_barn_per_mev) > 0.0):
            raise ValueError(
                "convert_sqe_to_asym_downscatter_sab received sigma_barn <= 0 "
                "for a nonzero S(Q,E) channel."
            )
        return alpha, beta_downscatter_abs, np.zeros_like(sqe_barn_per_mev, dtype=float)
    sab_asym_downscatter = (4.0 * np.pi * kT_meV / sigma_barn) * sqe_barn_per_mev
    return alpha, beta_downscatter_abs, sab_asym_downscatter


def compute_last_nonzero_beta(
    sab_alpha_beta: np.ndarray,
    beta_downscatter_abs: np.ndarray,
    tol: float = 0.0,
) -> float | None:
    """Return the last beta row carrying nonzero weight in an ``(alpha, beta)`` SAB array."""
    sab_arr = np.asarray(sab_alpha_beta, dtype=float)
    beta_arr = np.asarray(beta_downscatter_abs, dtype=float)
    if sab_arr.ndim != 2:
        raise ValueError("Expected a 2D (n_alpha, n_beta) SAB array.")
    if sab_arr.shape[1] != len(beta_arr):
        raise ValueError("Beta-axis length does not match the SAB array.")
    row_max = np.max(sab_arr, axis=0)
    nz = np.flatnonzero(row_max > tol)
    if nz.size == 0:
        return None
    return float(beta_arr[nz[-1]])


def parse_scattering_lengths(
    symbols: list[str],
    scattering_lengths_json: str | None,
    scattering_lengths_file: str | None,
) -> dict[str, float]:
    """Resolve coherent scattering lengths in Angstrom keyed by element symbol."""
    scattering_lengths = dict(_FALLBACK_C_SCATTERING_LENGTHS_ANGSTROM)
    if scattering_lengths_file:
        with open(scattering_lengths_file, encoding="ascii") as handle:
            scattering_lengths.update({str(k): float(v) for k, v in json.load(handle).items()})
    if scattering_lengths_json:
        scattering_lengths.update({str(k): float(v) for k, v in json.loads(scattering_lengths_json).items()})
    missing = sorted(set(symbols) - set(scattering_lengths))
    if missing:
        raise ValueError(f"Missing coherent scattering lengths for {missing}.")
    return scattering_lengths


def parse_incoherent_cross_sections(
    symbols: list[str],
    incoherent_cross_sections_json: str | None,
    incoherent_cross_sections_file: str | None,
) -> dict[str, float]:
    """Resolve incoherent cross sections in barn keyed by element symbol."""
    sigma_inc = dict(_FALLBACK_C_INCOHERENT_CROSS_SECTIONS_BARN)
    if incoherent_cross_sections_file:
        with open(incoherent_cross_sections_file, encoding="ascii") as handle:
            sigma_inc.update({str(k): float(v) for k, v in json.load(handle).items()})
    if incoherent_cross_sections_json:
        sigma_inc.update({str(k): float(v) for k, v in json.loads(incoherent_cross_sections_json).items()})
    missing = sorted(set(symbols) - set(sigma_inc))
    if missing:
        raise ValueError(f"Missing incoherent cross sections for {missing}.")
    return sigma_inc


def reshape_mesh_eigenvectors(mesh_eigenvectors: np.ndarray, num_atoms: int) -> np.ndarray:
    """Reshape phonopy mesh eigenvectors to ``(n_q, n_branches, n_atoms, 3)``."""
    n_q = mesh_eigenvectors.shape[0]
    n_branches = num_atoms * 3
    expected_last = n_branches
    if mesh_eigenvectors.ndim == 3 and mesh_eigenvectors.shape == (n_q, expected_last, n_branches):
        return mesh_eigenvectors.transpose(0, 2, 1).reshape(n_q, n_branches, num_atoms, 3)
    if mesh_eigenvectors.ndim == 4 and mesh_eigenvectors.shape == (n_q, n_branches, num_atoms, 3):
        return mesh_eigenvectors
    raise ValueError(f"Unexpected mesh eigenvector shape: {mesh_eigenvectors.shape}")


def derive_required_multiphonon_order(
    max_q_ang_inv: float,
    thermal_mats: np.ndarray,
    requested_order: int,
    margin_sigma: float = 6.0,
    hard_cap: int = 2000,
) -> tuple[int, int, float, float]:
    """Physically-required multiphonon order to converge the incoherent Poisson sum.

    The incoherent multiphonon order distribution is Poisson with mean
    ``2W = Q^2 * (u_hat . U . u_hat)``; its upper tail is captured to ~``Phi(margin_sigma)``
    by ``n >= 2W + margin_sigma * sqrt(2W)``. We size to the worst case
    ``2W_max = Q_max^2 * U_max`` (largest thermal-displacement eigenvalue, i.e. the soft
    axis) so that every sampled direction reaches the free-gas limit. The order is the only
    convergence lever once the per-order weighting is the bounded Poisson form (the signed
    work-grid extent does not enter, since orders whose support spills past the output grid
    contribute nothing to it).

    Returns ``(effective_order, required_order, two_w_max, u_max)`` where ``effective_order``
    is ``max(requested, required)`` clamped to ``hard_cap``.
    """
    eigvals = np.linalg.eigvalsh(np.asarray(thermal_mats, dtype=float))  # (n_atom, 3)
    u_max = float(np.max(eigvals)) if eigvals.size else 0.0
    requested_order = max(2, int(requested_order))
    if u_max <= 0.0 or max_q_ang_inv <= 0.0:
        return requested_order, requested_order, 0.0, u_max
    two_w_max = (float(max_q_ang_inv) ** 2) * u_max
    required = int(math.ceil(two_w_max + margin_sigma * math.sqrt(two_w_max))) + 2
    required = max(2, required)
    effective = min(max(requested_order, required), int(hard_cap))
    return effective, required, two_w_max, u_max
