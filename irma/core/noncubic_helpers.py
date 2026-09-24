"""Pure setup-path helpers for the noncubic inelastic engine.

These are dependency-light utilities — grid loading, deck-value parsing,
unit/sigma inference, the S(Q,E) -> asymmetric-SAB conversion, and the
multiphonon-order sizing rule — factored out of ``noncubic_engine`` so that
module can stay focused on the multiprocessing kernels and the compute-phase
orchestration. None of these touch the ``WORKER_STATE`` global or the
worker pool; ``noncubic_engine`` imports them for its own phases.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from irma.core.constants import (
    BK as _BK_EV_PER_K,
    HBAR2_OVER_2MN_MEV_A2,
    AMASSN as NEUTRON_MASS_AMU,
)

from irma.core.grids import effective_temperature_bound_ratio

ANG2_TO_BARN = 1e8
KB_MEV_PER_K = _BK_EV_PER_K * 1.0e3

# Margin shared by the multiphonon order rule and its energy-reach check:
# the order covers the Poisson(2W) phonon count to this many standard
# deviations, and the reach check asks for the recoil ridge plus this many
# thermal widths. One constant keeps the two statements in step.
MULTIPHONON_MARGIN_SIGMAS = 6.0

# Carbon-only fallback tables, used when no per-site values or JSON tables are
# given (the engine CLI, and spectra configs without b_coh_fm).
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
    """Site groups as primitive-site index arrays; ``None`` means one group of all sites.

    The groups come from IRMA's own atom-type grouping or from an NCrystal
    config that ``irma.ncrystal.build`` has already validated.
    """
    if site_groups is None:
        return (np.arange(num_sites, dtype=np.intp),)
    return tuple(np.asarray(group, dtype=np.intp) for group in site_groups)


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

    Here ``sqe_barn_per_mev`` is the powder-averaged ``(sigma/4pi) S(Q,E)`` in
    barn/sr/meV, with no k'/k factor (``d^2 sigma / (dOmega dE')`` is k'/k
    times it). The ``4*pi`` converts from the internal
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


def parse_element_table(
    symbols: list[str],
    table_json: str | None,
    table_file: str | None,
    fallback: dict[str, float],
    label: str,
) -> dict[str, float]:
    """Per-element values: ``fallback``, updated by the JSON file, then the JSON string."""
    table = dict(fallback)
    if table_file:
        with open(table_file, encoding="ascii") as handle:
            table.update({str(k): float(v) for k, v in json.load(handle).items()})
    if table_json:
        table.update({str(k): float(v) for k, v in json.loads(table_json).items()})
    missing = sorted(set(symbols) - set(table))
    if missing:
        raise ValueError(f"Missing {label} for {missing}.")
    return table


def reshape_mesh_eigenvectors(mesh_eigenvectors: np.ndarray, num_atoms: int) -> np.ndarray:
    """Reshape phonopy mesh eigenvectors to ``(n_q, n_branches, n_atoms, 3)``."""
    n_q = mesh_eigenvectors.shape[0]
    n_branches = num_atoms * 3
    if mesh_eigenvectors.shape != (n_q, n_branches, n_branches):
        raise ValueError(f"Unexpected mesh eigenvector shape: {mesh_eigenvectors.shape}")
    return mesh_eigenvectors.transpose(0, 2, 1).reshape(n_q, n_branches, num_atoms, 3)


def derive_required_multiphonon_order(
    max_q_ang_inv: float,
    thermal_mats: np.ndarray,
    requested_order: int,
    margin_sigma: float = MULTIPHONON_MARGIN_SIGMAS,
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


@dataclass(frozen=True)
class MultiphononReach:
    """How far in energy transfer the truncated multiphonon sum reaches,
    against how far the table needs it to (all energies in meV)."""
    reach_mev: float        # largest transfer the truncated sum can produce
    needed_mev: float       # recoil ridge + margin widths at Q_max, capped at the grid top
    ridge_mev: float        # recoil energy of the atom that sets needed_mev
    width_mev: float        # that atom's thermal width (an upper bound)
    atom_index: int | None  # that atom; None when no reach is needed
    capped: bool            # needed_mev was capped at the top of the energy grid

    @property
    def short(self) -> bool:
        return self.reach_mev < self.needed_mev


def multiphonon_energy_reach(
    order: int,
    max_mode_energy_mev: float,
    max_q_ang_inv: float,
    masses_amu,
    temperature_k: float,
    grid_top_mev: float,
    margin_sigma: float = MULTIPHONON_MARGIN_SIGMAS,
) -> MultiphononReach:
    """Energy reach of the truncated multiphonon sum against what the table needs.

    ``order`` phonons carry at most ``order * max_mode_energy_mev`` (one phonon
    below order 2): the reach. At wavevector Q the incoherent law of an atom
    of mass M is centred on its recoil energy E_R = hbar^2 Q^2 / 2M, with a
    thermal width sqrt(2 E_R kT_eff), and the ridge is highest at the largest
    Q in the table. The needed reach is the largest E_R + margin_sigma * width
    over the atoms, capped at ``grid_top_mev``, the top of the requested
    energy grid. Past it the law is a Gaussian tail too small for any cross
    section to see, so comparing the reach with the grid top instead raises
    false alarms for every atom heavier than a few mass units.

    kT_eff uses the bound of ``effective_temperature_bound_ratio``, so the
    width is never underestimated. With that bound, every order that meets
    ``derive_required_multiphonon_order`` at the same Q and margin also meets
    this reach, for every atom: E_R = 2W * e_bar and width^2 = 2W * 2 e_bar
    kT_eff, where e_bar = 1 / <coth(E/2kT)/E> <= E_max tanh(E_max/2kT)
    because coth(E/2kT)/E decreases with E; so E_R <= 2W E_max and
    width <= sqrt(2W) E_max. The check is therefore a guard on the order
    rule: it fires only when the order is below that rule's requirement.
    """
    e_max = float(max_mode_energy_mev)
    reach = e_max * (int(order) if int(order) >= 2 else 1)
    q = float(max_q_ang_inv)
    temperature = float(temperature_k)
    top = float(grid_top_mev)
    masses = np.asarray(masses_amu, dtype=float).ravel()
    if e_max <= 0.0 or q <= 0.0 or top <= 0.0 or masses.size == 0:
        return MultiphononReach(reach, 0.0, 0.0, 0.0, None, False)
    kt_eff = (effective_temperature_bound_ratio(e_max * 1.0e-3, temperature)
              * KB_MEV_PER_K * temperature)
    ridge = HBAR2_OVER_2MN_MEV_A2 * q * q * NEUTRON_MASS_AMU / masses
    width = np.sqrt(2.0 * ridge * kt_eff)
    need = ridge + float(margin_sigma) * width
    atom = int(np.argmax(need))
    needed = float(need[atom])
    return MultiphononReach(
        reach_mev=reach, needed_mev=min(needed, top),
        ridge_mev=float(ridge[atom]), width_mev=float(width[atom]),
        atom_index=atom, capped=needed > top)
