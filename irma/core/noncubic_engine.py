#!/usr/bin/env python3
"""Noncubic S(Q,E) and downscatter-side asymmetric S(alpha,|beta|) for inelastic_mode 1/2.

IRMA's noncubic inelastic engine, called in-process through
run_noncubic_sab_inprocess; parse_args/main are a diagnostic CLI
(``python -m irma.core.noncubic_engine``).

One-phonon terms (exact harmonic):

- coherent n=1, the (UV) term of Squires Sec. 3.7: atom amplitudes are summed
  before squaring; the diagonal (self) and interference pieces are kept too;
- incoherent n=1, the (UV0) term of Squires Sec. 3.9;
- the incoherent-approximation n=1 term: the same self kernel scaled with
  sigma_total instead of sigma_inc (mode 1, and the n=1 partner of the
  multiphonon tail).

Multiphonon tail (orders n >= 2, incoherent approximation, Squires Sec. 3.10):
a per-atom signed-energy self kernel normalized so its unit-Q integral equals
u_hat . U_d . u_hat, the recursion T_n = (T_1 * T_{n-1}) / n, then the
Debye-Waller and cross-section factors, with the powder average taken after
the fixed-direction convolution. Exact coherent multiphonon scattering is not
implemented.

Units and conventions: the sqe_* arrays are (sigma / 4 pi) S(Q,E) in
barn / sr / meV, without the k_f/k_i factor (irma.spectra applies it). The
S(alpha,beta) arrays are

    S_asym_downscatter(alpha, beta) = (4 pi kT / sigma_b) S(Q, E_tr),

with beta = E_tr / kT and alpha = hbar^2 Q^2 / (2 M kT). One sigma_b is used
for the coherent total, diagonal and interference pieces.
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np

# Physical unit constants. Values are identical to phonopy.units (which phonopy
# is deprecating); hardcoded here so this module imports WITHOUT phonopy. phonopy
# is only needed at compute time (mesh / thermal-displacement / q-point objects),
# imported lazily in the functions that use it, so inelastic_mode=1/2 stays the
# only path that requires it.
from irma.core.constants import (
    BK as _BK_EV_PER_K,
    AMU_KG as AMU,
    ECHARGE_C as EV,
    HBAR_EV_S as Hbar,
    THZ_TO_EV as THzToEv,
    HBAR2_OVER_2MN_MEV_A2,
    AMASSN as NEUTRON_MASS_AMU,
)
from irma.core.noncubic_helpers import (  # re-exported for back-compat
    ANG2_TO_BARN,
    KB_MEV_PER_K,
    _FALLBACK_C_SCATTERING_LENGTHS_ANGSTROM,
    _FALLBACK_C_INCOHERENT_CROSS_SECTIONS_BARN,
    load_grid_from_text,
    load_grid_from_oclimax_csv,
    infer_mass_ratio,
    infer_sigma_barn,
    normalize_site_groups,
    convert_sqe_to_asym_downscatter_sab,
    parse_scattering_lengths,
    parse_incoherent_cross_sections,
    reshape_mesh_eigenvectors,
    derive_required_multiphonon_order,
    multiphonon_energy_reach,
    MULTIPHONON_MARGIN_SIGMAS,
)
from irma.core.noncubic_numerics import (  # re-exported for back-compat
    UNIFORM_GRID_RTOL,
    MAX_MULTIPHONON_WORK_BINS,
    HARD_WORK_BIN_LIMIT,
    fibonacci_sphere,
    precompute_histogram_lookup,
    bincount_add,
    build_signed_energy_grid,
    centers_to_edges,
    infer_uniform_spacing_or_none,
    build_uniform_positive_work_grid,
    build_gain_output_grid,
    build_rebin_matrix,
    rebin_energy_axis,
)

Angstrom = 1e-10
THz = 1000000000000.0


# Single-element fallback tables for the STANDALONE CLI driver only. The
# production engine path (mode 1/2 ENDF generation) always passes
# site_scattering_lengths_angstrom / incoherent overrides explicitly and a
# scattering-lengths JSON, so these are never consulted on the validated path.
# Any non-Carbon standalone run must supply --scattering-lengths-json/-file (or
# the incoherent equivalents); otherwise resolution fails loudly with a clear
# "Missing ..." ValueError. 6.646e-5 A is the natural-C coherent length;
# 0.001 barn is a placeholder incoherent value.

# --- spawn-pool worker layer (lives in noncubic_workers) ---------------------
# The ProcessPoolExecutor kernels (accumulate_*_block), the WORKER_STATE
# global + set_worker_state, the shared-memory staging/attach helpers, the
# thread-limiting pool initializer, and the projection / q-sampling /
# precompute helpers live in noncubic_workers so this module
# stays navigable. Re-imported here so compute_from_args and existing
# ``from irma.core.noncubic_engine import ...`` callers keep working
# unchanged. WORKER_STATE itself is intentionally NOT re-exported:
# set_worker_state (called from compute_from_args) rebinds it in
# noncubic_workers, run_blocks stages THAT dict into shared memory for the
# spawned workers, and the serial path's kernels read it there — a copy bound
# here would only go stale.
from irma.core.noncubic_workers import (  # noqa: F401  (re-exported for callers)
    NATIVE_THREAD_ENV_VARS,
    BARN_PER_M2,
    limit_native_threads_to_one,
    _pool_worker_init,
    _dispatch_block,
    share_worker_state,
    release_shared_state,
    contract_real_symmetric_projection_components,
    build_star_averaged_projection_components,
    precompute_directional_multiphonon_orders,
    multiphonon_seed_area_deficit,
    set_worker_state,
    principal_weighted_coherent_partition,
    _batched_qpoints_eigh,
    accumulate_coherent_block,
    accumulate_incoherent_shell_block,
    accumulate_incoherent_multiphonon_direction_block,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse a reduced CLI focused on exact n=1 plus the Section 3.10 multiphonon model."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phonopy-yaml", default="phonopy.yaml")
    parser.add_argument(
        "--force-constants",
        default=None,
        help="Force-constants file (text FORCE_CONSTANTS or "
             "force_constants.hdf5). Default: discovered next to "
             "phonopy.yaml (force_constants.hdf5, FORCE_CONSTANTS, "
             "FORCE_SETS, or embedded in the yaml).",
    )
    parser.add_argument(
        "--force-sets",
        default=None,
        help="FORCE_SETS file to build force constants from (alternative "
             "to --force-constants).",
    )
    parser.add_argument(
        "--born",
        default=None,
        help="Path to a phonopy BORN file; enables the non-analytical-term "
             "correction (LO-TO splitting) in every mesh-based mode sum.",
    )
    parser.add_argument("--material-name", default="material")
    parser.add_argument("--temperature", type=float, default=296.0)
    parser.add_argument("--mesh", nargs=3, type=int, default=[40, 40, 40])
    parser.add_argument("--q-min", type=float, default=0.25)
    parser.add_argument("--q-max", type=float, default=30.0)
    parser.add_argument("--dq", type=float, default=0.25)
    parser.add_argument("--e-min", type=float, default=0.0)
    parser.add_argument("--e-max", type=float, default=210.0)
    parser.add_argument("--de", type=float, default=1.0)
    parser.add_argument(
        "--grid-from-oclimax-csv",
        default=None,
        help="Use the exact Q and E grid centers from an OCLIMAX CSV export.",
    )
    parser.add_argument(
        "--q-grid-file",
        default=None,
        help="Optional text file with Q-bin centers in 1/Angstrom. Overrides --q-min/--q-max/--dq.",
    )
    parser.add_argument(
        "--e-grid-file",
        default=None,
        help="Optional text file with E-bin centers in meV. Overrides --e-min/--e-max/--de.",
    )
    parser.add_argument(
        "--num-directions",
        type=int,
        default=10000,
        help="Number of Fibonacci-sphere directions used for the coherent one-phonon powder average.",
    )
    parser.add_argument(
        "--multiphonon-num-directions",
        type=int,
        default=1000,
        help="Number of golden-spiral directions for the multiphonon powder average "
             "(the anisotropic multiphonon depends on direction; this sizes its angular "
             "quadrature). Converged by ~50-100 for graphite-like crystals; default 1000 "
             "is well inside the converged regime. Cost is linear in the count.",
    )
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--q-chunk-size", type=int, default=20000)
    parser.add_argument("--multiphonon-max-order", type=int, default=100)
    parser.add_argument(
        "--auto-multiphonon-order",
        action="store_true",
        help=(
            "Opt in to automatic sizing of the multiphonon order: the engine raises "
            "--multiphonon-max-order (never below it) to the value required to converge the "
            "incoherent Poisson(2W) sum for the anisotropic Debye-Waller factor at the grid's "
            "highest Q. Off by default, in which case the requested order is honored verbatim "
            "and a warning is printed if it is below that requirement."
        ),
    )
    parser.add_argument(
        "--emit-gain-side",
        action="store_true",
        help=(
            "Additionally compute the DIRECT energy-gain side S(Q, E<0) with "
            "explicit Bose annihilation factors (no detailed-balance mirror) and "
            "surface it as sqe_*_gain_barn_per_meV. NS-bridge-only: the ENDF/SAB "
            "tape outputs are unaffected. Off by default."
        ),
    )
    parser.add_argument("--output-prefix", default="noncubic_sqe_with_multiphonon")
    parser.add_argument("--scattering-lengths-json", default=None)
    parser.add_argument("--scattering-lengths-file", default=None)
    parser.add_argument("--incoherent-cross-sections-json", default=None)
    parser.add_argument("--incoherent-cross-sections-file", default=None)
    parser.add_argument("--sab-mass-ratio", type=float, default=None)
    parser.add_argument(
        "--sab-sigma-barn",
        type=float,
        default=None,
        help=(
            "Optional bound total cross section used in the S(Q,E)->S(alpha,beta) "
            "conversion. Required for principal-scatterer exports of mixed materials."
        ),
    )
    parser.add_argument(
        "--coherent-partition-mode",
        choices=("auto", "exact-total", "principal-xs-weighted"),
        default="auto",
        help=(
            "How the coherent one-phonon term is mapped onto the exported MT4 law. "
            "'exact-total' keeps the full mixed-material coherent intensity; "
            "'principal-xs-weighted' keeps the exact principal self term and assigns "
            "cross-species interference to the principal group with coherent-strength "
            "weights; 'auto' uses exact-total for a single group and principal-xs-weighted "
            "when multiple site groups are provided."
        ),
    )
    parser.add_argument(
        "--represented-principal-site-count",
        type=int,
        default=None,
        help=(
            "Number of phonopy primitive-cell sites represented by the exported "
            "principal-scatterer MT4 law. Defaults to the full primitive-cell "
            "site count for standalone single-species use."
        ),
    )
    args = parser.parse_args(argv)
    if not args.temperature > 0.0:
        parser.error("--temperature must be > 0 K (got %r)" % (args.temperature,))
    if args.sab_mass_ratio is not None and not args.sab_mass_ratio > 0.0:
        parser.error(
            "--sab-mass-ratio must be > 0 (got %r)" % (args.sab_mass_ratio,)
        )
    return args




def compute_from_args(
    args: argparse.Namespace,
    *,
    q_grid_ang_inv: np.ndarray | None = None,
    e_grid_mev: np.ndarray | None = None,
    context: dict[str, object] | None = None,
) -> tuple[dict[str, object], dict[str, object], dict[str, object] | None]:
    """Run the full noncubic S(Q,E) computation for one parsed argument set.

    Runs the stages in order — mesh/grid setup, site grouping, coherent
    one-phonon, incoherent one-phonon + multiphonon, S(alpha,beta)
    conversion, metadata — with one spawned worker pool shared by all of
    them. ``q_grid_ang_inv``/``e_grid_mev`` override the grids parsed from
    ``args``; ``context`` supplies a prebuilt (cached) compute context so
    repeated runs against the same phonon model skip the mesh eigensolve.
    Returns ``(output_arrays, metadata, elastic_state)``: the named S(Q,E)
    component arrays, the provenance/units metadata dict, and the
    per-species inputs for the generalized elastic channel (None when the
    mesh context provides no thermal-displacement data).
    """
    limit_native_threads_to_one()
    # One spawned pool serves every compute phase of the run; run_blocks
    # creates it on first use and the finally clause retires it.
    pool_holder: dict = {}
    try:
        # --- Mesh, Q/E grids, thermal displacements and the direction quadrature.
        start = time.time()
        multiphonon_num_directions = int(args.multiphonon_num_directions)

        if q_grid_ang_inv is None or e_grid_mev is None:
            if args.grid_from_oclimax_csv:
                q_grid_ang_inv, e_grid_mev = load_grid_from_oclimax_csv(Path(args.grid_from_oclimax_csv))
            else:
                if args.q_grid_file:
                    q_grid_ang_inv = load_grid_from_text(Path(args.q_grid_file))
                else:
                    q_grid_ang_inv = np.arange(args.q_min, args.q_max + 0.5 * args.dq, args.dq, dtype=float)
                if args.e_grid_file:
                    e_grid_mev = load_grid_from_text(Path(args.e_grid_file))
                else:
                    e_grid_mev = np.arange(args.e_min, args.e_max + 0.5 * args.de, args.de, dtype=float)
        else:
            q_grid_ang_inv = np.asarray(q_grid_ang_inv, dtype=float)
            e_grid_mev = np.asarray(e_grid_mev, dtype=float)

        if context is None:
            from irma.core.noncubic_inelastic_context import build_compute_context

            context = build_compute_context(args, q_grid_ang_inv, e_grid_mev)

        requested_inelastic_mode = int(getattr(args, "inelastic_mode", 0) or 0)
        if requested_inelastic_mode not in (0, 1, 2):
            raise ValueError("inelastic_mode must be 0, 1, or 2 in the noncubic inelastic driver.")
        need_coherent_n1 = requested_inelastic_mode in (0, 2)
        need_exact_incoherent_n1 = requested_inelastic_mode in (0, 2)
        need_incoherent_approx_n1 = requested_inelastic_mode in (0, 1)

        q_grid_ang_inv = context["q_grid_ang_inv"]
        e_grid_mev = context["e_grid_mev"]
        q_edges_ang_inv = context["q_edges_ang_inv"]
        e_edges_mev = context["e_edges_mev"]
        e_bin_widths_mev = context["e_bin_widths_mev"]
        q_min_used = context["q_min_used"]
        q_max_used = context["q_max_used"]
        dq_used = context["dq_used"]
        e_min_used = context["e_min_used"]
        e_max_used = context["e_max_used"]
        de_used = context["de_used"]
        directions = context["directions"]
        mesh = context["mesh"]
        primitive = context["primitive"]
        frequency_factor_to_thz = context["frequency_factor_to_thz"]
        direction_red_basis = context["direction_red_basis"]
        mev_to_joule = context["mev_to_joule"]
        unit_conversion = context["unit_conversion"]
        scattering_lengths = context["scattering_lengths"]
        sigma_inc = context["sigma_inc"]
        sigma_coh_by_atom = context["sigma_coh_by_atom"]
        coherent_atom_prefactors = context["coherent_atom_prefactors"]
        sigma_inc_by_atom = context["sigma_inc_by_atom"]
        sigma_total_by_atom = context["sigma_total_by_atom"]
        incoherent_prefactors = context["incoherent_prefactors"]
        incoherent_approx_prefactors = context["incoherent_approx_prefactors"]
        incoherent_one_phonon_mesh_qpoints = context["incoherent_one_phonon_mesh_qpoints"]
        incoherent_one_phonon_mesh_weights = context["incoherent_one_phonon_mesh_weights"]
        incoherent_one_phonon_mode_energies_mev = context["incoherent_one_phonon_mode_energies_mev"]
        incoherent_one_phonon_mode_frequencies_thz = context["incoherent_one_phonon_mode_frequencies_thz"]
        incoherent_one_phonon_mode_weights = context["incoherent_one_phonon_mode_weights"]
        incoherent_one_phonon_mode_eigvecs_valid = context["incoherent_one_phonon_mode_eigvecs_valid"]
        max_mode_energy_mev = context["max_mode_energy_mev"]
        multiphonon_mode_energies_mev = context["multiphonon_mode_energies_mev"]
        multiphonon_mode_frequencies_thz = context["multiphonon_mode_frequencies_thz"]
        multiphonon_mode_weights = context["multiphonon_mode_weights"]
        multiphonon_mode_eigvecs_valid = context["multiphonon_mode_eigvecs_valid"]
        multiphonon_q_weight_norm = context["multiphonon_q_weight_norm"]
        multiphonon_mode_projection_components = context["multiphonon_mode_projection_components"]
        multiphonon_star_counts = context["multiphonon_star_counts"]
        num_jobs = context["num_jobs"]
        coherent_blocks = context["coherent_blocks"]
        shell_blocks = context["shell_blocks"]
        multiphonon_dir_chunk_size = context["multiphonon_dir_chunk_size"]

        # Thermal-displacement tensors on the full mesh. The ENDF driver passes
        # the tensors it already computed for this temperature; the spectra and
        # NCrystal paths compute them here. The cutoff summary reports what a
        # user phonon-energy cutoff removed.
        thermal_mats = getattr(args, "precomputed_thermal_mats", None)
        # The diagnostic CLI (parse_args) has no cutoff option.
        cutoff_mev = float(getattr(args, "min_phonon_energy_mev", 0.0))
        if thermal_mats is None or cutoff_mev > 0.0:
            from irma.core.phonopy_io import PhonopyMeshData
            mesh_data = PhonopyMeshData(
                qpoints=np.asarray(mesh.qpoints, dtype=float),
                frequencies_ev=np.asarray(mesh.frequencies, dtype=float) * THzToEv,
                eigenvectors=reshape_mesh_eigenvectors(
                    np.asarray(mesh.eigenvectors), len(primitive.masses)),
                weights=np.asarray(mesh.weights, dtype=float),
                masses_amu=np.asarray(primitive.masses, dtype=float),
                atom_symbols=[str(s) for s in primitive.symbols],
                atom_positions=np.asarray(primitive.scaled_positions, dtype=float),
                min_phonon_energy_mev=cutoff_mev,
                phonopy_mesh_object=mesh,
            )
        if thermal_mats is None:
            from irma.core.phonopy_io import compute_thermal_displacement_matrices
            print("Precomputing anisotropic thermal displacement matrices...", flush=True)
            thermal_mats = compute_thermal_displacement_matrices(mesh_data, args.temperature)
        else:
            thermal_mats = np.asarray(thermal_mats, dtype=float)
        phonon_cutoff_summary = None
        if cutoff_mev > 0.0:
            from irma.core.phonopy_io import (
                format_phonon_cutoff_summary, phonon_cutoff_summary as _summary)
            phonon_cutoff_summary = _summary(mesh_data, args.temperature)
            for _line in format_phonon_cutoff_summary(phonon_cutoff_summary):
                print(_line, flush=True)
        incoherent_one_phonon_mode_occupancies = 1.0 / np.expm1(np.clip(
            incoherent_one_phonon_mode_frequencies_thz * THzToEv
            / (_BK_EV_PER_K * args.temperature), 0.0, 700.0))

        # --- Site groups, principal-scatterer bookkeeping and the multiphonon order.
        # The incoherent multiphonon sum is Poisson(2W = Q^2 u.U.u); at the grid's
        # largest Q it converges once the order reaches about 2W_max + 6 sqrt(2W_max).
        # Card 3 nphon is honored unless auto-sizing (Card 6g) raises it.
        max_q_for_order = float(np.max(q_grid_ang_inv))
        order = args.multiphonon_max_order
        _, required_order, two_w_max, _ = derive_required_multiphonon_order(
            max_q_for_order, thermal_mats, order)
        if args.auto_multiphonon_order:
            new_order = max(order, min(required_order, 2000))
            if new_order > order:
                print(f"multiphonon: auto-sizing order {order} -> {new_order} "
                      f"(2W_max = {two_w_max:.0f} at Q_max = {max_q_for_order:.1f} 1/A)",
                      flush=True)
            order = args.multiphonon_max_order = new_order
        if order < required_order:
            print(f"WARNING: multiphonon order {order} is below the ~{required_order} needed "
                  f"at Q_max = {max_q_for_order:.1f} 1/A; the high-Q law is truncated. "
                  "Raise Card 3 nphon or set Card 6g auto_order = 1.", flush=True)
        # The sum must also reach the recoil ridge at the largest Q plus a few
        # thermal widths; any order that meets the Poisson rule does.
        energy_reach = multiphonon_energy_reach(
            order, max_mode_energy_mev, max_q_for_order,
            primitive.masses, args.temperature, float(np.max(e_grid_mev)))
        needed_multiphonon_beta_support = (
            energy_reach.needed_mev / (KB_MEV_PER_K * args.temperature))
        if energy_reach.short:
            print(f"WARNING: multiphonon order {order} reaches "
                  f"{energy_reach.reach_mev / 1e3:.2f} eV of energy transfer, short of the "
                  f"{energy_reach.needed_mev / 1e3:.2f} eV needed at Q_max = "
                  f"{max_q_for_order:.1f} 1/A (recoil ridge plus "
                  f"{MULTIPHONON_MARGIN_SIGMAS:g} thermal widths); the law is truncated "
                  "past it.", flush=True)
        represented_principal_site_count = args.represented_principal_site_count
        if represented_principal_site_count is None:
            represented_principal_site_count = len(primitive.symbols)
        represented_principal_site_count = int(represented_principal_site_count)
        if represented_principal_site_count < 1:
            raise ValueError("represented_principal_site_count must be a positive integer.")

        site_groups = normalize_site_groups(
            getattr(args, "site_groups", None),
            len(primitive.symbols),
        )
        principal_group_index = int(getattr(args, "principal_group_index", 0) or 0)
        if principal_group_index < 0 or principal_group_index >= len(site_groups):
            raise ValueError("principal_group_index is out of range for the provided site groups.")
        principal_site_indices = site_groups[principal_group_index]
        principal_site_mask = np.zeros(len(primitive.symbols), dtype=bool)
        principal_site_mask[principal_site_indices] = True
        group_coherent_weights = np.array(
            [float(np.sum(sigma_coh_by_atom[group])) for group in site_groups],
            dtype=float,
        )
        coherent_partition_mode = str(args.coherent_partition_mode)
        if coherent_partition_mode == "auto":
            coherent_partition_mode = "exact-total" if len(site_groups) == 1 else "principal-xs-weighted"
        if coherent_partition_mode not in ("exact-total", "principal-xs-weighted"):
            raise ValueError("coherent_partition_mode must be 'auto', 'exact-total', or 'principal-xs-weighted'.")
        export_incoherent_prefactors = incoherent_prefactors.copy()
        export_incoherent_prefactors[~principal_site_mask] = 0.0
        export_incoherent_approx_prefactors = incoherent_approx_prefactors.copy()
        export_incoherent_approx_prefactors[~principal_site_mask] = 0.0
        export_multiphonon_sigma_total_scale = np.zeros_like(sigma_total_by_atom, dtype=float)
        for group_indices in site_groups:
            group_scale = 1.0 / float(len(group_indices))
            export_multiphonon_sigma_total_scale[group_indices] = (
                sigma_total_by_atom[group_indices] / (4.0 * np.pi) * group_scale
            )
        export_multiphonon_sigma_total_scale[~principal_site_mask] = 0.0
        print(
            "Export grouping: "
            f"{len(site_groups)} group(s), principal group={principal_group_index}, "
            f"principal sites={len(principal_site_indices)}, "
            f"coherent partition={coherent_partition_mode}",
            flush=True,
        )

        # --- Coherent one-phonon S(Q,E) over the direction quadrature.
        one_phonon_energy_jacobian_mev_per_thz = THzToEv * 1000.0
        one_phonon_principal_site_normalization = 1.0 / float(represented_principal_site_count)
        one_phonon_creation_scale = (
            one_phonon_energy_jacobian_mev_per_thz * one_phonon_principal_site_normalization
        )
        incoherent_one_phonon_q_weight_norm = float(np.sum(incoherent_one_phonon_mesh_weights))
        incoherent_one_phonon_hist_lookup = precompute_histogram_lookup(
            incoherent_one_phonon_mode_energies_mev,
            e_edges_mev,
            e_bin_widths_mev,
        )
        incoherent_one_phonon_mode_creation_prefactors = (
            (incoherent_one_phonon_mode_occupancies + 1.0)
            * unit_conversion
            * mev_to_joule
            * BARN_PER_M2
            * one_phonon_creation_scale
            * incoherent_one_phonon_mode_weights
            / incoherent_one_phonon_q_weight_norm
            / incoherent_one_phonon_mode_frequencies_thz
        )
        # Energy-gain (annihilation) one-phonon prefactor: the creation prefactor
        # with (occ+1) -> occ; everything else identical. Built only when the gain
        # side is requested; consumed by the incoherent worker at -energy.
        emit_gain_side = bool(args.emit_gain_side)
        incoherent_one_phonon_mode_absorption_prefactors = (
            (incoherent_one_phonon_mode_occupancies
             * unit_conversion
             * mev_to_joule
             * BARN_PER_M2
             * one_phonon_creation_scale
             * incoherent_one_phonon_mode_weights
             / incoherent_one_phonon_q_weight_norm
             / incoherent_one_phonon_mode_frequencies_thz)
            if emit_gain_side else None
        )
        # Energy-gain grids. The worker grid is the full mirror -e_grid[::-1] (same
        # length as the loss grid, so worker stacks are rectangular); the output gain
        # grid is strictly negative.
        if emit_gain_side:
            e_gain_grid_mev = -e_grid_mev[::-1]
            e_gain_edges_mev = centers_to_edges(e_gain_grid_mev)
            e_gain_bin_widths_mev = np.diff(e_gain_edges_mev)
            gain_num_positive = int(np.count_nonzero(e_grid_mev > 0.0))
            e_gain_out_mev, e_gain_out_edges_mev = build_gain_output_grid(e_grid_mev)
            incoherent_one_phonon_gain_hist_lookup = precompute_histogram_lookup(
                -incoherent_one_phonon_mode_energies_mev,
                e_gain_edges_mev,
                e_gain_bin_widths_mev,
            )
        else:
            e_gain_grid_mev = None
            e_gain_edges_mev = None
            e_gain_bin_widths_mev = None
            gain_num_positive = 0
            e_gain_out_mev = None
            e_gain_out_edges_mev = None
            incoherent_one_phonon_gain_hist_lookup = None
        multiphonon_mode_energies_mev = incoherent_one_phonon_mode_energies_mev
        multiphonon_mode_frequencies_thz = incoherent_one_phonon_mode_frequencies_thz
        multiphonon_mode_weights = incoherent_one_phonon_mode_weights
        multiphonon_mode_occupancies = incoherent_one_phonon_mode_occupancies
        multiphonon_mode_eigvecs_valid = incoherent_one_phonon_mode_eigvecs_valid


        # The pool (pool_holder, created lazily by run_blocks) carries no
        # per-stage state of its own: each stage stages its state in shared
        # memory and the tasks reference it by generation, so reuse cannot
        # leak one stage's context into the next.
        def run_blocks(worker, initial, block_list, label: str):
            """Map ``worker`` over ``block_list`` and accumulate into ``initial``.

            Serial when num_jobs == 1; otherwise dispatches through the shared
            spawned pool (created lazily on first use, reused by every phase)
            with the current WORKER_STATE staged in shared memory. Blocks are
            accumulated in list order, so the floating-point sum — and any tape
            derived from it — is identical for every worker count.
            """
            result = initial
            phase_start = time.time()
            print(
                f"{label}: starting {len(block_list)} block(s) with {num_jobs} worker(s)...",
                flush=True,
            )
            if num_jobs == 1:
                for block_index, block in enumerate(block_list, start=1):
                    try:
                        result += worker(block)
                    except Exception as exc:
                        raise RuntimeError(
                            f"{label}: worker failed on block "
                            f"{block_index}/{len(block_list)}") from exc
                    print(
                        f"{label}: block {block_index}/{len(block_list)} done "
                        f"after {time.time() - phase_start:.1f} s",
                        flush=True,
                    )
            else:
                # The pool uses the SPAWN start method — the one start method
                # that exists on every platform (fork does not exist on
                # Windows, and forking from a threaded process is hazardous on
                # macOS). The large read-only state (mesh eigenvectors, grids,
                # lookups) does NOT travel by pickle: share_worker_state stages
                # every ndarray in multiprocessing.shared_memory once per
                # phase, plus the pickled remainder (scalars, the phonopy
                # dynamical matrix) in one more block, and every task carries
                # only a tiny generation reference — _dispatch_block attaches
                # zero-copy read-only views on the first task of a new
                # generation and reuses them for the rest of the phase, so the
                # state stays one-copy in RAM regardless of ncpu.
                #
                # ProcessPoolExecutor rather than multiprocessing.Pool: a worker
                # killed by the OS (e.g. the out-of-memory killer) raises
                # BrokenProcessPool here, where Pool.imap would wait on the dead
                # worker's result forever. Ordered map keeps the accumulation in
                # fixed block order, so the floating-point sum (and therefore
                # the tape) stays bitwise-reproducible from run to run.
                #
                # Import BrokenProcessPool by name: referencing it through
                # ``cf.process`` would rely on accessing ``cf.ProcessPoolExecutor``
                # first to trigger the submodule import as a side effect, which is
                # fragile if this block is ever reordered.
                import concurrent.futures as cf
                from concurrent.futures.process import BrokenProcessPool
                from functools import partial as _partial
                from irma.core import noncubic_workers as _ncw
                pool = pool_holder.get("pool")
                if pool is None:
                    ctx = mp.get_context("spawn")
                    pool = cf.ProcessPoolExecutor(
                        max_workers=num_jobs,
                        mp_context=ctx,
                        initializer=_pool_worker_init)
                    pool_holder["pool"] = pool
                state_ref, shm_handles = share_worker_state(_ncw.WORKER_STATE)
                try:
                    results_iter = pool.map(
                        _partial(_dispatch_block, state_ref, worker), block_list)
                    for block_index in range(1, len(block_list) + 1):
                        try:
                            block_partial = next(results_iter)
                        except BrokenProcessPool as exc:
                            raise RuntimeError(
                                f"{label}: a worker process died on block "
                                f"{block_index}/{len(block_list)}. Two usual "
                                f"causes: (1) the out-of-memory killer — "
                                f"reduce Card 6f ncpu or the mesh/direction "
                                f"counts; (2) a driver script that calls IRMA "
                                f"at module level — the spawn start method "
                                f"re-imports the main module in every worker, "
                                f"so the entry point must be wrapped in "
                                f"'if __name__ == \"__main__\":'"
                            ) from exc
                        except Exception as exc:
                            raise RuntimeError(
                                f"{label}: worker failed on block "
                                f"{block_index}/{len(block_list)}") from exc
                        result += block_partial
                        print(
                            f"{label}: block {block_index}/{len(block_list)} done "
                            f"after {time.time() - phase_start:.1f} s",
                            flush=True,
                        )
                finally:
                    release_shared_state(shm_handles)
            print(f"{label}: completed in {time.time() - phase_start:.1f} s", flush=True)
            return result

        if need_coherent_n1:
            print("Accumulating coherent one-phonon contribution...", flush=True)
            # The worker returns a 6-stack (loss total/diag/interf + their gain
            # siblings) when the gain side is on; the accumulator must match.
            n_coh_stack = 6 if emit_gain_side else 3
            coherent_initial = np.zeros(
                (n_coh_stack, len(q_grid_ang_inv), len(e_grid_mev)), dtype=float)
            coherent_state = {
                "num_q": len(q_grid_ang_inv),
                "num_e": len(e_grid_mev),
                "q_grid_ang_inv": q_grid_ang_inv,
                "directions": directions,
                "direction_red_basis": direction_red_basis,
                "dynamical_matrix": mesh.dynamical_matrix,
                "frequency_factor_to_thz": frequency_factor_to_thz,
                "min_phonon_energy_mev": cutoff_mev,
                "thermal_mats": thermal_mats,
                "positions_t": primitive.scaled_positions.T,
                "coherent_atom_prefactors": coherent_atom_prefactors,
                "unit_conversion": unit_conversion,
                "temperature": args.temperature,
                "e_edges_mev": e_edges_mev,
                "e_bin_widths_mev": e_bin_widths_mev,
                "mev_to_joule": mev_to_joule,
                "one_phonon_creation_scale": one_phonon_creation_scale,
                "coherent_partition_mode": coherent_partition_mode,
                "coherent_group_site_indices": site_groups,
                "principal_group_index": principal_group_index,
                "group_coherent_weights": group_coherent_weights,
            }
            if emit_gain_side:
                coherent_state.update({
                    "emit_gain_side": True,
                    "e_gain_edges_mev": e_gain_edges_mev,
                    "e_gain_bin_widths_mev": e_gain_bin_widths_mev,
                })
            set_worker_state(coherent_state)
            coherent_components = run_blocks(
                accumulate_coherent_block,
                coherent_initial,
                coherent_blocks,
                "coherent n=1",
            )
            sqe_coherent = coherent_components[0]
            sqe_coherent_diagonal = coherent_components[1]
            sqe_coherent_interference = coherent_components[2]
            # Gain siblings (full mirror grid; sliced to the strictly-negative
            # output grid in the SAB-conversion stage).
            sqe_coherent_gain = coherent_components[3] if emit_gain_side else None
        else:
            coherent_initial = None      # only the coherent branch allocates one
            sqe_coherent = np.zeros((len(q_grid_ang_inv), len(e_grid_mev)), dtype=float)
            sqe_coherent_diagonal = np.zeros_like(sqe_coherent)
            sqe_coherent_interference = np.zeros_like(sqe_coherent)
            sqe_coherent_gain = (np.zeros((len(q_grid_ang_inv), len(e_gain_grid_mev)),
                                          dtype=float) if emit_gain_side else None)

        # --- Incoherent one-phonon and incoherent-approximation multiphonon terms.
        # When the gain side is on the worker returns a (2, q, e) loss/gain stack,
        # so the accumulator carries the extra leading slice.
        if emit_gain_side:
            incoherent_initial = np.zeros((2, len(q_grid_ang_inv), len(e_grid_mev)),
                                          dtype=float)
        else:
            incoherent_initial = np.zeros((len(q_grid_ang_inv), len(e_grid_mev)),
                                          dtype=float)

        def _incoherent_worker_state(prefactors):
            """Worker-state dict for an incoherent one-phonon pass.

            The exact and incoherent-approximation passes use an identical
            layout; only the prefactor table differs.
            """
            st = {
                "num_q": len(q_grid_ang_inv),
                "num_e": len(e_grid_mev),
                "directions": directions,
                "q_grid_ang_inv": q_grid_ang_inv,
                "thermal_mats": thermal_mats,
                "e_grid_mev": e_grid_mev,
                "e_edges_mev": e_edges_mev,
                "e_bin_widths_mev": e_bin_widths_mev,
                "incoherent_prefactors": prefactors,
                "mesh_mode_energies_mev": incoherent_one_phonon_mode_energies_mev,
                "mesh_mode_bose_prefactors": incoherent_one_phonon_mode_creation_prefactors,
                "mesh_mode_eigvecs": incoherent_one_phonon_mode_eigvecs_valid,
                "hist_valid_indices": incoherent_one_phonon_hist_lookup[0],
                "hist_bin_indices": incoherent_one_phonon_hist_lookup[1],
                "hist_inv_bin_widths": incoherent_one_phonon_hist_lookup[2],
            }
            if emit_gain_side:
                st.update({
                    "emit_gain_side": True,
                    "mesh_mode_absorption_prefactors": incoherent_one_phonon_mode_absorption_prefactors,
                    "e_gain_grid_mev": e_gain_grid_mev,
                    "e_gain_edges_mev": e_gain_edges_mev,
                    "e_gain_bin_widths_mev": e_gain_bin_widths_mev,
                    "hist_gain_valid_indices": incoherent_one_phonon_gain_hist_lookup[0],
                    "hist_gain_bin_indices": incoherent_one_phonon_gain_hist_lookup[1],
                    "hist_gain_inv_bin_widths": incoherent_one_phonon_gain_hist_lookup[2],
                })
            return st

        if need_exact_incoherent_n1:
            print("Accumulating incoherent one-phonon contribution...", flush=True)
            incoherent_n1_worker = accumulate_incoherent_shell_block
            worker_state = _incoherent_worker_state(export_incoherent_prefactors)
            set_worker_state(worker_state)
            _inc_res = run_blocks(
                incoherent_n1_worker,
                incoherent_initial,
                shell_blocks,
                "incoherent n=1",
            )
            if emit_gain_side:
                sqe_incoherent, sqe_incoherent_gain = _inc_res[0], _inc_res[1]
            else:
                sqe_incoherent, sqe_incoherent_gain = _inc_res, None
        else:
            sqe_incoherent = np.zeros((len(q_grid_ang_inv), len(e_grid_mev)), dtype=float)
            sqe_incoherent_gain = (np.zeros((len(q_grid_ang_inv), len(e_gain_grid_mev)),
                                            dtype=float) if emit_gain_side else None)
        if need_incoherent_approx_n1:
            print("Accumulating incoherent-approximation one-phonon contribution...", flush=True)
            incoherent_approx_n1_worker = accumulate_incoherent_shell_block
            worker_state = _incoherent_worker_state(export_incoherent_approx_prefactors)
            set_worker_state(worker_state)
            _inc_approx_res = run_blocks(
                incoherent_approx_n1_worker,
                incoherent_initial * 0.0,
                shell_blocks,
                "incoherent-approx n=1",
            )
            if emit_gain_side:
                sqe_incoherent_approx_n1_term = _inc_approx_res[0]
                sqe_incoherent_approx_n1_term_gain = _inc_approx_res[1]
            else:
                sqe_incoherent_approx_n1_term = _inc_approx_res
                sqe_incoherent_approx_n1_term_gain = None
        else:
            sqe_incoherent_approx_n1_term = None
            sqe_incoherent_approx_n1_term_gain = None
        sqe_one_phonon_total = sqe_coherent + sqe_incoherent
        # Gain one-phonon total (on the full mirror grid; sliced to the strictly-
        # negative output grid in the SAB-conversion stage).
        sqe_one_phonon_total_gain = (
            sqe_coherent_gain + sqe_incoherent_gain if emit_gain_side else None)

        sqe_multiphonon_incoherent_approx = None
        sqe_one_phonon_total_plus_incoherent_approx_multiphonon = None
        sqe_incoherent_approx_n1_term_plus_incoherent_approx_multiphonon = None
        sqe_multiphonon_gain_work = None
        sqe_multiphonon_gain = None

        if args.multiphonon_max_order >= 2:
            print(
                f"Accumulating multiphonon background through order {args.multiphonon_max_order}...",
                flush=True,
            )
            if np.size(multiphonon_mode_energies_mev) == 0:
                raise ValueError(
                    "multiphonon: the model has no positive phonon modes to build "
                    "the seed from, so no work grid can be sized")
            e_work_grid_mev, de_work_mev = build_uniform_positive_work_grid(
                e_grid_mev,
                phonon_max_energy_mev=float(np.max(multiphonon_mode_energies_mev)),
            )
            e_work_edges_mev = centers_to_edges(e_work_grid_mev, lower_bound=0.0)
            e_signed_grid_mev, positive_slice = build_signed_energy_grid(e_work_grid_mev)
            e_signed_edges_mev = centers_to_edges(e_signed_grid_mev)
            e_signed_bin_widths_mev = np.diff(e_signed_edges_mev)
            multiphonon_directions = fibonacci_sphere(multiphonon_num_directions)
            signed_emission_lookup = precompute_histogram_lookup(
                multiphonon_mode_energies_mev,
                e_signed_edges_mev,
                e_signed_bin_widths_mev,
            )
            signed_absorption_lookup = precompute_histogram_lookup(
                -multiphonon_mode_energies_mev,
                e_signed_edges_mev,
                e_signed_bin_widths_mev,
            )

            # Build the signed unit-Q^2 self kernel from the displacement sum rule.
            # The later sigma_total scaling is normalized per equivalent scatterer
            # in the primitive cell, so this kernel remains a pure per-atom self
            # object before cross-section factors are applied.
            multiphonon_unit_conversion = (
                Hbar * EV / Angstrom**2 / (2.0 * 1.0e12 * 2.0 * np.pi * AMU)
            )
            multiphonon_base_prefactors = np.array([1.0 / mass for mass in primitive.masses], dtype=float)
            multiphonon_creation_prefactors = (
                (multiphonon_mode_occupancies + 1.0)
                * multiphonon_unit_conversion
                * multiphonon_mode_weights
                / multiphonon_q_weight_norm
                / multiphonon_mode_frequencies_thz
            )
            multiphonon_absorption_prefactors = (
                multiphonon_mode_occupancies
                * multiphonon_unit_conversion
                * multiphonon_mode_weights
                / multiphonon_q_weight_norm
                / multiphonon_mode_frequencies_thz
            )
            seed_deficit = multiphonon_seed_area_deficit(
                multiphonon_base_prefactors, multiphonon_mode_eigvecs_valid,
                multiphonon_creation_prefactors, multiphonon_absorption_prefactors,
                signed_emission_lookup[0], signed_absorption_lookup[0], thermal_mats)
            if seed_deficit > 1.0e-5:
                print(
                    f"WARNING: the multiphonon one-phonon seed differs from the "
                    f"Debye-Waller displacement by up to {seed_deficit * 100:.2f}% "
                    f"(tolerance 1e-05): phonon modes fall outside the multiphonon "
                    f"work grid, so the multiphonon background loses area. Extend "
                    f"the energy grid (decks: the beta grid; spectra: "
                    f"grid.e_max_meV) above the highest phonon energy.",
                    flush=True,
                )

            if emit_gain_side:
                multiphonon_initial = np.zeros(
                    (2, len(q_grid_ang_inv), len(e_work_grid_mev)), dtype=float
                )
            else:
                multiphonon_initial = np.zeros(
                    (len(q_grid_ang_inv), len(e_work_grid_mev)), dtype=float
                )
            # Cap each worker's order tables (block_dirs x n_atoms x orders x
            # n_signed_bins doubles) at 2 GiB. When the cap binds, the summation
            # grouping, and so the last bits of the result, change.
            _n_atoms_mp = int(multiphonon_mode_eigvecs_valid.shape[1])
            _per_dir_bytes = (_n_atoms_mp
                              * (int(args.multiphonon_max_order) + 1)
                              * len(e_signed_grid_mev) * 8)
            _budget_bytes = 2 * 1024**3
            _cap = max(1, _budget_bytes // max(1, _per_dir_bytes))
            if _cap < multiphonon_dir_chunk_size:
                print(
                    f"multiphonon: capping direction block size "
                    f"{multiphonon_dir_chunk_size} -> {_cap} to keep "
                    f"per-block kernel tables under "
                    f"{_budget_bytes / 1024**3:.0f} GiB "
                    f"({_per_dir_bytes / 1024**2:.1f} MiB per direction)",
                    flush=True,
                )
                multiphonon_dir_chunk_size = _cap
            multiphonon_direction_blocks = [
                np.arange(
                    start_index,
                    min(len(multiphonon_directions), start_index + multiphonon_dir_chunk_size),
                    dtype=int,
                )
                for start_index in range(0, len(multiphonon_directions), multiphonon_dir_chunk_size)
            ]
            set_worker_state(
                {
                    "num_q": len(q_grid_ang_inv),
                    "num_e": len(e_work_grid_mev),
                    "directions": multiphonon_directions,
                    "base_prefactors": multiphonon_base_prefactors,
                    "mesh_mode_energies_mev": multiphonon_mode_energies_mev,
                    "mesh_mode_emission_prefactors": multiphonon_creation_prefactors,
                    "mesh_mode_absorption_prefactors": multiphonon_absorption_prefactors,
                    "mesh_mode_eigvecs": multiphonon_mode_eigvecs_valid,
                    "mesh_mode_projection_components": multiphonon_mode_projection_components,
                    "e_signed_grid_mev": e_signed_grid_mev,
                    "de_mev": de_work_mev,
                        "max_order": args.multiphonon_max_order,
                    "positive_slice": positive_slice,
                    "thermal_mats": thermal_mats,
                    "q_grid_ang_inv": q_grid_ang_inv,
                    "multiphonon_sigma_total_scale": export_multiphonon_sigma_total_scale,
                    "num_total_dirs": len(multiphonon_directions),
                    "sigma0_emission_lookup": signed_emission_lookup,
                    "sigma0_absorption_lookup": signed_absorption_lookup,
                    "emit_gain_side": emit_gain_side,
                }
            )
            multiphonon_results = run_blocks(
                accumulate_incoherent_multiphonon_direction_block,
                multiphonon_initial,
                multiphonon_direction_blocks,
                "multiphonon",
            )
            if emit_gain_side:
                sqe_multiphonon_work = multiphonon_results[0]
                sqe_multiphonon_gain_work = multiphonon_results[1]
            else:
                sqe_multiphonon_work = multiphonon_results
            sqe_multiphonon_incoherent_approx = rebin_energy_axis(
                sqe_multiphonon_work,
                e_work_edges_mev,
                e_edges_mev,
            )
            sqe_one_phonon_total_plus_incoherent_approx_multiphonon = (
                sqe_one_phonon_total + sqe_multiphonon_incoherent_approx
            )
            if sqe_incoherent_approx_n1_term is not None:
                sqe_incoherent_approx_n1_term_plus_incoherent_approx_multiphonon = (
                    sqe_incoherent_approx_n1_term + sqe_multiphonon_incoherent_approx
                )
            # Energy-gain multiphonon: rebin the negative-slice convolution (on the
            # mirror work grid -e_work[::-1]) to the strictly-negative OUTPUT grid.
            if emit_gain_side and sqe_multiphonon_gain_work is not None:
                gm_work_edges_mev = centers_to_edges(-e_work_grid_mev[::-1])
                sqe_multiphonon_gain = rebin_energy_axis(
                    sqe_multiphonon_gain_work, gm_work_edges_mev, e_gain_out_edges_mev
                )

        # Energy-gain output arrays: the one-phonon gain sliced to the strictly
        # negative output grid, plus the rebinned multiphonon gain.
        sqe_coherent_gain_out = None
        sqe_incoherent_gain_out = None
        sqe_one_phonon_total_gain_out = None
        sqe_incoherent_approx_n1_term_gain_out = None
        sqe_one_phonon_total_plus_incoherent_approx_multiphonon_gain = None
        sqe_incoherent_approx_n1_term_plus_incoherent_approx_multiphonon_gain = None
        if emit_gain_side:
            npos = gain_num_positive
            sqe_coherent_gain_out = sqe_coherent_gain[:, :npos]
            sqe_incoherent_gain_out = sqe_incoherent_gain[:, :npos]
            sqe_one_phonon_total_gain_out = sqe_one_phonon_total_gain[:, :npos]
            if sqe_incoherent_approx_n1_term_gain is not None:
                sqe_incoherent_approx_n1_term_gain_out = (
                    sqe_incoherent_approx_n1_term_gain[:, :npos])
            if sqe_multiphonon_gain is not None:
                sqe_one_phonon_total_plus_incoherent_approx_multiphonon_gain = (
                    sqe_one_phonon_total_gain_out + sqe_multiphonon_gain)
                if sqe_incoherent_approx_n1_term_gain_out is not None:
                    sqe_incoherent_approx_n1_term_plus_incoherent_approx_multiphonon_gain = (
                        sqe_incoherent_approx_n1_term_gain_out + sqe_multiphonon_gain)

        # --- Conversion of the S(Q,E) maps to asymmetric downscatter S(alpha,beta).
        # Components by name: output keys are sqe_<name>_barn_per_meV and
        # sab_asym_downscatter_<name>; gain-side keys sqe_<name>_gain_barn_per_meV.
        loss = {
            "coherent": sqe_coherent,
            "coherent_diagonal": sqe_coherent_diagonal,
            "coherent_interference": sqe_coherent_interference,
            "incoherent": sqe_incoherent,
            "one_phonon_total": sqe_one_phonon_total,
            "incoherent_approx_n1_term": sqe_incoherent_approx_n1_term,
            "multiphonon_incoherent_approx": sqe_multiphonon_incoherent_approx,
            "one_phonon_total_plus_incoherent_approx_multiphonon":
                sqe_one_phonon_total_plus_incoherent_approx_multiphonon,
            "incoherent_approx_n1_term_plus_incoherent_approx_multiphonon":
                sqe_incoherent_approx_n1_term_plus_incoherent_approx_multiphonon,
        }
        loss = {k: v for k, v in loss.items() if v is not None}
        gain = {}
        if emit_gain_side:
            gain = {
                "coherent": sqe_coherent_gain_out,
                "incoherent": sqe_incoherent_gain_out,
                "one_phonon_total": sqe_one_phonon_total_gain_out,
                "incoherent_approx_n1_term": sqe_incoherent_approx_n1_term_gain_out,
                "multiphonon_incoherent_approx": sqe_multiphonon_gain,
                "one_phonon_total_plus_incoherent_approx_multiphonon":
                    sqe_one_phonon_total_plus_incoherent_approx_multiphonon_gain,
                "incoherent_approx_n1_term_plus_incoherent_approx_multiphonon":
                    sqe_incoherent_approx_n1_term_plus_incoherent_approx_multiphonon_gain,
            }
            gain = {k: v for k, v in gain.items() if v is not None}
        for name, arr in list(loss.items()) + [(k + "_gain", v) for k, v in gain.items()]:
            if not np.all(np.isfinite(arr)):
                raise RuntimeError(
                    f"sqe_{name} contains non-finite values after the accumulation "
                    "phase (numerical overflow in a worker block).")

        sab_sigma_barn_override = args.sab_sigma_barn
        if sab_sigma_barn_override is None:
            if scattering_lengths is None or sigma_inc is None:
                raise ValueError(
                    "sab_sigma_barn must be provided when site-specific scattering data "
                    "are used for SAB export."
                )
            sigma_coh_barn, sigma_inc_barn, sigma_total_barn = infer_sigma_barn(
                list(primitive.symbols),
                scattering_lengths,
                sigma_inc,
            )
        else:
            # Per-atom convention: the override is the principal type's
            # sigma_coh + sigma_inc, and the sqe arrays already carry the
            # 1/represented_principal_site_count normalization.
            sigma_coh_barn = float(
                np.sum(sigma_coh_by_atom[principal_site_indices])
            ) / float(represented_principal_site_count)
            sigma_inc_barn = float(
                np.sum(sigma_inc_by_atom[principal_site_indices])
            ) / float(represented_principal_site_count)
            sigma_total_barn = float(sab_sigma_barn_override)
        mass_ratio = infer_mass_ratio(primitive.masses, args.sab_mass_ratio)
        sigma_for = {
            "coherent": sigma_coh_barn,
            "coherent_diagonal": sigma_coh_barn,
            "coherent_interference": sigma_coh_barn,
            "incoherent": sigma_inc_barn,
        }
        output_arrays: dict[str, object] = {
            "q_ang_inv": q_grid_ang_inv,
            "q_bin_edges_ang_inv": q_edges_ang_inv,
            "e_mev": e_grid_mev,
            "e_bin_edges_mev": e_edges_mev,
            "sampled_directions": directions,
        }
        for name, arr in loss.items():
            alpha, beta_downscatter_abs, sab = convert_sqe_to_asym_downscatter_sab(
                arr, q_grid_ang_inv, e_grid_mev, args.temperature,
                sigma_for.get(name, sigma_total_barn), mass_ratio)
            output_arrays[f"sqe_{name}_barn_per_meV"] = arr
            output_arrays[f"sab_asym_downscatter_{name}"] = sab
        output_arrays["alpha"] = alpha
        output_arrays["beta_downscatter_abs"] = beta_downscatter_abs
        # Energy-gain (E<0) outputs for the spectra bridge only; they are not
        # converted to S(alpha,beta) or written to tapes.
        for name, arr in gain.items():
            output_arrays[f"sqe_{name}_gain_barn_per_meV"] = arr
        if emit_gain_side:
            output_arrays["e_gain_mev"] = e_gain_out_mev

        # --- Run metadata and the elastic state.
        metadata = {
            "material_name": args.material_name,
            "phonopy_yaml": str(Path(args.phonopy_yaml).resolve()),
            "force_constants": (
                None if args.force_constants is None
                else str(Path(args.force_constants).resolve())
            ),
            "force_sets": (
                None if args.force_sets is None
                else str(Path(args.force_sets).resolve())
            ),
            "temperature_K": args.temperature,
            "mesh": list(args.mesh),
            "incoherent_one_phonon_weighted_mesh_qpoints": int(len(incoherent_one_phonon_mesh_qpoints)),
            "incoherent_one_phonon_weighted_mesh_weight_sum": int(np.sum(incoherent_one_phonon_mesh_weights)),
            "multiphonon_star_averaged_projection_components": bool(
                multiphonon_mode_projection_components is not None
            ),
            "q_min_A^-1": q_min_used,
            "q_max_A^-1": q_max_used,
            "dq_A^-1": dq_used,
            "e_min_meV": e_min_used,
            "e_max_meV": e_max_used,
            "de_meV": de_used,
            "grid_from_oclimax_csv": str(Path(args.grid_from_oclimax_csv).resolve()) if args.grid_from_oclimax_csv else None,
            "q_grid_file": str(Path(args.q_grid_file).resolve()) if args.q_grid_file else None,
            "e_grid_file": str(Path(args.e_grid_file).resolve()) if args.e_grid_file else None,
            "jobs": num_jobs,
            "num_directions": args.num_directions,
            "multiphonon_num_directions": multiphonon_num_directions,
            "multiphonon_max_order": args.multiphonon_max_order,
            "min_phonon_energy_meV": cutoff_mev,
            "phonon_cutoff": phonon_cutoff_summary,
            "multiphonon_model": "incoherent_approximation",
            "max_mode_energy_meV": max_mode_energy_mev,
            "needed_multiphonon_beta_support": needed_multiphonon_beta_support,
            "one_phonon_energy_jacobian_meV_per_THz": one_phonon_energy_jacobian_mev_per_thz,
            "represented_principal_site_count": represented_principal_site_count,
            "one_phonon_principal_site_normalization": one_phonon_principal_site_normalization,
            "one_phonon_creation_scale": one_phonon_creation_scale,
            "coherent_partition_mode": coherent_partition_mode,
            "principal_group_index": principal_group_index,
            "principal_group_site_count": int(len(principal_site_indices)),
            "site_group_sizes": [int(len(group)) for group in site_groups],
            "group_coherent_weights": [float(w) for w in group_coherent_weights],
            "coherent_interference_pair_weighting": "w_p/(w_p+w_o) per pair",
            "output_units": "barn / sr / meV",
            "sab_output": {
                "coherent_sigma_b_barn": sigma_coh_barn,
                "incoherent_sigma_b_barn": sigma_inc_barn,
                "total_sigma_b_barn": sigma_total_barn,
            },
            "elapsed_seconds": time.time() - start,
        }
        if multiphonon_star_counts is not None:
            metadata["multiphonon_star_count_min"] = int(np.min(multiphonon_star_counts))
            metadata["multiphonon_star_count_max"] = int(np.max(multiphonon_star_counts))

        print(f"Elapsed time: {time.time() - start:.2f} s")

        # The elastic state (anisotropic Debye-Waller tensors and primitive
        # geometry) lets the spectra and NCrystal paths build the elastic line
        # from the same phonon calculation.
        elastic_state = {
            "thermal_displacement_matrices_ang2": np.asarray(thermal_mats, dtype=float),
            "primitive_lattice_ang": np.asarray(primitive.cell, dtype=float),
            "primitive_scaled_positions": np.asarray(
                primitive.scaled_positions, dtype=float),
            "primitive_symbols": [str(s) for s in primitive.symbols],
            "primitive_masses_amu": np.asarray(primitive.masses, dtype=float),
            "temperature_k": float(args.temperature),
        }
        return output_arrays, metadata, elastic_state
    finally:
        if pool_holder.get("pool") is not None:
            pool_holder["pool"].shutdown()


def write_results(
    output_prefix: str | Path,
    output_arrays: dict[str, object],
    metadata: dict[str, object],
) -> tuple[Path, Path]:
    """Write the computed arrays and metadata next to ``output_prefix``.

    Produces ``<prefix>.npz`` (compressed arrays) and ``<prefix>.json``
    (metadata) and returns both paths.
    """
    output_prefix_path = Path(output_prefix)
    npz_path = output_prefix_path.with_suffix(".npz")
    json_path = output_prefix_path.with_suffix(".json")
    np.savez_compressed(npz_path, **output_arrays)
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    print(f"Wrote {npz_path}")
    print(f"Wrote {json_path}")
    return npz_path, json_path


def run_noncubic_sab_inprocess(
    *,
    inelastic_mode: int | None = None,
    phonopy_yaml: str | Path,
    force_constants: str | Path | None = None,
    force_sets: str | Path | None = None,
    born_path: str | Path | None = None,
    temperature_k: float,
    mesh: tuple[int, int, int] | list[int],
    q_grid_ang_inv: np.ndarray,
    e_grid_mev: np.ndarray,
    output_prefix: str | Path,
    sab_mass_ratio: float,
    num_directions: int = 10000,
    multiphonon_num_directions: int = 1000,
    jobs: int = 1,
    multiphonon_max_order: int = 100,
    auto_multiphonon_order: bool = False,
    min_phonon_energy_mev: float = 0.0,
    material_name: str = "material",
    represented_principal_site_count: int | None = None,
    principal_group_index: int = 0,
    site_groups: list[list[int]] | tuple[tuple[int, ...], ...] | None = None,
    coherent_partition_mode: str = "auto",
    sab_sigma_barn: float | None = None,
    site_scattering_lengths_angstrom: list[float] | tuple[float, ...] | np.ndarray | None = None,
    site_incoherent_cross_sections_barn: list[float] | tuple[float, ...] | np.ndarray | None = None,
    scattering_lengths_json: str | None = None,
    scattering_lengths_file: str | None = None,
    incoherent_cross_sections_json: str | None = None,
    incoherent_cross_sections_file: str | None = None,
    context: dict[str, object] | None = None,
    write_output_files: bool = True,
    precomputed_thermal_mats: np.ndarray | None = None,
    emit_gain_side: bool = False,
) -> dict[str, object]:
    """Run the standalone noncubic SAB builder inside the current process.

    ``represented_principal_site_count`` keeps the MT4 reduction explicit: the
    microscopic kernel is accumulated over all represented primitive-cell sites
    and then reduced to the per-principal-scatterer law IRMA writes in MF7.

    ``precomputed_thermal_mats``: the (n_atoms, 3, 3) U_ij array [Angstrom^2]
    for THIS temperature from the engine's
    ``compute_thermal_displacement_matrices`` (same full mesh, same mode
    floor); when given, the compute phase reuses it instead of re-running
    phonopy's ThermalDisplacementMatrices.
    """
    args = argparse.Namespace(
        phonopy_yaml=str(phonopy_yaml),
        force_constants=None if force_constants is None else str(force_constants),
        force_sets=None if force_sets is None else str(force_sets),
        born=None if born_path is None else str(born_path),
        material_name=str(material_name),
        temperature=float(temperature_k),
        mesh=[int(mesh[0]), int(mesh[1]), int(mesh[2])],
        q_min=0.25,
        q_max=30.0,
        dq=0.25,
        e_min=0.0,
        e_max=210.0,
        de=1.0,
        grid_from_oclimax_csv=None,
        q_grid_file=None,
        e_grid_file=None,
        num_directions=int(num_directions),
        multiphonon_num_directions=int(multiphonon_num_directions),
        jobs=int(jobs),
        q_chunk_size=20000,
        multiphonon_max_order=int(multiphonon_max_order),
        auto_multiphonon_order=bool(auto_multiphonon_order),
        min_phonon_energy_mev=float(min_phonon_energy_mev),
        inelastic_mode=0 if inelastic_mode is None else int(inelastic_mode),
        represented_principal_site_count=(
            None
            if represented_principal_site_count is None
            else int(represented_principal_site_count)
        ),
        principal_group_index=int(principal_group_index),
        site_groups=site_groups,
        coherent_partition_mode=str(coherent_partition_mode),
        sab_sigma_barn=(
            None
            if sab_sigma_barn is None
            else float(sab_sigma_barn)
        ),
        site_scattering_lengths_angstrom=site_scattering_lengths_angstrom,
        site_incoherent_cross_sections_barn=site_incoherent_cross_sections_barn,
        output_prefix=str(output_prefix),
        scattering_lengths_json=scattering_lengths_json,
        scattering_lengths_file=scattering_lengths_file,
        incoherent_cross_sections_json=incoherent_cross_sections_json,
        incoherent_cross_sections_file=incoherent_cross_sections_file,
        sab_mass_ratio=float(sab_mass_ratio),
        precomputed_thermal_mats=precomputed_thermal_mats,
        emit_gain_side=bool(emit_gain_side),
    )
    output_arrays, metadata, elastic_state = compute_from_args(
        args,
        q_grid_ang_inv=np.asarray(q_grid_ang_inv, dtype=float),
        e_grid_mev=np.asarray(e_grid_mev, dtype=float),
        context=context,
    )
    result = {
        "output_arrays": output_arrays,
        "metadata": metadata,
        "elastic_state": elastic_state,
    }
    if write_output_files:
        npz_path, json_path = write_results(output_prefix, output_arrays, metadata)
        result["npz_path"] = npz_path
        result["json_path"] = json_path
    return result
def main(argv: list[str] | None = None) -> None:
    """CLI entry point: parse arguments, compute, write the result files."""
    args = parse_args(argv)
    output_arrays, metadata, _elastic_state = compute_from_args(args)
    write_results(args.output_prefix, output_arrays, metadata)


if __name__ == "__main__":
    main()
