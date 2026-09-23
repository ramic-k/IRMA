"""Temperature-independent setup for the noncubic inelastic driver.

The context has two layers. The model layer (:func:`build_model_context`)
depends only on the phonon model and the mesh: the phonopy load, both mesh
eigensolves, the reduced-mesh mode arrays and the star-averaged projections.
The grid layer (the rest of :func:`build_compute_context`) holds the Q/E
edges, directions, scattering prefactors and block partitions, and is cheap
to rebuild. ``standalone_sab.get_or_build_context`` caches both.
"""

from __future__ import annotations

import numpy as np

from irma.core.noncubic_engine import (
    AMU,
    ANG2_TO_BARN,
    EV,
    THz,
    THzToEv,
    argparse,
    build_star_averaged_projection_components,
    centers_to_edges,
    fibonacci_sphere,
    infer_uniform_spacing_or_none,
    _FALLBACK_C_INCOHERENT_CROSS_SECTIONS_BARN,
    _FALLBACK_C_SCATTERING_LENGTHS_ANGSTROM,
    parse_element_table,
    reshape_mesh_eigenvectors,
)

# Fixed so that the multiphonon block partition, and with it the floating-point
# summation order, does not depend on the number of jobs.
_MULTIPHONON_DIR_CHUNK = 25


def _site_values(site_values, symbols, table_json, table_file, fallback, label):
    """Return ``(per-element table or None, per-site array)``.

    Explicit per-site values win; otherwise the per-element table is resolved
    and looked up for each site.
    """
    if site_values is not None:
        values = np.asarray(site_values, dtype=float)
        if len(values) != len(symbols):
            raise ValueError(
                f"Per-site {label} must match the number of primitive-cell atoms.")
        return None, values
    table = parse_element_table(symbols, table_json, table_file, fallback, label)
    return table, np.array([table[s] for s in symbols], dtype=float)


def build_model_context(
    args: argparse.Namespace,
    preloaded_full_mesh: object = None,
) -> dict[str, object]:
    """Build the phonopy MODEL layer of the compute context.

    Everything returned here is a function of the phonon model and the mesh
    dimensions ONLY: ``args.phonopy_yaml`` / ``args.force_constants`` /
    ``args.force_sets`` / ``args.born`` / ``args.mesh``. No temperature, Q/E
    grid, direction count or job count enters, so one model context can be
    shared by every grid-layer rebuild of the same model (e.g. the per-
    temperature rebuilds of a lat=0 multi-temperature deck).

    ``preloaded_full_mesh`` (optional): an already-run full-Monkhorst-Pack
    phonopy ``Mesh`` for the SAME model and mesh dimensions (the MT2 path's
    ``load_phonopy_mesh`` result carries one as ``phonopy_mesh_object``).
    When given, the duplicate full-mesh eigensolve is skipped;
    the symmetry-reduced mode-sum mesh is always run fresh.
    """
    from irma.core.phonopy_io import load_phonopy, validate_min_phonon_energy_mev
    min_phonon_energy_mev = validate_min_phonon_energy_mev(
        getattr(args, "min_phonon_energy_mev", 0.0))

    # NAC, when present, reaches both mesh runs and so every mode sum.
    print("Loading phonopy object...")
    phonon = load_phonopy(args.phonopy_yaml, born_path=args.born,
                          force_constants_filename=args.force_constants,
                          force_sets_filename=args.force_sets)
    # Full Monkhorst-Pack mesh with eigenvectors. The MT2 path (load_phonopy_mesh)
    # runs the identical mesh and passes it in as preloaded_full_mesh, which
    # skips the duplicate eigensolve.
    from irma.core.phonopy_io import openmp_unpinned_serial_setup
    if preloaded_full_mesh is not None:
        mesh = preloaded_full_mesh
        print(
            f"Reusing the MT2 full phonopy mesh {tuple(args.mesh)} "
            "(eigenvectors shared; skipping the duplicate eigensolve)..."
        )
    else:
        print(
            f"Running phonopy mesh {tuple(args.mesh)} with eigenvectors and full mesh symmetry disabled..."
        )
        with openmp_unpinned_serial_setup():
            phonon.run_mesh(args.mesh, with_eigenvectors=True,
                            is_mesh_symmetry=False)
        mesh = phonon.mesh
    # Read before the symmetry-reduced run below replaces phonon.mesh, so both
    # runs share one phonopy.load.
    _full_frequencies = np.asarray(mesh.frequencies, dtype=float)
    _full_eigenvectors = np.asarray(mesh.eigenvectors)
    print(
        f"Running phonopy mesh {tuple(args.mesh)} again with symmetry reduction for incoherent one-phonon mode sums..."
    )
    with openmp_unpinned_serial_setup():
        phonon.run_mesh(
            args.mesh,
            with_eigenvectors=True,
            is_mesh_symmetry=True,
        )
    incoherent_one_phonon_mesh = phonon.mesh

    # An Angstrom copy of the geometry; the live phonon object keeps its
    # calculator-unit cell, which its force constants are paired with.
    from irma.core.phonopy_io import angstrom_primitive
    primitive = angstrom_primitive(phonon)
    rec_lat_no_2pi = np.linalg.inv(primitive.cell)

    mesh_frequencies = _full_frequencies
    mesh_eigenvectors = reshape_mesh_eigenvectors(_full_eigenvectors, len(primitive.masses))
    incoherent_one_phonon_mesh_qpoints = np.asarray(incoherent_one_phonon_mesh.qpoints, dtype=float)
    incoherent_one_phonon_mesh_frequencies = np.asarray(
        incoherent_one_phonon_mesh.frequencies,
        dtype=float,
    )
    incoherent_one_phonon_mesh_eigenvectors = reshape_mesh_eigenvectors(
        np.asarray(incoherent_one_phonon_mesh.eigenvectors),
        len(primitive.masses),
    )
    incoherent_one_phonon_mesh_weights = np.asarray(
        incoherent_one_phonon_mesh.weights, dtype=float)
    print(
        "Using symmetry-reduced weighted mesh with "
        f"{len(incoherent_one_phonon_mesh_qpoints)} q-points "
        f"(weight sum {int(np.sum(incoherent_one_phonon_mesh_weights))}) "
        "for incoherent one-phonon and multiphonon mode sums..."
    )
    incoherent_one_phonon_n_branches = incoherent_one_phonon_mesh_frequencies.shape[1]
    incoherent_one_phonon_mode_energies_mev = (
        incoherent_one_phonon_mesh_frequencies * THzToEv * 1000.0
    ).reshape(-1)
    incoherent_one_phonon_mode_frequencies_thz = incoherent_one_phonon_mesh_frequencies.reshape(-1)
    incoherent_one_phonon_mode_weights = np.repeat(
        incoherent_one_phonon_mesh_weights,
        incoherent_one_phonon_n_branches,
    )
    incoherent_one_phonon_mode_eigvecs = incoherent_one_phonon_mesh_eigenvectors.reshape(
        -1,
        len(primitive.masses),
        3,
    )
    # Two-tier mode floor (see constants.py): the Goldstone guard applies
    # only at Gamma q-points; elsewhere a 1-ueV overflow guard keeps any
    # genuinely soft physics. Same mode set as the DOS tensor and the
    # thermal-displacement matrices.
    from irma.core.phonopy_io import mode_floor_mask
    incoherent_one_phonon_valid_modes = mode_floor_mask(
        incoherent_one_phonon_mode_energies_mev,
        incoherent_one_phonon_mesh_qpoints,
        incoherent_one_phonon_n_branches,
        min_phonon_energy_mev)
    incoherent_one_phonon_mode_energies_mev = incoherent_one_phonon_mode_energies_mev[
        incoherent_one_phonon_valid_modes
    ]
    incoherent_one_phonon_mode_frequencies_thz = incoherent_one_phonon_mode_frequencies_thz[
        incoherent_one_phonon_valid_modes
    ]
    incoherent_one_phonon_mode_weights = incoherent_one_phonon_mode_weights[
        incoherent_one_phonon_valid_modes
    ]
    incoherent_one_phonon_mode_eigvecs_valid = incoherent_one_phonon_mode_eigvecs[
        incoherent_one_phonon_valid_modes
    ]

    # The one-phonon spectrum ceiling, with the same mode floor as the mode sums.
    n_branches = mesh_frequencies.shape[1]
    mesh_mode_energies_mev = (mesh_frequencies * THzToEv * 1000.0).reshape(-1)
    valid_modes = mode_floor_mask(
        mesh_mode_energies_mev, mesh.qpoints, n_branches,
        min_phonon_energy_mev)
    mesh_mode_energies_mev = mesh_mode_energies_mev[valid_modes]
    max_mode_energy_mev = float(np.max(mesh_mode_energies_mev)) if len(mesh_mode_energies_mev) else 0.0

    multiphonon_mode_projection_components = None
    multiphonon_star_counts = None
    if len(incoherent_one_phonon_mesh_qpoints) < len(mesh_frequencies):
        multiphonon_projection_components_all, multiphonon_star_counts = build_star_averaged_projection_components(
            mesh_eigenvectors,
            incoherent_one_phonon_mesh,
        )
        multiphonon_mode_projection_components = multiphonon_projection_components_all.reshape(
            -1,
            len(primitive.masses),
            6,
        )[incoherent_one_phonon_valid_modes]
        print(
            "Using star-averaged projection tensors for reduced-mesh multiphonon "
            f"(star count range {int(np.min(multiphonon_star_counts))}..{int(np.max(multiphonon_star_counts))})."
        )

    # The MODEL's own eigenvalue->THz factor. phonopy already applies it to
    # mesh.frequencies, but the coherent one-phonon path solves the dynamical
    # matrix itself (noncubic_workers._batched_qpoints_eigh) and so must be
    # handed the same factor: it is calculator-specific (15.633302 vasp,
    # 108.970772 qe, 21.490680 abinit, 112.105157 cp2k, ...) because the
    # dynamical matrix is built from native-unit force constants and a
    # native-unit cell.
    frequency_factor_to_thz = float(phonon.unit_conversion_factor)

    return {
        "mesh": mesh,
        "primitive": primitive,
        "rec_lat_no_2pi": rec_lat_no_2pi,
        "frequency_factor_to_thz": frequency_factor_to_thz,
        "min_phonon_energy_mev": min_phonon_energy_mev,
        "incoherent_one_phonon_mesh_qpoints": incoherent_one_phonon_mesh_qpoints,
        "incoherent_one_phonon_mesh_weights": incoherent_one_phonon_mesh_weights,
        "incoherent_one_phonon_mode_energies_mev": incoherent_one_phonon_mode_energies_mev,
        "incoherent_one_phonon_mode_frequencies_thz": incoherent_one_phonon_mode_frequencies_thz,
        "incoherent_one_phonon_mode_weights": incoherent_one_phonon_mode_weights,
        "incoherent_one_phonon_mode_eigvecs_valid": incoherent_one_phonon_mode_eigvecs_valid,
        "max_mode_energy_mev": max_mode_energy_mev,
        "multiphonon_mode_projection_components": multiphonon_mode_projection_components,
        "multiphonon_star_counts": multiphonon_star_counts,
    }


def build_compute_context(
    args: argparse.Namespace,
    q_grid_ang_inv: np.ndarray,
    e_grid_mev: np.ndarray,
    preloaded_full_mesh: object = None,
    model_context: dict[str, object] | None = None,
) -> dict[str, object]:
    """Build the reusable, temperature-independent context for SAB evaluation.

    ``preloaded_full_mesh`` (optional): an already-run full-Monkhorst-Pack
    phonopy ``Mesh`` for the SAME model and mesh dimensions (the MT2 path's
    ``load_phonopy_mesh`` result carries one as ``phonopy_mesh_object``).
    When given, the duplicate full-mesh eigensolve is skipped;
    the symmetry-reduced mode-sum mesh is always run fresh.

    ``model_context`` (optional): an already-built MODEL layer from
    :func:`build_model_context` for the SAME model and mesh dimensions.
    When given, the phonopy load, both mesh eigensolves and the star-average
    are skipped entirely and only the grid layer is rebuilt — the caller
    (``standalone_sab.get_or_build_context``) is responsible for keying it by
    the model inputs. The model arrays are reused by reference, so the
    compute phase sees bit-identical inputs to a full rebuild.

    The returned dict carries the model layer under ``"_model_context"`` so
    cache owners can store/reuse it without rebuilding.
    """
    q_grid_ang_inv = np.asarray(q_grid_ang_inv, dtype=float)
    e_grid_mev = np.asarray(e_grid_mev, dtype=float)
    q_edges_ang_inv = centers_to_edges(q_grid_ang_inv, lower_bound=0.0)
    # First energy bin clamped to the physical support: for an energy grid
    # starting at 0, an unclamped first bin would span [-de/2, +de/2]
    # although deposits only land at E >= 0, under-weighting the tabulated
    # DENSITY at the first point by ~2x (deposited weight = density x
    # width is conserved either way; the clamp only moves the S value
    # tabulated at the first energy point onto the correct bin width).
    # Grids that genuinely start negative keep unclamped edges.
    e_edges_mev = centers_to_edges(
        e_grid_mev, lower_bound=0.0 if float(e_grid_mev[0]) >= 0.0 else None)
    e_bin_widths_mev = np.diff(e_edges_mev)
    q_min_used = float(q_grid_ang_inv.min())
    q_max_used = float(q_grid_ang_inv.max())
    dq_used = infer_uniform_spacing_or_none(q_grid_ang_inv)
    e_min_used = float(e_grid_mev.min())
    e_max_used = float(e_grid_mev.max())
    de_used = infer_uniform_spacing_or_none(e_grid_mev)

    directions = fibonacci_sphere(args.num_directions)

    if model_context is None:
        model_context = build_model_context(
            args, preloaded_full_mesh=preloaded_full_mesh)
    else:
        print(
            "Reusing cached noncubic model context (phonopy load, mesh "
            "eigensolves and star-averaged projections skipped).",
            flush=True,
        )
    primitive = model_context["primitive"]
    rec_lat_no_2pi = model_context["rec_lat_no_2pi"]

    # Reduced-coordinate direction basis: the coherent worker forms each
    # sample's reduced q as |Q| * direction_red_basis[direction].
    direction_red_basis = np.linalg.solve(rec_lat_no_2pi, directions.T).T / (2.0 * np.pi)
    num_coherent_samples = len(q_grid_ang_inv) * len(directions)

    mev_to_joule = EV * 1e-3
    unit_conversion = 1.0 / (AMU * (2 * np.pi * THz) ** 2)

    symbols = list(primitive.symbols)
    scattering_lengths, site_scattering_lengths = _site_values(
        getattr(args, "site_scattering_lengths_angstrom", None), symbols,
        args.scattering_lengths_json, args.scattering_lengths_file,
        _FALLBACK_C_SCATTERING_LENGTHS_ANGSTROM, "coherent scattering lengths")
    sigma_inc, site_incoherent_cross_sections = _site_values(
        getattr(args, "site_incoherent_cross_sections_barn", None), symbols,
        args.incoherent_cross_sections_json, args.incoherent_cross_sections_file,
        _FALLBACK_C_INCOHERENT_CROSS_SECTIONS_BARN, "incoherent cross sections")

    sigma_coh_by_atom = np.array(
        [4.0 * np.pi * bcoh**2 * ANG2_TO_BARN for bcoh in site_scattering_lengths],
        dtype=float,
    )
    coherent_atom_prefactors = np.array(
        [
            bcoh / np.sqrt(2.0 * mass)
            for bcoh, mass in zip(site_scattering_lengths, primitive.masses)
        ],
        dtype=np.complex128,
    )
    sigma_inc_by_atom = np.asarray(site_incoherent_cross_sections, dtype=float)
    sigma_total_by_atom = sigma_coh_by_atom + sigma_inc_by_atom
    incoherent_prefactors = np.array(
        [
            sigma_inc_atom / (4.0 * np.pi * ANG2_TO_BARN) / (2.0 * mass)
            for sigma_inc_atom, mass in zip(sigma_inc_by_atom, primitive.masses)
        ],
        dtype=float,
    )
    incoherent_approx_prefactors = np.array(
        [
            sigma_total_by_atom[i] / (4.0 * np.pi * ANG2_TO_BARN) / (2.0 * mass)
            for i, mass in enumerate(primitive.masses)
        ],
        dtype=float,
    )

    num_jobs = max(1, args.jobs)
    chunk_size = max(1, args.q_chunk_size)
    shell_chunk_size = max(1, min(chunk_size, int(np.ceil(len(q_grid_ang_inv) / num_jobs))))

    coherent_blocks = [
        np.arange(start_index, min(num_coherent_samples, start_index + chunk_size), dtype=int)
        for start_index in range(0, num_coherent_samples, chunk_size)
    ]
    shell_blocks = [
        np.arange(start_index, min(len(q_grid_ang_inv), start_index + shell_chunk_size), dtype=int)
        for start_index in range(0, len(q_grid_ang_inv), shell_chunk_size)
    ]
    multiphonon_num_directions = (
        max(1, int(args.num_directions))
        if args.multiphonon_num_directions is None
        else max(1, int(args.multiphonon_num_directions))
    )
    multiphonon_dir_chunk_size = max(
        1, min(multiphonon_num_directions, _MULTIPHONON_DIR_CHUNK))

    return {
        "q_grid_ang_inv": q_grid_ang_inv,
        "e_grid_mev": e_grid_mev,
        "q_edges_ang_inv": q_edges_ang_inv,
        "e_edges_mev": e_edges_mev,
        "e_bin_widths_mev": e_bin_widths_mev,
        "q_min_used": q_min_used,
        "q_max_used": q_max_used,
        "dq_used": dq_used,
        "e_min_used": e_min_used,
        "e_max_used": e_max_used,
        "de_used": de_used,
        "directions": directions,
        "direction_red_basis": direction_red_basis,
        # MODEL layer (shared by reference with model_context — see
        # build_model_context for the keys: mesh, primitive, rec_lat_no_2pi,
        # the incoherent_one_phonon_* arrays, max_mode_energy_mev and the
        # multiphonon_* mode/projection arrays).
        **model_context,
        "mev_to_joule": mev_to_joule,
        "unit_conversion": unit_conversion,
        "scattering_lengths": scattering_lengths,
        "sigma_inc": sigma_inc,
        "site_scattering_lengths_angstrom": site_scattering_lengths,
        "site_incoherent_cross_sections_barn": site_incoherent_cross_sections,
        "sigma_coh_by_atom": sigma_coh_by_atom,
        "coherent_atom_prefactors": coherent_atom_prefactors,
        "sigma_inc_by_atom": sigma_inc_by_atom,
        "sigma_total_by_atom": sigma_total_by_atom,
        "incoherent_prefactors": incoherent_prefactors,
        "incoherent_approx_prefactors": incoherent_approx_prefactors,
        "num_jobs": num_jobs,
        "chunk_size": chunk_size,
        "shell_chunk_size": shell_chunk_size,
        "coherent_blocks": coherent_blocks,
        "shell_blocks": shell_blocks,
        "multiphonon_dir_chunk_size": multiphonon_dir_chunk_size,
        "_model_context": model_context,
    }
