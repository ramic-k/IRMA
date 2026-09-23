"""Temperature-independent setup for the noncubic inelastic driver.

Two cache layers:

* MODEL layer (:func:`build_model_context`) — a pure function of the phonon
  model and mesh dimensions (phonopy.yaml + force constants + BORN + mesh):
  the phonopy load, both mesh eigensolves, the reduced-mesh mode arrays and
  the star-averaged projection tensors. Independent of temperature, of the
  Q/E grids, and of the sampling/job controls.
* GRID layer (the rest of :func:`build_compute_context`) — the Q/E edges,
  direction quadrature, Q-vector geometry, scattering prefactors and block
  partitions. Cheap numpy work, rebuilt per grid digest.

A lat=1 deck has temperature-independent (alpha, beta)-scaled grids, so the
whole context is reused across temperature cards. A lat=0 deck's physical
Q/E grids change with kT, so every temperature is a full-context miss — but
the model layer is grid-independent, so only the grid layer is rebuilt
(see ``standalone_sab.get_or_build_context`` for the keying/eviction).
"""

from __future__ import annotations

import time

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
    parse_incoherent_cross_sections,
    parse_scattering_lengths,
    reshape_mesh_eigenvectors,
)

# Fixed multiphonon direction-block size. The multiphonon blocks overlap on
# the full (Q, E) grid and are accumulated in fixed order with one BLAS
# contraction per block, so the PARTITION — not just the accumulation order —
# determines the floating-point summation tree. The chunk size must
# therefore be a CONSTANT, never derived from Card 6f ncpu (or from the
# machine's core count through the ncpu clamp): a jobs-dependent chunk
# would move tape bytes at the last bit with the worker count, and the
# ENDF writer's 6-significant-figure rounding leaves only ~1e-15 of
# margin over that drift. A fixed, jobs-independent chunk makes
# parallel == serial structurally bitwise-identical. 25 keeps every
# pinned test configuration on its pinned partition (mpdir<=25 decks are
# a single block; the tape-gauge small profile splits at exactly
# ceil(100/4)=25) and still yields 40 blocks at the
# validation campaign's mpdir=1000 for pool balance (8 at the 200 default). The engine's 2-GiB
# kernel-table memory cap can only shrink the block further and is itself
# jobs-independent.
_MULTIPHONON_DIR_CHUNK = 25


def _stage_printer(setup_start: float):
    """Return a finish_stage(label) closure printing elapsed stage times."""
    state = {"t": setup_start}

    def finish_stage(label: str) -> None:
        """Print the elapsed time of the setup stage that just finished."""
        now = time.perf_counter()
        print(
            f"  Context setup: {label} in {now - state['t']:.2f} s",
            flush=True,
        )
        state["t"] = now

    return finish_stage


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
    import phonopy

    finish_stage = _stage_printer(time.perf_counter())
    from irma.core.phonopy_io import validate_min_phonon_energy_mev
    min_phonon_energy_mev = validate_min_phonon_energy_mev(
        getattr(args, "min_phonon_energy_mev", 0.0))

    # Non-analytical-term correction (Born effective charges): applied to
    # BOTH mesh runs so LO-TO splitting reaches the coherent one-phonon,
    # incoherent one-phonon, and multiphonon mode sums alike. An explicit
    # BORN path (Card 6f use_born=1) wins; otherwise NAC embedded in the
    # named phonopy.yaml is honored; phonopy's fallback of auto-reading
    # ./BORN from the process working directory is disabled (is_nac=False)
    # so the applied physics never depends on the run cwd.
    import os

    from irma.core.phonopy_io import (
        isolated_phonopy_cwd,
        phonopy_yaml_embeds_nac,
        pinned_primitive_matrix_kwargs,
        reject_unsafe_phonopy_yaml,
        resolve_force_constants_source,
    )

    # Absolutize before the isolated_phonopy_cwd pin below.
    phonopy_yaml = os.path.abspath(str(args.phonopy_yaml))
    # TRUST BOUNDARY (SEC-1): refuse a phonopy.yaml carrying code-executing
    # YAML tags BEFORE anything (embeds_nac scan aside, phonopy.load below)
    # touches it with phonopy's unsafe YAML loader.
    reject_unsafe_phonopy_yaml(phonopy_yaml)
    born_path = getattr(args, "born", None)
    if born_path is not None:
        born_path = os.path.abspath(str(born_path))
    embeds_nac = phonopy_yaml_embeds_nac(phonopy_yaml)
    apply_nac = (born_path is not None) or embeds_nac
    # Force constants: honor explicit caller paths; otherwise discover them
    # next to the yaml (embedded, else hdf5 > text > FORCE_SETS). phonopy's
    # fallback of searching the process working directory must never decide
    # which model gets computed — isolated_phonopy_cwd pins cwd to an empty
    # scratch directory during the loads because phonopy probes the cwd
    # even when explicit paths are given.
    fc_filename = getattr(args, "force_constants", None)
    fs_filename = getattr(args, "force_sets", None)
    if fc_filename is None and fs_filename is None:
        fc_kwargs = resolve_force_constants_source(phonopy_yaml)
        fc_filename = fc_kwargs.get("force_constants_filename")
        fs_filename = fc_kwargs.get("force_sets_filename")
    else:
        fc_filename = None if fc_filename is None else os.path.abspath(str(fc_filename))
        fs_filename = None if fs_filename is None else os.path.abspath(str(fs_filename))
    if born_path is not None:
        print(f"Loading phonopy object with NAC (BORN: {born_path})...")
    elif embeds_nac:
        print("Loading phonopy object with NAC (embedded in phonopy.yaml)...")
    else:
        print("Loading phonopy object...")
    # Same (C) backend as load_phonopy_mesh, so both loaders give identical arrays.
    with isolated_phonopy_cwd():
        phonon = phonopy.load(
            phonopy_yaml,
            force_constants_filename=fc_filename,
            force_sets_filename=fs_filename,
            born_filename=born_path,
            is_nac=apply_nac,
            **pinned_primitive_matrix_kwargs(phonopy_yaml),
            lang="C",
        )
    if born_path is not None and phonon.nac_params is None:
        raise RuntimeError(
            f"BORN corrections were requested but phonopy.load returned no "
            f"NAC parameters from {born_path}."
        )
    if born_path is None and embeds_nac and phonon.nac_params is None:
        # NAC keys declared in the yaml but unusable: refuse to guess.
        raise RuntimeError(
            f"{phonopy_yaml} embeds NAC keys but phonopy could not "
            f"parse NAC parameters from them; fix the file or supply an "
            f"explicit BORN file (Card 6f use_born=1)."
        )
    # Full Monkhorst-Pack mesh with eigenvectors. The MT2 directional-DW path
    # (load_phonopy_mesh) computes the IDENTICAL run — same resolved force
    # constants, same NAC policy, same run_mesh(with_eigenvectors=True,
    # is_mesh_symmetry=False), pinned single-thread BLAS — so when the caller
    # hands that Mesh in (preloaded_full_mesh) the eigensolve is
    # reused instead of repeated; the symmetry-reduced run below still runs on
    # the freshly loaded phonon object. A dimension mismatch falls back to the
    # fresh run rather than trusting a wrong cache.
    _full_mesh = None
    if preloaded_full_mesh is not None:
        n_q_expected = int(np.prod([int(m) for m in args.mesh]))
        _freqs_pre = np.asarray(preloaded_full_mesh.frequencies, dtype=float)
        if _freqs_pre.shape[0] == n_q_expected:
            _full_mesh = preloaded_full_mesh
            mesh = _full_mesh          # context["mesh"] = the full-MP mesh
            print(
                f"Reusing the MT2 full phonopy mesh {tuple(args.mesh)} "
                "(eigenvectors shared; skipping the duplicate eigensolve)..."
            )
            finish_stage("full phonopy mesh reused")
        else:
            print(
                f"NOTE: preloaded full mesh has {_freqs_pre.shape[0]} q-points, "
                f"expected {n_q_expected}; running the full mesh fresh."
            )
    if _full_mesh is None:
        print(
            f"Running phonopy mesh {tuple(args.mesh)} with eigenvectors and full mesh symmetry disabled..."
        )
        from irma.core.phonopy_io import openmp_unpinned_serial_setup
        with openmp_unpinned_serial_setup():
            phonon.run_mesh(args.mesh, with_eigenvectors=True,
                            is_mesh_symmetry=False)
        mesh = phonon.mesh
        assert mesh is not None
        finish_stage("full phonopy mesh loaded")
        # The full-mesh arrays are extracted into plain numpy BEFORE re-running
        # the mesh, so the second (symmetry-reduced) run can reuse the SAME
        # phonopy object: one phonopy.load / force-constants parse instead of
        # two.
        _full_mesh = phonon.mesh
    _full_frequencies = np.asarray(_full_mesh.frequencies, dtype=float)
    _full_eigenvectors = np.asarray(_full_mesh.eigenvectors)
    print(
        f"Running phonopy mesh {tuple(args.mesh)} again with symmetry reduction for incoherent one-phonon mode sums..."
    )
    from irma.core.phonopy_io import openmp_unpinned_serial_setup
    with openmp_unpinned_serial_setup():
        phonon.run_mesh(
            args.mesh,
            with_eigenvectors=True,
            is_mesh_symmetry=True,
        )
    incoherent_one_phonon_mesh = phonon.mesh
    assert incoherent_one_phonon_mesh is not None
    finish_stage("incoherent/mode-sum mesh loaded")

    # UNITS. phonopy keeps the cell in the CALCULATOR's native length unit and
    # phonopy.load never converts it, so `phonon.primitive.cell` is bohr for a
    # qe/abinit/siesta/wien2k/... model. angstrom_primitive() is the single
    # conversion point (irma.core.phonopy_io): it returns an IRMA-owned COPY of
    # the geometry with the lattice in Angstrom, leaving the live phonopy
    # object's native-unit cell alone -- phonopy pairs that cell with force
    # constants in the same native units and with the calculator's frequency
    # factor, so mutating it would corrupt the frequencies. Everything that
    # reads context["primitive"] downstream (masses, symbols, fractional
    # positions, and the `primitive_lattice_ang` of the elastic state) is
    # therefore genuinely in the units its name claims, and rec_lat_no_2pi is
    # genuinely 1/Angstrom -- which matters because q_red (the reduced q at
    # which the dynamical matrix and the structure-factor phases are evaluated)
    # is |Q|[1/A] x cell/2pi.
    from irma.core.phonopy_io import angstrom_primitive
    primitive = angstrom_primitive(phonon, phonopy_yaml)
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
        getattr(
            incoherent_one_phonon_mesh,
            "weights",
            np.ones(incoherent_one_phonon_mesh_frequencies.shape[0]),
        ),
        dtype=float,
    )
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

    # Full-mesh mode bookkeeping. Only max_mode_energy_mev (the one-phonon
    # spectrum ceiling, with the same per-q mode floor as the live mode
    # sums) goes into the context. Flattened full-mesh mode arrays or a
    # fancy-indexed full-mesh eigenvector copy would cost O(100 MB) per
    # context and have no consumer — the live mode sums all read the
    # incoherent_one_phonon_* reduced-mesh arrays — so they are never
    # built.
    n_branches = mesh_frequencies.shape[1]
    mesh_mode_energies_mev = (mesh_frequencies * THzToEv * 1000.0).reshape(-1)
    valid_modes = mode_floor_mask(
        mesh_mode_energies_mev, _full_mesh.qpoints, n_branches,
        min_phonon_energy_mev)
    mesh_mode_energies_mev = mesh_mode_energies_mev[valid_modes]
    max_mode_energy_mev = float(np.max(mesh_mode_energies_mev)) if len(mesh_mode_energies_mev) else 0.0
    finish_stage("mesh mode arrays prepared")

    # NOTE: only the q-weight norm is kept (it seeds multiphonon_q_weight_norm
    # below). The per-temperature histogram lookups, signed work grids and
    # multiphonon direction set are built by compute_from_args, not here.
    incoherent_one_phonon_q_weight_norm = float(np.sum(incoherent_one_phonon_mesh_weights))
    multiphonon_mode_energies_mev = incoherent_one_phonon_mode_energies_mev
    multiphonon_mode_frequencies_thz = incoherent_one_phonon_mode_frequencies_thz
    multiphonon_mode_weights = incoherent_one_phonon_mode_weights
    multiphonon_mode_eigvecs_valid = incoherent_one_phonon_mode_eigvecs_valid
    multiphonon_q_weight_norm = incoherent_one_phonon_q_weight_norm
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
    finish_stage("multiphonon projection data prepared")

    # NOTE: the full-mesh frequency/weight/eigenvector arrays (and the
    # flattened mesh_mode_* views of them) are deliberately NOT exported:
    # no compute phase reads them (the live mode sums use the
    # incoherent_one_phonon_* reduced-mesh arrays), and the eigenvector
    # copies alone would be O(100 MB)+ of dead state per cached context.
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
        "incoherent_one_phonon_mesh_frequencies": incoherent_one_phonon_mesh_frequencies,
        "incoherent_one_phonon_mesh_eigenvectors": incoherent_one_phonon_mesh_eigenvectors,
        "incoherent_one_phonon_mesh_weights": incoherent_one_phonon_mesh_weights,
        "incoherent_one_phonon_mode_energies_mev": incoherent_one_phonon_mode_energies_mev,
        "incoherent_one_phonon_mode_frequencies_thz": incoherent_one_phonon_mode_frequencies_thz,
        "incoherent_one_phonon_mode_weights": incoherent_one_phonon_mode_weights,
        "incoherent_one_phonon_mode_eigvecs_valid": incoherent_one_phonon_mode_eigvecs_valid,
        "max_mode_energy_mev": max_mode_energy_mev,
        "multiphonon_mode_energies_mev": multiphonon_mode_energies_mev,
        "multiphonon_mode_frequencies_thz": multiphonon_mode_frequencies_thz,
        "multiphonon_mode_weights": multiphonon_mode_weights,
        "multiphonon_mode_eigvecs_valid": multiphonon_mode_eigvecs_valid,
        "multiphonon_q_weight_norm": multiphonon_q_weight_norm,
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
    setup_start = time.perf_counter()
    finish_stage = _stage_printer(setup_start)

    print(
        "  Context setup: preparing reusable phonopy/mesh state...",
        flush=True,
    )

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

    finish_stage("grid sampling prepared")

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
    finish_stage("coherent geometry prepared")

    print("Evaluating coherent one-phonon dynamic structure factor for "
          f"{num_coherent_samples} Q-vectors...")

    mev_to_joule = EV * 1e-3
    unit_conversion = 1.0 / (AMU * (2 * np.pi * THz) ** 2)

    site_scattering_lengths = getattr(args, "site_scattering_lengths_angstrom", None)
    if site_scattering_lengths is None:
        scattering_lengths = parse_scattering_lengths(
            list(primitive.symbols),
            args.scattering_lengths_json,
            args.scattering_lengths_file,
        )
        site_scattering_lengths = np.array(
            [scattering_lengths[s] for s in primitive.symbols],
            dtype=float,
        )
    else:
        scattering_lengths = None
        site_scattering_lengths = np.asarray(site_scattering_lengths, dtype=float)
        if len(site_scattering_lengths) != len(primitive.symbols):
            raise ValueError(
                "site_scattering_lengths_angstrom must match the number of primitive-cell atoms."
            )

    site_incoherent_cross_sections = getattr(args, "site_incoherent_cross_sections_barn", None)
    if site_incoherent_cross_sections is None:
        sigma_inc = parse_incoherent_cross_sections(
            list(primitive.symbols),
            args.incoherent_cross_sections_json,
            args.incoherent_cross_sections_file,
        )
        site_incoherent_cross_sections = np.array(
            [sigma_inc[s] for s in primitive.symbols],
            dtype=float,
        )
    else:
        sigma_inc = None
        site_incoherent_cross_sections = np.asarray(site_incoherent_cross_sections, dtype=float)
        if len(site_incoherent_cross_sections) != len(primitive.symbols):
            raise ValueError(
                "site_incoherent_cross_sections_barn must match the number of primitive-cell atoms."
            )

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
    # NOTE: the authoritative multiphonon sigma_total scale is rebuilt downstream
    # per CARD-6d ATOM-TYPE GROUP (export_multiphonon_sigma_total_scale in
    # noncubic_engine.compute_from_args, using 1/len(group_indices)), and that
    # per-group version is what the worker reads.
    # An element-SYMBOL-multiplicity normalization is NOT the contract and would
    # diverge from the per-group convention whenever two inequivalent Card 6d
    # groups share one element symbol, so it is deliberately not computed/exported
    # here to avoid a future-edit footgun.
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
    finish_stage("scattering prefactors prepared")

    num_jobs = max(1, args.jobs)
    chunk_size = max(1, args.q_chunk_size)
    shell_chunk_size = max(1, min(chunk_size, int(np.ceil(len(q_grid_ang_inv) / num_jobs))))

    print("  Context setup: partitioning coherent direction blocks...", flush=True)
    coherent_blocks = [
        np.arange(start_index, min(num_coherent_samples, start_index + chunk_size), dtype=int)
        for start_index in range(0, num_coherent_samples, chunk_size)
    ]
    print("  Context setup: partitioning incoherent shell blocks...", flush=True)
    shell_blocks = [
        np.arange(start_index, min(len(q_grid_ang_inv), start_index + shell_chunk_size), dtype=int)
        for start_index in range(0, len(q_grid_ang_inv), shell_chunk_size)
    ]
    multiphonon_num_directions = (
        max(1, int(args.num_directions))
        if args.multiphonon_num_directions is None
        else max(1, int(args.multiphonon_num_directions))
    )
    # Jobs-INDEPENDENT partition (see _MULTIPHONON_DIR_CHUNK above): tape
    # bytes must not vary with Card 6f ncpu, so the chunk is a fixed constant
    # rather than ceil(mpdir/num_jobs).
    multiphonon_dir_chunk_size = max(
        1, min(multiphonon_num_directions, _MULTIPHONON_DIR_CHUNK))
    # The multiphonon direction set, work/signed grids, signed histogram
    # lookups and base prefactors are built by compute_from_args, not cached
    # here.
    finish_stage("block partitions prepared")

    print(
        f"  Context setup: complete in {time.perf_counter() - setup_start:.2f} s",
        flush=True,
    )
    print("  Context setup: returning cached arrays to compute phase.", flush=True)

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
