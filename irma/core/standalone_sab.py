"""Helpers for the in-process noncubic SAB workflow used by inelastic_mode=1/2.

Progress prints here go to STDOUT (flushed), deliberately sharing the channel
the rest of the noncubic engine uses (noncubic_engine / noncubic_inelastic_
context), so a TSL deck run logs on a single stream. Note the spectra CLI
routes ITS provenance lines to stderr, so a modes-1/2 spectra run splits
progress across both streams; routing the whole engine through one channel
would be an engine-wide change, not something this module can do alone.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np

from irma.core.noncubic_inelastic import NoncubicInelasticControls

# The α↔Q/β↔E convention lives in one shared place (irma.core.sab_grids) so
# the ENDF/standalone path and the NCrystal exporter cannot drift; the
# private alias keeps this module's call sites terse.
from irma.core.sab_grids import irma_grid_to_physical_qe as _irma_grid_to_physical_qe

_CONTEXT_CACHE: dict[tuple[object, ...], dict[str, object]] = {}


def _pick_sab_key(
    output_arrays,
    multiphonon_max_order: int,
    inelastic_mode: int,
) -> str:
    """Select the SAB array key for the given mode and (effective) order.

    The key is a pure function of ``(inelastic_mode, multiphonon_max_order)``.
    ``output_arrays`` is used only to validate that the selected key is
    actually present, turning a desync between this mapping and the engine's
    population guards into a clear error instead of a bare KeyError at the
    later dereference.
    """
    if inelastic_mode == 2:
        if multiphonon_max_order >= 2:
            key = "sab_asym_downscatter_one_phonon_total_plus_incoherent_approx_multiphonon"
        else:
            key = "sab_asym_downscatter_one_phonon_total"
    elif inelastic_mode == 1:
        if multiphonon_max_order >= 2:
            key = "sab_asym_downscatter_incoherent_approx_n1_term_plus_incoherent_approx_multiphonon"
        else:
            key = "sab_asym_downscatter_incoherent_approx_n1_term"
    else:
        raise ValueError(
            f"_pick_sab_key: inelastic_mode must be 1 or 2, got {inelastic_mode}")
    if output_arrays is not None and key not in output_arrays:
        raise KeyError(
            f"_pick_sab_key selected '{key}' for inelastic_mode={inelastic_mode}, "
            f"multiphonon_max_order={multiphonon_max_order}, but it is not present "
            "in the engine output arrays — the SAB-key mapping and the engine's "
            "array-population guards have desynchronized."
        )
    return key


def _file_identity(path):
    """``(path, size, mtime_ns)`` -- cheap content identity for cache keys.

    A path alone is not identity: a regenerated FORCE_CONSTANTS at the same
    path in a long-lived process (a GUI session running several decks) must
    MISS the model cache, not silently reuse the stale phonon model.
    size + mtime_ns catches every regeneration without hashing files that
    can reach hundreds of MB. A vanished file keys as (-1, -1): the build
    itself will raise the real error."""
    if path is None:
        return None
    p = str(path)
    try:
        st = os.stat(p)
        return (p, st.st_size, st.st_mtime_ns)
    except OSError:
        return (p, -1, -1)


def _model_input_identities(phonopy_yaml, force_constants, force_sets, born):
    """Identity tuple of every file the phonon model is built from.

    When no explicit force-constants source is given, phonopy_io's resolver
    picks a sibling of the yaml (force_constants.hdf5 / FORCE_CONSTANTS /
    FORCE_SETS, or the yaml itself when FCs are embedded) -- that file's
    identity must be in the key too, or rewriting it next to an unchanged
    yaml would silently reuse a stale cached model."""
    ids = [_file_identity(phonopy_yaml), _file_identity(force_constants),
           _file_identity(force_sets), _file_identity(born)]
    if force_constants is None and force_sets is None:
        from irma.core.phonopy_io import resolve_force_constants_source
        try:
            resolved = resolve_force_constants_source(str(phonopy_yaml))
        except (OSError, ValueError):
            resolved = {"unresolved": True}
        ids.append(tuple(sorted(
            (k, _file_identity(v) if isinstance(v, str) else v)
            for k, v in resolved.items())))
    return tuple(ids)


def _grid_digest(values: np.ndarray) -> str:
    """SHA-1 of a float64 grid's exact bytes, for cache keys."""
    arr = np.ascontiguousarray(np.asarray(values, dtype=np.float64))
    return hashlib.sha1(arr.view(np.uint8)).hexdigest()


def get_or_build_context(*, phonopy_yaml, force_constants, force_sets, born,
                         mesh_dim, grid_key, q_grid_ang_inv, e_grid_mev,
                         num_directions, multiphonon_num_directions, num_jobs,
                         sigma_mev,
                         multiphonon_max_order,
                         min_phonon_energy_mev=0.0,
                         scattering_lengths_json=None,
                         incoherent_cross_sections_json=None,
                         site_scattering_lengths_angstrom=None,
                         site_incoherent_cross_sections_barn=None,
                         context_cache=None, preloaded_full_mesh=None):
    """Digest-keyed lookup/build of the temperature-independent compute context.

    One key discipline for every consumer (the ENDF standalone path and the
    spectra bridge): identical model + mesh + grids + sampling -> the cached
    context; anything else -> a fresh build. ``build_compute_context`` consumes
    none of sigma_mev / multiphonon_max_order; they
    stay in the key as a deliberately conservative over-key (a hit can never
    return wrong arrays). ``preloaded_full_mesh`` skips the duplicate full-mesh
    eigensolve on a build.

    Two cache layers: besides the full-context entry,
    the MODEL layer (phonopy load + mesh eigensolves + star-averaged
    projections — see ``build_model_context``) is cached separately under a
    key of the model inputs only. A lat=0 multi-temperature deck misses the
    full-context key at every temperature (its physical Q/E grids scale with
    kT) but hits the model key, so only the cheap grid layer is rebuilt. The
    model arrays are reused by reference — bit-identical inputs reach the
    compute phase.
    """
    from irma.core.noncubic_inelastic import build_compute_context

    input_ids = _model_input_identities(phonopy_yaml, force_constants,
                                        force_sets, born)
    model_key = (
        "__noncubic_model__",
        input_ids,
        tuple(int(x) for x in mesh_dim),
        float(min_phonon_energy_mev),
    )
    context_key = (
        input_ids,
        tuple(int(x) for x in mesh_dim),
        grid_key,
        int(num_directions),
        int(multiphonon_num_directions),
        int(max(1, int(num_jobs))),
        float(sigma_mev),
        int(multiphonon_max_order),
        float(min_phonon_energy_mev),
        scattering_lengths_json,
        incoherent_cross_sections_json,
        None if site_scattering_lengths_angstrom is None else _grid_digest(
            np.asarray(site_scattering_lengths_angstrom, dtype=float)),
        None if site_incoherent_cross_sections_barn is None else _grid_digest(
            np.asarray(site_incoherent_cross_sections_barn, dtype=float)),
    )
    cache = _CONTEXT_CACHE if context_cache is None else context_cache
    context = cache.get(context_key)
    if context is None:
        model_context = cache.get(model_key)
        print("Building noncubic compute context...", flush=True)
        args_context = type("Args", (), {
            "phonopy_yaml": str(phonopy_yaml),
            "force_constants": None if force_constants is None else str(force_constants),
            "force_sets": None if force_sets is None else str(force_sets),
            "born": None if born is None else str(born),
            "mesh": [int(mesh_dim[0]), int(mesh_dim[1]), int(mesh_dim[2])],
            "num_directions": int(num_directions),
            "scattering_lengths_json": scattering_lengths_json,
            "scattering_lengths_file": None,
            "incoherent_cross_sections_json": incoherent_cross_sections_json,
            "incoherent_cross_sections_file": None,
            "site_scattering_lengths_angstrom": site_scattering_lengths_angstrom,
            "site_incoherent_cross_sections_barn": site_incoherent_cross_sections_barn,
            "jobs": int(max(1, int(num_jobs))),
            "q_chunk_size": 20000,
            "multiphonon_num_directions": int(multiphonon_num_directions),
            "multiphonon_max_order": int(multiphonon_max_order),
            "min_phonon_energy_mev": float(min_phonon_energy_mev),
        })()
        if model_context is None:
            # Positional/keyword form kept stub-compatible: tests monkeypatch
            # build_compute_context with (args, q, e, preloaded_full_mesh=None)
            # fakes, so the model_context kwarg is only passed on a model hit.
            context = build_compute_context(args_context, q_grid_ang_inv, e_grid_mev,
                                            preloaded_full_mesh=preloaded_full_mesh)
        else:
            context = build_compute_context(args_context, q_grid_ang_inv, e_grid_mev,
                                            preloaded_full_mesh=preloaded_full_mesh,
                                            model_context=model_context)
        cache[context_key] = context
        # Cache the model layer (phonopy/mesh/star-average state) under its
        # model-inputs-only key so the next grid-key miss for the same model
        # (lat=0: every further temperature card) rebuilds only the grid
        # layer. Shared by reference with the full context — no extra memory.
        built_model = context.get("_model_context") if isinstance(context, dict) else None
        if built_model is not None:
            cache[model_key] = built_model
        print("Noncubic compute context cached.", flush=True)
    else:
        print("Reusing cached noncubic inelastic context.", flush=True)
    # Keep-only-current-key eviction (same size-one policy as the spectra
    # bridge, forward.py): within a run, context reuse is strictly sequential —
    # lat=1 multi-temperature decks re-hit ONE temperature-independent key,
    # while lat=0 decks produce a NEW key per temperature and never revisit an
    # old one. Without this, a lat=0 multi-temperature deck accumulates one
    # full context (O(100 MB) at production sampling) per temperature in
    # crystal_info's cache for the whole run_leapr call. The current model
    # layer is kept alongside (same arrays by reference, no extra memory).
    for stale in [k for k in cache if k not in (context_key, model_key)]:
        del cache[stale]
    return context, context_key


def _last_nonzero_beta(beta_downscatter_abs: np.ndarray, sab_qe: np.ndarray) -> float | None:
    """Return the last nonzero beta support in an ``(alpha, beta)`` SAB array."""
    beta_arr = np.asarray(beta_downscatter_abs, dtype=float)
    sab_arr = np.asarray(sab_qe, dtype=float)
    if sab_arr.ndim != 2:
        raise ValueError("Expected a 2D SAB array.")
    if sab_arr.shape[1] != len(beta_arr):
        raise ValueError("Beta grid length does not match SAB shape.")
    row_max = np.max(sab_arr, axis=0)
    nz = np.flatnonzero(row_max > 0.0)
    if nz.size == 0:
        return None
    return float(beta_arr[nz[-1]])


def run_noncubic_standalone_sab(
    *,
    alpha: np.ndarray,
    beta: np.ndarray,
    lat: int,
    temperature_k: float,
    awr: float,
    phonopy_yaml_path: str,
    mesh_dim: list[int] | tuple[int, int, int],
    born_path: str | None = None,
    num_jobs: int,
    workdir: str | None = None,
    sigma_mev: float | None = None,
    inelastic_mode: int = 1,
    represented_principal_site_count: int | None = None,
    principal_group_index: int = 0,
    site_groups: list[list[int]] | tuple[tuple[int, ...], ...] | None = None,
    coherent_partition_mode: str = "auto",
    sab_sigma_barn: float | None = None,
    site_scattering_lengths_angstrom: list[float] | tuple[float, ...] | np.ndarray | None = None,
    site_incoherent_cross_sections_barn: list[float] | tuple[float, ...] | np.ndarray | None = None,
    scattering_lengths_json: str | None = None,
    incoherent_cross_sections_json: str | None = None,
    controls: NoncubicInelasticControls,
    context_cache: dict[tuple[object, ...], dict[str, object]] | None = None,
    preloaded_full_mesh: object = None,
    precomputed_thermal_mats: np.ndarray | None = None,
) -> dict[str, object]:
    """Run the in-process noncubic SAB driver and convert to IRMA internal form.

    ``represented_principal_site_count`` is the number of phonopy primitive-cell
    sites represented by the exported principal-scatterer MT4 law. The
    standalone builder accumulates one-phonon intensity over those explicit
    sites first and only then reduces to the per-principal-scatterer S(a,b)
    convention expected by IRMA MF7/MT4.

    ``preloaded_full_mesh``: an already-run full-MP phonopy ``Mesh`` for the
    same model/mesh (the MT2 loader's ``PhonopyMeshData.phonopy_mesh_object``);
    forwarded to ``build_compute_context`` so a context-cache MISS skips the
    duplicate full-mesh eigensolve.
    """
    if controls is None:
        raise ValueError("run_noncubic_standalone_sab requires explicit NoncubicInelasticControls.")

    workdir_path = Path(workdir or Path.cwd())
    workdir_path.mkdir(parents=True, exist_ok=True)

    phonopy_yaml = Path(phonopy_yaml_path).resolve()
    # hdf5 > text FORCE_CONSTANTS > FORCE_SETS next to the yaml, or embedded
    # in the yaml itself ({} -> phonopy reads them from the yaml); raises if
    # no source exists. Resolved here so the failure is immediate and the
    # context cache key reflects the actual model source.
    from irma.core.phonopy_io import resolve_force_constants_source
    fc_kwargs = resolve_force_constants_source(phonopy_yaml)
    force_constants = fc_kwargs.get("force_constants_filename")
    force_sets = fc_kwargs.get("force_sets_filename")
    born = None
    if born_path is not None:
        born = Path(born_path).resolve()
        if not born.exists():
            raise FileNotFoundError(
                f"BORN corrections were requested but the BORN file does not "
                f"exist: {born}"
            )

    multiphonon_max_order = int(controls.multiphonon_max_order)
    auto_multiphonon_order = bool(getattr(controls, "auto_multiphonon_order", False))
    min_phonon_energy_mev = float(
        getattr(controls, "min_phonon_energy_mev", 0.0))
    num_directions = int(controls.num_directions)
    multiphonon_num_directions = int(controls.multiphonon_num_directions)
    sigma_mev = 0.0 if sigma_mev is None else float(sigma_mev)

    q_grid_ang_inv, e_grid_mev, alpha_abs_expected, beta_downscatter_abs_expected = _irma_grid_to_physical_qe(
        alpha, beta, lat, temperature_k, awr
    )

    coherent_summary = "cohavg=directions"

    print(
        "Starting in-process noncubic SAB driver: "
        f"mode={inelastic_mode}, mesh={tuple(int(x) for x in mesh_dim)}, "
        f"nac={'on (' + born.name + ')' if born is not None else 'off'}, "
        f"{coherent_summary}, "
        f"multiphonon_max_order={multiphonon_max_order}"
        f"{' [auto-size]' if auto_multiphonon_order else ''}, jobs={max(1, int(num_jobs))}"
    , flush=True)
    from irma.core.noncubic_inelastic import run_noncubic_sab_inprocess

    if int(lat) == 1:
        grid_key = (
            int(lat),
            "alpha-beta-scaled",
            _grid_digest(alpha),
            _grid_digest(beta),
            float(awr),
        )
    else:
        grid_key = (
            int(lat),
            "physical-qe",
            _grid_digest(q_grid_ang_inv),
            _grid_digest(e_grid_mev),
        )

    context, _context_key = get_or_build_context(
        phonopy_yaml=phonopy_yaml, force_constants=force_constants,
        force_sets=force_sets, born=born, mesh_dim=mesh_dim,
        grid_key=grid_key, q_grid_ang_inv=q_grid_ang_inv,
        e_grid_mev=e_grid_mev, num_directions=num_directions,
        multiphonon_num_directions=multiphonon_num_directions,
        num_jobs=num_jobs,
        sigma_mev=sigma_mev, multiphonon_max_order=multiphonon_max_order,
        min_phonon_energy_mev=min_phonon_energy_mev,
        scattering_lengths_json=scattering_lengths_json,
        incoherent_cross_sections_json=incoherent_cross_sections_json,
        site_scattering_lengths_angstrom=site_scattering_lengths_angstrom,
        site_incoherent_cross_sections_barn=site_incoherent_cross_sections_barn,
        context_cache=context_cache, preloaded_full_mesh=preloaded_full_mesh)

    print("Launching noncubic SAB compute phase...", flush=True)
    result = run_noncubic_sab_inprocess(
        inelastic_mode=inelastic_mode,
        phonopy_yaml=phonopy_yaml,
        force_constants=force_constants,
        force_sets=force_sets,
        born_path=born,
        temperature_k=temperature_k,
        mesh=mesh_dim,
        q_grid_ang_inv=q_grid_ang_inv,
        e_grid_mev=e_grid_mev,
        output_prefix=workdir_path / "noncubic_sab_unused",
        sab_mass_ratio=awr,
        num_directions=num_directions,
        multiphonon_num_directions=multiphonon_num_directions,
        jobs=max(1, int(num_jobs)),
        sigma_mev=sigma_mev,
        multiphonon_max_order=multiphonon_max_order,
        auto_multiphonon_order=auto_multiphonon_order,
        min_phonon_energy_mev=min_phonon_energy_mev,
        represented_principal_site_count=represented_principal_site_count,
        principal_group_index=principal_group_index,
        site_groups=site_groups,
        coherent_partition_mode=coherent_partition_mode,
        sab_sigma_barn=sab_sigma_barn,
        site_scattering_lengths_angstrom=site_scattering_lengths_angstrom,
        site_incoherent_cross_sections_barn=site_incoherent_cross_sections_barn,
        scattering_lengths_json=scattering_lengths_json,
        incoherent_cross_sections_json=incoherent_cross_sections_json,
        context=context,
        write_output_files=False,
        precomputed_thermal_mats=precomputed_thermal_mats,
    )
    print("Noncubic SAB compute phase complete.", flush=True)

    output_arrays = result["output_arrays"]
    # Auto-sizing resolves the multiphonon order INSIDE compute_from_args, so the SAB-key
    # selection must follow the resolved (effective) order rather than the deck value: with
    # auto-sizing ON and a deck nphon < 2, the engine still produces multiphonon arrays and we
    # must select the multiphonon key, not the one-phonon-only key.
    effective_multiphonon_max_order = int(
        result.get("metadata", {}).get("multiphonon_max_order", multiphonon_max_order)
    )
    sab_key = _pick_sab_key(
        output_arrays,
        effective_multiphonon_max_order,
        inelastic_mode,
    )
    alpha_abs = np.asarray(output_arrays["alpha"], dtype=float)
    beta_downscatter_abs = np.asarray(output_arrays["beta_downscatter_abs"], dtype=float)
    sab_downscatter = np.asarray(output_arrays[sab_key], dtype=float)

    if not np.allclose(alpha_abs, alpha_abs_expected, rtol=1.0e-8, atol=1.0e-10):
        raise ValueError("Standalone alpha grid does not match the IRMA-requested physical alpha grid")
    if not np.allclose(beta_downscatter_abs, beta_downscatter_abs_expected, rtol=1.0e-8, atol=1.0e-10):
        raise ValueError("Standalone beta grid does not match the IRMA-requested physical beta grid")

    # The engine output orientation is contractually (alpha, beta): every
    # sab_* array is allocated (num_q, num_e) and convert_sqe_to_asym_downscatter_sab
    # is a shape-preserving scalar multiply. We assert that contract rather than
    # inferring orientation from the shape — shape inference is ambiguous on a
    # square (nalpha == nbeta) grid, where a transposed array would be silently
    # accepted as (alpha, beta) and re-transposed.
    expected_qe_shape = (len(alpha_abs), len(beta_downscatter_abs))
    if sab_downscatter.shape != expected_qe_shape:
        raise ValueError(
            f"Unexpected standalone SAB shape {sab_downscatter.shape}; the engine "
            f"contract is (alpha, beta) = {expected_qe_shape}"
        )
    sab_qe = sab_downscatter

    last_nonzero_beta = _last_nonzero_beta(beta_downscatter_abs, sab_qe)
    requested_beta_max = float(beta_downscatter_abs[-1]) if len(beta_downscatter_abs) else 0.0
    if (
        last_nonzero_beta is not None
        and requested_beta_max > 0.0
        and last_nonzero_beta < 0.95 * requested_beta_max
    ):
        print(
            "WARNING: Selected standalone SAB array "
            f"{sab_key} becomes identically zero above beta={last_nonzero_beta:.6f} "
            f"while IRMA requested beta_max={requested_beta_max:.6f}. "
            "This usually means multiphonon_max_order (Card 3 nphon) is too small "
            "for the requested grid — raise it or set Card 6g auto_order=1. "
            "THERMR's short-collision-time extension covers transfers beyond the "
            "tabulated law using the tape's effective temperature."
        , flush=True)

    ssm_internal = sab_qe.T

    return {
        "ssm_internal": ssm_internal,
        "sab_downscatter_qe": sab_qe,
        "alpha_abs": alpha_abs,
        "beta_downscatter_abs": beta_downscatter_abs,
        "stdout": "",
        "stderr": "",
        "selected_sab_key": sab_key,
        "last_nonzero_beta": last_nonzero_beta,
        # Surface the engine's anisotropic Debye-Waller state (thermal-displacement
        # matrices + primitive geometry) so consumers (e.g. the NCrystal exporter)
        # can build the anisotropic-DW elastic line from the SAME phonon
        # calculation as the inelastic kernel, with no second phonopy load.
        "elastic_state": result.get("elastic_state"),
    }
