"""Helpers for the in-process noncubic SAB workflow used by inelastic_mode=1/2."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from irma.core.noncubic_helpers import MULTIPHONON_MARGIN_SIGMAS
from irma.core.noncubic_inelastic import NoncubicInelasticControls

# The α↔Q/β↔E convention lives in one shared place (irma.core.sab_grids) so
# the ENDF/standalone path and the NCrystal exporter cannot drift; the
# private alias keeps this module's call sites terse.
from irma.core.sab_grids import irma_grid_to_physical_qe as _irma_grid_to_physical_qe

_CONTEXT_CACHE: dict[tuple[object, ...], dict[str, object]] = {}


def _pick_sab_key(multiphonon_max_order: int, inelastic_mode: int) -> str:
    """Engine output key of the S(alpha, beta) law for the mode and effective order."""
    base = {1: "incoherent_approx_n1_term", 2: "one_phonon_total"}[inelastic_mode]
    if multiphonon_max_order >= 2:
        base += "_plus_incoherent_approx_multiphonon"
    return "sab_asym_downscatter_" + base


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
                         min_phonon_energy_mev=0.0,
                         scattering_lengths_json=None,
                         incoherent_cross_sections_json=None,
                         site_scattering_lengths_angstrom=None,
                         site_incoherent_cross_sections_barn=None,
                         context_cache=None, preloaded_full_mesh=None):
    """Return the cached compute context for these inputs, or build it.

    The cache keeps the current context and, under a key of the model inputs
    alone, its model layer (phonopy load, mesh eigensolves, star-averaged
    projections; see ``build_model_context``). A lat=0 multi-temperature deck
    has new Q/E grids at every temperature, so it misses the context but
    reuses the model layer. ``preloaded_full_mesh`` skips the duplicate
    full-mesh eigensolve on a build.
    """
    from irma.core.noncubic_inelastic_context import build_compute_context

    input_ids = _model_input_identities(phonopy_yaml, force_constants,
                                        force_sets, born)
    mesh_key = tuple(int(x) for x in mesh_dim)
    model_key = ("__noncubic_model__", input_ids, mesh_key,
                 float(min_phonon_energy_mev))
    context_key = (
        input_ids,
        mesh_key,
        grid_key,
        int(num_directions),
        int(multiphonon_num_directions),
        max(1, int(num_jobs)),
        float(min_phonon_energy_mev),
        scattering_lengths_json,
        incoherent_cross_sections_json,
        None if site_scattering_lengths_angstrom is None else _grid_digest(
            site_scattering_lengths_angstrom),
        None if site_incoherent_cross_sections_barn is None else _grid_digest(
            site_incoherent_cross_sections_barn),
    )
    cache = _CONTEXT_CACHE if context_cache is None else context_cache
    context = cache.get(context_key)
    if context is None:
        args_context = SimpleNamespace(
            phonopy_yaml=str(phonopy_yaml),
            force_constants=None if force_constants is None else str(force_constants),
            force_sets=None if force_sets is None else str(force_sets),
            born=None if born is None else str(born),
            mesh=list(mesh_key),
            num_directions=int(num_directions),
            scattering_lengths_json=scattering_lengths_json,
            scattering_lengths_file=None,
            incoherent_cross_sections_json=incoherent_cross_sections_json,
            incoherent_cross_sections_file=None,
            site_scattering_lengths_angstrom=site_scattering_lengths_angstrom,
            site_incoherent_cross_sections_barn=site_incoherent_cross_sections_barn,
            jobs=max(1, int(num_jobs)),
            q_chunk_size=20000,
            multiphonon_num_directions=int(multiphonon_num_directions),
            min_phonon_energy_mev=float(min_phonon_energy_mev),
        )
        context = build_compute_context(args_context, q_grid_ang_inv, e_grid_mev,
                                        preloaded_full_mesh=preloaded_full_mesh,
                                        model_context=cache.get(model_key))
        cache[context_key] = context
        if context.get("_model_context") is not None:
            cache[model_key] = context["_model_context"]
    else:
        print("Reusing cached noncubic inelastic context.", flush=True)
    # Keep only the current context and its model layer: reuse within a run
    # is sequential, and one context can hold O(100 MB).
    for stale in [k for k in cache if k not in (context_key, model_key)]:
        del cache[stale]
    return context


def _last_nonzero_beta(beta_downscatter_abs: np.ndarray, sab_qe: np.ndarray) -> float | None:
    """Largest beta at which some alpha row of an ``(alpha, beta)`` array is positive."""
    nz = np.flatnonzero(np.max(sab_qe, axis=0) > 0.0)
    return float(beta_downscatter_abs[nz[-1]]) if nz.size else None


def _truncation_warning(sab_key, beta_downscatter_abs, sab_qe, needed_beta):
    """Warning text when the computed law ends before the reach it needs, else None.

    ``needed_beta`` is the engine's needed multiphonon reach, the recoil ridge
    at the largest Q plus the margin widths capped at the grid top, in the
    grid's own kT units. Zeros beyond it are the negligible tail of the law,
    not a truncation, so the grid top is the wrong reference. The comparison
    uses the last grid point at or below the needed reach, so a coarse grid
    step past the reach cannot raise a false alarm.
    """
    last = _last_nonzero_beta(beta_downscatter_abs, sab_qe)
    needed = float(needed_beta or 0.0)
    if last is None or not needed > 0.0:
        return None
    beta = np.asarray(beta_downscatter_abs, dtype=float)
    covered = beta[beta <= needed * (1.0 + 1.0e-12)]
    if covered.size == 0 or last >= covered[-1]:
        return None
    return (
        f"Selected standalone SAB array {sab_key} becomes identically zero "
        f"above beta={last:.6f}, short of beta={needed:.6f}, the recoil ridge "
        f"at the largest Q plus {MULTIPHONON_MARGIN_SIGMAS:g} thermal widths, "
        f"below which the law still has weight, so the computed array was cut "
        f"short. Check the multiphonon order warnings above; if the order "
        f"meets the requirement, this points to a fault in the calculation "
        f"worth reporting.")


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
    auto_multiphonon_order = bool(controls.auto_multiphonon_order)
    min_phonon_energy_mev = float(controls.min_phonon_energy_mev)
    num_directions = int(controls.num_directions)
    multiphonon_num_directions = int(controls.multiphonon_num_directions)

    q_grid_ang_inv, e_grid_mev, _, _ = _irma_grid_to_physical_qe(
        alpha, beta, lat, temperature_k, awr)

    print(
        "Starting in-process noncubic SAB driver: "
        f"mode={inelastic_mode}, mesh={tuple(int(x) for x in mesh_dim)}, "
        f"nac={'on (' + born.name + ')' if born is not None else 'off'}, "
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

    context = get_or_build_context(
        phonopy_yaml=phonopy_yaml, force_constants=force_constants,
        force_sets=force_sets, born=born, mesh_dim=mesh_dim,
        grid_key=grid_key, q_grid_ang_inv=q_grid_ang_inv,
        e_grid_mev=e_grid_mev, num_directions=num_directions,
        multiphonon_num_directions=multiphonon_num_directions,
        num_jobs=num_jobs,
        min_phonon_energy_mev=min_phonon_energy_mev,
        scattering_lengths_json=scattering_lengths_json,
        incoherent_cross_sections_json=incoherent_cross_sections_json,
        site_scattering_lengths_angstrom=site_scattering_lengths_angstrom,
        site_incoherent_cross_sections_barn=site_incoherent_cross_sections_barn,
        context_cache=context_cache, preloaded_full_mesh=preloaded_full_mesh)

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
        sab_mass_ratio=awr,
        num_directions=num_directions,
        multiphonon_num_directions=multiphonon_num_directions,
        jobs=max(1, int(num_jobs)),
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

    output_arrays = result["output_arrays"]
    # Auto-sizing resolves the multiphonon order INSIDE compute_from_args, so the SAB-key
    # selection must follow the resolved (effective) order rather than the deck value: with
    # auto-sizing ON and a deck nphon < 2, the engine still produces multiphonon arrays and we
    # must select the multiphonon key, not the one-phonon-only key.
    effective_multiphonon_max_order = int(
        result.get("metadata", {}).get("multiphonon_max_order", multiphonon_max_order)
    )
    sab_key = _pick_sab_key(effective_multiphonon_max_order, inelastic_mode)
    beta_downscatter_abs = np.asarray(output_arrays["beta_downscatter_abs"], dtype=float)
    sab_qe = np.asarray(output_arrays[sab_key], dtype=float)

    truncation_warning = _truncation_warning(
        sab_key, beta_downscatter_abs, sab_qe,
        result.get("metadata", {}).get("needed_multiphonon_beta_support"))
    if truncation_warning:
        print(f"WARNING: {truncation_warning}", flush=True)

    return {
        "ssm_internal": sab_qe.T,
        "sab_downscatter_qe": sab_qe,
        "alpha_abs": np.asarray(output_arrays["alpha"], dtype=float),
        "beta_downscatter_abs": beta_downscatter_abs,
        "selected_sab_key": sab_key,
        # Surface the engine's anisotropic Debye-Waller state (thermal-displacement
        # matrices + primitive geometry) so consumers (e.g. the NCrystal exporter)
        # can build the anisotropic-DW elastic line from the SAME phonon
        # calculation as the inelastic kernel, with no second phonopy load.
        "elastic_state": result.get("elastic_state"),
    }
