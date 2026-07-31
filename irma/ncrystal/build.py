"""Orchestrator: IRMA mode-2 → per-principal ``.irmapack`` set.

``build_packs(cfg)`` is the one public entry. For each principal scatterer (one
per chemical species; graphite → 1, BeO → 2) it runs the in-repo standalone
mode-2 SAB driver, converts to the pack convention, attaches the IRMA-native
anisotropic-DW elastic line, stamps provenance, and returns the packs plus the
``@CUSTOM_IRMA`` NCMAT snippet that wires them into a material.

This reimplements zero physics — the S(α,β) law is exactly what
``run_noncubic_standalone_sab`` returns (the same engine output that feeds the
NJOY tape). IRMA is the single producer and reference.
"""
from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np

from irma.core.constants import AMASSN
from irma.core.grids import (
    generate_alpha_grid, generate_beta_grid_for_iint,
    grid_reference_temperature_K)
from .config import NCrystalExportConfig
from .convert import pack_from_irma_sab
from .ncmat import assemble_material_ncmat
from .pack import IRMAPack, write_pack
from .provenance import collect_provenance


# fm → Angstrom for the engine's per-site coherent scattering length input.
_FM_TO_ANGSTROM = 1.0e-5


@dataclasses.dataclass
class PrincipalGroup:
    """One principal scatterer: a species and the primitive sites it owns."""
    symbol: str
    site_indices: list[int]
    mass_amu: float
    awr: float
    sigma_bound_b: float
    b_coh_fm: float
    sigma_inc_b: float


def _load_mesh_and_freq_max_eV(mat):
    """Load the phonopy mesh the engine resolves (same force constants + NAC)
    and return ``(freq_max_eV, mesh_data)``.

    Single source for the automatic grid's phonon-spectrum top: the max mesh
    frequency in eV (negative entries are imaginary modes near gamma; max()
    picks the highest real optical mode). Surfaces the dynamical-instability
    warning the ENDF driver emits on its mesh, so a pack is never baked from
    an unstable model silently."""
    from irma.core.phonopy_io import load_phonopy_mesh, warn_dynamic_instability
    mesh_data = load_phonopy_mesh(
        str(mat.phonopy_yaml), tuple(int(m) for m in mat.mesh), born_path=mat.born)
    warn_dynamic_instability(mesh_data.frequencies_ev, mesh_data.qpoints)
    freq_max = float(np.max(np.asarray(mesh_data.frequencies_ev, dtype=float)))
    return freq_max, mesh_data


def _estimate_freq_max_eV(mat) -> float:
    """Max phonon frequency [eV] for the automatic grid (thin wrapper)."""
    return _load_mesh_and_freq_max_eV(mat)[0]


def _auto_beta_grid(cfg, t_ref, recoil_awr):
    """Build the shared converged beta grid (generate_beta_grid_for_iint,
    lin-lin) AND return a phonopy ``Mesh`` to reuse downstream.

    NCrystal's SAB kernel always interpolates S(alpha,beta) LINEARLY in beta
    (NCSABEval LINLIN), so the baked grid must carry the lin-lin (iint=1)
    step cap below the recoil ridge (DELTA_BETA_MAX_LINLIN) — the same
    treatment every other auto-grid caller uses (the GUI deck writers and
    irma.mlip.emit). A pure-log upper tail would let a linearly-interpolated
    law overshoot the free-atom cross section at high incident energy.
    ``recoil_awr`` must be the SMALLEST principal-species AWR in the export:
    the ridge beta = 4*beta_max/awr sits highest for the lightest species, so
    capping up to its ridge keeps the one shared grid safe for every species.

    Returns ``(beta, preloaded_full_mesh_or_None)``. The mesh is loaded ONCE, and
    only when freq_max must be estimated -- returning it as ``preloaded_full_mesh``
    lets every per-species engine call skip a duplicate full-mesh eigensolve
    (the mesh is material-level, identical for all species). With freq_max pinned
    no mesh is loaded and the preload is None."""
    preloaded = None
    if cfg.freq_max_eV:
        freq_max = float(cfg.freq_max_eV)
    else:
        freq_max, mesh_data = _load_mesh_and_freq_max_eV(cfg.material)
        preloaded = mesh_data.phonopy_mesh_object
    beta = generate_beta_grid_for_iint(
        freq_max, t_ref, iint=1, awr=float(recoil_awr),
        n_lower=cfg.n_lower, n_phonon=cfg.n_phonon,
        n_upper=cfg.n_upper, beta_max_eV=cfg.beta_max_eV)
    return beta, preloaded


def _group_grids(cfg, awr, auto_beta, t_ref):
    """The (alpha, beta) grid for one principal species. EXPLICIT -> the config
    grids. AUTOMATIC -> the converged ENDF-style grid: generate_alpha_grid
    (linear in Q at alpha_dq_invA up to alpha_qcut_invA, then a log tail) over the
    shared ``auto_beta``. alpha is awr-dependent (per species); beta is shared."""
    if cfg.grid_mode == "explicit":
        return list(cfg.alpha_grid), list(cfg.beta_grid)
    alpha = generate_alpha_grid(
        auto_beta, float(awr), t_ref, dq_ang_inv=cfg.alpha_dq_invA,
        q_cut_ang_inv=cfg.alpha_qcut_invA, n_log=cfg.alpha_nlog)
    return alpha.tolist(), auto_beta.tolist()


# -- primitive-cell geometry --------------------------------------------------
def load_primitive_info(phonopy_yaml: str | Path, born: str | Path | None = None):
    """Return ``(symbols, masses_amu, scaled_positions, lattice)`` for the
    phonopy primitive cell, without a mesh eigensolve.

    Symbols/positions/masses are read straight from the phonopy primitive so the
    exporter groups sites by species exactly as the engine sees them.
    """
    import inspect

    from phonopy import load as phonopy_load

    # Route this load through the same C-backend guard as the mesh loaders
    # (phonopy_io.load_phonopy_mesh, noncubic_inelastic_context.build_model_context)
    # so all phonopy loading uses one pattern. This call only reads primitive
    # geometry (no eigensolve, so no rayon threadpool is spun up), but pinning the
    # C backend keeps the loading path uniform and fork-safe on phonopy>=4. Only
    # pass `lang` when this phonopy accepts it.
    backend_kwargs = (
        {"lang": "C"}
        if "lang" in inspect.signature(phonopy_load).parameters
        else {}
    )
    # isolated_phonopy_cwd: only primitive GEOMETRY is consumed here (symbols,
    # masses, positions, lattice -- none of which depend on force constants or
    # NAC), but phonopy.load still probes the process cwd for FORCE_SETS/
    # FORCE_CONSTANTS/BORN as a fallback. Pinning to an empty scratch dir keeps
    # the invariant every other phonopy load in the tree holds: what gets
    # loaded never depends on where IRMA runs. Absolutize first, as the pin
    # changes the cwd relative paths would resolve against.
    from irma.core.phonopy_io import (
        isolated_phonopy_cwd,
        pinned_primitive_matrix_kwargs,
        reject_unsafe_phonopy_yaml,
    )
    import os as _os
    phonopy_yaml_abs = _os.path.abspath(str(phonopy_yaml))
    born_abs = _os.path.abspath(str(born)) if born is not None else None
    # TRUST BOUNDARY (SEC-1): refuse a phonopy.yaml carrying code-executing
    # YAML tags BEFORE phonopy's unsafe loader parses it below.
    reject_unsafe_phonopy_yaml(phonopy_yaml_abs)
    with isolated_phonopy_cwd():
        ph = phonopy_load(phonopy_yaml_abs, log_level=0,
                          is_nac=born is not None,
                          born_filename=born_abs,
                          **pinned_primitive_matrix_kwargs(phonopy_yaml_abs),
                          **backend_kwargs)
    prim = ph.primitive
    if hasattr(prim, "get_chemical_symbols"):
        symbols = list(prim.get_chemical_symbols())
        masses = np.asarray(prim.get_masses(), float)
        positions = np.asarray(prim.get_scaled_positions(), float)
        lattice = np.asarray(prim.get_cell(), float)
    else:
        symbols = list(prim.symbols)
        masses = np.asarray(prim.masses, float)
        positions = np.asarray(prim.scaled_positions, float)
        lattice = np.asarray(prim.cell, float)
    return symbols, masses, positions, lattice


def site_groups_by_species(symbols: list[str]) -> list[list[int]]:
    """Group primitive-site indices by chemical species, first-appearance order.

    Matches the ENDF convention (one principal scatterer per species) and the
    BeO reference example (``[[0,1],[2,3]]``). One group per distinct symbol.
    """
    order: list[str] = []
    groups: dict[str, list[int]] = {}
    for idx, sym in enumerate(symbols):
        if sym not in groups:
            groups[sym] = []
            order.append(sym)
        groups[sym].append(idx)
    return [groups[sym] for sym in order]


def resolve_principal_groups(cfg: NCrystalExportConfig):
    """Resolve principal groups + per-site neutron arrays + cell geometry.

    Returns ``(groups, site_groups, site_b_coh_angstrom, site_sigma_inc_barn,
    symbols, scaled_positions, lattice_ang)`` — the last two are the phonopy
    primitive cell, used to write a matching base ``.ncmat``. Each species in the
    structure must have a matching scatterer in
    ``material.scatterers`` carrying at least ``sigma_bound_b`` and ``awr`` (or a
    mass-derived awr); ``b_coh_fm``/``sigma_inc_b`` are required for the elastic
    line and the coherent partition.
    """
    mat = cfg.material
    symbols, masses, scaled_positions, lattice = load_primitive_info(
        mat.phonopy_yaml, mat.born)
    site_groups = ([[int(i) for i in g] for g in cfg.site_groups]
                   if cfg.site_groups is not None
                   else site_groups_by_species(symbols))

    by_symbol = {s.symbol: s for s in mat.scatterers}
    n_atoms = len(symbols)
    # Explicit site_groups must cover every primitive site exactly once: an out-of-
    # range index would crash (or, negative, silently pick the last atom = wrong
    # species), and omitting sites would silently drop a species' scattering while the
    # per-atom fraction (len(site_indices)/n_atoms) still divides by ALL atoms ->
    # under-normalized cross sections. The default grouping always covers all sites.
    if cfg.site_groups is not None:
        flat = [i for g in site_groups for i in g]
        if any(not (0 <= i < n_atoms) for i in flat):
            raise ValueError(
                f"explicit site_groups index out of range [0, {n_atoms}); got {flat}")
        if sorted(flat) != list(range(n_atoms)):
            raise ValueError(
                f"explicit site_groups must cover every primitive site exactly once "
                f"(0..{n_atoms - 1}); got sites {sorted(flat)}. Omitted sites are silently "
                "dropped while per-atom fractions still divide by all atoms.")
    site_b_coh_ang = [0.0] * n_atoms
    site_sigma_inc = [0.0] * n_atoms
    groups: list[PrincipalGroup] = []
    for g in site_groups:
        sym = symbols[g[0]]
        if any(symbols[i] != sym for i in g):
            raise ValueError(
                f"site group {g} mixes species; groups must be single-species "
                "(one principal scatterer per species)")
        sc = by_symbol.get(sym)
        if sc is None:
            raise ValueError(
                f"material.scatterers has no entry for species {sym!r} "
                f"(structure species: {sorted(set(symbols))})")
        # (the reverse mismatch -- a scatterer row for a species the
        # structure does not contain -- is checked once after the loop)
        if sc.sigma_bound_b is None:
            raise ValueError(f"scatterer {sym!r} is missing sigma_bound_b")
        mass = float(np.mean([masses[i] for i in g]))
        awr = float(sc.awr) if sc.awr is not None else mass / AMASSN
        # The neutron constants are a hard requirement REGARDLESS of
        # cfg.elastic (the config validates YAML paths; this guards
        # direct-construction callers): the inelastic engine derives its
        # channel weights from them, so substituting 0.0 bakes an identically
        # zero S(alpha,beta) while bound_xs advertises real physics
        # (review NC-1).
        if sc.b_coh_fm is None or sc.sigma_inc_b is None:
            missing = [f for f in ("b_coh_fm", "sigma_inc_b")
                       if getattr(sc, f) is None]
            raise ValueError(
                f"scatterer {sym!r} is missing {', '.join(missing)}: the "
                "mode-1/2 inelastic engine derives its channel weights from "
                "b_coh_fm and sigma_inc_b, so both are required even for an "
                "inelastic-only pack (elastic=False)")
        b_coh_fm = float(sc.b_coh_fm)
        sigma_inc_b = float(sc.sigma_inc_b)
        for i in g:
            site_b_coh_ang[i] = b_coh_fm * _FM_TO_ANGSTROM
            site_sigma_inc[i] = sigma_inc_b
        groups.append(PrincipalGroup(
            symbol=sym, site_indices=[int(i) for i in g], mass_amu=mass, awr=awr,
            sigma_bound_b=float(sc.sigma_bound_b), b_coh_fm=b_coh_fm,
            sigma_inc_b=sigma_inc_b))
    # One principal group per species: an explicit site_groups that splits a
    # species across >1 group yields groups with the SAME symbol, which collide
    # on the symbol-derived pack filename (one silently overwrites the other,
    # possibly the elastic-bearing one) and break the per-species summation.
    # A scatterer row for a species the structure does not contain used to
    # be SILENTLY ignored (review NC-2) -- a typo'd symbol meant the intended
    # species ran with defaults while the user believed their constants were
    # in effect.
    configured = {s.symbol for s in cfg.material.scatterers}
    structural = set(symbols)
    extra = sorted(configured - structural)
    if extra:
        raise ValueError(
            f"material.scatterers lists species {extra} that the phonopy "
            f"structure does not contain (structure species: "
            f"{sorted(structural)}); remove the row(s) or fix the symbol")
    syms = [g.symbol for g in groups]
    if len(set(syms)) != len(syms):
        dups = sorted({s for s in syms if syms.count(s) > 1})
        raise ValueError(
            f"explicit site_groups split species {dups} across multiple groups; "
            "the per-species pack-summation contract requires exactly one "
            "principal group per species (pack filenames derive from the species "
            "symbol and would collide, silently overwriting a pack). Merge each "
            "species into a single group.")
    return (groups, site_groups, site_b_coh_ang, site_sigma_inc, symbols,
            scaled_positions, lattice)


def _coherent_bearing_index(groups: list[PrincipalGroup]) -> int:
    """Which principal pack carries the (whole-structure) coherent Bragg comb.

    The coherent structure factor F(hkl)=Σ_sites b_coh·e^{-W}·e^{iφ} is a single
    crystal-wide quantity; to avoid double-counting when packs are summed, the
    full comb lives in exactly ONE pack — the species with the largest coherent
    weight n·b_coh² — and the others contribute only their incoherent DW line.
    (Monoatomic → the single pack.)  See the overnight decision log: the
    alternative is a per-species principal-xs-weighted split.
    """
    weights = [len(g.site_indices) * g.b_coh_fm ** 2 for g in groups]
    return int(np.argmax(weights)) if weights else 0


def build_packs(cfg: NCrystalExportConfig, *, pack_path_prefix=None,
                progress=print) -> tuple[list[IRMAPack], str]:
    """Build the per-principal pack set + a complete, loadable ``.ncmat``.

    Returns ``(packs, material_ncmat_text)``. The NCMAT carries the phonopy
    primitive cell (so the pack's anisotropic-DW tensor positions match the
    material's atom sites exactly) plus the ``@CUSTOM_IRMA`` section.

    ``pack_path_prefix`` controls how the ``@CUSTOM_IRMA`` section references
    the pack files: ``None`` -> bare relative names; a directory -> ``<dir>/<name>``
    (NCrystal resolves the pack path via ``createTextData`` against CWD / its data
    path, so :func:`write_packs` passes the absolute output dir to make the NCMAT
    loadable from anywhere).
    """
    # Pin native (BLAS/OMP) threads to 1 up front, EXPLICITLY -- exactly as the
    # engine's compute_from_args does. The auto-grid path runs a phonopy mesh
    # eigensolve IN THIS (parent) process before the engine's jobs>1 fork; a
    # multi-threaded BLAS pool spun up there would leave idle pthreads that the
    # fork inherits (the fork-after-threads deadlock hazard). Do not rely on the
    # import-time setdefault firing first or on threadpoolctl being installed.
    from irma.core.noncubic_workers import limit_native_threads_to_one
    limit_native_threads_to_one()

    if cfg.gain_side == "asym":
        # The full-asymmetric ('sab') table is a deferred opt-in (see the design
        # spec); the current path always bakes the validated scaled-symmetric
        # half-table. Fail loudly rather than silently ignoring the request.
        raise NotImplementedError(
            "gain_side='asym' (full-asymmetric S table) is not implemented yet; "
            "use the default gain_side='scaled_sym' (downscatter half-table; "
            "NCrystal reconstructs the gain side by detailed balance).")
    (groups, site_groups, site_b_coh_ang, site_sigma_inc, symbols,
     scaled_positions, lattice) = resolve_principal_groups(cfg)
    # 'exact-total' gives every per-principal pack the WHOLE-crystal coherent
    # one-phonon total; with >1 group, summing the packs over-counts that channel
    # by the number of species. Only valid for a single principal group.
    if cfg.coherent_partition_mode == "exact-total" and len(groups) > 1:
        raise ValueError(
            f"coherent_partition_mode='exact-total' is only correct for a single "
            f"principal group, but this export resolves to {len(groups)} "
            f"({[g.symbol for g in groups]}): each pack would carry the whole-crystal "
            "coherent one-phonon total and summing them double-counts it. Use "
            "'principal-xs-weighted' (default) or 'auto'.")
    coh_index = _coherent_bearing_index(groups)
    n_atoms = len(symbols)

    # symbol -> (coherent scattering length [sqrt(barn) = b_coh_fm/10, NCrystal's
    # coherentScatLen() unit], sigma_inc [barn]). The coherent-bearing pack stamps
    # these onto each tensor site so the C++ coherent F(hkl) + incoherent DW use the
    # config-specified neutron data instead of NCrystal's atom DB. Keyed per-symbol,
    # mirroring resolve_principal_groups' `by_symbol` (one scatterer per species): the
    # engine's inelastic SAB and this map therefore use the SAME b_coh/sigma_inc, so
    # the C++ elastic is consistent with the engine (not a third atom-DB value).
    neutron_by_symbol = {g.symbol: (g.b_coh_fm / 10.0, g.sigma_inc_b) for g in groups}

    # Automatic grid: build the SAME converged beta grid the ENDF evaluator uses
    # with the lin-lin (iint=1) treatment — NCrystal interpolates linearly in
    # beta, so the tail carries the DELTA_BETA_MAX_LINLIN recoil-ridge cap. The
    # grid is shared across species (the per-species alpha is generated in the
    # loop), so the cap is sized to the LIGHTEST species' recoil ridge — the
    # highest one — which covers every heavier species too. lat=1 anchors the
    # grid units at THERM=0.0253 eV (grid_reference_temperature_K), exactly as
    # the deck writer does. Skipped for an explicit grid so an explicit export
    # needs no extra phonopy mesh pass.
    t_ref = grid_reference_temperature_K(cfg.lat, float(cfg.material.temperature_K))
    auto_beta, preloaded_mesh = (
        (None, None) if cfg.grid_mode == "explicit"
        else _auto_beta_grid(cfg, t_ref, min(g.awr for g in groups)))

    packs: list[IRMAPack] = []
    # Geometry + dynamics for the NCMAT: prefer the EXACT positions/lattice the
    # coherent-bearing pack baked its DW tensors from (guarantees the 1e-6
    # position match in the plugin); fall back to the phonopy primitive cell for
    # an inelastic-only export. The engine's thermal-displacement matrices give
    # the per-element Debye temperature NCrystal needs to construct the crystal.
    # The whole-crystal elastic line (coherent Bragg comb over ALL sites + the
    # full incoherent DW) is carried by exactly ONE pack — the coherent-bearing
    # principal — and the other packs are inelastic-only. This is verified
    # double-count-free end-to-end on BeO: the C++ builds F(hkl) from
    # the pack's full tensor set + NC::Info b_coh, so a single pack already
    # captures the cross-species interference; spreading the tensors across packs
    # would double-count the coherent line (the C++ does not weight the coherent
    # tensor path by elastic_scale).
    coh_elastic_state = None
    for gi, group in enumerate(groups):
        # alpha is awr-dependent (per species) in the auto grid; explicit grids
        # are shared. beta is one shared grid either way (the auto tail's
        # lin-lin cap is already sized to the lightest species above).
        alpha, beta = _group_grids(cfg, group.awr, auto_beta, t_ref)
        progress(f"[irma.ncrystal] principal scatterer {gi + 1}/{len(groups)}: "
                 f"{group.symbol} (sites {group.site_indices}; grid {cfg.grid_mode}, "
                 f"{len(alpha)} alpha x {len(beta)} beta)")
        pack, elastic_state = _build_pack_for_group(
            cfg, gi, group, site_groups, site_b_coh_ang, site_sigma_inc, n_atoms,
            alpha, beta, attach_elastic=(gi == coh_index), progress=progress,
            preloaded_full_mesh=preloaded_mesh, neutron_by_symbol=neutron_by_symbol)
        packs.append(pack)
        if gi == coh_index and elastic_state is not None:
            coh_elastic_state = elastic_state

    pack_names = [f"{cfg.material_id}__{g.symbol}.irmapack" for g in groups]
    pack_filenames = ([str(Path(pack_path_prefix) / n) for n in pack_names]
                      if pack_path_prefix is not None else pack_names)
    masses_by_symbol = {g.symbol: g.mass_amu for g in groups}
    if coh_elastic_state is not None:
        pos = np.asarray(coh_elastic_state["primitive_scaled_positions"], float)
        syms = list(coh_elastic_state["primitive_symbols"])
        lat = np.asarray(coh_elastic_state["primitive_lattice_ang"], float)
    else:
        pos, syms, lat = scaled_positions, symbols, lattice
    debye_temps = _debye_temperatures(
        coh_elastic_state, syms, masses_by_symbol, float(cfg.material.temperature_K))
    material_ncmat = assemble_material_ncmat(
        lattice_ang=lat, scaled_positions=pos, symbols=syms,
        pack_filenames=pack_filenames, debye_temperatures=debye_temps)
    return packs, material_ncmat


def _debye_temperatures(elastic_state, symbols, masses_by_symbol, temperature_K):
    """Per-element Debye temperature from the engine's mean-squared displacements.

    Averages each element's site MSDs (Tr(U)/3) and maps to a Debye temperature
    so the NCMAT has a valid MSD source. Falls back to 300 K per element when no
    elastic state is available.
    """
    from .ncmat import debye_temperature_from_msd
    out: dict[str, float] = {}
    if elastic_state is None:
        return {s: 300.0 for s in dict.fromkeys(symbols)}
    U = np.asarray(elastic_state["thermal_displacement_matrices_ang2"], float)
    es_symbols = list(elastic_state["primitive_symbols"])
    msd_site = (U[:, 0, 0] + U[:, 1, 1] + U[:, 2, 2]) / 3.0
    for sym in dict.fromkeys(es_symbols):
        idx = [i for i, s in enumerate(es_symbols) if s == sym]
        msd = float(np.mean(msd_site[idx])) if idx else 0.0
        mass = float(masses_by_symbol.get(sym, 0.0))
        out[sym] = debye_temperature_from_msd(msd, mass, temperature_K)
    return out


def _build_pack_for_group(cfg, group_index, group, site_groups, site_b_coh_ang,
                          site_sigma_inc, n_atoms, alpha, beta,
                          *, attach_elastic, progress, preloaded_full_mesh=None,
                          neutron_by_symbol=None):
    """Build one principal pack; returns ``(pack, elastic_state)`` (the engine's
    anisotropic-DW state, or ``None`` if elastic was not computed). ``attach_elastic``
    decides whether this pack carries the (whole-crystal) elastic block."""
    from irma.core.noncubic_inelastic import NoncubicInelasticControls
    from irma.core.standalone_sab import run_noncubic_standalone_sab

    mat = cfg.material
    controls = NoncubicInelasticControls(
        num_directions=cfg.num_directions,
        multiphonon_num_directions=cfg.multiphonon_num_directions,
        multiphonon_max_order=cfg.effective_multiphonon_max_order,
        auto_multiphonon_order=cfg.auto_multiphonon_order)

    result = run_noncubic_standalone_sab(
        alpha=np.asarray(alpha, float),
        beta=np.asarray(beta, float),
        lat=int(cfg.lat),
        temperature_k=float(mat.temperature_K),
        awr=float(group.awr),
        phonopy_yaml_path=str(mat.phonopy_yaml),
        mesh_dim=tuple(int(m) for m in mat.mesh),
        born_path=mat.born,
        num_jobs=int(cfg.jobs) if cfg.jobs else 1,
        sigma_mev=0.0,
        controls=controls,
        inelastic_mode=int(cfg.inelastic_mode),
        represented_principal_site_count=len(group.site_indices),
        principal_group_index=int(group_index),
        site_groups=site_groups,
        coherent_partition_mode=str(cfg.coherent_partition_mode),
        sab_sigma_barn=float(group.sigma_bound_b),
        site_scattering_lengths_angstrom=site_b_coh_ang,
        site_incoherent_cross_sections_barn=site_sigma_inc,
        preloaded_full_mesh=preloaded_full_mesh,
        workdir=None)

    # The SAB table keeps the engine's full-sigma normalization
    # (sab_sigma_barn = group.sigma_bound_b); the per-atom weighting for a
    # multi-species material is applied ONLY through the pack's advertised
    # bound_xs (atom_fraction * sigma_bound_b, below). Do NOT insert a
    # rescale_sab_to_bound_xs call here: with source == target it is a
    # no-op whose provenance line would claim a transform that never
    # happened; the helper exists for genuine convention changes only.
    sab_downscatter = result["sab_downscatter_qe"]
    atom_fraction = len(group.site_indices) / float(n_atoms)

    meta = collect_provenance(
        phonopy_yaml=mat.phonopy_yaml, mesh=mat.mesh,
        temperature_K=mat.temperature_K, num_directions=cfg.num_directions,
        multiphonon_num_directions=cfg.multiphonon_num_directions,
        multiphonon_max_order=cfg.multiphonon_max_order,
        inelastic_mode=cfg.inelastic_mode,
        born=mat.born,
        extra={
            "principal_symbol": group.symbol,
            "principal_group_index": str(group_index),
            "coherent_partition_mode": cfg.coherent_partition_mode,
            # The table keeps the engine's full-sigma normalization; the
            # species atom fraction below is folded into the advertised
            # bound_xs, so bound_xs = atom_fraction * sigma_bound_b.
            "atom_fraction": f"{atom_fraction:.17g}",
            "irma_selected_sab_key": str(result.get("selected_sab_key")),
            "grid_mode": cfg.grid_mode,
            "grid_spec": (
                "explicit" if cfg.grid_mode == "explicit"
                else (f"auto(ENDF): freq_max_eV="
                      f"{cfg.freq_max_eV if cfg.freq_max_eV else 'phonopy'} "
                      f"n_lower/n_phonon/n_upper={cfg.n_lower}/{cfg.n_phonon}/"
                      f"{cfg.n_upper} beta_max_eV={cfg.beta_max_eV} "
                      f"alpha_dq={cfg.alpha_dq_invA}/qcut={cfg.alpha_qcut_invA}/"
                      f"nlog={cfg.alpha_nlog}")),
        })

    # Per-atom normalization: NCrystal cross sections are PER ATOM, but the C++
    # plugin sums the per-principal packs at weight 1.0, so a naive multi-species
    # material reads per-formula-unit (~N_atoms x too high). Scale this pack's
    # inelastic bound_xs by the species atom fraction (sites in this group / total
    # cell atoms) so sum_i f_i*sigma_inel,i = per-atom-average. The coherent F(hkl)
    # is already a per-atom whole-crystal quantity (built in C++ from b_coh + the
    # full cell), so only the inelastic bound_xs is scaled. Monatomic -> fraction=1.
    # (atom_fraction computed above, next to the normalization note.)
    pack = pack_from_irma_sab(
        material_id=f"{cfg.material_id}__{group.symbol}",
        temperature_K=float(mat.temperature_K),
        bound_xs_barn=atom_fraction * float(group.sigma_bound_b),
        element_mass_amu=float(group.mass_amu),
        alpha_mass_ratio=float(group.awr),
        alpha_grid=result["alpha_abs"],
        beta_downscatter_abs=result["beta_downscatter_abs"],
        sab_asym_downscatter=sab_downscatter,
        metadata=meta)

    # Elastic block: the coherent-bearing pack holds the FULL primitive-cell
    # anisotropic U-tensor set, from which the C++ plugin builds the coherent
    # structure factor F(hkl) over all sites (so cross-species interference is
    # exact) AND the per-site incoherent DW. The full physical elastic line
    # (both channels) is always emitted — isolate a channel at scatter time via
    # NCrystal's comp=coh_elas / comp=incoh_elas if needed.
    elastic_state = result.get("elastic_state")
    if cfg.elastic and attach_elastic:
        _attach_elastic(pack, elastic_state, n_atoms, neutron_by_symbol or {},
                        progress=progress,
                        incoherent_elastic_mode=cfg.incoherent_elastic_mode)
    return pack, elastic_state


def _attach_elastic(pack, elastic_state, n_atoms, neutron_by_symbol, *, progress,
                    incoherent_elastic_mode="isotropic") -> None:
    """Populate the (whole-crystal) elastic block from the engine's
    anisotropic-DW state — structure mode: full primitive-cell U tensors, no
    scalar MSD / incoherent-xs (the C++ derives both channels from the tensors +
    NC::Info per-species data). Both coherent and incoherent are always enabled.
    ``incoherent_elastic_mode = 'directional'`` is carried as a pack field
    (``incoherent_elastic_mode``, within the schema-2 format) so the C++ samples
    the orientation-averaged incoherent-elastic Debye-Waller factor instead of
    the trace/3 scalar collapse.
    """
    if elastic_state is None:
        # elastic=true is a hard output contract (review NC-3): silently
        # writing an inelastic-only pack (and dropping a requested
        # directional mode with it) let a defensive branch masquerade as
        # success. Both branches are unreachable through the normal engine
        # path; if they fire, something upstream is genuinely broken.
        raise RuntimeError(
            "elastic export requested but the engine surfaced no "
            "elastic_state; refusing to write a pack that silently omits "
            "the requested elastic block"
            + (" (and the requested directional incoherent-elastic mode)"
               if incoherent_elastic_mode == "directional" else ""))
    U = np.asarray(elastic_state["thermal_displacement_matrices_ang2"], float)
    symbols = list(elastic_state["primitive_symbols"])
    frac = np.asarray(elastic_state["primitive_scaled_positions"], float)
    if U.shape[0] != n_atoms:
        raise RuntimeError(
            f"elastic_state has {U.shape[0]} sites but the primitive has "
            f"{n_atoms}; refusing to write a pack with a mismatched or "
            "missing elastic block (review NC-3)")

    pack.elastic_coherent = True            # full physical elastic (both channels)
    # Whole-structure per-site anisotropic U tensors (row-major 3x3 flattened),
    # symbols and fractional positions — the coherent structure factor and the
    # per-site incoherent DW both read these. elastic_scale is left at the C++
    # default (1.0) — the per-atom normalization is automatic.
    pack.elastic_u_tensors_a2 = [float(v) for v in U.reshape(n_atoms, 9).reshape(-1)]
    pack.elastic_u_symbols = [str(s) for s in symbols]
    pack.elastic_u_frac_positions = [float(v) for v in frac.reshape(-1)]
    # Per-tensor-site neutron data (config b_coh/sigma_inc), in the SAME site order as
    # the tensors above, so the C++ coherent F(hkl) + incoherent DW use exactly what
    # the IRMA config specified rather than NCrystal's atom DB. Looked up per-symbol.
    try:
        pack.elastic_u_coherent_scatlen_sqrtbarn = [
            float(neutron_by_symbol[s][0]) for s in symbols]
        pack.elastic_u_incoherent_xs_barn = [
            float(neutron_by_symbol[s][1]) for s in symbols]
    except KeyError as exc:
        raise ValueError(
            f"elastic site symbol {exc.args[0]!r} has no scatterer neutron data "
            "(b_coh_fm / sigma_inc_b); cannot stamp the per-site elastic neutron "
            "arrays the C++ plugin requires alongside the U tensors.") from None
    # Structure mode: leave elastic_msd_a2 / elastic_incoherent_xs_barn unset
    # (None). The anisotropic tensors supersede the scalar-MSD pair.
    pack.incoherent_elastic_mode = str(incoherent_elastic_mode)


def write_packs(cfg: NCrystalExportConfig, outdir: str | Path, *,
                progress=print) -> tuple[list[Path], Path]:
    """Build and write the pack set + a complete ``<material_id>.ncmat`` to
    ``outdir``.

    Returns ``(pack_paths, ncmat_path)``. The ``.ncmat`` is loadable as-is
    (``NCrystal.createScatter("<material_id>.ncmat;temp=<bakeT>")`` with the IRMA
    plugin installed): it carries the phonopy structure + ``@CUSTOM_IRMA``
    referencing the pack files (relative names; they sit beside it in ``outdir``).
    Pass ``;temp=`` equal to ``cfg.material.temperature_K`` (the pack bake
    temperature); the plugin rejects a temperature mismatch and does not
    interpolate. Without ``;temp=`` NCrystal defaults to 293.15 K, which only
    matches a pack baked at 293.15 K.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    outdir_resolved = outdir.resolve()

    def _within_outdir(path: Path) -> Path:
        """Assert ``path`` resolves inside ``outdir`` and return it."""
        # material_id is validated as a safe stem in NCrystalExportConfig; re-assert
        # every write lands inside outdir as defense in depth, so a future caller
        # that bypasses that validation still cannot escape the directory.
        if not path.resolve().is_relative_to(outdir_resolved):
            raise ValueError(
                f"refusing to write {path} outside the output directory "
                f"{outdir_resolved}")
        return path

    packs, material_ncmat = build_packs(
        cfg, pack_path_prefix=outdir_resolved, progress=progress)
    pack_paths: list[Path] = []
    for pack in packs:
        # material_id is "<id>__<symbol>"; file mirrors it.
        path = _within_outdir(outdir / f"{pack.material_id}.irmapack")
        write_pack(pack, path)
        pack_paths.append(path)
        progress(f"[irma.ncrystal] wrote {path}")
    ncmat_path = _within_outdir(outdir / f"{cfg.material_id}.ncmat")
    ncmat_path.write_text(material_ncmat, encoding="utf-8")
    progress(f"[irma.ncrystal] wrote {ncmat_path}")
    return pack_paths, ncmat_path
