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
import os
from pathlib import Path

import numpy as np

from irma.core.constants import AMASSN
from irma.core.grids import (
    describe_beta_grid, generate_alpha_grid, generate_beta_grid,
    grid_reference_temperature_K)
from .config import NCrystalExportConfig
from .convert import pack_from_irma_sab
from .ncmat import assemble_material_ncmat
from .pack import IRMAPack, write_pack
from .provenance import collect_provenance


# fm → Angstrom for the engine's per-site coherent scattering length input.
_FM_TO_ANGSTROM = 1.0e-5


def resolve_jobs(jobs):
    """Worker count for the engine's parallel direction/shell sums.

    ``jobs`` unset (``None``, the config default that a blank GUI field
    produces) means AUTO: every CPU core. An explicit value caps it — the
    lever for a memory-limited machine, since each worker holds its own copy
    of the phonon arrays.
    """
    return int(jobs) if jobs else (os.cpu_count() or 1)


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


def _auto_beta_grid(cfg, t_ref, recoil_awr, progress=print):
    """Build the shared converged beta grid (generate_beta_grid, lin-lin)
    AND return a phonopy ``Mesh`` to reuse downstream.

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
    beta = generate_beta_grid(
        freq_max, t_ref, iint=1, awr=float(recoil_awr),
        n_lower=cfg.n_lower, n_phonon=cfg.n_phonon,
        n_upper=cfg.n_upper, beta_max_eV=cfg.beta_max_eV,
        evaluation_temperatures_K=[float(cfg.material.temperature_K)])
    progress("  " + describe_beta_grid(beta, t_ref, 1))
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
def load_primitive_info(phonopy_yaml: str | Path):
    """Return ``(symbols, masses_amu, scaled_positions, lattice_ang)`` for the
    phonopy primitive cell, without a mesh eigensolve (the lattice in Angstrom,
    converted by ``phonopy_io.angstrom_primitive`` as in the engine)."""
    from irma.core.phonopy_io import angstrom_primitive, load_phonopy
    ph = load_phonopy(phonopy_yaml, geometry_only=True)
    prim = angstrom_primitive(ph)
    return (list(prim.symbols), np.asarray(prim.masses, float),
            np.asarray(prim.scaled_positions, float),
            np.asarray(prim.cell, float))


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
    """Resolve the principal groups (one per species) and the per-site neutron
    arrays -> ``(groups, site_groups, site_b_coh_angstrom, site_sigma_inc_barn,
    symbols)``. Every structure species needs a ``material.scatterers`` row and
    every row a structure species."""
    mat = cfg.material
    symbols, masses, _positions, _lattice = load_primitive_info(mat.phonopy_yaml)
    site_groups = site_groups_by_species(symbols)
    by_symbol = {s.symbol: s for s in mat.scatterers}
    extra = sorted(set(by_symbol) - set(symbols))
    if extra:
        raise ValueError(
            f"material.scatterers lists species {extra} that the phonopy "
            f"structure does not contain (structure species: "
            f"{sorted(set(symbols))}); remove the row(s) or fix the symbol")
    n_atoms = len(symbols)
    site_b_coh_ang = [0.0] * n_atoms
    site_sigma_inc = [0.0] * n_atoms
    groups: list[PrincipalGroup] = []
    for g in site_groups:
        sym = symbols[g[0]]
        sc = by_symbol.get(sym)
        if sc is None:
            raise ValueError(
                f"material.scatterers has no entry for species {sym!r} "
                f"(structure species: {sorted(set(symbols))})")
        mass = float(np.mean([masses[i] for i in g]))
        awr = float(sc.awr) if sc.awr is not None else mass / AMASSN
        b_coh_fm = float(sc.b_coh_fm)
        sigma_inc_b = float(sc.sigma_inc_b)
        for i in g:
            site_b_coh_ang[i] = b_coh_fm * _FM_TO_ANGSTROM
            site_sigma_inc[i] = sigma_inc_b
        groups.append(PrincipalGroup(
            symbol=sym, site_indices=[int(i) for i in g], mass_amu=mass, awr=awr,
            sigma_bound_b=float(sc.sigma_bound_b), b_coh_fm=b_coh_fm,
            sigma_inc_b=sigma_inc_b))
    return groups, site_groups, site_b_coh_ang, site_sigma_inc, symbols


def _coherent_bearing_index(groups: list[PrincipalGroup]) -> int:
    """Which principal pack carries the (whole-structure) coherent Bragg edges.

    The coherent structure factor F(hkl)=Σ_sites b_coh·e^{-W}·e^{iφ} is a single
    crystal-wide quantity; to avoid double-counting when packs are summed, the
    full Bragg-edge structure lives in exactly ONE pack — the species with the
    largest coherent weight n·b_coh² — and the others contribute only their
    incoherent DW line.  (Monoatomic → the single pack.)  See the overnight
    decision log: the alternative is a per-species principal-xs-weighted split.
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

    (groups, site_groups, site_b_coh_ang, site_sigma_inc,
     symbols) = resolve_principal_groups(cfg)
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
        else _auto_beta_grid(cfg, t_ref, min(g.awr for g in groups),
                             progress=progress))

    packs: list[IRMAPack] = []
    # The whole-crystal elastic line (Bragg edges over all sites + the full
    # incoherent DW) is carried by exactly one pack, the coherent-bearing
    # principal: the C++ builds F(hkl) from that pack's full tensor set, so
    # spreading the tensors across packs would double-count it. The NCMAT
    # takes that pack's exact geometry (the plugin matches sites to 1e-6).
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
        if gi == coh_index:
            coh_elastic_state = elastic_state

    pack_names = [f"{cfg.material_id}__{g.symbol}.irmapack" for g in groups]
    pack_filenames = ([str(Path(pack_path_prefix) / n) for n in pack_names]
                      if pack_path_prefix is not None else pack_names)
    masses_by_symbol = {g.symbol: g.mass_amu for g in groups}
    debye_temps = _debye_temperatures(
        coh_elastic_state, masses_by_symbol, float(cfg.material.temperature_K))
    material_ncmat = assemble_material_ncmat(
        lattice_ang=np.asarray(coh_elastic_state["primitive_lattice_ang"], float),
        scaled_positions=np.asarray(coh_elastic_state["primitive_scaled_positions"], float),
        symbols=list(coh_elastic_state["primitive_symbols"]),
        pack_filenames=pack_filenames, debye_temperatures=debye_temps)
    return packs, material_ncmat


def _debye_temperatures(elastic_state, masses_by_symbol, temperature_K):
    """Per-element Debye temperature from the engine's mean-squared
    displacements (each element's site average of Tr(U)/3), so the NCMAT has a
    valid MSD source."""
    from .ncmat import debye_temperature_from_msd
    U = np.asarray(elastic_state["thermal_displacement_matrices_ang2"], float)
    es_symbols = list(elastic_state["primitive_symbols"])
    msd_site = (U[:, 0, 0] + U[:, 1, 1] + U[:, 2, 2]) / 3.0
    out: dict[str, float] = {}
    for sym in dict.fromkeys(es_symbols):
        idx = [i for i, s in enumerate(es_symbols) if s == sym]
        out[sym] = debye_temperature_from_msd(
            float(np.mean(msd_site[idx])), float(masses_by_symbol[sym]), temperature_K)
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
        auto_multiphonon_order=cfg.auto_multiphonon_order,
        min_phonon_energy_mev=cfg.min_phonon_energy_meV)

    result = run_noncubic_standalone_sab(
        alpha=np.asarray(alpha, float),
        beta=np.asarray(beta, float),
        lat=cfg.lat,
        temperature_k=float(mat.temperature_K),
        awr=group.awr,
        phonopy_yaml_path=str(mat.phonopy_yaml),
        mesh_dim=tuple(mat.mesh),
        born_path=mat.born,
        num_jobs=resolve_jobs(cfg.jobs),
        controls=controls,
        inelastic_mode=cfg.inelastic_mode,
        represented_principal_site_count=len(group.site_indices),
        principal_group_index=group_index,
        site_groups=site_groups,
        coherent_partition_mode=cfg.coherent_partition_mode,
        sab_sigma_barn=group.sigma_bound_b,
        site_scattering_lengths_angstrom=site_b_coh_ang,
        site_incoherent_cross_sections_barn=site_sigma_inc,
        preloaded_full_mesh=preloaded_full_mesh)

    # The SAB table keeps the engine's full-sigma normalization; the per-atom
    # weighting of a multi-species material goes only into the pack's
    # advertised bound_xs (atom_fraction * sigma_bound_b, below).
    sab_downscatter = result["sab_downscatter_qe"]
    atom_fraction = len(group.site_indices) / float(n_atoms)

    meta = collect_provenance(
        phonopy_yaml=mat.phonopy_yaml, mesh=mat.mesh,
        temperature_K=mat.temperature_K, num_directions=cfg.num_directions,
        multiphonon_num_directions=cfg.multiphonon_num_directions,
        multiphonon_max_order=cfg.multiphonon_max_order,
        min_phonon_energy_meV=cfg.min_phonon_energy_meV,
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
        _attach_elastic(pack, elastic_state, neutron_by_symbol,
                        incoherent_elastic_mode=cfg.incoherent_elastic_mode)
    return pack, elastic_state


def _attach_elastic(pack, elastic_state, neutron_by_symbol, *,
                    incoherent_elastic_mode="isotropic") -> None:
    """Stamp the whole-crystal elastic block from the engine's anisotropic-DW
    state: the per-site U tensors (row-major 3x3), symbols, fractional
    positions and per-site neutron data (the config's b_coh [sqrt(barn)] and
    sigma_inc, in the tensor site order). The C++ builds both elastic channels
    from these; ``incoherent_elastic_mode='directional'`` has it sample the
    orientation-averaged incoherent Debye-Waller factor."""
    U = np.asarray(elastic_state["thermal_displacement_matrices_ang2"], float)
    symbols = [str(s) for s in elastic_state["primitive_symbols"]]
    frac = np.asarray(elastic_state["primitive_scaled_positions"], float)
    pack.elastic_u_tensors_a2 = [float(v) for v in U.reshape(-1)]
    pack.elastic_u_symbols = symbols
    pack.elastic_u_frac_positions = [float(v) for v in frac.reshape(-1)]
    pack.elastic_u_coherent_scatlen_sqrtbarn = [
        float(neutron_by_symbol[s][0]) for s in symbols]
    pack.elastic_u_incoherent_xs_barn = [float(neutron_by_symbol[s][1]) for s in symbols]
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
    packs, material_ncmat = build_packs(
        cfg, pack_path_prefix=outdir.resolve(), progress=progress)
    pack_paths: list[Path] = []
    for pack in packs:
        # material_id is "<id>__<symbol>"; file mirrors it.
        path = outdir / f"{pack.material_id}.irmapack"
        write_pack(pack, path)
        pack_paths.append(path)
        progress(f"[irma.ncrystal] wrote {path}")
    ncmat_path = outdir / f"{cfg.material_id}.ncmat"
    ncmat_path.write_text(material_ncmat, encoding="utf-8")
    progress(f"[irma.ncrystal] wrote {ncmat_path}")
    return pack_paths, ncmat_path
