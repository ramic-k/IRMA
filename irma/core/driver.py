"""IRMA's main LEAPR-style driver: ``run_leapr`` and its phonopy/MT4 helpers.

The card-by-card deck format this driver parses is documented in
``docs/input-reference.md`` ("Input deck reference" in the manual); the
``iel=10`` Card 6b-6g block is parsed by :mod:`irma.core.crystal_cards`.

Extracted verbatim from :mod:`irma.core.engine` to slim that module.
``run_leapr`` is IRMA's primary entry point (parse a LEAPR deck -> build
S(alpha, beta) + elastic terms -> write an ENDF-6 File 7 tape);
``_noncubic_mt4_step`` and ``_store_directional_species_dw`` support the
``iel=10`` ``inelastic_mode=1/2`` path. :mod:`irma.core.engine` re-imports all
three, so the public API, the CLI, and the test suite that import them from
there are unaffected. Kept byte-for-byte identical so the ENDF tape is unchanged.
"""
import dataclasses
import json
import os
from math import sqrt
from pathlib import Path

import numpy as np

from irma.core.constants import BK, THERM, WL2EKIN, _Z_TO_SYMBOL
from irma.core.deck import (
    DeckError,
    parse_leapr_input, TokenReader,
    _read_temperature_detail_cards,
)
from irma.core.kernels import (
    contin, contin_cubic_trace_dos, trans, discre, coldh, skold_approx,
)
from irma.core.crystal import (
    compute_bragg_edges_general, coher,
    _build_atom_types_expanded, _group_phonopy_atoms_by_type,
    _average_site_quantity, _compute_per_species_msd,
    _order_site_groups_by_card6d_positions, _site_tensors_uniform,
)
from irma.core.crystal_cards import _parse_crystal_cards
from irma.core.endf_writer import write_endf_output, _LN_FLOAT_MAX


def _noncubic_mt4_step(crystal_info, ssm, itemp, alpha, beta, nalpha,
                       nbeta, lat, arat, tev, tempr_arr, ntempr, awr,
                       nphon, tbeta, output_file):
    """One temperature of the mode-1/2 path: compute the in-process
    noncubic SAB, inject it into ssm[:, :, itemp], and refresh the
    per-species Debye-Waller traces. Returns (f0, tbar, deltab,
    F_matrix_all) for the ENDF bookkeeping."""
    from irma.core.phonopy_io import (
        compute_dos_tensor,
        compute_thermal_displacement_matrices,
        mode_floor_mask,
        thermal_displacements_to_f_matrix,
    )
    nc_mesh = crystal_info['nc_mesh_data']
    nc_ncpu_run = crystal_info.get('nc_ncpu', 1)
    principal_site_indices = crystal_info['principal_nc_site_indices']
    site_groups = crystal_info['nc_atom_type_site_groups']
    atom_types_expanded = crystal_info['nc_atom_types_expanded']
    awr_by_atom = crystal_info['nc_awr_by_atom']

    # Energy grid for the DOS tensor; the Gaussian smearing always uses
    # compute_dos_tensor's auto width (2 x grid spacing). freq_max is the
    # largest mode energy (np.max): imaginary modes are reported as negative,
    # so as long as any positive mode exists np.max returns it and instability
    # does not stretch the grid (a fully-imaginary model would give a negative
    # freq_max, flagged by the warning below).
    _freqs = nc_mesh.frequencies_ev                          # (N_q, n_branches), eV
    # Stability diagnostics, split the way the mode floor itself is: an imaginary
    # mode AT Gamma is acoustic-sum-rule / NAC numerical noise (routine and
    # expected), but an imaginary mode at a NON-Gamma q-point is a genuine
    # finite-wavevector dynamical instability and is surfaced even when
    # small — a single loose threshold for both cases would hide it.
    _is_gamma_q = np.all(np.abs(nc_mesh.qpoints) < 1.0e-9, axis=1)
    _nongamma_imag = (_freqs < -1.0e-6) & ~_is_gamma_q[:, None]
    # Severity tags (WARNING:/NOTE:) start at column 0 so `grep '^WARNING'`
    # over a run log finds them; only plain progress lines carry nesting
    # indentation.
    if np.any(_nongamma_imag):
        print(f"WARNING: {int(np.count_nonzero(_nongamma_imag))} imaginary "
              "phonon mode(s) at non-Gamma q-points (min "
              f"{np.min(_freqs[_nongamma_imag])*1000:.3f} meV) -- the phonon model "
              "is DYNAMICALLY UNSTABLE (a finite-wavevector soft mode); review the "
              "structure and force constants before trusting this evaluation.")
    elif np.any(_freqs < -1.0e-4):
        print("WARNING: imaginary modes near Gamma (min "
              f"{np.min(_freqs)*1000:.2f} meV) beyond the acoustic-sum-rule noise "
              "band; they are excluded from the grid, but review the phonon model.")
    _dropped = ~mode_floor_mask(_freqs.reshape(-1) * 1.0e3, nc_mesh.qpoints,
                                nc_mesh.n_branches)
    if np.any(_dropped):
        _below = _freqs.reshape(-1)[_dropped] * 1000.0
        print(f"NOTE: {int(_dropped.sum())} mode(s) below the DOS floor "
              f"excluded (range [{_below.min():.3f}, {_below.max():.3f}] meV).")
    freq_max_ev = float(np.max(_freqs))
    if freq_max_ev <= 0.0:
        # No positive modes at all: a fully unstable / unphysical phonon model.
        # np.max would be <= 0 and the DOS grid linspace(0, freq_max*1.1) would
        # be degenerate/reversed -- fail loudly instead of emitting garbage.
        raise DeckError(
            "the phonon model has no positive modes (max frequency "
            f"{freq_max_ev * 1000.0:.3f} meV <= 0): it is fully unstable and a "
            "DOS grid cannot be built. Check the structure and force constants.")
    freq_max_ev *= 1.1  # 10% headroom
    # The 1000-point cap only coarsens the grid for spectra extending above
    # ~1 eV (where it slightly smooths the Teff bookkeeping integrals);
    # hydrogenous and ordinary moderator spectra are unaffected.
    nc_n_freq = max(500, min(1000, int(round(freq_max_ev / 0.001)) + 1))

    print(f"    Non-cubic DOS tensor: freq_max={freq_max_ev*1000:.1f} meV, "
          f"n_freq={nc_n_freq}, sigma=auto")

    # The DOS tensor's inputs (mesh, freq grid) are temperature-independent:
    # compute it once and reuse it for every temperature card.
    _dt_key = (float(freq_max_ev), int(nc_n_freq))
    _dt_cache = crystal_info.get('nc_dos_tensor_cache')
    if _dt_cache is not None and _dt_cache[0] == _dt_key:
        dos_tensor_all, energy_grid_ev = _dt_cache[1]
        print("    Reusing the temperature-independent DOS tensor.")
    else:
        dos_tensor_all, energy_grid_ev = compute_dos_tensor(
            nc_mesh, freq_max_ev, nc_n_freq)
        crystal_info['nc_dos_tensor_cache'] = (
            _dt_key, (dos_tensor_all, energy_grid_ev))
    thermal_mats_all = compute_thermal_displacement_matrices(
        nc_mesh, tempr_arr[itemp])
    F_matrix_all = thermal_displacements_to_f_matrix(
        thermal_mats_all, awr_by_atom, tev)
    dos_tensor_d = dos_tensor_all[principal_site_indices]

    from irma.core.standalone_sab import run_noncubic_standalone_sab

    inelastic_mode_value = crystal_info.get('inelastic_mode', 0)
    # Bookkeeping pass only: tbar/deltab feed the SCT/Teff records and come
    # from start() alone, so the full nphon-order expansion (formerly run into
    # a discarded scratch array, seconds-to-minutes per site per temperature)
    # is skipped — bit-identical returns.
    _f0_scratch, tbar, deltab = contin_cubic_trace_dos(
        None, alpha, beta, nalpha, nbeta,
        lat, arat, tev, dos_tensor_d, energy_grid_ev,
        nphon, tbeta, 0, expand_ssm=False)
    f0 = float(np.mean([
        np.trace(F_matrix_all[d_idx]) / 3.0
        for d_idx in principal_site_indices
    ]))
    standalone_result = run_noncubic_standalone_sab(
        alpha=alpha[:nalpha],
        beta=beta[:nbeta],
        lat=lat,
        temperature_k=tempr_arr[itemp],
        awr=awr,
        # MT4 is stored per principal scatterer, but the standalone
        # one-phonon kernel is accumulated over all represented
        # phonopy sites for that scatterer.
        represented_principal_site_count=len(principal_site_indices),
        principal_group_index=crystal_info['principal_atom_idx'],
        site_groups=site_groups,
        coherent_partition_mode="auto",
        sab_sigma_barn=(
            crystal_info['atom_types'][crystal_info['principal_atom_idx']]['sigma_coh']
            + crystal_info['atom_types'][crystal_info['principal_atom_idx']]['sigma_inc']
        ),
        phonopy_yaml_path=crystal_info['nc_phonopy_yaml_path'],
        mesh_dim=crystal_info['nc_mesh_dim'],
        born_path=crystal_info.get('nc_born_path'),
        num_jobs=nc_ncpu_run,
        workdir=os.path.dirname(os.path.abspath(output_file)) or ".",
        inelastic_mode=inelastic_mode_value,
        site_scattering_lengths_angstrom=[
            float(at['b_coh']) * 1.0e-5 for at in atom_types_expanded
        ],
        site_incoherent_cross_sections_barn=[
            float(at['sigma_inc']) for at in atom_types_expanded
        ],
        scattering_lengths_json=json.dumps({
            _Z_TO_SYMBOL[at['Z']]: float(at['b_coh']) * 1.0e-5
            for at in crystal_info['atom_types']
            if at['Z'] in _Z_TO_SYMBOL
        }),
        incoherent_cross_sections_json=json.dumps({
            _Z_TO_SYMBOL[at['Z']]: float(at['sigma_inc'])
            for at in crystal_info['atom_types']
            if at['Z'] in _Z_TO_SYMBOL
        }),
        controls=crystal_info['nc_inelastic_controls'],
        context_cache=crystal_info.setdefault("nc_inelastic_context_cache", {}),
        # The MT2 loader already ran the identical full-MP mesh
        # with eigenvectors; hand its Mesh to the context builder so a cache
        # miss skips the duplicate eigensolve.
        preloaded_full_mesh=getattr(nc_mesh, "phonopy_mesh_object", None),
        # This temperature's U_ij was just computed above on the
        # same mesh with the same mode floor; the compute phase reuses it.
        precomputed_thermal_mats=thermal_mats_all,
    )
    ssm[:, :, itemp] = standalone_result['ssm_internal']
    print(
        f"    Inelastic mode {inelastic_mode_value} MT4: injected in-process SAB "
        f"({standalone_result['selected_sab_key']})"
    )

    return f0, tbar, deltab, F_matrix_all


def _store_directional_species_dw(crystal_info, itemp, ntempr, tev,
                                  tempr_arr, isecs, F_matrix_all):
    """Store the per-species averaged Debye-Waller F-matrices that
    _build_cef_coherent uses for directional attenuation
    (inelastic_mode=1/2). F_matrix_all comes from the MT4 step: the guard
    below is true exactly when the caller's use_phonopy_dw was true, so the
    matrices are always supplied."""
    if not (crystal_info is not None
            and crystal_info.get('inelastic_mode', 0) in (1, 2)
            and isecs == 0
            and crystal_info.get('nc_mesh_data') is not None):
        return
    nc_mesh_ci = crystal_info['nc_mesh_data']
    F_matrix_ci = F_matrix_all

    site_groups = crystal_info.get('nc_atom_type_site_groups')
    if site_groups is None:
        atom_types_exp = _build_atom_types_expanded(crystal_info, nc_mesh_ci)
        site_groups = _group_phonopy_atoms_by_type(
            crystal_info['atom_types'], atom_types_exp)
        crystal_info['nc_atom_type_site_groups'] = site_groups

    # Average per-species F-matrix (F_matrix_ci shape: n_phonopy_atoms × 3 × 3)
    atom_types_list = crystal_info['atom_types']
    nsp = len(atom_types_list)
    F_species = np.zeros((nsp, 3, 3))
    for si in range(nsp):
        F_species[si] = _average_site_quantity(
            F_matrix_ci, site_groups[si],
            f"atom type {si+1} Debye-Waller matrix")
    if 'F_species_per_temp' not in crystal_info:
        crystal_info['F_species_per_temp'] = [None] * ntempr
    crystal_info['F_species_per_temp'][itemp] = F_species

    # Site-resolved tensors for the directional coherent-elastic path (review
    # finding P3): keep the per-SITE F-matrices, ordered to match crystal.py's
    # site_terms flattening (species, then Card 6d position order), plus an
    # exact-uniformity flag. When every group's tensors are bitwise identical
    # the elastic kernels keep the species-averaged fast path (byte-identical
    # tapes -- graphite/Be/BeO); otherwise they sum site-resolved
    # DW-attenuated complex amplitudes per plane.
    F_arr = np.asarray(F_matrix_ci)
    mesh_positions = nc_mesh_ci.atom_positions
    ordered_groups = _order_site_groups_by_card6d_positions(
        atom_types_list, site_groups, mesh_positions)
    F_sites = []
    uniform = True
    for si in range(nsp):
        idx = list(site_groups[si])
        group_uniform = _site_tensors_uniform(F_arr, idx)
        uniform = uniform and group_uniform
        if ordered_groups is not None:
            F_sites.append(F_arr[ordered_groups[si]])
        elif group_uniform:
            F_sites.append(F_arr[idx])  # uniform group: order is irrelevant
        else:
            raise ValueError(
                f"Card 6d atom type {si+1}: cannot pair its fractional "
                f"positions one-to-one with the phonopy sites of its "
                f"group (even modulo a shared origin shift), and the "
                f"per-site Debye-Waller tensors are not identical -- the "
                f"site-resolved directional elastic cannot align tensors "
                f"with structure-factor phases. Check that the Card 6d "
                f"positions match the phonopy primitive cell.")
    if 'F_sites_per_temp' not in crystal_info:
        crystal_info['F_sites_per_temp'] = [None] * ntempr
    crystal_info['F_sites_per_temp'][itemp] = F_sites
    crystal_info['dir_tensors_uniform'] = (
        crystal_info.get('dir_tensors_uniform', True) and uniform)


def extinction_provenance_comment(ext):
    """One MF1/MT451 free-text line recording the crystalline-extinction provenance.

    The tape then carries, in its human-readable description, that it is a
    sample-specific extinction-corrected evaluation and which model/parameters
    produced it. ``ext`` is the ``coherent_extinction`` config dict.
    """
    return (f"' extinction: {ext['model']} l={ext['l']:g} g={ext['g']:g} "
            f"L={ext['L']:g} {ext['dist']}/{ext['recipe']} (CrysXT models)'")


@dataclasses.dataclass(frozen=True)
class LeaprResult:
    """Summary of one ``run_leapr`` evaluation, for programmatic callers.

    The tape at ``output_file`` remains the authoritative product; this
    object only surfaces the run's headline numbers so sweeps and notebooks
    do not have to re-parse the tape or capture stdout. For a two-pass
    mixed-moderator deck (``nss>0`` with a bound ``b7<=0`` secondary),
    ``teff_K`` and ``dw_lambda`` carry the FINAL scatterer pass's values
    (the secondary's), mirroring NJOY's ``tempf``/``dwpix`` arrays; the
    principal's Teff is on the tape as Teff0.
    """

    output_file: str
    title: str
    ntempr: int
    temperatures_K: tuple
    nedge: int
    iel: int
    isym: int
    teff_K: tuple
    dw_lambda: tuple


def run_leapr(input_file: str | Path, output_file: str | Path) -> LeaprResult:
    """Run a LEAPR-style thermal-scattering calculation and write an ENDF tape.

    This is IRMA's primary entry point: it parses a LEAPR-style input deck,
    builds the thermal scattering law S(alpha, beta) and the elastic terms,
    and writes an ENDF-6 File 7 tape (MF7/MT2 elastic, MF7/MT4 inelastic,
    MF1/MT451 comments). It covers the classic cubic path (``iel=0``-``6``)
    and the generalized ``iel=10`` path (SEF/MEF elastic and the directional
    ``inelastic_mode=1/2`` engines).

    Args:
        input_file: Path to the LEAPR-style ``.input``/``.leapr`` deck.
        output_file: Path the ENDF-6 tape is written to (overwritten if it
            exists).

    Returns:
        A :class:`LeaprResult` summary (output path, temperatures, edge
        count, effective temperatures). The primary result is the ENDF tape
        at ``output_file``; progress is printed to stdout. Earlier releases
        returned ``None`` — callers that tested ``result is None`` must
        update.

    Raises:
        DeckError: The deck is malformed or self-inconsistent (the message
            names the offending card).
        RuntimeError: A required runtime dependency or model file is missing
            (e.g. the phonopy mesh for ``inelastic_mode=1/2``).
    """

    # Parse input
    tokens, raw_lines, start_line, token_lines = parse_leapr_input(input_file)
    reader = TokenReader(tokens, token_lines=token_lines, raw_lines=raw_lines,
                         filename=input_file)
    if not tokens:
        raise DeckError(f"no LEAPR cards found in {input_file} "
                        "(empty file or nothing after the 'leapr' line)")

    # Card 1: output unit (ignored; the ENDF tape is written directly)
    reader.card("Card 1 (nout)")
    reader.read_ints(1)  # Card 1 nout (output unit): consumed but ignored

    # Card 2: title
    reader.card("Card 2 (title)")
    title = reader.read_string(allow_numeric=True)
    print(f"IRMA: {title}")

    # Card 3: run control
    reader.card("Card 3 (ntempr iprint nphon)")
    vals = reader.read_ints(3, defaults=[1, 1, 100])
    ntempr, iprint, nphon = vals
    reader.require(ntempr >= 1, f"ntempr must be >= 1, got {ntempr}")
    reader.require(nphon >= 1, f"nphon must be >= 1, got {nphon}")

    # Card 4: ENDF output control
    reader.card("Card 4 (mat za isabt ilog smin [iint])")
    fvals = reader.read_floats(6, defaults=[0, 0, 0, 0, 1.0e-75, 0])
    mat = reader.to_int(fvals[0], "mat")
    # za is the ENDF ZA designator (1000*Z+A) and is integer-valued by
    # definition; coerce it like the iel=10 integer fields so a fractional
    # ZA is rejected with Card 4 context instead of emitted as a malformed
    # ENDF float.
    za = reader.to_int(fvals[1], "za")
    isabt = reader.to_int(fvals[2], "isabt")
    ilog = reader.to_int(fvals[3], "ilog")
    smin = fvals[4]
    # iint: interpolation form of the written MF7/MT4 S(alpha,beta) law.
    # 0 = log-lin (ENDF INT=4, classic/NJOY-faithful, default); 1 = lin-lin
    # (INT=2). Coherent one-phonon laws have structural near-zeros that log
    # interpolation floors (biasing the cross section low in the thermal
    # minimum); lin-lin preserves them. Honored by INT-aware THERMR. Pair
    # iint=1 with a Q/alpha grid converged near sharp coherent peaks.
    iint = reader.to_int(fvals[5], "iint")
    reader.require(mat >= 1, f"mat must be >= 1, got {mat}")
    reader.require(za > 0, f"za must be > 0, got {za}")
    reader.require(isabt in (0, 1), f"isabt must be 0 or 1, got {isabt}")
    reader.require(ilog in (0, 1), f"ilog must be 0 or 1, got {ilog}")
    reader.require(iint in (0, 1), f"iint must be 0 or 1, got {iint}")
    # smin is the S(alpha,beta) zero-suppression cutoff. The cutoff
    # semantics match NJOY (values below smin are dropped), but a negative
    # or non-finite cutoff is a foot-gun NJOY does not guard, so reject it
    # at parse time.
    reader.require(np.isfinite(smin) and smin >= 0.0,
                   f"smin (S cutoff) must be finite and >= 0, got {smin!r}")

    print(f"  ntempr={ntempr}, iprint={iprint}, nphon={nphon}")
    print(f"  mat={mat}, za={za}, isabt={isabt}, ilog={ilog}, iint={iint}")

    # Card 5: principal scatterer control
    reader.card("Card 5 (awr spr npr iel ncold nsk)")
    fvals = reader.read_floats(6, defaults=[0, 0, 0, 0, 0, 0])
    awr = fvals[0]
    spr = fvals[1]
    npr = reader.to_int(fvals[2], "npr")
    iel = reader.to_int(fvals[3], "iel")
    ncold = reader.to_int(fvals[4], "ncold")
    nsk = reader.to_int(fvals[5], "nsk")
    reader.require(awr > 0, f"awr (mass ratio) must be > 0, got {awr}")
    reader.require(spr > 0, f"spr (free-atom cross section) must be > 0, got {spr}")
    reader.require(npr >= 1, f"npr (principal atom count) must be >= 1, got {npr}")
    reader.require(iel in (0, 1, 2, 3, 4, 5, 6, 10),
                   f"iel must be 0-6 (built-in coherent elastic) or 10 "
                   f"(generalized), got {iel}")
    reader.require(0 <= ncold <= 4, f"ncold must be 0-4, got {ncold}")
    reader.require(0 <= nsk <= 2, f"nsk must be 0-2, got {nsk}")
    if nsk == 1 and ncold == 0:
        # NJOY2016 parity: nsk=1 (Vineyard) only prints a banner there too --
        # leapr.f90 reads the S(kappa)/cfrac cards for any nsk>0 but its only
        # skold call is gated on nsk==2, so the Vineyard option is
        # unimplemented upstream as well. Accept the deck (NJOY input
        # compatibility) but say loudly that the physics is a no-op.
        print("WARNING: nsk=1 (Vineyard) is accepted for NJOY input "
              "compatibility but is not implemented in NJOY2016 or IRMA; the "
              "Card 17-19 S(kappa)/cfrac data will be read and ignored. Use "
              "nsk=2 (Skold) for an applied pair-correlation correction.",
              flush=True)

    print(f"  awr={awr}, spr={spr}, npr={npr}, iel={iel}, ncold={ncold}, nsk={nsk}")

    # Card 6: secondary scatterer control
    reader.card("Card 6 (nss b7 aws sps mss)")
    fvals = reader.read_floats(5, defaults=[0, 0, 0, 0, 0])
    nss = reader.to_int(fvals[0], "nss")
    b7 = fvals[1]
    aws = fvals[2]
    sps = fvals[3]
    mss = reader.to_int(fvals[4], "mss")
    reader.require(nss in (0, 1), f"nss must be 0 or 1, got {nss}")
    if nss > 0:
        reader.require(b7 in (0.0, 1.0, 2.0),
                       f"b7 (secondary scatterer type) must be 0 (bound, "
                       f"two-pass), 1 (free gas), or 2 (diffusion); got {b7}")
        reader.require(aws > 0, f"aws (secondary mass ratio) must be > 0, got {aws}")
        reader.require(sps > 0, f"sps (secondary cross section) must be > 0, got {sps}")
        reader.require(mss >= 1, f"mss (secondary atom count) must be >= 1, got {mss}")
        # The TWO-PASS merge (only taken for a bound b7<=0 secondary) preserves
        # only the symmetric law (ssm) across passes; it would silently drop a
        # cold-hydrogen/deuterium asymmetric law (ssp, allocated when ncold>0).
        # An analytic secondary (b7=1 free gas / b7=2 diffusion) is single-pass
        # and does NOT enter that merge, so it is left alone.
        reader.require(
            not (ncold > 0 and b7 <= 0.0),
            "a bound two-pass secondary scatterer (b7<=0) cannot be combined with "
            "ncold>0 (cold-hydrogen/deuterium asymmetric law): the two-pass merge "
            "carries only the symmetric law and would drop the asymmetric ssp. "
            "Use an analytic secondary (b7=1 free gas / b7=2 diffusion) or drop "
            "the secondary scatterer.")
        print(f"  nss={nss}, b7={b7}, aws={aws}, sps={sps}, mss={mss}")

    # ---- Generalized coherent elastic cards (iel=10) ----
    crystal_info = None  # will hold parsed crystal/elastic data if iel==10

    if iel == 10:
        crystal_info = _parse_crystal_cards(reader, za, nphon,
                                            ncold=ncold, nsk=nsk, nss=nss,
                                            b7=b7)

    # Card 7: alpha, beta control
    reader.card("Card 7 (nalpha nbeta lat)")
    vals = reader.read_ints(3, defaults=[0, 0, 0])
    nalpha, nbeta, lat = vals
    reader.require(nalpha >= 1, f"nalpha must be >= 1, got {nalpha}")
    # Every interpolation/convolution consumer of the beta grid (trans
    # bracket slopes, coldh/sint tables, NJOY's own sbfill) assumes at
    # least a two-point bracket; nbeta=1 crashes downstream.
    reader.require(nbeta >= 2, f"nbeta must be >= 2, got {nbeta}")
    reader.require(lat in (0, 1), f"lat must be 0 or 1, got {lat}")

    print(f"  nalpha={nalpha}, nbeta={nbeta}, lat={lat}")

    def _check_grid(name, grid, min_ok):
        """Require a strictly increasing grid with a valid first value."""
        reader.require(grid[0] >= min_ok if min_ok == 0.0 else grid[0] > 0.0,
                       f"{name} values must be {'>= 0' if min_ok == 0.0 else '> 0'} "
                       f"(first is {grid[0]:g})")
        d = np.diff(grid)
        if len(d) and not np.all(d > 0):
            bad = int(np.argmax(d <= 0))
            reader.require(False,
                           f"{name} grid must be strictly increasing "
                           f"({name}[{bad+2}]={grid[bad+1]:g} follows "
                           f"{name}[{bad+1}]={grid[bad]:g})")

    # Card 8: alpha values
    reader.card(f"Card 8 ({nalpha} alpha values)")
    alpha = reader.read_float_array(nalpha)
    _check_grid("alpha", alpha, min_ok=1e-300)

    # Card 9: beta values
    reader.card(f"Card 9 ({nbeta} beta values)")
    beta = reader.read_float_array(nbeta)
    _check_grid("beta", beta, min_ok=0.0)

    # Allocate storage
    ssm = np.zeros((nbeta, nalpha, ntempr))
    ssp = None
    if ncold != 0:
        ssp = np.zeros((nbeta, nalpha, ntempr))

    tempr_arr = np.zeros(ntempr)
    dwpix = np.zeros(ntempr)
    dwp1 = np.zeros(ntempr)
    tempf = np.zeros(ntempr)
    tempf1 = np.zeros(ntempr)

    arat = 1.0

    # Loop over scatterers and temperatures
    isecs = 0
    idone = False
    ssm_principal = None

    while not idone:
        if isecs == 0:
            print("\n  Principal scatterer...")
        else:
            arat = aws / awr
            print(f"\n  Secondary scatterer (alpha scaled by {arat:.3f})...")

        # Temperature-detail-card state (continuous phonon spectrum p1, the
        # translational/diffusion weights, discrete oscillators, and any
        # Skold/cold-hydrogen data). In LEAPR a NEGATIVE temperature card means
        # "reuse the PREVIOUS temperature's scattering-law inputs unchanged and
        # only recompute the law at the new |T|" — the deck therefore omits the
        # detail block for those temperatures. This state is initialized once
        # per scatterer pass and overwritten only when a temperature actually
        # supplies a fresh detail block (the first temperature, or any positive
        # temperature); negative temperatures fall through and reuse the carried
        # forward values. (The defaults below also cover the iel=10 phonopy
        # inelastic modes, which never read these cards.)
        delta1 = 0.0
        np1 = 0
        p1 = None
        twt = 0.0
        c_diff = 0.0
        tbeta = 1.0
        nd = 0
        bdel = None
        adel = None
        ska = None
        nka = 0
        dka = 0.0
        cfrac = 0.0

        for itemp in range(ntempr):
            reader.card(f"Card 10 (temperature {itemp+1} of {ntempr})")
            temp = reader.read_floats(1)[0]
            reader.require(temp != 0.0,
                           "temperature must be nonzero (negative reuses the "
                           "previous temperature's spectrum)")
            tempr_arr[itemp] = abs(temp)
            tev = BK * abs(temp)
            print(f"  Temperature {itemp+1}: {abs(temp):.2f} K")

            # Early warning for the cold-hydrogen asymmetric-law overflow
            # regime (isym=1, linear ilog=0 storage). The writer evaluates
            # S*exp(beta'/2) with beta' = beta*THERM/(k*T) at lat=1, and
            # math.exp overflows float64 once beta'/2 > ln(DBL_MAX) ~ 709.78
            # -- at cryogenic T a modest Card 9 beta reaches that. The
            # writer recovers representable points in log space and raises a
            # DeckError for unrecoverable ones, but only AFTER the expensive
            # kernel run; say so now, while the deck is still being parsed.
            if ncold > 0 and isabt == 0 and ilog == 0:
                _sc_t = THERM / tev if lat == 1 else 1.0
                _be_max = beta[nbeta - 1] * _sc_t
                if _be_max / 2.0 > _LN_FLOAT_MAX:
                    print(f"WARNING: Card 10 T={abs(temp):g} K enters the "
                          f"cold-hydrogen asymmetric-law overflow regime: the "
                          f"Card 9 beta grid reaches beta*THERM/(k*T) = "
                          f"{_be_max:.1f} (lat={lat}), past the float64 limit "
                          f"2*ln(DBL_MAX) = {2.0 * _LN_FLOAT_MAX:.1f} for the "
                          f"linear (ilog=0) asymmetric law S*exp(beta/2). "
                          f"Points recoverable in log space will be written; "
                          f"points whose S has underflowed to 0 will fail the "
                          f"ENDF write AFTER the kernel run. Set ilog=1 "
                          f"(LLN=1 log storage) on Card 4 -- see "
                          f"examples/lln_low_temperature_demo.py.",
                          flush=True)

            if crystal_info is not None and crystal_info.get('inelastic_mode', 0) in (1, 2):
                if itemp == 0:
                    print("    Inelastic mode 1/2: using noncubic in-process MT4 path; "
                          "legacy continuous DOS and temperature-detail cards are not read")
            elif itemp == 0 or temp >= 0.0:
                (
                    delta1, np1, p1, twt, c_diff, tbeta,
                    nd, bdel, adel, ska, nka, dka, cfrac,
                ) = _read_temperature_detail_cards(reader, nsk, ncold)
            else:
                # Negative T: reuse the previous temperature's detail block
                # (standard LEAPR convention) — no cards are consumed here.
                print("    (negative T: reusing previous temperature's "
                      "scattering-law inputs)")

            # Incoherent inelastic S(α,β): either the default cubic path or the
            # external standalone SAB path with directional DW support.
            use_phonopy_dw = (
                crystal_info is not None
                and crystal_info.get('inelastic_mode', 0) in (1, 2)
                and isecs == 0
                and crystal_info.get('nc_mesh_data') is not None
            )
            if use_phonopy_dw:
                f0, tbar, deltab, F_matrix_all = _noncubic_mt4_step(
                    crystal_info, ssm, itemp, alpha, beta, nalpha,
                    nbeta, lat, arat, tev, tempr_arr, ntempr, awr,
                    nphon, tbeta, output_file)
            else:
                f0, tbar, deltab = contin(ssm[:, :, itemp], alpha, beta,
                                          nalpha, nbeta, lat, arat, tev,
                                          p1, np1, delta1, tbeta, nphon,
                                          iprint)
            # NOTE (modes 1/2): dwpix and tempf do not come from a single
            # start() call (unlike the classic path). f0 is the
            # principal-species Tr(F)/3 from the phonopy thermal-displacement
            # matrices, while tbar comes from the scratch DOS-tensor contin.
            # Both describe the same phonon model via the exact (TDM) and the
            # binned-DOS routes respectively.
            dwpix[itemp] = f0
            tempf[itemp] = tbar * tempr_arr[itemp]

            print(f"    DW lambda = {f0:.6f}, T_eff = {tempf[itemp]:.3f}")

            # Directional DW for coherent elastic (inelastic_mode=1/2)
            # Store per-species averaged F-matrix so _build_cef_coherent can use it.
            _store_directional_species_dw(
                crystal_info, itemp, ntempr, tev, tempr_arr, isecs,
                F_matrix_all if use_phonopy_dw else None)

            # Translational part
            if twt > 0.0:
                trans(ssm[:, :, itemp], alpha, beta, nalpha, nbeta,
                      lat, arat, tev, twt, c_diff, tbeta, f0, deltab,
                      tbar, iprint)
                tempf[itemp] = (tbeta * tempf[itemp] + twt * tempr_arr[itemp]) / (tbeta + twt)
                print(f"    After trans: T_eff = {tempf[itemp]:.3f}")

            # Discrete oscillators
            if nd > 0:
                dwpix[itemp], tempf[itemp] = discre(
                    ssm[:, :, itemp], alpha, beta, nalpha, nbeta,
                    lat, arat, tev, twt, tbeta, nd, bdel, adel,
                    dwpix[itemp], tempf[itemp], tempr_arr[itemp], iprint)
                print(f"    After discre: DW = {dwpix[itemp]:.6f}, T_eff = {tempf[itemp]:.3f}")

            # Cold hydrogen/deuterium. ssp is already allocated upfront
            # whenever ncold != 0 (== ncold > 0, since ncold is validated to
            # 0..4), so it is never None here.
            if ncold > 0:
                coldh(ssm[:, :, itemp], ssp[:, :, itemp],
                      alpha, beta, nalpha, nbeta, lat, arat, tev,
                      twt, tbeta, ncold, ska, nka, dka,
                      tempf[itemp], tempr_arr[itemp], iprint)

            # Skold option
            if nsk == 2 and ncold == 0:
                skold_approx(ssm, alpha, beta, nalpha, nbeta,
                            itemp, ntempr, lat, arat, awr, tev,
                            ska, nka, dka, cfrac)

        # Save or merge for secondary scatterer
        if nss == 0 or b7 > 0.0 or isecs > 0:
            idone = True
        else:
            isecs += 1
            ssm_principal = ssm.copy()
            for itemp in range(ntempr):
                tempf1[itemp] = tempf[itemp]
                dwp1[itemp] = dwpix[itemp]

    # Merge mixed moderator if needed
    if nss != 0 and b7 <= 0.0 and ssm_principal is not None:
        sb = spr * ((1.0 + awr) / awr)**2
        sbs = sps * ((1.0 + aws) / aws)**2
        srat = sbs / sb
        # vectorized mixed-moderator merge over the full (nbeta, nalpha, ntempr)
        # array; byte-identical to the triple loop (per-element srat*ssm +
        # ssm_principal in float64), assigned in place to keep the same array.
        ssm[:] = srat * ssm + ssm_principal

    # Coherent elastic
    bragg = []
    nedge = 0
    if iel == 10 and crystal_info is not None:
        # Generalized Bragg edge calculation
        # Compute dcutoff from emax: d_min = sqrt(WL2EKIN / (4*emax))

        emax_bragg = 5.0
        dcutoff = sqrt(WL2EKIN / (4.0 * emax_bragg)) * 0.95  # 5% margin
        bragg_data, nedge, species_corr, bragg_dir_terms = compute_bragg_edges_general(
            crystal_info['crystal'], emax=emax_bragg, dcutoff=dcutoff)
        # Convert from numpy array to list of (E, delta) tuples
        bragg = [(bragg_data[i, 0], bragg_data[i, 1]) for i in range(nedge)]
        crystal_info['species_corr'] = species_corr
        crystal_info['bragg_dir_terms'] = bragg_dir_terms
        print(f"  Generalized: found {nedge} Bragg edges below 5 eV")

        # Compute per-species MSD/DW for elastic scattering. In the
        # phonopy-backed modes the directional tensors supersede these
        # lambdas (resolve_species_dw takes the use_dir_dw branch), so the
        # inherited-lambda warning is suppressed there.
        _compute_per_species_msd(
            crystal_info, tempr_arr, ntempr, dwpix,
            directional_dw=crystal_info.get('inelastic_mode', 0) in (1, 2))

    elif iel > 0 and iel != 10:
        bragg, nedge = coher(iel, npr, 5.0)
        print(f"  Found {nedge} Bragg edges below 5 eV")

    # Set iel=-1 for incoherent elastic if no translational and no coherent
    if iel == 0 and twt == 0.0:
        iel = -1

    # Determine symmetry
    isym = 0
    if ncold != 0:
        isym = 1
    if isabt == 1:
        isym += 2

    # Read comment cards for MF1/MT451
    comments = reader.read_comment_strings()
    # Stamp the extinction provenance into the MF1/MT451 free-text description so
    # the tape records that it is a sample-specific, extinction-corrected
    # evaluation. Append only when a comment block is present (>=5 header records),
    # so the line lands in the description region, not a header field.
    _ext = (crystal_info or {}).get('coherent_extinction')
    if _ext and len(comments) >= 5:
        comments = list(comments) + [extinction_provenance_comment(_ext)]
    if comments:
        print(f"  Read {len(comments)} comment cards")

    # Write ENDF output
    print("\n  Writing ENDF output...")
    write_endf_output(output_file, mat, za, awr, spr, npr, iel, ncold, nss,
                      b7, aws, sps, mss, nalpha, nbeta, lat,
                      alpha, beta, ssm, ssp, tempr_arr, ntempr,
                      dwpix, dwp1, tempf, tempf1,
                      bragg, nedge, isym, ilog, smin, iprint,
                      iint=iint, comments=comments, crystal_info=crystal_info)

    print("  IRMA complete.")
    return LeaprResult(
        output_file=str(output_file),
        title=str(title),
        ntempr=int(ntempr),
        temperatures_K=tuple(float(t) for t in tempr_arr),
        nedge=int(nedge),
        iel=int(iel),
        isym=int(isym),
        teff_K=tuple(float(t) for t in tempf),
        dw_lambda=tuple(float(w) for w in dwpix),
    )
