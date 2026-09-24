"""IRMA's main LEAPR-style driver: ``run_leapr`` and its phonopy/MT4 helpers.

The card-by-card deck format this driver parses is documented in
``docs/input-reference.md`` ("Input deck reference" in the manual); the
``iel=10`` Card 6b-6g block is parsed by :mod:`irma.core.crystal_cards`.
"""
import dataclasses
from pathlib import Path

import numpy as np

from irma.core.constants import BK
from irma.core.deck import (
    DeckError,
    parse_leapr_input, TokenReader,
    _read_temperature_detail_cards,
)
from irma.core.kernels import (
    contin, mean_tbar, trans, discre, coldh, skold_approx,
)
from irma.core.crystal import (
    compute_bragg_edges_general, coher,
    _compute_per_species_msd,
    _order_site_groups_by_card6d_positions, _site_tensors_uniform,
)
from irma.core.crystal_cards import _parse_crystal_cards
from irma.core.endf_writer import write_endf_output


def _noncubic_atom_dos(nc_mesh):
    """Stability warnings, then the per-atom DOS for the mode-1/2 T_eff record."""
    from irma.core.phonopy_io import compute_atom_dos, warn_dynamic_instability
    warn_dynamic_instability(nc_mesh.frequencies_ev, nc_mesh.qpoints)
    freq_max_ev = float(np.max(nc_mesh.frequencies_ev))
    if freq_max_ev <= 0.0:
        raise DeckError(
            f"the phonon model has no positive modes (max {freq_max_ev * 1000.0:.3f} "
            "meV); check the force constants")
    freq_max_ev *= 1.1  # 10% headroom
    n_freq = max(500, min(1000, int(round(freq_max_ev / 0.001)) + 1))
    print(f"    Non-cubic DOS: freq_max={freq_max_ev*1000:.1f} meV, "
          f"n_freq={n_freq}")
    return compute_atom_dos(nc_mesh, freq_max_ev, n_freq)


def _noncubic_mt4_step(crystal_info, ssm, itemp, alpha, beta, nalpha, nbeta,
                       lat, tev, temperature_k, awr, nphon, atom_dos,
                       energy_grid_ev):
    """One temperature of the mode-1/2 path: compute the in-process
    noncubic SAB and inject it into ssm[:, :, itemp]. Returns (f0, tbar,
    deltab, F_matrix_all) for the ENDF bookkeeping."""
    from irma.core.phonopy_io import (
        compute_thermal_displacement_matrices,
        thermal_displacements_to_f_matrix,
    )
    from irma.core.standalone_sab import run_noncubic_standalone_sab

    nc_mesh = crystal_info['nc_mesh_data']
    principal_site_indices = crystal_info['principal_nc_site_indices']
    atom_types_expanded = crystal_info['nc_atom_types_expanded']
    principal = crystal_info['atom_types'][crystal_info['principal_atom_idx']]

    thermal_mats_all = compute_thermal_displacement_matrices(nc_mesh, temperature_k)
    F_matrix_all = thermal_displacements_to_f_matrix(
        thermal_mats_all, crystal_info['nc_awr_by_atom'], tev)
    # tbar and deltab for the SCT/Teff records.
    tbar, deltab = mean_tbar(
        atom_dos[principal_site_indices], energy_grid_ev, tev, 1.0)
    f0 = float(np.mean([
        np.trace(F_matrix_all[d_idx]) / 3.0
        for d_idx in principal_site_indices
    ]))
    inelastic_mode_value = crystal_info['inelastic_mode']
    standalone_result = run_noncubic_standalone_sab(
        alpha=alpha[:nalpha],
        beta=beta[:nbeta],
        lat=lat,
        temperature_k=temperature_k,
        awr=awr,
        # MT4 is per principal scatterer; the one-phonon kernel is summed over
        # all phonopy sites of that scatterer.
        represented_principal_site_count=len(principal_site_indices),
        principal_group_index=crystal_info['principal_atom_idx'],
        site_groups=crystal_info['nc_atom_type_site_groups'],
        sab_sigma_barn=principal['sigma_coh'] + principal['sigma_inc'],
        phonopy_yaml_path=crystal_info['nc_phonopy_yaml_path'],
        mesh_dim=crystal_info['nc_mesh_dim'],
        born_path=crystal_info['nc_born_path'],
        num_jobs=crystal_info['nc_ncpu'],
        inelastic_mode=inelastic_mode_value,
        site_scattering_lengths_angstrom=[
            float(at['b_coh']) * 1.0e-5 for at in atom_types_expanded
        ],
        site_incoherent_cross_sections_barn=[
            float(at['sigma_inc']) for at in atom_types_expanded
        ],
        controls=crystal_info['nc_inelastic_controls'],
        context_cache=crystal_info.setdefault("nc_inelastic_context_cache", {}),
        preloaded_full_mesh=nc_mesh.phonopy_mesh_object,
        precomputed_thermal_mats=thermal_mats_all,
    )
    ssm[:, :, itemp] = standalone_result['ssm_internal']
    print(
        f"    Inelastic mode {inelastic_mode_value} MT4: injected in-process SAB "
        f"({standalone_result['selected_sab_key']})"
    )

    return f0, tbar, deltab, F_matrix_all


def _store_directional_species_dw(crystal_info, itemp, ntempr, F_matrix_all):
    """Store the per-species averaged and per-site Debye-Waller F-matrices
    that _build_sef_coherent uses for directional attenuation
    (inelastic_mode=1/2), rotated into the comb's frame when Card 6c is the
    phonopy cell (``nc_frame_rotation``)."""
    site_groups = crystal_info['nc_atom_type_site_groups']
    atom_types_list = crystal_info['atom_types']
    nsp = len(atom_types_list)
    F_arr = np.asarray(F_matrix_all)
    M = crystal_info.get('nc_frame_rotation')
    if M is not None:
        F_arr = M.T @ F_arr @ M
    F_species = np.zeros((nsp, 3, 3))
    for si in range(nsp):
        F_species[si] = F_arr[site_groups[si]].mean(axis=0)
    crystal_info.setdefault('F_species_per_temp', [None] * ntempr)[itemp] = F_species

    # Per-site F-matrices in crystal.py's site_terms order (species, then
    # Card 6d position order). When every group's tensors are identical the
    # elastic kernels keep the species-averaged path.
    ordered_groups = _order_site_groups_by_card6d_positions(
        atom_types_list, site_groups, crystal_info['nc_mesh_data'].atom_positions)
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
                f"Card 6d atom type {si+1}: cannot pair its positions with the "
                f"phonopy sites of its group, and the per-site Debye-Waller "
                f"tensors differ; check that the Card 6d positions match the "
                f"phonopy primitive cell.")
    crystal_info.setdefault('F_sites_per_temp', [None] * ntempr)[itemp] = (
        np.concatenate(F_sites, axis=0))
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
        at ``output_file``; progress is printed to stdout.

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
    reader.card("Card 4 (mat za isabt ilog smin [iint nver lrel])")
    fvals = reader.read_floats(8, defaults=[0, 0, 0, 0, 1.0e-75, 0, 8, 1])
    mat = reader.to_int(fvals[0], "mat")
    za = reader.to_int(fvals[1], "za")
    isabt = reader.to_int(fvals[2], "isabt")
    ilog = reader.to_int(fvals[3], "ilog")
    smin = fvals[4]
    # iint: MF7/MT4 interpolation, 0 = log-lin (INT=4, NJOY default), 1 = lin-lin (INT=2).
    iint = reader.to_int(fvals[5], "iint")
    # nver, lrel: MF1/MT451 library version and release (ENDF/B-VIII.1 -> 8 1).
    nver = reader.to_int(fvals[6], "nver")
    lrel = reader.to_int(fvals[7], "lrel")
    reader.require(mat >= 1, f"mat must be >= 1, got {mat}")
    reader.require(za > 0, f"za must be > 0, got {za}")
    reader.require(isabt in (0, 1), f"isabt must be 0 or 1, got {isabt}")
    reader.require(ilog in (0, 1), f"ilog must be 0 or 1, got {ilog}")
    reader.require(iint in (0, 1), f"iint must be 0 or 1, got {iint}")
    reader.require(nver >= 1, f"nver must be >= 1, got {nver}")
    reader.require(lrel >= 0, f"lrel must be >= 0, got {lrel}")
    reader.require(np.isfinite(smin) and smin >= 0.0,
                   f"smin (S cutoff) must be finite and >= 0, got {smin!r}")

    print(f"  ntempr={ntempr}, iprint={iprint}, nphon={nphon}")
    print(f"  mat={mat}, za={za}, isabt={isabt}, ilog={ilog}, iint={iint}, "
          f"nver={nver}, lrel={lrel}")

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
        print("WARNING: nsk=1 (Vineyard) is read for NJOY compatibility but is "
              "not implemented (as in NJOY2016); use nsk=2.", flush=True)

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
        # The two-pass merge carries only the symmetric law, not ssp.
        reader.require(
            not (ncold > 0 and b7 <= 0.0),
            "a bound two-pass secondary scatterer (b7<=0) cannot be combined "
            "with ncold>0; use b7=1 or 2, or nss=0")
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

    def _check_grid(name, grid, allow_zero):
        """Require a strictly increasing grid with a valid first value."""
        reader.require(grid[0] >= 0.0 if allow_zero else grid[0] > 0.0,
                       f"{name} values must be {'>= 0' if allow_zero else '> 0'} "
                       f"(first is {grid[0]:g})")
        bad = np.flatnonzero(np.diff(grid) <= 0)
        if bad.size:
            i = int(bad[0])
            reader.require(False,
                           f"{name} grid must be strictly increasing "
                           f"({name}[{i+2}]={grid[i+1]:g} follows "
                           f"{name}[{i+1}]={grid[i]:g})")

    # Card 8: alpha values
    reader.card(f"Card 8 ({nalpha} alpha values)")
    alpha = reader.read_float_array(nalpha)
    _check_grid("alpha", alpha, allow_zero=False)

    # Card 9: beta values
    reader.card(f"Card 9 ({nbeta} beta values)")
    beta = reader.read_float_array(nbeta)
    _check_grid("beta", beta, allow_zero=True)

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

    phonopy_mt4 = iel == 10 and crystal_info['inelastic_mode'] in (1, 2)
    if phonopy_mt4:
        atom_dos, energy_grid_ev = _noncubic_atom_dos(
            crystal_info['nc_mesh_data'])

    # A bound (b7 <= 0) secondary scatterer is a second pass over the
    # temperatures, merged with the principal's law afterwards.
    npass = 2 if nss != 0 and b7 <= 0.0 else 1
    ssm_principal = None

    for isecs in range(npass):
        if isecs == 0:
            arat = 1.0
            print("\n  Principal scatterer...")
        else:
            arat = aws / awr
            print(f"\n  Secondary scatterer (alpha scaled by {arat:.3f})...")

        # Temperature-detail-card state. A negative temperature card reuses
        # the previous temperature's detail block, so the state carries over.
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
            # The coherent-elastic edge thinning uses the first (coldest)
            # temperature, and THERMR reads the MT4 temperatures in order.
            reader.require(itemp == 0 or abs(temp) > tempr_arr[itemp - 1],
                           f"temperatures must increase: {abs(temp):g} K follows "
                           f"{tempr_arr[itemp - 1]:g} K")
            tempr_arr[itemp] = abs(temp)
            tev = BK * abs(temp)
            print(f"  Temperature {itemp+1}: {abs(temp):.2f} K")

            if phonopy_mt4:
                if itemp == 0:
                    print("    Inelastic mode 1/2: using noncubic in-process MT4 path; "
                          "the temperature-detail cards (Cards 11-19) are not read")
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

            if phonopy_mt4:
                f0, tbar, deltab, F_matrix_all = _noncubic_mt4_step(
                    crystal_info, ssm, itemp, alpha, beta, nalpha, nbeta,
                    lat, tev, tempr_arr[itemp], awr, nphon,
                    atom_dos, energy_grid_ev)
                _store_directional_species_dw(crystal_info, itemp, ntempr,
                                              F_matrix_all)
            else:
                f0, tbar, deltab = contin(ssm[:, :, itemp], alpha, beta,
                                          nalpha, nbeta, lat, arat, tev,
                                          p1, np1, delta1, tbeta, nphon)
            # Modes 1/2: f0 is the principal's Tr(F)/3 from the thermal-
            # displacement matrices, tbar comes from the DOS-tensor start().
            dwpix[itemp] = f0
            tempf[itemp] = tbar * tempr_arr[itemp]

            print(f"    DW lambda = {f0:.6f}, T_eff = {tempf[itemp]:.3f}")

            # Translational part
            if twt > 0.0:
                trans(ssm[:, :, itemp], alpha, beta, nalpha, nbeta,
                      lat, arat, tev, twt, c_diff, tbeta, f0, deltab,
                      tbar)
                tempf[itemp] = (tbeta * tempf[itemp] + twt * tempr_arr[itemp]) / (tbeta + twt)
                print(f"    After trans: T_eff = {tempf[itemp]:.3f}")

            # Discrete oscillators
            if nd > 0:
                dwpix[itemp], tempf[itemp] = discre(
                    ssm[:, :, itemp], alpha, beta, nalpha, nbeta,
                    lat, arat, tev, twt, tbeta, nd, bdel, adel,
                    dwpix[itemp], tempf[itemp], tempr_arr[itemp])
                print(f"    After discre: DW = {dwpix[itemp]:.6f}, T_eff = {tempf[itemp]:.3f}")

            # Cold hydrogen/deuterium
            if ncold > 0:
                coldh(ssm[:, :, itemp], ssp[:, :, itemp],
                      alpha, beta, nalpha, nbeta, lat, arat, tev,
                      twt, tbeta, ncold, ska, nka, dka,
                      tempf[itemp], tempr_arr[itemp])

            # Skold option
            if nsk == 2 and ncold == 0:
                skold_approx(ssm, alpha, nalpha, nbeta, itemp, lat, arat,
                             awr, tev, ska, nka, dka, cfrac)

        if isecs == 0 and npass == 2:
            ssm_principal = ssm.copy()
            tempf1[:] = tempf
            dwp1[:] = dwpix

    # Merge mixed moderator if needed
    if npass == 2:
        sb = spr * ((1.0 + awr) / awr)**2
        sbs = sps * ((1.0 + aws) / aws)**2
        srat = sbs / sb
        ssm[:] = srat * ssm + ssm_principal

    # Coherent elastic
    bragg = []
    nedge = 0
    if iel == 10:
        bragg_data, nedge, species_corr, bragg_dir_terms = compute_bragg_edges_general(
            crystal_info['crystal'], emax=5.0)
        # Convert from numpy array to list of (E, delta) tuples
        bragg = [(bragg_data[i, 0], bragg_data[i, 1]) for i in range(nedge)]
        crystal_info['species_corr'] = species_corr
        crystal_info['bragg_dir_terms'] = bragg_dir_terms
        print(f"  Generalized: found {nedge} Bragg edges below 5 eV")

        # Per-species Debye-Waller lambdas; modes 1/2 use the directional
        # tensors instead.
        if phonopy_mt4:
            print("    Per-species elastic Debye-Waller comes from the phonopy "
                  "displacement tensors (inelastic_mode 1/2).")
        else:
            _compute_per_species_msd(crystal_info, tempr_arr, ntempr, dwpix)

    elif iel > 0:
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
    for n, text in enumerate(comments, 1):
        bad = next((ch for ch in text if ord(ch) > 127), None)
        # ENDF-6 is fixed-column ASCII: one wider character shifts MAT/MF/MT
        reader.require(bad is None,
                       f"comment card {n} contains the non-ASCII character "
                       f"{bad!r}; ENDF-6 tapes are ASCII")
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
    write_endf_output(output_file, mat, za, awr, spr, npr, iel, nss,
                      b7, aws, sps, mss, nalpha, nbeta, lat,
                      alpha, beta, ssm, ssp, tempr_arr, ntempr,
                      dwpix, dwp1, tempf, tempf1,
                      bragg, nedge, isym, ilog, smin,
                      iint=iint, comments=comments, crystal_info=crystal_info,
                      nver=nver, lrel=lrel)

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
