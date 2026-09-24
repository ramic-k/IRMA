"""Crystal/elastic card parsing for the generalized ``iel=10`` path.

The card-by-card reference for the Cards 6b-6g parsed here (fields, defaults,
mode interactions, worked decks) is ``docs/input-reference.md``.
"""
import os
from math import cos, radians, pi

import numpy as np

from irma.core.deck import DeckError
from irma.core.crystal import (
    AtomSite, CrystalStructure,
    _build_atom_types_expanded, _group_phonopy_atoms_by_type,
    lattice_to_cell_params, standard_frame_rotation,
)
from irma.core.noncubic_inelastic import NoncubicInelasticControls


def _parse_extinction_card(reader, elastic_mode):
    """Parse the optional crystalline-extinction card (off by default).

        extinction <model> l=<Å> g=<rad⁻¹> L=<Å> [dist=<...>] [rec=cls|std] [rmse_tol=<frac>]

    Placed at the end of the iel=10 elastic block (after Cards 6d/6e, or 6g for
    inelastic_mode=1/2), before Card 7. Returns the ``coherent_extinction`` config
    dict consumed by the ENDF writer, or ``None`` if the card is absent.
    Extinction is a *sample* property — l (crystallite
    size), g (mosaic), L (grain) come from a fit-to-transmission or microstructure.
    See ``irma/core/extinction.py`` for the model definitions and references.
    """
    tok = reader.peek_token()
    if not (isinstance(tok, str) and tok.lower() == "extinction"):
        return None

    from irma.core.extinction import (
        EXTINCTION_MODELS, RECIPES, _SABINE_DIST, _BC_DIST)

    reader.card("extinction card (model l= g= L= [dist=] [rec=] [rmse_tol=])")
    toks = reader.read_card_tokens()                 # includes the 'extinction' keyword
    reader.require(
        len(toks) >= 2,
        "extinction card needs a model name, e.g. "
        "'extinction BC_mix l=8550 g=170 L=75750'")
    model = toks[1]
    reader.require(
        isinstance(model, str) and model in EXTINCTION_MODELS,
        f"extinction model must be one of {EXTINCTION_MODELS}, got {model!r}")

    l = g = L = 0.0
    dist = None
    recipe = "std"
    rmse_tol = 1e-3
    seen = set()
    for t in toks[2:]:
        reader.require(
            isinstance(t, str) and "=" in t,
            f"extinction field {t!r} must be key=value (e.g. l=8550, dist=Gauss)")
        key, _, val = t.partition("=")
        if key not in ("l", "L"):           # l (crystallite) vs L (grain) are case-sensitive
            key = key.lower()
        # Reject a repeated key rather than silently letting the last token win,
        # which would hide a typo (e.g. `l=8550 l=0`) that changes the sample.
        reader.require(key not in seen,
                       f"extinction field {key!r} given more than once")
        seen.add(key)

        def _f(name):
            """Parse field ``name`` as float, accepting Fortran D-exponents."""
            try:
                # accept Fortran D-exponent notation (e.g. 1.0d4), as the deck
                # tokenizer does elsewhere -- see irma/core/deck.py.
                return float(val.lower().replace('d', 'e'))
            except (ValueError, AttributeError):
                reader.require(False, f"extinction {name} must be a number, got {val!r}")
        if key == "l":
            l = _f("l")
        elif key == "L":
            L = _f("L")
        elif key == "g":
            g = _f("g")
        elif key == "dist":
            dist = val
        elif key == "rec":
            recipe = val
        elif key == "rmse_tol":
            rmse_tol = _f("rmse_tol")
        else:
            reader.require(
                False,
                f"unknown extinction field {key!r} "
                f"(allowed: l, g, L, dist, rec, rmse_tol)")

    # ---- validation ----
    for nm, v in (("l", l), ("g", g), ("L", L)):
        reader.require(np.isfinite(v) and v >= 0.0,
                       f"extinction {nm} must be finite and >= 0, got {v}")
    reader.require(recipe in RECIPES,
                   f"extinction rec must be one of {RECIPES}, got {recipe!r}")
    reader.require(rmse_tol > 0.0 and np.isfinite(rmse_tol),
                   f"extinction rmse_tol must be a positive, finite number, "
                   f"got {rmse_tol}")
    if model.startswith("Sabine"):
        allowed, default = set(_SABINE_DIST), "rect"
    else:
        allowed, default = set(_BC_DIST), "Gauss"
    dist = default if dist is None else dist
    reader.require(dist in allowed,
                   f"extinction dist for {model} must be one of {sorted(allowed)}, "
                   f"got {dist!r}")
    # A combination a model does not handle would silently give y=1.
    reader.require(
        l > 0.0 or (g > 0.0 and L > 0.0),
        f"extinction {model}: no active mechanism — set l>0 (primary) or "
        f"g>0 with L>0 (secondary)")
    if model in ("BC_mix", "BC_mod"):
        reader.require(
            l > 0.0 and g > 0.0 and L > 0.0,
            f"extinction {model} needs l>0, g>0 and L>0 (got l={l:g} g={g:g} "
            f"L={L:g}); for primary only, use BC_pure with only l set")
    elif model == "BC_pure":
        reader.require(
            not (l > 0.0 and g > 0.0),
            f"extinction BC_pure is primary OR secondary, not both: set l>0 with "
            f"g=0, or l=0 with g>0 and L>0 (got l={l:g} g={g:g})")

    print(f"  Coherent-elastic EXTINCTION ON: model={model}, l={l:g} Å, g={g:g}, "
          f"L={L:g} Å, dist={dist}, rec={recipe}, rmse_tol={rmse_tol:g}")
    return {"model": model, "l": l, "g": g, "L": L,
            "dist": dist, "recipe": recipe, "rmse_tol": rmse_tol}


def _dw_frame_rotation(card6c, lattice_ang):
    """Rotation taking the phonopy Debye-Waller tensors into the frame of the
    Card 6c comb (a along x, b in the xy-plane), or None.

    Applied only when Card 6c is the phonopy cell (lengths within 1e-3
    relative, angles within 0.01 degrees): Card 6c may describe another cell,
    and then the tensors stay as they are, with a NOTE.
    """
    M = standard_frame_rotation(lattice_ang)
    if M is None:
        return None
    ph = lattice_to_cell_params(lattice_ang)
    if (all(abs(x - y) <= 1e-3 * y for x, y in zip(card6c[:3], ph[:3]))
            and all(abs(x - y) <= 0.01 for x, y in zip(card6c[3:], ph[3:]))):
        return M
    print("  NOTE: the phonopy cell is not oriented with a along x and b in the "
          "xy-plane, and Card 6c describes a different cell, so the "
          "Debye-Waller tensors of the coherent elastic are not rotated")
    return None


def _parse_crystal_cards(reader, za, nphon, ncold=0, nsk=0, nss=0, b7=0.0):
    """Parse the generalized-elastic card block (iel=10): Cards 6b-6g.

    Reads the elastic/inelastic mode controls, lattice, atom types and
    positions, the Card 6e partial spectra (mode 0), and — for inelastic_mode=1/2 —
    the phonopy mesh controls (loading the mesh once for all
    temperatures). Returns the populated crystal_info dict.
    """
    # Card 6b: elastic_mode nat nspec inelastic_mode [bins_per_decade threshold_eV]
    reader.card("Card 6b (elastic_mode nat nspec inelastic_mode [grouping])")
    fvals = reader.read_floats(6, defaults=[0, 0, 0, 0, 0, 0])
    elastic_mode = reader.to_int(fvals[0], "elastic_mode")   # 1=SEF, 2=MEF
    nat = reader.to_int(fvals[1], "nat")            # number of distinct atom types
    nspec = reader.to_int(fvals[2], "nspec")          # number of partial phonon spectra (Card 6e blocks)
    inelastic_mode = reader.to_int(fvals[3], "inelastic_mode")   # 0=isotropic, 1/2=in-process noncubic SAB
    # Bragg-edge grouping knob (field 5) + threshold in eV (field 6).
    edge_group_bins_per_decade = reader.to_int(fvals[4], "bins_per_decade")  # 0 = off (keep all edges)
    reader.require(
        edge_group_bins_per_decade >= 0,
        f"Card 6b bins_per_decade (field 5) must be >= 0 (0 = grouping off), "
        f"got {edge_group_bins_per_decade}")
    reader.require(
        fvals[5] >= 0.0,
        f"Card 6b grouping threshold (field 6) must be >= 0, got {fvals[5]:g}")
    edge_group_threshold_ev = (
        float(fvals[5]) if fvals[5] > 0 else 1.0             # ENDF-102 default: 1 eV
    )

    reader.require(elastic_mode in (1, 2),
                   f"elastic_mode must be 1 (SEF) or 2 (MEF), got {elastic_mode}")
    reader.require(nat >= 1, f"nat must be >= 1, got {nat}")
    reader.require(inelastic_mode in (0, 1, 2),
                   f"inelastic_mode must be 0, 1, or 2; got {inelastic_mode}")
    # Modes 1/2 have no S(kappa) tables and no secondary-scatterer pass.
    if inelastic_mode in (1, 2):
        reader.require(
            ncold == 0 and nsk == 0,
            f"ncold/nsk pair-correlation options are not available with "
            f"inelastic_mode={inelastic_mode} "
            f"(got ncold={ncold}, nsk={nsk})")
        reader.require(
            nss == 0,
            f"a secondary scatterer (Card 6 nss={nss}) is not supported "
            f"with inelastic_mode={inelastic_mode}")
    # The two-pass merge leaves the secondary's Debye-Waller data in place,
    # and the iel=10 MT2 builder would use it for the principal.
    reader.require(
        not (nss > 0 and b7 <= 0.0),
        f"a bound two-pass secondary scatterer (Card 6 nss={nss}, b7={b7:g}) "
        f"is not supported with iel=10, which has no secondary-scatterer "
        f"Debye-Waller path; use b7=1 or 2, or drop the secondary")
    # Modes 1/2 take MT4 and the Debye-Waller factors from phonopy.
    reader.require(
        not (inelastic_mode in (1, 2) and nspec != 0),
        f"nspec must be 0 when inelastic_mode={inelastic_mode}: "
        f"Phonopy provides MT4 and the Debye-Waller factors, so Card 6e "
        f"partial spectra are not used (got nspec={nspec})")
    reader.require(nspec >= 0, f"nspec must be >= 0, got {nspec}")

    print(f"  Generalized elastic: elastic_mode={elastic_mode} "
          f"({'SEF' if elastic_mode == 1 else 'MEF'}), nat={nat}, nspec={nspec}, "
          f"inelastic_mode={inelastic_mode}")

    # Card 6c: lattice parameters
    reader.card("Card 6c (lattice a b c alpha beta gamma)")
    fvals = reader.read_floats(6)
    latt_a, latt_b, latt_c = fvals[0], fvals[1], fvals[2]
    latt_alpha, latt_beta, latt_gamma = fvals[3], fvals[4], fvals[5]

    print(f"  Lattice: a={latt_a}, b={latt_b}, c={latt_c}, "
          f"alpha={latt_alpha}, beta={latt_beta}, gamma={latt_gamma}")

    # The cell volume clips a negative metric term to zero, so reject an
    # unphysical cell here.
    for nm, v in (("a", latt_a), ("b", latt_b), ("c", latt_c)):
        reader.require(v > 0.0, f"lattice {nm} must be > 0, got {v}")
    for nm, ang in (("alpha", latt_alpha), ("beta", latt_beta),
                    ("gamma", latt_gamma)):
        reader.require(0.0 < ang < 180.0,
                       f"lattice angle {nm} must be in (0, 180) degrees, "
                       f"got {ang}")
    ca = cos(radians(latt_alpha))
    cb = cos(radians(latt_beta))
    cg = cos(radians(latt_gamma))
    metric = 1.0 - ca * ca - cb * cb - cg * cg + 2.0 * ca * cb * cg
    reader.require(metric > 0.0,
                   "lattice angles do not form a valid cell "
                   f"(metric determinant term {metric:.6g} <= 0); "
                   f"alpha={latt_alpha}, beta={latt_beta}, gamma={latt_gamma}")

    # Card 6d: atom types (repeated nat times)
    atom_types = []
    sites = []
    total_atoms_in_cell = 0

    for iat in range(nat):
        # First line: Z A awr_i b_coh sigma_inc npos
        reader.card(f"Card 6d (atom type {iat+1}: Z A awr b_coh sigma_inc npos)")
        fvals = reader.read_floats(6)
        at_Z = reader.to_int(fvals[0], "Z")
        at_A = reader.to_int(fvals[1], "A")
        at_awr = fvals[2]
        at_b_coh = fvals[3]       # fm
        at_sigma_inc = fvals[4]   # barns
        at_npos = reader.to_int(fvals[5], "npos")
        reader.require(at_Z >= 1, f"Z must be >= 1, got {at_Z}")
        # A only identifies the nuclide (matched to Card 4 za and Card 6e);
        # A = 0 is ENDF's natural element.
        reader.require(at_A >= 0, f"A must be >= 0 (0 = natural "
                                  f"element), got {at_A}")
        reader.require(at_awr > 0.0, f"awr must be > 0, got {at_awr}")
        reader.require(at_sigma_inc >= 0.0,
                       f"sigma_inc must be >= 0, got {at_sigma_inc}")
        reader.require(at_npos >= 1, f"npos must be >= 1, got {at_npos}")

        # Read fractional coordinates (npos × 3 values)
        reader.card(f"Card 6d (atom type {iat+1}: {at_npos} fractional positions)")
        coords_flat = reader.read_float_array(at_npos * 3)
        positions = []
        for ip in range(at_npos):
            x = coords_flat[3 * ip]
            y = coords_flat[3 * ip + 1]
            z = coords_flat[3 * ip + 2]
            positions.append((x, y, z))

        sigma_coh = 4.0 * pi * at_b_coh**2 * 0.01  # barns (1 barn = 100 fm²)
        total_atoms_in_cell += at_npos

        atom_types.append({
            'Z': at_Z, 'A': at_A, 'awr': at_awr,
            'b_coh': at_b_coh, 'sigma_inc': at_sigma_inc,
            'sigma_coh': sigma_coh,
            'npos': at_npos, 'positions': positions,
        })

        sites.append(AtomSite(b_coh_fm=at_b_coh, positions=positions))

        print(f"    Atom {iat+1}: Z={at_Z}, A={at_A}, awr={at_awr:.4f}, "
              f"b_coh={at_b_coh:.4f} fm, sigma_inc={at_sigma_inc:.4f} b, "
              f"sigma_coh={sigma_coh:.4f} b, npos={at_npos}")

    za_Z, za_A = int(za) // 1000, int(za) % 1000
    # Merge Card 6d rows of the principal (Z, A) into one group: the MT4 law
    # accumulates over one principal group, and the SEF coherent comb counts
    # the principal once (a split principal doubled it in mode 0).
    _matches = [i for i, at in enumerate(atom_types)
                if at['Z'] == za_Z and at['A'] == za_A]
    if len(_matches) > 1:
        _first = atom_types[_matches[0]]
        for i in _matches[1:]:
            at = atom_types[i]
            reader.require(
                at['awr'] == _first['awr']
                and at['b_coh'] == _first['b_coh']
                and at['sigma_inc'] == _first['sigma_inc'],
                f"Card 6d atom types {_matches[0] + 1} and {i + 1} both "
                f"carry the principal nuclide Z={za_Z} A={za_A} but "
                f"differ in awr/b_coh/sigma_inc; the principal's entries "
                f"are merged into one group and must be identical")
        _merged_positions = []
        for i in _matches:
            _merged_positions.extend(atom_types[i]['positions'])
        _first['positions'] = _merged_positions
        _first['npos'] = len(_merged_positions)
        for i in reversed(_matches[1:]):
            del atom_types[i]
            del sites[i]
        sites[_matches[0]] = AtomSite(b_coh_fm=_first['b_coh'],
                                      positions=_merged_positions)
        print(f"    merged {len(_matches)} Card 6d entries of the principal "
              f"nuclide (Z={za_Z}, A={za_A}) into one group with "
              f"{_first['npos']} positions")

    # Compute fractions
    for at in atom_types:
        at['fraction'] = at['npos'] / total_atoms_in_cell

    # Card 6e: partial phonon spectra (per-species Debye-Waller input for
    # inelastic_mode=0; rejected above for modes 1/2).
    partial_spectra = []
    for isp in range(nspec):
        # First line: Z A delta_s ni_s
        reader.card(f"Card 6e (partial spectrum {isp+1}: Z A delta ni)")
        fvals = reader.read_floats(4)
        sp_Z = reader.to_int(fvals[0], "Z")
        sp_A = reader.to_int(fvals[1], "A")
        sp_delta = fvals[2]
        sp_ni = reader.to_int(fvals[3], "ni")
        # Same validity rules as the classic Card 11/12 spectrum: these
        # spectra feed the per-species Debye-Waller integrals directly.
        reader.require(sp_delta > 0.0,
                       f"delta (spectrum spacing, eV) must be > 0, "
                       f"got {sp_delta:g}")
        reader.require(sp_ni >= 2,
                       f"ni (number of spectrum points) must be >= 2, "
                       f"got {sp_ni}")

        # Read phonon spectrum values
        reader.card(f"Card 6e (partial spectrum {isp+1}: {sp_ni} rho values)")
        sp_rho = reader.read_float_array(sp_ni)
        reader.require(bool(np.all(sp_rho >= 0.0)), "rho values must be >= 0")
        reader.require(bool(np.any(sp_rho > 0.0)), "rho values are all zero")

        partial_spectra.append({
            'Z': sp_Z, 'A': sp_A,
            'delta': sp_delta, 'ni': sp_ni,
            'rho': sp_rho,
        })
        print(f"    Partial spectrum {isp+1}: Z={sp_Z}, A={sp_A}, "
              f"delta={sp_delta:.6e}, ni={sp_ni}")

    # Build crystal structure for Bragg edge calculation
    crystal = CrystalStructure(
        a=latt_a, b=latt_b, c=latt_c,
        alpha=latt_alpha, beta=latt_beta, gamma=latt_gamma,
        sites=sites,
    )

    # Match principal scatterer (za) to crystal atom type
    from irma.core.crystal_input import principal_mismatch_message
    mismatch = principal_mismatch_message(int(za), atom_types)
    reader.require(mismatch is None, mismatch)
    principal_atom_idx = next(i for i, at in enumerate(atom_types)
                              if at['Z'] == za_Z and at['A'] == za_A)

    print(f"  Principal scatterer (za={za}) matches atom type "
          f"{principal_atom_idx+1}: Z={za_Z}, A={za_A}, "
          f"fraction={atom_types[principal_atom_idx]['fraction']:.4f}")

    # Designated-coherent (DC) atom for SEF (Eq. 26 of Ramic et al., NIM-A 1027
    # (2022) 166227); a single-type cell has no channel competition (f_i=1
    # would divide by zero).
    dc_atom_idx = None
    if elastic_mode == 1 and len(atom_types) > 1:
        min_inc_criterion = float('inf')
        for i, at in enumerate(atom_types):
            f_i = at['fraction']
            criterion = f_i / (1.0 - f_i) * at['sigma_inc']
            if criterion < min_inc_criterion:
                min_inc_criterion = criterion
                dc_atom_idx = i

        dc_at = atom_types[dc_atom_idx]
        print(f"  DC atom (min incoherent contribution): type {dc_atom_idx+1}, "
              f"Z={dc_at['Z']}, A={dc_at['A']}, "
              f"f_DC={dc_at['fraction']:.4f}")

        if principal_atom_idx == dc_atom_idx:
            print("  -> Principal scatterer is the DC atom -> LTHR=1 (coherent elastic)")
        else:
            print("  -> Principal scatterer is not the DC atom -> LTHR=2 (incoherent elastic)")

    # Match each Card 6e spectrum to exactly one Card 6d atom type by (Z, A).
    reader.card("Card 6e (partial spectrum atom matching)")
    keys = [(at['Z'], at['A']) for at in atom_types]
    for at in atom_types:
        at['spectrum_idx'] = None
    for i, sp in enumerate(partial_spectra):
        key = (sp['Z'], sp['A'])
        reader.require(
            key in keys,
            f"Card 6e partial spectrum {i+1} (Z={key[0]}, A={key[1]}) "
            f"does not match any Card 6d atom type")
        reader.require(
            keys.count(key) == 1,
            f"duplicate Card 6d atom type Z={key[0]}, A={key[1]} with a "
            f"Card 6e partial spectrum for the same (Z, A)")
        at = atom_types[keys.index(key)]
        reader.require(
            at['spectrum_idx'] is None,
            f"duplicate Card 6e partial spectrum for Z={key[0]}, A={key[1]}")
        at['spectrum_idx'] = i

    # Store everything for later use
    crystal_info = {
        'elastic_mode': elastic_mode,
        'nat': len(atom_types),     # after the principal merge
        'inelastic_mode': inelastic_mode,
        'crystal': crystal,
        'atom_types': atom_types,
        'partial_spectra': partial_spectra,
        'principal_atom_idx': principal_atom_idx,
        'dc_atom_idx': dc_atom_idx,
        # Coherent-elastic Bragg-edge grouping (ENDF-102 7.2.2); 0 = off.
        'coh_edge_group_bins_per_decade': edge_group_bins_per_decade,
        'coh_edge_group_threshold_ev': edge_group_threshold_ev,
    }
    if edge_group_bins_per_decade > 0:
        print(f"  Coherent-elastic Bragg-edge grouping ON: "
              f"{edge_group_bins_per_decade} bins/decade above "
              f"{edge_group_threshold_ev:g} eV (ENDF-102 7.2.2)")

    # ---- Card 6f: phonopy mesh parameters (inelastic_mode=1/2) ----
    # The mesh is loaded once here and reused for all temperatures.
    if inelastic_mode in (1, 2):
        # Card 6f-1: phonopy_yaml_path
        reader.card("Card 6f-1 (phonopy.yaml path)")
        phonopy_yaml_path = reader.read_string()
        print(f"  Non-cubic: phonopy_yaml = {phonopy_yaml_path}")

        # Card 6f-2: mesh_nx mesh_ny mesh_nz ncpu use_born
        reader.card("Card 6f-2 (mesh_nx mesh_ny mesh_nz ncpu use_born)")
        fvals_nc = reader.read_card_floats()
        reader.require(
            len(fvals_nc) == 5,
            "Card 6f-2 needs 5 values: mesh_nx mesh_ny mesh_nz ncpu use_born")
        mesh_nx = reader.to_int(fvals_nc[0], "mesh_nx")
        mesh_ny = reader.to_int(fvals_nc[1], "mesh_ny")
        mesh_nz = reader.to_int(fvals_nc[2], "mesh_nz")
        nc_ncpu = reader.to_int(fvals_nc[3], "ncpu")
        nc_use_born = reader.to_int(fvals_nc[4], "use_born")
        reader.require(mesh_nx >= 1 and mesh_ny >= 1 and mesh_nz >= 1,
                       f"mesh dimensions must all be >= 1, got "
                       f"{mesh_nx}x{mesh_ny}x{mesh_nz}")
        reader.require(nc_ncpu >= 1, f"ncpu must be >= 1, got {nc_ncpu}")
        reader.require(nc_use_born in (0, 1),
                       f"use_born must be 0 or 1, got {nc_use_born}")
        # Guard against a self-inflicted OOM from starting far more
        # workers than there are cores: clamp to the available
        # CPU count with a warning rather than failing late.
        cpu_avail = os.cpu_count() or 1
        if nc_ncpu > cpu_avail:
            print(f"WARNING: Card 6f ncpu={nc_ncpu} exceeds available cores "
                  f"({cpu_avail}); clamping to {cpu_avail}.")
            nc_ncpu = cpu_avail

        print(f"  Non-cubic: mesh={mesh_nx}×{mesh_ny}×{mesh_nz}, "
              f"ncpu={nc_ncpu}, use_born={nc_use_born}")

        # Card 6f-3: born_path (only if use_born=1)
        born_path_nc = None
        if nc_use_born == 1:
            reader.card("Card 6f-3 (BORN file path)")
            born_path_nc = reader.read_string()
            print(f"  Non-cubic: BORN path = {born_path_nc}")

        # Optional one-value card: the minimum phonon energy in meV. A 2- or
        # 3-value card here is Card 6g.
        reader.card("optional minimum phonon energy / Card 6g")
        fvals_nc_ctrl = reader.read_card_floats()
        min_phonon_energy_mev = 0.0
        if len(fvals_nc_ctrl) == 1:
            min_phonon_energy_mev = float(fvals_nc_ctrl[0])
            reader.require(min_phonon_energy_mev >= 0.0,
                           "minimum phonon energy must be >= 0 (meV)")
            reader.card("Card 6g (ndir mpdir [auto])")
            fvals_nc_ctrl = reader.read_card_floats()

        # Card 6g: ndir mpdir [auto_multiphonon_order]; the order is Card 3 nphon.
        reader.require(
            len(fvals_nc_ctrl) in (2, 3),
            "Card 6g must contain 2 values plus an optional 3rd value: "
            "ndir mpdir [auto_multiphonon_order]")
        num_directions_nc = reader.to_int(fvals_nc_ctrl[0], "ndir")
        multiphonon_num_directions_nc = reader.to_int(fvals_nc_ctrl[1], "mpdir")
        multiphonon_max_order_nc = int(nphon)
        auto_multiphonon_order_nc = (
            reader.to_int(fvals_nc_ctrl[2], "auto_order") if len(fvals_nc_ctrl) == 3 else 0
        )
        reader.require(num_directions_nc >= 1 and multiphonon_num_directions_nc >= 1,
                       "Card 6g direction counts must be >= 1.")
        reader.require(auto_multiphonon_order_nc in (0, 1),
                       "Card 6g auto_multiphonon_order (3rd field) must be 0 "
                       "(honor Card 3 nphon) or 1 (auto-size).")
        nc_inelastic_controls = NoncubicInelasticControls(
            num_directions=num_directions_nc,
            multiphonon_num_directions=multiphonon_num_directions_nc,
            multiphonon_max_order=multiphonon_max_order_nc,
            auto_multiphonon_order=bool(auto_multiphonon_order_nc),
            min_phonon_energy_mev=min_phonon_energy_mev,
        )
        print(
            "  Non-cubic controls: "
            f"ndir={nc_inelastic_controls.num_directions}, "
            f"mpdir={nc_inelastic_controls.multiphonon_num_directions}, "
            f"mporder(from nphon)={nc_inelastic_controls.multiphonon_max_order}"
            f"{' [auto-size]' if nc_inelastic_controls.auto_multiphonon_order else ''}"
        )

        # Load phonopy mesh once (temperature-independent). Only the load
        # itself is wrapped as a runtime failure; the deck-vs-model
        # consistency checks below raise DeckError with card context.
        try:
            from irma.core.phonopy_io import load_phonopy_mesh
            nc_mesh_data = load_phonopy_mesh(
                phonopy_yaml_path, [mesh_nx, mesh_ny, mesh_nz],
                born_path=born_path_nc,
                min_phonon_energy_mev=min_phonon_energy_mev)
        except DeckError:
            raise   # deck problems keep their card/line context
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load phonopy mesh for non-cubic inelastic: {exc}") from exc
        if min_phonon_energy_mev > 0.0:
            from irma.core.phonopy_io import mode_floor_mask
            freq = np.asarray(nc_mesh_data.frequencies_ev, dtype=float)
            kept = mode_floor_mask(freq.reshape(-1) * 1.0e3, nc_mesh_data.qpoints,
                                   freq.shape[1], min_phonon_energy_mev)
            reader.require(
                np.any(kept),
                f"the minimum phonon energy {min_phonon_energy_mev:g} meV "
                f"removes every phonon mode; if this one-value card is a "
                f"Card 6g missing its mpdir field, write 'ndir mpdir'")

        # Deck-vs-phonopy-model consistency: mismatches here are CARD 6d
        # problems (wrong positions, wrong species, wrong principal ZA),
        # not phonopy failures — report them as deck errors.
        try:
            nc_atom_types_expanded = _build_atom_types_expanded(
                crystal_info, nc_mesh_data)
            nc_atom_type_site_groups = _group_phonopy_atoms_by_type(
                atom_types, nc_atom_types_expanded)
        except DeckError:
            raise
        except ValueError as exc:
            reader.require(False, f"Card 6d does not match the phonopy "
                                  f"model: {exc}")
        principal_nc_site_indices = list(nc_atom_type_site_groups[principal_atom_idx])
        crystal_info['nc_mesh_data'] = nc_mesh_data
        crystal_info['nc_frame_rotation'] = _dw_frame_rotation(
            (latt_a, latt_b, latt_c, latt_alpha, latt_beta, latt_gamma),
            nc_mesh_data.lattice_ang)
        crystal_info['nc_ncpu'] = nc_ncpu
        crystal_info['nc_phonopy_yaml_path'] = phonopy_yaml_path
        crystal_info['nc_mesh_dim'] = [mesh_nx, mesh_ny, mesh_nz]
        crystal_info['nc_born_path'] = born_path_nc
        crystal_info['nc_atom_types_expanded'] = nc_atom_types_expanded
        crystal_info['nc_awr_by_atom'] = np.array(
            [at['awr'] for at in nc_atom_types_expanded], dtype=float)
        crystal_info['nc_atom_type_site_groups'] = nc_atom_type_site_groups
        crystal_info['principal_nc_site_indices'] = principal_nc_site_indices
        crystal_info['nc_inelastic_controls'] = nc_inelastic_controls
        print(f"  Non-cubic: principal scatterer spans "
              f"{len(principal_nc_site_indices)} phonopy site(s)")

    # Optional crystalline-extinction card (off by default; last card of the
    # iel=10 elastic block, before Card 7). Sets crystal_info['coherent_extinction'].
    ext_cfg = _parse_extinction_card(reader, elastic_mode)
    if ext_cfg is not None:
        # SEF writes MT2 through the incoherent elastic builder, which ignores
        # extinction, for a single-atom principal with sigma_coh <= sigma_inc
        # or a polyatomic principal that is not the designated-coherent atom.
        if elastic_mode == 1:
            p = atom_types[principal_atom_idx]
            coherent = (p['sigma_coh'] > p['sigma_inc'] if len(atom_types) == 1
                        else principal_atom_idx == dc_atom_idx)
            reader.require(
                coherent,
                "extinction would be a silent no-op: SEF writes this principal "
                "through the incoherent elastic builder (sigma_coh <= sigma_inc, "
                "or not the designated-coherent atom); use elastic_mode=2 (MEF)")
        crystal_info['coherent_extinction'] = ext_cfg

    return crystal_info
