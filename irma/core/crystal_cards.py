"""Crystal/elastic card parsing for the generalized ``iel=10`` path.

``_parse_crystal_cards`` was extracted verbatim from :mod:`irma.core.engine`;
the ``run_leapr`` driver still calls it (and re-imports it) when ``iel == 10``.
Kept byte-for-byte identical so the emitted ENDF tape is unchanged.

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
)
from irma.core.noncubic_inelastic import NoncubicInelasticControls


def _parse_extinction_card(reader, elastic_mode):
    """Parse the OPTIONAL crystalline-extinction card (off by default).

        extinction <model> l=<Å> g=<rad⁻¹> L=<Å> [dist=<...>] [rec=cls|std] [rmse_tol=<frac>]

    Placed at the end of the iel=10 elastic block (after Cards 6d/6e, or 6g for
    inelastic_mode=1/2), before Card 7. Returns the ``coherent_extinction`` config
    dict consumed by the ENDF writer, or ``None`` if the card is absent (so every
    existing deck is unchanged). Extinction is a *sample* property — l (crystallite
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
    reader.require(
        elastic_mode in (1, 2),
        f"extinction needs a coherent-elastic channel (Card 6b elastic_mode=1 SEF "
        f"or 2 MEF), got elastic_mode={elastic_mode}")
    for nm, v in (("l", l), ("g", g), ("L", L)):
        reader.require(np.isfinite(v) and v >= 0.0,
                       f"extinction {nm} must be finite and >= 0, got {v}")
    reader.require(recipe in RECIPES,
                   f"extinction rec must be one of {RECIPES}, got {recipe!r}")
    reader.require(
        recipe != "lux",
        "extinction rec=lux is not implemented (its 1e-6 precision is far below the "
        "tape tolerance); use rec=std (default) or rec=cls")
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
    # Per-model parameter requirements that match what each model actually
    # computes. A combination the model does not handle silently returns y=1 (a
    # no-op "extinction" tape stamped as corrected), so reject it at parse time.
    # The handled (l, g, L) combinations are:
    #   BC_pure : primary (l>0, g=0) OR secondary (l=0, g>0, L>0) OR primary+grain
    #             type-II (l>0, g=0, L>0) — never coupled primary AND secondary.
    #   BC_mix / BC_mod : coupled — both need l>0 AND g>0 AND L>0, because the
    #             secondary term is parameterised by the crystallite size l (it
    #             divides by lambda/l), so l=0 silently disables the whole model.
    reader.require(
        l > 0.0 or (g > 0.0 and L > 0.0),
        f"extinction {model}: no active mechanism — set l>0 (primary) or "
        f"g>0 with L>0 (secondary)")
    if model in ("BC_mix", "BC_mod"):
        reader.require(
            l > 0.0 and g > 0.0 and L > 0.0,
            f"extinction {model} couples primary and secondary extinction and needs "
            f"l>0, g>0 and L>0 (its secondary term is parameterised by the "
            f"crystallite size l as well as the mosaic g and grain L); got "
            f"l={l:g} g={g:g} L={L:g}. For primary-only use BC_pure with only l set.")
    elif model == "BC_pure":
        reader.require(
            not (l > 0.0 and g > 0.0),
            f"extinction BC_pure is primary OR secondary, not both: set l>0 with g=0 "
            f"(primary; optional grain L for type-II) OR l=0 with g>0 and L>0 "
            f"(secondary); got l={l:g} g={g:g}")

    print(f"  Coherent-elastic EXTINCTION ON: model={model}, l={l:g} Å, g={g:g}, "
          f"L={L:g} Å, dist={dist}, rec={recipe}, rmse_tol={rmse_tol:g}")
    return {"model": model, "l": l, "g": g, "L": L,
            "dist": dist, "recipe": recipe, "rmse_tol": rmse_tol}


def _parse_crystal_cards(reader, za, nphon, ncold=0, nsk=0, nss=0, b7=0.0):
    """Parse the generalized-elastic card block (iel=10): Cards 6b-6g.

    Reads the elastic/inelastic mode controls, lattice, atom types and
    positions, any legacy partial spectra, and — for inelastic_mode=1/2 —
    the phonopy mesh controls (loading the mesh once for all
    temperatures). Returns the populated crystal_info dict.
    """
    # Card 6b: elastic_mode, nat, nspec, inelastic_mode
    # inelastic_mode=0 → isotropic DW, cubic inelastic (no phonopy required)
    # inelastic_mode=1 → directional DW for coherent elastic + in-process
    #                    noncubic SAB using incoherent-approx
    #                    n=1 plus incoherent-approx multiphonons in MT4
    # inelastic_mode=2 → directional DW for coherent elastic + in-process
    #                    noncubic SAB using exact n=1
    #                    (coherent + incoherent) plus incoherent-approx
    #                    multiphonons in MT4
    # Optional trailing fields 5-6 control coherent-elastic Bragg-edge
    # GROUPING (ENDF-102 sec 7.2.2): above a threshold energy the dense,
    # tiny "stair steps" of S(E,T) may be grouped to reduce the number of
    # steps while preserving the average cross section. Absent/0 = off, so
    # existing 4-field Card 6b decks are unchanged.
    reader.card("Card 6b (elastic_mode nat nspec inelastic_mode [grouping])")
    fvals = reader.read_floats(6, defaults=[0, 0, 0, 0, 0, 0])
    # elastic_mode: 1=CEF, 2=MEF. "CEF" (single-channel elastic format) is
    # labelled SEF in the docs/GUI as of 2026-07-24; the numeric deck field and
    # the internal "CEF" spelling are unchanged. See endf_writer.py for the note.
    elastic_mode = reader.to_int(fvals[0], "elastic_mode")   # 1=CEF/SEF, 2=MEF
    nat = reader.to_int(fvals[1], "nat")            # number of distinct atom types
    nspec = reader.to_int(fvals[2], "nspec")          # number of partial phonon spectra (Card 6e blocks)
    inelastic_mode = reader.to_int(fvals[3], "inelastic_mode")   # 0=isotropic, 1/2=in-process noncubic SAB
    # Bragg-edge grouping knob (field 5) + threshold in eV (field 6).
    edge_group_bins_per_decade = reader.to_int(fvals[4], "bins_per_decade")  # 0 = off (keep all edges)
    # Sign-guard both grouping fields: a negative bins_per_decade would
    # silently disable grouping (the only downstream gate is > 0) and a
    # negative threshold would be silently replaced by the 1 eV default,
    # masking a sign typo on an opt-in expert knob.
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
    # The phonopy-backed MT4 path never builds the S(kappa) tables that
    # coldh/skold consume, and the secondary-scatterer pass has no noncubic
    # counterpart: reject the combinations here, before the (expensive)
    # phonopy mesh load, instead of crashing mid-run.
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
    # The generalized (iel=10) MF7/MT2 builder reads only the last-computed
    # Debye-Waller array, and the bound (b7=0) TWO-PASS merge leaves that
    # array holding the SECONDARY scatterer's data (the principal's is kept
    # in dwp1, which the generalized builder never sees — unlike the classic
    # iel=1-6 builder, which averages the two). The whole elastic section
    # would silently be built from the wrong species, so reject the
    # combination here. An analytic secondary (b7=1 free gas / b7=2
    # diffusion) is single-pass and leaves the principal's Debye-Waller data
    # in place, so it stays legal.
    reader.require(
        not (nss > 0 and b7 <= 0.0),
        f"a bound two-pass secondary scatterer (Card 6 nss={nss}, b7={b7:g}) "
        f"is not supported with generalized elastic (Card 5 iel=10): the "
        f"generalized elastic builder has no secondary-scatterer "
        f"Debye-Waller path, so MF7/MT2 would be built from the secondary "
        f"scatterer's Debye-Waller data instead of the principal's. Use an "
        f"analytic secondary (Card 6 b7=1 free gas / b7=2 diffusion) or "
        f"drop the secondary scatterer.")
    # The Phonopy-backed noncubic inelastic modes build MT4 and the
    # directional Debye-Waller factors from the phonon mesh, so Card 6e
    # partial spectra have no role there.
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

    # Validate Card 6c at parse time. Downstream the unit-cell volume clips a
    # negative metric term with max(0.0, ...) (crystal.py volume property), so
    # an unphysical cell (non-positive edge, out-of-range angle, or a
    # degenerate/imaginary-volume angle triple) can silently collapse to a
    # zero-volume cell instead of failing. Reject it here with Card-6c context.
    for nm, v in (("a", latt_a), ("b", latt_b), ("c", latt_c)):
        reader.require(np.isfinite(v) and v > 0.0,
                       f"lattice {nm} must be finite and > 0, got {v}")
    for nm, ang in (("alpha", latt_alpha), ("beta", latt_beta),
                    ("gamma", latt_gamma)):
        reader.require(np.isfinite(ang) and 0.0 < ang < 180.0,
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
        # A is an identity key, never a physical quantity (awr/b_coh/
        # sigma_inc are explicit on this card): it is matched against
        # Card 4's za = 1000*Z + A to find the principal scatterer, and
        # against Card 6e's (Z, A). A = 0 is ENDF's natural-element
        # code, so a natural composition can be named honestly instead
        # of borrowing an isotope's mass number.
        reader.require(at_A >= 0, f"A must be >= 0 (0 = natural "
                                  f"element), got {at_A}")
        reader.require(at_awr > 0.0, f"awr must be > 0, got {at_awr}")
        reader.require(np.isfinite(at_b_coh),
                       f"b_coh must be finite, got {at_b_coh}")
        reader.require(at_sigma_inc >= 0.0 and np.isfinite(at_sigma_inc),
                       f"sigma_inc must be >= 0, got {at_sigma_inc}")
        reader.require(at_npos >= 1, f"npos must be >= 1, got {at_npos}")

        # Read fractional coordinates (npos × 3 values)
        reader.card(f"Card 6d (atom type {iat+1}: {at_npos} fractional positions)")
        coords_flat = reader.read_float_array(at_npos * 3)
        reader.require(bool(np.all(np.isfinite(coords_flat))),
                       "fractional coordinates must all be finite")
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

    # inelastic_mode=1/2: merge Card 6d atom types that share the PRINCIPAL
    # (Z,A) into ONE group rather than rejecting such decks.
    # The MT4 law is accumulated over one principal group; a split principal
    # would mask the law to the first group while normalizing by every
    # matching group's sites. The same crystal is exactly expressible as one
    # entry carrying all of the element's positions, so do that merge here —
    # BEFORE the crystal/sites/fractions are built, keeping every downstream
    # per-group structure (species_corr, directional DW, site groups) aligned.
    if inelastic_mode in (1, 2):
        _za_Z = int(za) // 1000
        _za_A = int(za) % 1000
        _matches = [i for i, at in enumerate(atom_types)
                    if at['Z'] == _za_Z and at['A'] == _za_A]
        if len(_matches) > 1:
            _first = atom_types[_matches[0]]
            for i in _matches[1:]:
                at = atom_types[i]
                reader.require(
                    at['awr'] == _first['awr']
                    and at['b_coh'] == _first['b_coh']
                    and at['sigma_inc'] == _first['sigma_inc'],
                    f"Card 6d atom types {_matches[0] + 1} and {i + 1} both "
                    f"carry the principal nuclide Z={_za_Z} A={_za_A} but "
                    f"differ in awr/b_coh/sigma_inc; with inelastic_mode=1/2 "
                    f"the principal's entries are merged into one group and "
                    f"must be identical")
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
            print(f"    inelastic_mode={inelastic_mode}: merged "
                  f"{len(_matches)} Card 6d entries of the principal nuclide "
                  f"(Z={_za_Z}, A={_za_A}) into one group with "
                  f"{_first['npos']} positions (the MT4 law accumulates over "
                  f"every represented site)")

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
    za_Z = int(za) // 1000
    za_A = int(za) % 1000
    principal_matches = [i for i, at in enumerate(atom_types)
                         if at['Z'] == za_Z and at['A'] == za_A]
    principal_atom_idx = principal_matches[0] if principal_matches else None

    if principal_atom_idx is None:
        from irma.core.crystal_input import principal_mismatch_message
        reader.require(False, principal_mismatch_message(int(za), atom_types))
    # For inelastic_mode=1/2 a multi-entry principal was merged into one
    # group right after the Card 6d parse, so the match is unique.
    assert inelastic_mode not in (1, 2) or len(principal_matches) == 1

    print(f"  Principal scatterer (za={za}) matches atom type "
          f"{principal_atom_idx+1}: Z={za_Z}, A={za_A}, "
          f"fraction={atom_types[principal_atom_idx]['fraction']:.4f}")

    # DC atom selection for CEF (Eq 26 from paper). len(atom_types), not the
    # parsed nat: the F3 principal merge above can reduce the type count to 1,
    # and a single-type cell has no channel competition (f_i=1 would divide by
    # zero in Eq 26).
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
            print("  -> Principal scatterer IS the DC atom -> LTHR=1 (coherent elastic)")
        else:
            print("  -> Principal scatterer is NOT the DC atom -> LTHR=2 (incoherent elastic)")

    # Match partial spectra to atom types. The matching is by (Z, A), so
    # ambiguity in either direction would silently drop or overwrite a
    # user-supplied per-species DOS (wrong DW / MT2 / MT4 on a
    # legitimate-looking tape): reject duplicate (Z, A) on Card 6e, reject
    # any Card 6e spectrum that matched no Card 6d atom type (mirroring the
    # principal-scatterer membership check above), and reject duplicate
    # Card 6d (Z, A) ONLY when a Card 6e spectrum actually targets that
    # (Z, A) — two atom types sharing an isotope at distinct sites is a
    # valid configuration (resolved by position downstream) as long as no
    # per-species spectrum needs an unambiguous match.
    reader.card("Card 6e (partial spectrum atom matching)")
    spec_za_set = {(sp['Z'], sp['A']) for sp in partial_spectra}
    seen_atom_za = set()
    for at in atom_types:
        za_key = (at['Z'], at['A'])
        reader.require(
            za_key not in seen_atom_za or za_key not in spec_za_set,
            f"duplicate Card 6d atom type Z={at['Z']}, A={at['A']} with a "
            f"Card 6e partial spectrum for the same (Z, A): the spectrum "
            f"match is by (Z, A) and cannot resolve duplicate atom types")
        seen_atom_za.add(za_key)
        at['spectrum_idx'] = None
    seen_spec_za = set()
    for i, sp in enumerate(partial_spectra):
        sp_za = (sp['Z'], sp['A'])
        reader.require(
            sp_za not in seen_spec_za,
            f"duplicate Card 6e partial spectrum for Z={sp['Z']}, A={sp['A']}: "
            f"only one spectrum per atom type is allowed")
        seen_spec_za.add(sp_za)
        matched = False
        for j, at in enumerate(atom_types):
            if sp['Z'] == at['Z'] and sp['A'] == at['A']:
                at['spectrum_idx'] = i
                matched = True
                break
        reader.require(
            matched,
            f"Card 6e partial spectrum {i+1} (Z={sp['Z']}, A={sp['A']}) "
            f"does not match any Card 6d atom type")

    # Store everything for later use
    crystal_info = {
        'elastic_mode': elastic_mode,
        # EFFECTIVE type count: the F3 principal merge can reduce it below the
        # parsed Card 6b nat, and the MF7/MT2 writer branches single-vs-poly
        # atomic on this value (a merged single-type cell must take the
        # Eq 24/25 single-atom CEF path, identical to the unsplit deck).
        'nat': len(atom_types),
        'nspec': nspec,
        'inelastic_mode': inelastic_mode,
        'crystal': crystal,
        'atom_types': atom_types,
        'partial_spectra': partial_spectra,
        'principal_atom_idx': principal_atom_idx,
        'dc_atom_idx': dc_atom_idx,
        'total_atoms_in_cell': total_atoms_in_cell,
        # Coherent-elastic Bragg-edge grouping (ENDF-102 7.2.2); 0 = off.
        'coh_edge_group_bins_per_decade': edge_group_bins_per_decade,
        'coh_edge_group_threshold_ev': edge_group_threshold_ev,
    }
    if edge_group_bins_per_decade > 0:
        print(f"  Coherent-elastic Bragg-edge grouping ON: "
              f"{edge_group_bins_per_decade} bins/decade above "
              f"{edge_group_threshold_ev:g} eV (ENDF-102 7.2.2)")

    # ---- Card 6f: phonopy mesh parameters (inelastic_mode=1/2) ----
    # Parsed after Card 6d (before Card 7); the mesh is loaded once here
    # and reused across all temperatures.
    nc_mesh_data = None   # PhonopyMeshData, set below if phonopy DW is enabled
    nc_ncpu = 1
    nc_use_born = 0

    nc_inelastic_controls = None

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
            "Card 6f mesh line must contain exactly 5 values: "
            "mesh_nx mesh_ny mesh_nz ncpu use_born "
            "(DOS-tensor smoothing always uses the auto width of "
            "2 × the energy-grid spacing).")
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

        # Optional Card 6f-cutoff: a single minimum phonon energy in meV.
        # A 2/3-value card here is the legacy/current Card 6g, so old decks
        # remain unambiguous and unchanged.
        reader.card("optional minimum phonon energy / Card 6g")
        fvals_nc_ctrl = reader.read_card_floats()
        min_phonon_energy_mev = 0.0
        if len(fvals_nc_ctrl) == 1:
            min_phonon_energy_mev = float(fvals_nc_ctrl[0])
            reader.require(
                np.isfinite(min_phonon_energy_mev)
                and min_phonon_energy_mev >= 0.0,
                "minimum phonon energy must be finite and nonnegative (meV).")
            reader.card("Card 6g (ndir mpdir [auto])")
            fvals_nc_ctrl = reader.read_card_floats()

        # Card 6g: noncubic inelastic controls.
        # Decks must provide:
        #   num_directions multiphonon_num_directions [auto_multiphonon_order] /
        # The multiphonon maximum order is the standard Card 3 nphon, which is
        # honored verbatim by default. The optional 3rd field opts into
        # automatic sizing of that order from the anisotropic Debye-Waller
        # physics (0 = honor the deck nphon, 1 = auto-size).
        # The coherent powder average always uses golden-spiral direction
        # sampling (equivalent to Euphonic's 'golden' method) and the
        # incoherent powder average is always the exact numerical
        # orientational average; there are no method selectors. Transfers
        # beyond the tabulated law are handled by THERMR's
        # short-collision-time extension (driven by the tape's effective
        # temperature), not by IRMA.
        reader.require(
            len(fvals_nc_ctrl) != 4,
            "Card 6g takes at most 3 fields: num_directions "
            "multiphonon_num_directions [auto_multiphonon_order]. There is "
            "no powder-method selector (the exact numerical powder average "
            "is always used) - a 4-field card has one field too many, "
            "e.g. '10000 1000 0 1 /' should read '10000 1000 1 /'.")
        reader.require(
            len(fvals_nc_ctrl) in (2, 3),
            "Card 6g must contain 2 values plus an optional 3rd value: "
            "num_directions multiphonon_num_directions "
            "[auto_multiphonon_order] "
            "(to extend the law's support, raise Card 3 nphon or set "
            "auto_multiphonon_order=1).")
        num_directions_nc = reader.to_int(fvals_nc_ctrl[0], "ndir")
        multiphonon_num_directions_nc = reader.to_int(fvals_nc_ctrl[1], "mpdir")
        multiphonon_max_order_nc = int(nphon)
        auto_multiphonon_order_nc = (
            reader.to_int(fvals_nc_ctrl[2], "auto_order") if len(fvals_nc_ctrl) == 3 else 0
        )
        reader.require(num_directions_nc >= 1 and multiphonon_num_directions_nc >= 1,
                       "Card 6g direction counts must be >= 1.")
        reader.require(multiphonon_max_order_nc >= 0,
                       "Card 3 nphon must be >= 0.")
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
            f"{' [auto-size]' if nc_inelastic_controls.auto_multiphonon_order else ''}, "
            f"cohavg=directions"
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
        principal_nc_site_indices = [
            d_idx
            for si, at in enumerate(atom_types)
            if at['Z'] == za_Z and at['A'] == za_A
            for d_idx in nc_atom_type_site_groups[si]
        ]
        reader.require(
            bool(principal_nc_site_indices),
            f"Principal scatterer za={za} (Z={za_Z}, A={za_A}) has no "
            f"matching phonopy sites for the non-cubic calculation "
            f"(check Card 4 ZA against the Card 6d atom types)")
        crystal_info['nc_mesh_data'] = nc_mesh_data
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
        crystal_info['nc_min_phonon_energy_mev'] = min_phonon_energy_mev
        if min_phonon_energy_mev > 0.0:
            # Count only what the cutoff removes beyond the automatic floors;
            # the displacement consequence is reported per temperature by the
            # engine (phonon_cutoff_summary).
            from irma.core.phonopy_io import mode_floor_mask as _floor_mask
            _energies_mev = nc_mesh_data.frequencies_ev.reshape(-1) * 1000.0
            _n_branches = nc_mesh_data.frequencies_ev.shape[1]
            _baseline = _floor_mask(_energies_mev, nc_mesh_data.qpoints, _n_branches, 0.0)
            _kept = _floor_mask(_energies_mev, nc_mesh_data.qpoints, _n_branches,
                                min_phonon_energy_mev)
            _removed = _baseline & ~_kept
            _weights = np.repeat(np.asarray(nc_mesh_data.weights, dtype=float), _n_branches)
            print(
                f"User phonon-energy cutoff: {min_phonon_energy_mev:g} meV removes "
                f"{int(np.count_nonzero(_removed))} positive mode(s) beyond the "
                f"automatic floors ({100.0 * _weights[_removed].sum() / _weights.sum():.4f}% "
                f"of the mode weight); {int(np.count_nonzero(~_baseline))} mode(s) were "
                f"already excluded as imaginary or below the floors. No replacement "
                f"spectrum is added."
            )
        print(f"  Non-cubic: principal scatterer spans "
              f"{len(principal_nc_site_indices)} phonopy site(s)")

    # Optional crystalline-extinction card (off by default; last card of the
    # iel=10 elastic block, before Card 7). Sets crystal_info['coherent_extinction'].
    ext_cfg = _parse_extinction_card(reader, elastic_mode)
    if ext_cfg is not None:
        # Extinction attenuates the coherent Bragg edges, so only the
        # coherent-carrying MT2 builders apply it. SEF (elastic_mode=1)
        # routes to the INCOHERENT elastic builder — which never reads the
        # extinction config — when the single-atom principal has
        # sigma_coh <= sigma_inc, or when the polyatomic principal is not
        # the designated-coherent (DC) atom; the provenance stamp would
        # still claim an extinction-corrected tape. Both routing conditions
        # are fully determined by the Card 6b/6d data parsed above (they
        # mirror _build_generalized_elastic in endf_writer.py), so reject
        # the silent no-op here rather than at write time.
        if elastic_mode == 1:
            _p = atom_types[principal_atom_idx]
            if len(atom_types) == 1:
                reader.require(
                    _p['sigma_coh'] > _p['sigma_inc'],
                    f"extinction would be a silent no-op: SEF (Card 6b "
                    f"elastic_mode=1) with a single-atom material whose "
                    f"sigma_coh <= sigma_inc (Card 6d gives "
                    f"sigma_coh={_p['sigma_coh']:.4g} b, "
                    f"sigma_inc={_p['sigma_inc']:.4g} b) writes MF7/MT2 "
                    f"from the incoherent elastic builder, which does not "
                    f"apply extinction. Use elastic_mode=2 (MEF) to keep a "
                    f"coherent channel, or remove the extinction card.")
            else:
                reader.require(
                    principal_atom_idx == dc_atom_idx,
                    f"extinction would be a silent no-op: SEF (Card 6b "
                    f"elastic_mode=1) with a polyatomic material whose "
                    f"principal scatterer (Card 6d atom type "
                    f"{principal_atom_idx + 1}) is not the "
                    f"designated-coherent atom (type {dc_atom_idx + 1}) "
                    f"writes MF7/MT2 from the incoherent elastic builder, "
                    f"which does not apply extinction. Use elastic_mode=2 "
                    f"(MEF) to keep a coherent channel, or remove the "
                    f"extinction card.")
        crystal_info['coherent_extinction'] = ext_cfg

    return crystal_info
