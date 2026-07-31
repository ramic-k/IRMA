"""Tk-free deck-text helpers for the IRMA GUI.

These pure functions emit and parse the editable ``.input`` deck without
importing tkinter, so the data-integrity logic (quote doubling, comment
emission, and the import parser) is unit-testable on a headless runner and
the GUI import becomes transactional (parse fully, then apply to widgets).
"""

from math import cos, isfinite, radians


def _quote(text):
    """Quote a deck string field, doubling embedded single quotes.

    The tokenizer reads '' inside a quoted field as a literal quote
    (Fortran convention); emitting a raw ' instead silently terminates
    the field, truncating comment cards and file paths.
    """
    return "'" + text.replace("'", "''") + "'"


def fmt_array(arr, ncols=5):
    """Format a numeric array as deck card text, ncols values per line."""
    parts = []
    for i in range(0, len(arr), ncols):
        chunk = arr[i:i + ncols]
        parts.append(' '.join(f'{v:.6e}' for v in chunk))
    return '\n'.join(parts)


def emit_comment_lines(comments_raw):
    """Emit MF1/MT451 comment cards from the GUI comment text block.

    Each source line becomes one quoted comment card. The line text is
    preserved verbatim (interior and leading/trailing whitespace round-trips
    inside the quotes), with embedded single quotes doubled so apostrophes
    and slashes survive the tokenizer. The Tk text widget always appends a
    trailing newline; it is dropped so an otherwise-empty block emits no
    cards (matching the prior whitespace-trimmed behavior) without
    discarding intentional in-line padding.
    """
    if comments_raw.endswith("\n"):
        comments_raw = comments_raw[:-1]
    if not comments_raw.strip():
        return []
    return [f"{_quote(cline)} /" for cline in comments_raw.splitlines()]


def parse_atoms_text(text):
    """Parse Card 6d-style atom type lines from raw widget text.

    Pure text parsing (Tk-free) so the validation is headlessly
    testable; the GUI's `_parse_atoms` delegates here. Returns a list
    of dicts with keys: Z, A, awr, b_coh, sigma_inc, npos, positions.
    """
    atoms = []
    for line in text.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        parts = [float(x) for x in line.split()]
        if len(parts) < 6:
            raise ValueError(
                f"Atom line '{line}' must contain at least 6 fields "
                f"(Z A awr b_coh sigma_inc npos), got {len(parts)}.")
        Z = int(parts[0])
        A = int(parts[1])
        awr_at = parts[2]
        b_coh = parts[3]
        sigma_inc = parts[4]
        npos = int(parts[5])
        coords = parts[6:]
        if len(coords) < 3 * npos:
            raise ValueError(
                f"Atom line '{line}' declares npos={npos} but provides "
                f"{len(coords)} coordinate values (need {3 * npos}).")
        positions = []
        for i in range(npos):
            positions.append(
                (coords[3*i], coords[3*i+1], coords[3*i+2]))
        atoms.append({
            'Z': Z, 'A': A, 'awr': awr_at, 'b_coh': b_coh,
            'sigma_inc': sigma_inc, 'npos': npos,
            'positions': positions,
        })
    return atoms


def parse_deck_to_staging(reader, path):
    """Parse a IRMA/LEAPR deck into a plain staging dict (Tk-free).

    Builds the COMPLETE imported state and validates it before the GUI
    applies anything, so a malformed deck leaves the form untouched
    (transactional import). Engine-enforced ranges are checked here so the
    GUI rejects on import instead of exporting a deck the engine will
    reject. Raises DeckError/ValueError on any unsupported/malformed deck.
    """
    st = {}

    # Card 1: output unit (ignored)
    reader.read_ints(1)

    # Card 2: title (free text, numerics allowed) — preserved on round-trip.
    st['title'] = reader.read_string(allow_numeric=True)

    # Card 3: ntempr, iprint, nphon — iprint preserved on round-trip.
    ntempr, iprint, nphon = reader.read_ints(3, defaults=[1, 1, 100])
    st['ntempr'] = ntempr
    st['iprint'] = iprint
    st['nphon'] = nphon

    # Card 4: mat, za, isabt, ilog, smin, [iint]
    fvals = reader.read_floats(6, defaults=[0, 0, 0, 0, 1.0e-75, 0])
    st['mat'] = reader.to_int(fvals[0], "mat")
    st['za'] = fvals[1]
    # Integer-coded flags take the same checked conversion + range checks
    # the deck driver applies (driver.py, Card 4): a bare int() would
    # silently truncate a non-integral value (iint=1.9 imported as the
    # valid lin-lin flag 1) that the engine rejects.
    isabt = reader.to_int(fvals[2], "isabt")
    ilog = reader.to_int(fvals[3], "ilog")
    iint = reader.to_int(fvals[5], "iint")
    if isabt not in (0, 1):
        raise ValueError(f"isabt must be 0 or 1, got {isabt}.")
    if ilog not in (0, 1):
        raise ValueError(f"ilog must be 0 or 1, got {ilog}.")
    if iint not in (0, 1):
        raise ValueError(f"iint must be 0 or 1, got {iint}.")
    st['isabt'] = isabt
    st['ilog'] = ilog
    st['smin'] = fvals[4]
    st['iint'] = iint

    # Card 5: awr, spr, npr, iel, ncold, nsk
    fvals = reader.read_floats(6, defaults=[0, 0, 0, 0, 0, 0])
    st['awr'] = fvals[0]
    st['spr'] = fvals[1]
    st['npr'] = reader.to_int(fvals[2], "npr")
    iel = reader.to_int(fvals[3], "iel")
    ncold = reader.to_int(fvals[4], "ncold")
    nsk = reader.to_int(fvals[5], "nsk")
    st['iel'] = iel
    st['ncold'] = ncold
    st['nsk'] = nsk

    # Card 6: nss, b7, aws, sps, mss
    fvals = reader.read_floats(5, defaults=[0, 0, 0, 0, 0])
    nss = reader.to_int(fvals[0], "nss")
    b7 = fvals[1]
    aws = fvals[2]
    sps = fvals[3]
    mss = reader.to_int(fvals[4], "mss")

    if nss not in (0, 1):
        raise ValueError(f"nss must be 0 or 1, got {nss}.")
    two_pass = nss > 0 and b7 <= 0.0
    if two_pass and (ncold > 0 or nsk > 0):
        raise ValueError(
            "ncold/nsk together with a two-pass secondary scatterer "
            "is not supported in the GUI — edit the deck directly.")
    st['nss'] = nss
    st['aws'] = aws
    st['sps'] = sps
    st['b7'] = None
    st['mss'] = None
    if nss > 0:
        b7_int = int(round(b7))
        if b7_int not in (0, 1, 2):
            raise ValueError(
                f"Unsupported secondary-scatterer b7={b7:g}; the GUI "
                f"supports b7=0 (bound two-pass), 1 (free gas), or "
                f"2 (diffusion).")
        st['b7'] = b7_int
        st['mss'] = max(1, mss)

    # Generalized elastic cards (iel=10)
    nspec = 0
    inelastic_mode_loaded = 0   # classic decks have no Card 6b
    st['partial_spectra'] = []
    st['noncubic'] = None
    st['coherent_extinction'] = None
    if iel == 10:
        # A bound two-pass secondary scatterer with generalized elastic is
        # rejected by the engine (crystal_cards.py: the generalized MF7/MT2
        # builder has no secondary-scatterer Debye-Waller path); mirror it
        # here so the deck is refused on import, not at run time.
        if two_pass:
            raise ValueError(
                f"a bound two-pass secondary scatterer (Card 6 nss={nss}, "
                f"b7={b7:g}) is not supported with generalized elastic "
                f"(Card 5 iel=10): MF7/MT2 would be built from the secondary "
                f"scatterer's Debye-Waller data instead of the principal's. "
                f"Use an analytic secondary (Card 6 b7=1 free gas / "
                f"b7=2 diffusion) or drop the secondary scatterer.")

        # Card 6b: elastic_mode nat nspec inelastic_mode
        #          [edge_group_bins_per_decade] [edge_group_threshold_eV]
        # Guarded with the same ranges the engine enforces
        # (crystal_cards.py, Card 6b), so an invalid generalized-elastic
        # deck is rejected on import instead of exported and refused by
        # the engine.
        fvals = reader.read_floats(6, defaults=[0, 0, 0, 0, 0, 0])
        elastic_mode = reader.to_int(fvals[0], "elastic_mode")
        nat = reader.to_int(fvals[1], "nat")
        nspec = reader.to_int(fvals[2], "nspec")
        inelastic_mode_loaded = reader.to_int(fvals[3], "inelastic_mode")
        edge_group_bpd = reader.to_int(fvals[4], "bins_per_decade")
        if elastic_mode not in (1, 2):
            raise ValueError(f"Card 6b elastic_mode must be 1 (SEF) or "
                             f"2 (MEF), got {elastic_mode}.")
        if nat < 1:
            raise ValueError(f"Card 6b nat must be >= 1, got {nat}.")
        if inelastic_mode_loaded not in (0, 1, 2):
            raise ValueError(f"Card 6b inelastic_mode must be 0, 1, or 2; "
                             f"got {inelastic_mode_loaded}.")
        if edge_group_bpd < 0:
            raise ValueError(
                f"Card 6b bins_per_decade (field 5) must be >= 0 "
                f"(0 = grouping off), got {edge_group_bpd}.")
        if not fvals[5] >= 0.0:
            raise ValueError(f"Card 6b grouping threshold (field 6) must "
                             f"be >= 0, got {fvals[5]:g}.")
        if inelastic_mode_loaded in (1, 2):
            if nspec != 0:
                raise ValueError(
                    f"Card 6b nspec must be 0 when inelastic_mode="
                    f"{inelastic_mode_loaded}: Phonopy provides MT4 and the "
                    f"Debye-Waller factors, so Card 6e partial spectra are "
                    f"not used (got nspec={nspec}).")
            if ncold != 0 or nsk != 0:
                raise ValueError(
                    f"ncold/nsk pair-correlation options are not available "
                    f"with inelastic_mode={inelastic_mode_loaded} "
                    f"(got ncold={ncold}, nsk={nsk}).")
            if nss != 0:
                raise ValueError(
                    f"a secondary scatterer (Card 6 nss={nss}) is not "
                    f"supported with inelastic_mode={inelastic_mode_loaded}.")
        if nspec < 0:
            raise ValueError(f"Card 6b nspec must be >= 0, got {nspec}.")
        st['elastic_mode'] = elastic_mode
        st['edge_group_bpd'] = edge_group_bpd
        st['edge_group_thr'] = fvals[5]
        st['inelastic_mode'] = inelastic_mode_loaded

        # Card 6c: lattice parameters, validated with the engine's cell
        # checks (crystal_cards.py, Card 6c): positive finite edges, angles
        # in (0, 180), and a positive metric determinant term.
        fvals = reader.read_floats(6)
        for nm, v in (("a", fvals[0]), ("b", fvals[1]), ("c", fvals[2])):
            if not (isfinite(v) and v > 0.0):
                raise ValueError(f"Card 6c lattice {nm} must be finite "
                                 f"and > 0, got {v}.")
        for nm, ang in (("alpha", fvals[3]), ("beta", fvals[4]),
                        ("gamma", fvals[5])):
            if not (isfinite(ang) and 0.0 < ang < 180.0):
                raise ValueError(f"Card 6c lattice angle {nm} must be in "
                                 f"(0, 180) degrees, got {ang}.")
        ca = cos(radians(fvals[3]))
        cb = cos(radians(fvals[4]))
        cg = cos(radians(fvals[5]))
        metric = 1.0 - ca * ca - cb * cb - cg * cg + 2.0 * ca * cb * cg
        if not metric > 0.0:
            raise ValueError(
                f"Card 6c lattice angles do not form a valid cell (metric "
                f"determinant term {metric:.6g} <= 0); alpha={fvals[3]}, "
                f"beta={fvals[4]}, gamma={fvals[5]}.")
        st['lattice'] = list(fvals)

        # Card 6d: atom types, validated with the engine's per-type checks
        # (crystal_cards.py, Card 6d).
        atoms = []
        for iat in range(nat):
            fvals = reader.read_floats(6)
            at_Z = reader.to_int(fvals[0], "Z")
            at_A = reader.to_int(fvals[1], "A")
            at_awr = fvals[2]
            at_b_coh = fvals[3]
            at_sigma_inc = fvals[4]
            at_npos = reader.to_int(fvals[5], "npos")
            if at_Z < 1:
                raise ValueError(f"Card 6d Z must be >= 1, got {at_Z}.")
            if at_A < 1:
                raise ValueError(f"Card 6d A must be >= 1, got {at_A}.")
            if not at_awr > 0.0:
                raise ValueError(f"Card 6d awr must be > 0, got {at_awr}.")
            if not isfinite(at_b_coh):
                raise ValueError(f"Card 6d b_coh must be finite, "
                                 f"got {at_b_coh}.")
            if not (at_sigma_inc >= 0.0 and isfinite(at_sigma_inc)):
                raise ValueError(f"Card 6d sigma_inc must be >= 0, "
                                 f"got {at_sigma_inc}.")
            if at_npos < 1:
                raise ValueError(f"Card 6d npos must be >= 1, "
                                 f"got {at_npos}.")
            coords_flat = reader.read_float_array(at_npos * 3)
            if not all(isfinite(float(c)) for c in coords_flat):
                raise ValueError("Card 6d fractional coordinates must all "
                                 "be finite.")
            coords = [
                (coords_flat[3 * ip], coords_flat[3 * ip + 1],
                 coords_flat[3 * ip + 2])
                for ip in range(at_npos)
            ]
            atoms.append({
                'Z': at_Z, 'A': at_A, 'awr': at_awr, 'b_coh': at_b_coh,
                'sigma_inc': at_sigma_inc, 'npos': at_npos, 'coords': coords,
            })
        st['atoms'] = atoms

        # Card 6e: partial spectra (per-species DW, inelastic_mode=0),
        # validated with the engine's spectrum checks (crystal_cards.py,
        # Card 6e).
        for isp in range(nspec):
            fvals = reader.read_floats(4)
            sp_Z = reader.to_int(fvals[0], "Z")
            sp_A = reader.to_int(fvals[1], "A")
            sp_delta = fvals[2]
            sp_ni = reader.to_int(fvals[3], "ni")
            if not sp_delta > 0.0:
                raise ValueError(
                    f"Card 6e delta (spectrum spacing, eV) must be > 0, "
                    f"got {sp_delta:g}.")
            if sp_ni < 2:
                raise ValueError(
                    f"Card 6e ni (number of spectrum points) must be >= 2, "
                    f"got {sp_ni}.")
            sp_rho = [float(v) for v in reader.read_float_array(sp_ni)]
            if not all(v >= 0.0 for v in sp_rho):
                raise ValueError("Card 6e rho values must be >= 0.")
            if not any(v > 0.0 for v in sp_rho):
                raise ValueError("Card 6e rho values are all zero.")
            st['partial_spectra'].append({
                'Z': sp_Z, 'A': sp_A,
                'delta': sp_delta, 'ni': sp_ni, 'rho': sp_rho,
            })

        # Card 6f: phonopy mesh parameters (inelastic_mode=1/2)
        if inelastic_mode_loaded in (1, 2):
            nc = {}
            nc['yaml'] = reader.read_string()

            fvals_nc = reader.read_card_floats()
            if len(fvals_nc) != 5:
                raise ValueError(
                    "Card 6f mesh line must contain 5 values "
                    "(mesh_nx mesh_ny mesh_nz ncpu use_born); the "
                    "former 6th field sigma_beta was removed.")
            mesh_nx = reader.to_int(fvals_nc[0], "mesh_nx")
            mesh_ny = reader.to_int(fvals_nc[1], "mesh_ny")
            mesh_nz = reader.to_int(fvals_nc[2], "mesh_nz")
            nc_ncpu = reader.to_int(fvals_nc[3], "ncpu")
            use_born = reader.to_int(fvals_nc[4], "use_born")
            # Same range checks the engine enforces (engine.py:616-621),
            # so an invalid noncubic deck is rejected on import instead of
            # exported and failed at run time.
            if not (mesh_nx >= 1 and mesh_ny >= 1 and mesh_nz >= 1):
                raise ValueError(
                    f"Card 6f mesh dimensions must all be >= 1, got "
                    f"{mesh_nx}x{mesh_ny}x{mesh_nz}.")
            if nc_ncpu < 1:
                raise ValueError(f"Card 6f ncpu must be >= 1, got {nc_ncpu}.")
            if use_born not in (0, 1):
                raise ValueError(
                    f"Card 6f use_born must be 0 or 1, got {use_born}.")
            nc['mesh_nx'] = mesh_nx
            nc['mesh_ny'] = mesh_ny
            nc['mesh_nz'] = mesh_nz
            nc['ncpu'] = nc_ncpu
            nc['use_born'] = use_born
            nc['born_path'] = ''
            if use_born == 1:
                nc['born_path'] = reader.read_string()

            # Card 6g: required noncubic inelastic controls
            fvals_nc_ctrl = reader.read_card_floats()
            if len(fvals_nc_ctrl) == 4:
                raise ValueError(
                    "Card 6g takes at most 3 fields: ndir mpdir "
                    "[auto_multiphonon_order]. There is no powder-method "
                    "selector (the exact numerical powder average is always "
                    "used) - a 4-field card has one field too many, e.g. "
                    "'10000 1000 0 1 /' should read '10000 1000 1 /'.")
            if len(fvals_nc_ctrl) not in (2, 3):
                raise ValueError(
                    "Card 6g is required for inelastic_mode=1/2 and must "
                    "contain 2 values plus an optional 3rd value.")
            ndir = reader.to_int(fvals_nc_ctrl[0], "ndir")
            mpdir = reader.to_int(fvals_nc_ctrl[1], "mpdir")
            auto_order = (reader.to_int(fvals_nc_ctrl[2], "auto_order")
                          if len(fvals_nc_ctrl) == 3 else 0)
            # Same range checks the engine enforces (engine.py:669-678).
            if not (ndir >= 1 and mpdir >= 1):
                raise ValueError(
                    "Card 6g direction counts must be >= 1, got "
                    f"ndir={ndir}, mpdir={mpdir}.")
            if auto_order not in (0, 1):
                raise ValueError(
                    "Card 6g auto_multiphonon_order (3rd field) must be 0 "
                    f"(honor Card 3 nphon) or 1 (auto-size), got {auto_order}.")
            nc['ndir'] = ndir
            nc['mpdir'] = mpdir
            nc['auto_order'] = auto_order
            st['noncubic'] = nc

        # Optional extinction card (last card of the iel=10 block, before Card 7).
        # Reuses the engine's parser so the GUI and the engine accept identical
        # syntax; returns None when the card is absent.
        from irma.core.crystal_cards import _parse_extinction_card
        st['coherent_extinction'] = _parse_extinction_card(reader, elastic_mode)

    st['inelastic_mode'] = inelastic_mode_loaded

    # Card 7: nalpha, nbeta, lat
    nalpha, nbeta, lat = reader.read_ints(3, defaults=[0, 0, 0])
    st['nalpha'] = nalpha
    st['nbeta'] = nbeta
    st['lat'] = lat

    # Card 8/9: alpha, beta values
    st['alpha'] = list(reader.read_float_array(nalpha))
    st['beta'] = list(reader.read_float_array(nbeta))

    # Temperature loop — collect temps and first-temp phonon data.
    temperatures = []
    st['twt'] = "0.0"
    st['c_diff'] = "0.0"
    st['tbeta'] = "1.0"
    st['osc_e'] = []
    st['osc_w'] = []
    st['delta1'] = None
    st['rho'] = None
    st['dka'] = None
    st['ska'] = None
    st['cfrac'] = None
    n_detail_blocks = 0

    for itemp in range(ntempr):
        temp = reader.read_floats(1)[0]
        temperatures.append(abs(temp))

        if inelastic_mode_loaded in (1, 2):
            continue
        if itemp == 0 or temp >= 0.0:
            n_detail_blocks += 1
            if n_detail_blocks > 1:
                raise ValueError(
                    "This deck supplies a separate phonon-spectrum "
                    "block for more than one temperature (multiple "
                    "positive temperature cards). The GUI supports "
                    "only the shared-spectrum convention (first "
                    "temperature positive, the rest negative) — "
                    "edit the deck directly.")
            fvals = reader.read_floats(2)
            st['delta1'] = fvals[0]
            ni = reader.to_int(fvals[1], "ni")
            st['rho'] = list(reader.read_float_array(ni))

            fvals = reader.read_floats(3)
            st['twt'] = str(fvals[0])
            st['c_diff'] = str(fvals[1])
            st['tbeta'] = str(fvals[2])

            nd = reader.read_ints(1)[0]
            if nd > 0:
                bdel = reader.read_float_array(nd)
                adel = reader.read_float_array(nd)
                st['osc_e'] = list(bdel)
                st['osc_w'] = list(adel)

            if nsk > 0 or ncold > 0:
                fvals = reader.read_floats(2)
                nka = reader.to_int(fvals[0], "nka")
                st['dka'] = fvals[1]
                st['ska'] = list(reader.read_float_array(nka))

            if nsk > 0:
                st['cfrac'] = reader.read_floats(1)[0]

    st['temperatures'] = temperatures

    # Second pass: a bound (b7 <= 0) secondary scatterer repeats the whole
    # temperature block sequence with the secondary's own phonon model.
    st['two_pass'] = two_pass
    st['sec_delta'] = None
    st['sec_rho'] = None
    st['sec_twt'] = "0.0"
    st['sec_c'] = "0.0"
    st['sec_tbeta'] = "1.0"
    st['sec_osc_e'] = []
    st['sec_osc_w'] = []
    if two_pass:
        n_sec_blocks = 0
        for itemp in range(ntempr):
            temp = reader.read_floats(1)[0]
            if abs(abs(temp) - temperatures[itemp]) > 1e-6:
                raise ValueError(
                    f"Second-pass temperature {abs(temp):g} K does not "
                    f"match the principal pass "
                    f"({temperatures[itemp]:g} K).")
            if itemp == 0 or temp >= 0.0:
                n_sec_blocks += 1
                if n_sec_blocks > 1:
                    raise ValueError(
                        "The secondary pass supplies a separate phonon-"
                        "spectrum block for more than one temperature; "
                        "the GUI supports only the shared-spectrum "
                        "convention — edit the deck directly.")
                fvals = reader.read_floats(2)
                st['sec_delta'] = fvals[0]
                sec_ni = reader.to_int(fvals[1], "ni")
                st['sec_rho'] = list(reader.read_float_array(sec_ni))
                fvals = reader.read_floats(3)
                st['sec_twt'] = str(fvals[0])
                st['sec_c'] = str(fvals[1])
                st['sec_tbeta'] = str(fvals[2])
                sec_nd = reader.read_ints(1)[0]
                if sec_nd > 0:
                    st['sec_osc_e'] = list(reader.read_float_array(sec_nd))
                    st['sec_osc_w'] = list(reader.read_float_array(sec_nd))

    # Comment cards (MF1/MT451). The tokenizer already strips the quote
    # delimiters and resolves doubled quotes, preserving interior and
    # leading/trailing whitespace; do NOT strip the content here, so an
    # import -> export cycle is byte-faithful for column-aligned records.
    comments = reader.read_comment_strings()
    clean = []
    for c in comments:
        # Only an unquoted fallback token (rare) may still carry surrounding
        # quotes; strip exactly one matched pair without touching whitespace
        # inside it.
        if len(c) >= 2 and ((c[0] == "'" and c[-1] == "'") or
                            (c[0] == '"' and c[-1] == '"')):
            c = c[1:-1]
        clean.append(c)
    st['comments'] = clean

    st['path'] = path
    return st
