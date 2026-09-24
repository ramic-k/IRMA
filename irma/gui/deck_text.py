"""Tk-free deck-text helpers for the IRMA GUI.

These pure functions emit and parse the editable ``.input`` deck without
importing tkinter, so the data-integrity logic (quote doubling, comment
emission, and the import parser) is unit-testable on a headless runner and
the GUI import becomes transactional (parse fully, then apply to widgets).
"""

from irma.core.deck import _read_temperature_detail_cards


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
    cards without discarding intentional in-line padding.
    """
    if comments_raw.endswith("\n"):
        comments_raw = comments_raw[:-1]
    if not comments_raw.strip():
        return []
    return [f"{_quote(cline)} /" for cline in comments_raw.splitlines()]


def _whole(value, name, line):
    """An integer field of an atom line; a non-integral value is an error
    (the deck reader rejects it too)."""
    if value != int(value):
        raise ValueError(f"Atom line '{line}': {name} must be an integer, "
                         f"got {value:g}.")
    return int(value)


def parse_atoms_text(text):
    """Parse Card 6d-style atom type lines from raw widget text.

    Pure text parsing (Tk-free) so the validation is headlessly
    testable; the GUI deck writer calls it. Returns a list
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
        Z, A, npos = (_whole(parts[k], name, line)
                      for k, name in ((0, "Z"), (1, "A"), (5, "npos")))
        awr_at = parts[2]
        b_coh = parts[3]
        sigma_inc = parts[4]
        coords = parts[6:]
        if len(coords) != 3 * npos:
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

    Builds the COMPLETE imported state before the GUI applies anything, so
    a malformed deck leaves the form untouched (transactional import). Only
    the structure the form needs is checked here; the engine checks value
    ranges when the deck runs. Raises DeckError/ValueError on an
    unsupported or malformed deck.
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
    st['isabt'] = reader.to_int(fvals[2], "isabt")
    st['ilog'] = reader.to_int(fvals[3], "ilog")
    st['smin'] = fvals[4]
    st['iint'] = reader.to_int(fvals[5], "iint")

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
            "is not supported in the GUI — edit the input file directly.")
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
        # Card 6b: elastic_mode nat nspec inelastic_mode
        #          [edge_group_bins_per_decade] [edge_group_threshold_eV]
        fvals = reader.read_floats(6, defaults=[0, 0, 0, 0, 0, 0])
        elastic_mode = reader.to_int(fvals[0], "elastic_mode")
        nat = reader.to_int(fvals[1], "nat")
        nspec = reader.to_int(fvals[2], "nspec")
        inelastic_mode_loaded = reader.to_int(fvals[3], "inelastic_mode")
        edge_group_bpd = reader.to_int(fvals[4], "bins_per_decade")
        # the mode selects which cards follow
        if inelastic_mode_loaded not in (0, 1, 2):
            raise ValueError(f"Card 6b inelastic_mode must be 0, 1, or 2; "
                             f"got {inelastic_mode_loaded}.")
        # the form hides (and would silently clear) these fields in modes 1/2
        if inelastic_mode_loaded in (1, 2) and (ncold or nsk or nss):
            raise ValueError(
                f"ncold/nsk/secondary scatterer are not available with "
                f"inelastic_mode={inelastic_mode_loaded} (got ncold={ncold}, "
                f"nsk={nsk}, nss={nss}).")
        st['elastic_mode'] = elastic_mode
        st['edge_group_bpd'] = edge_group_bpd
        st['edge_group_thr'] = fvals[5]
        st['inelastic_mode'] = inelastic_mode_loaded

        # Card 6c: lattice parameters
        st['lattice'] = list(reader.read_floats(6))

        # Card 6d: atom types
        atoms = []
        for iat in range(nat):
            fvals = reader.read_floats(6)
            at_Z = reader.to_int(fvals[0], "Z")
            at_A = reader.to_int(fvals[1], "A")
            at_awr = fvals[2]
            at_b_coh = fvals[3]
            at_sigma_inc = fvals[4]
            at_npos = reader.to_int(fvals[5], "npos")
            coords_flat = reader.read_float_array(at_npos * 3)
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

        # Card 6e: partial spectra (per-species DW, inelastic_mode=0)
        for isp in range(nspec):
            fvals = reader.read_floats(4)
            sp_Z = reader.to_int(fvals[0], "Z")
            sp_A = reader.to_int(fvals[1], "A")
            sp_delta = fvals[2]
            sp_ni = reader.to_int(fvals[3], "ni")
            sp_rho = [float(v) for v in reader.read_float_array(sp_ni)]
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
                    "(mesh_nx mesh_ny mesh_nz ncpu use_born).")
            mesh_nx = reader.to_int(fvals_nc[0], "mesh_nx")
            mesh_ny = reader.to_int(fvals_nc[1], "mesh_ny")
            mesh_nz = reader.to_int(fvals_nc[2], "mesh_nz")
            nc_ncpu = reader.to_int(fvals_nc[3], "ncpu")
            use_born = reader.to_int(fvals_nc[4], "use_born")
            nc['mesh_nx'] = mesh_nx
            nc['mesh_ny'] = mesh_ny
            nc['mesh_nz'] = mesh_nz
            nc['ncpu'] = nc_ncpu
            nc['use_born'] = use_born
            nc['born_path'] = ''
            if use_born == 1:
                nc['born_path'] = reader.read_string()

            # Optional one-value minimum-phonon-energy card before Card 6g.
            # Without it the 2/3-value Card 6g follows directly.
            fvals_nc_ctrl = reader.read_card_floats()
            nc['min_phonon_energy_mev'] = 0.0
            if len(fvals_nc_ctrl) == 1:
                nc['min_phonon_energy_mev'] = float(fvals_nc_ctrl[0])
                fvals_nc_ctrl = reader.read_card_floats()

            # Card 6g: required noncubic inelastic controls
            if len(fvals_nc_ctrl) not in (2, 3):
                raise ValueError(
                    "Card 6g is required for inelastic_mode=1/2 and must "
                    "contain 2 values plus an optional 3rd value.")
            ndir = reader.to_int(fvals_nc_ctrl[0], "ndir")
            mpdir = reader.to_int(fvals_nc_ctrl[1], "mpdir")
            auto_order = (reader.to_int(fvals_nc_ctrl[2], "auto_order")
                          if len(fvals_nc_ctrl) == 3 else 0)
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
                    "This input file supplies a separate phonon-spectrum "
                    "block for more than one temperature (multiple "
                    "positive temperature cards). The GUI supports "
                    "only the shared-spectrum convention (first "
                    "temperature positive, the rest negative) — "
                    "edit the input file directly.")
            (delta, _ni, rho, twt, c_diff, tbeta, _nd, bdel, adel, ska, _nka,
             dka, cfrac) = _read_temperature_detail_cards(reader, nsk, ncold)
            st.update(
                delta1=delta, rho=list(rho), twt=str(twt), c_diff=str(c_diff),
                tbeta=str(tbeta), osc_e=[] if bdel is None else list(bdel),
                osc_w=[] if adel is None else list(adel),
                dka=None if ska is None else dka,
                ska=None if ska is None else list(ska),
                cfrac=cfrac if nsk > 0 else None)

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
                        "convention — edit the input file directly.")
                (delta, _ni, rho, twt, c_diff, tbeta, _nd, bdel, adel,
                 *_) = _read_temperature_detail_cards(reader, 0, 0)
                st.update(
                    sec_delta=delta, sec_rho=list(rho), sec_twt=str(twt),
                    sec_c=str(c_diff), sec_tbeta=str(tbeta),
                    sec_osc_e=[] if bdel is None else list(bdel),
                    sec_osc_w=[] if adel is None else list(adel))

    # Comment cards (MF1/MT451). The tokenizer already strips the quote
    # delimiters and resolves doubled quotes, preserving interior and
    # leading/trailing whitespace; the content is not stripped here, so an
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
