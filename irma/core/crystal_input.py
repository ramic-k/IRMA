"""Species -> Card 6d crystal-input helpers, shared by every prefill path.

A phonon model names elements, not isotopes: a phonopy.yaml (or an MLIP
bundle structure) records the symbol ``C``, never ``12-C``, and it says
nothing about enrichment. The default identity for a prefilled Card 6d row
is therefore the natural element, which ENDF codes as A = 0,
with that same table entry's natural-abundance constants -- identity and
physics then come from one entry and cannot disagree. This mirrors what
``irma mlip emit`` writes for an un-``--nuclide``d species.

Nuclides whose tabulated scattering length the source table marks
energy-dependent (resonance-region values, not static constants) are
refused: silently prefilling them would put a number in the deck that is
only valid at one energy.

Everything here is Tk-free and side-effect-free. Advisories are returned
as data (``warnings`` lists) so the caller -- CLI progress line, GUI
dialog, or a test -- decides how to show them; core never prints.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple


@dataclass(frozen=True)
class CrystalSpecies:
    """One Card 6d group: a distinct species and its positions in the cell."""

    symbol: str
    Z: int
    A: int                    # the ZA identity; 0 = natural element
    awr: float
    b_coh_fm: float
    sigma_inc_b: float
    positions: Tuple[Tuple[float, float, float], ...]

    @property
    def npos(self) -> int:
        return len(self.positions)

    @property
    def za(self) -> int:
        return 1000 * self.Z + self.A


def resolve_natural_element(symbol) -> Tuple[int, int, float, float, float]:
    """``(Z, A=0, awr, b_coh_fm, sigma_inc_b)`` for ``symbol``.

    The constants are the natural-abundance values of the SAME table entry
    that supplies the identity, so ``za = 1000*Z`` always pairs with the
    numbers written beside it.

    Raises ``KeyError`` for an unknown element symbol and ``ValueError``
    for a nuclide the table marks energy-dependent (message style shared
    with ``irma.mlip.emit.resolve_species``).
    """
    from irma.core.nuclear_data import lookup

    symbol = str(symbol).strip()
    base = lookup(symbol)          # natural-element entry (A = 0)
    if base.energy_dependent:
        raise ValueError(
            f"{symbol}: the tabulated scattering length is marked "
            f"ENERGY-DEPENDENT (resonance-region value) and is not a safe "
            f"prefill; enter b_coh and sigma_inc for your energy range by "
            f"hand instead")
    return (int(base.Z), 0, float(base.awr), float(base.b_coh_fm),
            float(base.sigma_inc_b))


def species_from_sites(symbols: Sequence[str],
                       scaled_positions) -> Tuple[List[CrystalSpecies],
                                                  List[str]]:
    """Group per-site ``(symbol, fractional position)`` into Card 6d rows.

    One row per DISTINCT species, in first-appearance order (the order the
    structure lists them, which is also the order ``irma mlip emit`` uses).
    Returns ``(species, warnings)``; the warnings are advisories about the
    natural-element assumption, not errors.
    """
    symbols = [str(s).strip() for s in symbols]
    positions = [tuple(float(x) for x in p) for p in scaled_positions]

    order: List[str] = []
    grouped: dict = {}
    for symbol, pos in zip(symbols, positions):
        if symbol not in grouped:
            order.append(symbol)
            grouped[symbol] = []
        grouped[symbol].append(pos)

    species, warnings = [], []
    for symbol in order:
        z, a, awr, b_coh, sigma_inc = resolve_natural_element(symbol)
        species.append(CrystalSpecies(
            symbol=symbol, Z=z, A=a, awr=awr, b_coh_fm=b_coh,
            sigma_inc_b=sigma_inc, positions=tuple(grouped[symbol])))
        warnings.append(
            f"{symbol}: filled as the natural element (za={1000 * z}, A=0) "
            f"with natural-abundance constants")
        if symbol == "H":
            # A phonon model cannot distinguish the two: deuterated samples
            # are routinely modelled with the symbol H, and the natural-H
            # constants (b_coh = -3.74 fm, sigma_inc = 80.3 b) differ strongly
            # from the D values (6.67 fm, 2.05 b).
            warnings.append(
                "H: a phonon model labels deuterium as H; if these sites are "
                "deuterium, replace the row's A and constants with the 2-H "
                "values by hand")
    return species, warnings


def format_card6d_row(species: CrystalSpecies) -> str:
    """One 'Atom Types in Unit Cell' row:
    ``Z A AWR b_coh sigma_inc npos x1 y1 z1 ...``."""
    return format_atom_row({
        "Z": species.Z, "A": species.A, "awr": species.awr,
        "b_coh": species.b_coh_fm, "sigma_inc": species.sigma_inc_b,
        "npos": species.npos, "positions": species.positions})


def format_lattice_fields(cellpar) -> Tuple[str, ...]:
    """The six lattice entries (a, b, c in Angstrom; angles in degrees)."""
    return tuple(f"{float(v):.6f}" for v in cellpar)


# ------------------------------------------------------------------------
# The principal scatterer against the Card 6d rows.
#
# Card 5's ZA names the nuclide the evaluation is written for; Card 6d's rows
# carry the identity and the scattering constants each site is computed with.
# The engine requires the principal (Z, A) to be one of the rows, so the
# ENDF identity and the physics come from the same nuclide. These helpers
# are the one place that rule and its wording live, for the engine, the
# deck validator, and the GUI alike. Rows are the dictionaries
# ``irma.gui.deck_text.parse_atoms_text`` and the deck stager produce
# (keys Z, A, awr, b_coh, sigma_inc, npos, positions).

# Constants equal to the table entry to this many significant digits count
# as the table's own numbers (a prefill), not the user's.
TABLE_MATCH_DIGITS = 6


def _same_digits(x, y):
    """True when x and y agree to TABLE_MATCH_DIGITS significant digits."""
    return f"{float(x):.{TABLE_MATCH_DIGITS}g}" == f"{float(y):.{TABLE_MATCH_DIGITS}g}"


def split_za(za):
    """``(Z, A)`` of an integer ZA; raises ValueError for a bad value."""
    za = int(za)
    if za <= 0:
        raise ValueError(f"za must be a positive integer, got {za}")
    return za // 1000, za % 1000


def nuclide_label(z, a):
    """'C' for a natural element, 'C-12' for an isotope, 'D' for 1-2."""
    from irma.core.nuclear_data import lookup
    if (int(z), int(a)) == (1, 2):
        return "D"
    try:
        symbol = lookup((int(z), 0)).symbol
    except KeyError:
        symbol = f"Z{int(z)}"
    return symbol if int(a) == 0 else f"{symbol}-{int(a)}"


def principal_row_match(za, rows):
    """Which Card 6d rows the principal ZA matches.

    Returns ``{"za", "Z", "A", "exact", "same_z"}``: ``exact`` the indices
    of rows with the principal's (Z, A), ``same_z`` the indices of rows of
    that element with any A. The deck is consistent when ``exact`` is not
    empty.
    """
    z, a = split_za(za)
    exact = [i for i, r in enumerate(rows) if int(r["Z"]) == z and int(r["A"]) == a]
    same_z = [i for i, r in enumerate(rows) if int(r["Z"]) == z]
    return {"za": int(za), "Z": z, "A": a, "exact": exact, "same_z": same_z}


def principal_mismatch_message(za, rows):
    """None when the principal ZA is one of the rows; else what to do."""
    match = principal_row_match(za, rows)
    if match["exact"]:
        return None
    z, a = match["Z"], match["A"]
    want = nuclide_label(z, a)
    head = (f"Card 4 ZA={int(za)} ({want}) requires a Card 6d atom row with "
            f"Z={z}, A={a}.")
    if not match["same_z"]:
        found = ", ".join(f"row {i + 1}: {nuclide_label(r['Z'], r['A'])}"
                          for i, r in enumerate(rows)) or "no rows"
        return (f"{head} No row has Z={z} (found {found}). Add a row for "
                f"{want} or change ZA to one of the rows' nuclides.")
    if len(match["same_z"]) == 1:
        i = match["same_z"][0]
        have = nuclide_label(rows[i]["Z"], rows[i]["A"])
        have_za = 1000 * int(rows[i]["Z"]) + int(rows[i]["A"])
        return (f"{head} Found row {i + 1}: Z={z}, A={int(rows[i]['A'])} "
                f"({have}). Either keep that composition and set ZA={have_za} "
                f"with matching Card 5 constants, or change row {i + 1}'s A "
                f"and its AWR, b_coh and sigma_inc to the {want} values (the "
                f"GUI's 'Apply ZA' button does this). No rows were changed.")
    listing = "; ".join(f"row {i + 1}: A={int(rows[i]['A'])} "
                        f"({nuclide_label(rows[i]['Z'], rows[i]['A'])})"
                        for i in match["same_z"])
    return (f"{head} Rows with Z={z}: {listing}. Set ZA to the row that is "
            f"the principal scatterer, or change that row's A and constants "
            f"to {want}. No rows were changed.")


def row_constants_match_table(row):
    """True when the row's AWR, b_coh and sigma_inc are the table entry's
    values for its own (Z, A), to ``TABLE_MATCH_DIGITS`` significant digits;
    False when they differ or the nuclide has no entry."""
    from irma.core.nuclear_data import lookup
    try:
        nuc = lookup((int(row["Z"]), int(row["A"])))
    except KeyError:
        return False
    return (_same_digits(row["awr"], nuc.awr)
            and _same_digits(row["b_coh"], nuc.b_coh_fm)
            and _same_digits(row["sigma_inc"], nuc.sigma_inc_b))


def relabel_row(row, za):
    """The row rewritten for nuclide ``za``: identity and constants from the
    table entry, positions and npos untouched.

    Returns ``(new_row, changes)`` where ``changes`` lists the fields that
    differ as ``(name, old, new)``. Raises ``KeyError`` when the nuclide has
    no tabulated constants and ``ValueError`` when the table marks them
    energy-dependent, in both cases before anything is changed.
    """
    from irma.core.nuclear_data import lookup
    z, a = split_za(za)
    if int(row["Z"]) != z:
        raise ValueError(f"row is Z={int(row['Z'])}, not Z={z}")
    nuc = lookup((z, a))          # KeyError: no entry
    if nuc.energy_dependent:
        raise ValueError(
            f"{nuclide_label(z, a)}: the tabulated scattering length is marked "
            f"ENERGY-DEPENDENT (resonance-region value) and is not a safe "
            f"prefill; enter b_coh and sigma_inc for your energy range by hand")
    new = dict(row)
    new.update({"A": a, "awr": float(nuc.awr), "b_coh": float(nuc.b_coh_fm),
                "sigma_inc": float(nuc.sigma_inc_b)})
    changes = [(name, row[name], new[name]) for name in ("A", "awr", "b_coh", "sigma_inc")
               if not _same_digits(row[name], new[name])]
    return new, changes


def format_atom_row(row):
    """One Card 6d line from a row dictionary (the GUI's atom-block format)."""
    coords = "  ".join(f"{x:.6f} {y:.6f} {z:.6f}" for x, y, z in row["positions"])
    return (f"{int(row['Z'])}  {int(row['A'])}  {float(row['awr']):.6f}  "
            f"{float(row['b_coh']):.6f}  {float(row['sigma_inc']):.6f}  "
            f"{int(row['npos'])}  {coords}")
