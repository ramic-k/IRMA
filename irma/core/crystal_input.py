"""Species -> Card 6d crystal-input helpers, shared by every prefill path.

A phonon model names ELEMENTS, not isotopes: a phonopy.yaml (or an MLIP
bundle structure) records the symbol ``C``, never ``12-C``, and it says
nothing about enrichment. The honest default identity for a prefilled
Card 6d row is therefore the natural element, which ENDF codes as A = 0,
with that same table entry's natural-abundance constants -- identity and
physics then come from one entry and cannot disagree. This mirrors what
``irma mlip emit`` writes for an un-``--nuclide``d species.

Nuclides whose tabulated scattering length the source table marks
ENERGY-DEPENDENT (resonance-region values, not static constants) are
refused: silently prefilling them would put a number in the deck that is
only valid at one energy.

Everything here is Tk-free and side-effect-free. Advisories are returned
as DATA (``warnings`` lists) so the caller -- CLI progress line, GUI
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
    if len(symbols) != len(positions):
        raise ValueError(
            f"symbols and positions disagree on the site count: "
            f"{len(symbols)} vs {len(positions)}")
    for pos in positions:
        if len(pos) != 3:
            raise ValueError(
                f"fractional position {pos!r} does not have 3 components")

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
            # constants (b_coh = -3.74 fm, sigma_inc = 80.3 b) are wildly
            # wrong for D (6.67 fm, 2.05 b).
            warnings.append(
                "H: a phonon model labels deuterium as H; if these sites are "
                "deuterium, replace the row's A and constants with the 2-H "
                "values by hand")
    return species, warnings


def format_card6d_row(species: CrystalSpecies) -> str:
    """One 'Atom Types in Unit Cell' row:
    ``Z A AWR b_coh sigma_inc npos x1 y1 z1 ...``."""
    coords = "  ".join(f"{x:.6f} {y:.6f} {z:.6f}"
                       for x, y, z in species.positions)
    return (f"{species.Z}  {species.A}  {species.awr:.6f}  "
            f"{species.b_coh_fm:.6f}  {species.sigma_inc_b:.6f}  "
            f"{species.npos}  {coords}")


def format_lattice_fields(cellpar) -> Tuple[str, ...]:
    """The six lattice entries (a, b, c in Angstrom; angles in degrees)."""
    values = [float(v) for v in cellpar]
    if len(values) != 6:
        raise ValueError(f"cellpar must have 6 entries, got {len(values)}")
    return tuple(f"{v:.6f}" for v in values)
