#!/usr/bin/env python3
"""Generate irma/core/nuclear_data.py from the periodictable package.

Dev-time only: periodictable is NOT an IRMA dependency. Rerun this script to
refresh the table after a periodictable upgrade, then commit the regenerated
module:

    python tools/generate_nuclear_data.py

Data source: periodictable.nsf (public domain), which reproduces the
Atominstitut/Rauch-Waschkowski neutron scattering-length table (Neutron Data
Booklet, 2nd ed., 2003) with Sears (1992) cross-references. The NCNR
n-lengths web page presents the same compilation and serves as a cross-check,
not the direct source.

Extracted per nuclide (natural elements carry A=0):
- b_coh_fm: real part of the bound coherent scattering length
- sigma_inc_b: bound incoherent cross section
- sigma_bound_b: 4*pi*b_coh^2/100 + sigma_inc (fm -> barn), cross-checked
  against the table's own coherent+incoherent columns; relative discrepancies
  above 1% are listed in the generated header. The b_c-derived value is what
  IRMA uses, matching the engine's own 4*pi*b^2*0.01 convention
  (crystal_cards.py), so the prefilled triple is self-consistent.
- awr: atomic mass / neutron mass (natural mass for A=0, isotopic otherwise)
- abundance_pct for isotopes (neutron-table abundance preferred; element mass
  table as fallback -- the element table is empty for e.g. uranium)
- energy_dependent: True when the upstream is_energy_dependent flag is set,
  when the tabulated scattering length carries an imaginary part (b_c_i --
  Sears marks such strong absorbers "E": Li-6, B, In, Lu-176, ...), or when
  the entry is in _SEARS_E_EXTRA (Sears "E" entries whose imaginary part the
  upstream table dropped outright: B-10, Dy-164). Such scattering lengths
  are resonance-region values that must not be used as static prefills
  without explicit user override.

MOST_ABUNDANT_A (the default ZA-identity map) is built from abundances over
ALL isotopes of each element, independent of whether the isotope has its own
scattering constants (the identity is a label; the constants used with it are
the natural-element ones). Elements whose isotopes all have zero/unknown
abundance (fully synthetic elements) get NO default: choosing one would be
arbitrary, so most_abundant_a() raises instead.

Entries missing b_c or sigma_inc are skipped and counted in the header.
"""
from __future__ import annotations

import hashlib
import math
import sys
from pathlib import Path

NEUTRON_MASS_U = 1.008664916

# Sears (1992) marks these "E" (strongly energy-dependent complex b), but the
# upstream nsf table carries them as plain real b_c with no flag and no b_c_i
# (10-B: -0.1-1.066i became -0.2; 164-Dy: 49.4-0.79i became 49.4). Labels use
# the generator's "<A>-<symbol>" form.
_SEARS_E_EXTRA = {"10-B", "164-Dy"}


def _abundance(iso) -> float:
    """Best-available natural abundance in percent (0.0 = none/synthetic)."""
    n = getattr(iso, "neutron", None)
    if n is not None and getattr(n, "abundance", None):
        return float(n.abundance)
    return float(iso.abundance or 0.0)


def main() -> None:
    import periodictable as pt

    out_path = Path(__file__).resolve().parents[1] / "irma" / "core" / "nuclear_data.py"
    script = Path(__file__).resolve()
    script_sha = hashlib.sha256(script.read_bytes()).hexdigest()
    source_tag = f"periodictable-{pt.__version__}/nsf"

    nuclides = {}       # (Z, A) -> dict
    skipped = []        # (label, reason)
    discrepancies = []  # (label, derived, table, rel)
    energy_dependent = []

    def add(z, a, symbol, mass, neutron, abundance_pct):
        b_c = getattr(neutron, "b_c", None)
        inc = getattr(neutron, "incoherent", None)
        label = symbol if a == 0 else f"{a}-{symbol}"
        if b_c is None:
            skipped.append((label, "no b_c"))
            return
        if inc is None:
            skipped.append((label, "no sigma_inc"))
            return
        edep = (bool(getattr(neutron, "is_energy_dependent", False))
                or bool(getattr(neutron, "b_c_i", None))
                or label in _SEARS_E_EXTRA)
        if edep:
            energy_dependent.append(label)
        coh = getattr(neutron, "coherent", None)
        sigma_bound = 0.04 * math.pi * b_c * b_c + inc
        if coh is not None:
            table_total = coh + inc
            if table_total > 0:
                rel = abs(sigma_bound - table_total) / table_total
                if rel > 0.01:
                    discrepancies.append((label, sigma_bound, table_total, rel))
        nuclides[(z, a)] = {
            "symbol": symbol,
            "awr": mass / NEUTRON_MASS_U,
            "b_coh_fm": float(b_c),
            "sigma_inc_b": float(inc),
            "sigma_bound_b": sigma_bound,
            "abundance_pct": abundance_pct,
            "energy_dependent": edep,
        }

    most_abundant = {}
    for el in pt.elements:
        if el.number == 0:      # the bare neutron pseudo-element
            continue
        if getattr(el, "neutron", None) is not None:
            add(el.number, 0, el.symbol, el.mass, el.neutron, None)
        best_a, best_ab = None, 0.0
        for iso in el:
            ab = _abundance(iso)
            if ab > best_ab:
                best_a, best_ab = iso.isotope, ab
            n = getattr(iso, "neutron", None)
            if n is None:
                continue
            add(el.number, iso.isotope, el.symbol, iso.mass, n, ab)
        if best_a is not None:
            most_abundant[el.number] = best_a

    lines = []
    w = lines.append
    w('"""Neutron scattering constants per nuclide. GENERATED FILE - DO NOT EDIT.')
    w("")
    w("Regenerate with tools/generate_nuclear_data.py (requires periodictable).")
    w(f"Source: periodictable {pt.__version__} (periodictable.nsf: Atominstitut/")
    w("Rauch-Waschkowski table, Neutron Data Booklet 2nd ed. 2003, with Sears")
    w("1992 cross-references; public domain). Cross-check reference: NIST NCNR")
    w("n-lengths page (same compilation). Citations: V.F. Sears, Neutron News 3")
    w("(1992) 26; H. Rauch and W. Waschkowski, Neutron Data Booklet, 2nd ed.")
    w("(2003); periodictable package documentation (nsf module).")
    w(f"Generator sha256: {script_sha}")
    w("")
    w("sigma_bound_b is derived as 4*pi*b_coh_fm^2/100 + sigma_inc_b, so the")
    w("(b_coh_fm, sigma_inc_b, sigma_bound_b) triple is always mutually")
    w("consistent -- IRMA's cards derive coherent quantities from b_coh, and an")
    w("inconsistent prefilled triple would contradict itself inside one deck.")
    w("Entries below differ by more than 1% from the source table's own")
    w("coherent+incoherent columns. Known causes include a b_c newer than the")
    w("Sears-era sigma_coh column (e.g. V, 13-C: the b_c-derived value is then")
    w("the current one) and strong/resonant absorbers whose scattering length")
    w("has a large imaginary part that a real b_c cannot represent (e.g.")
    w("157-Gd, 113-Cd, 149-Sm); the latter are NOT suitable prefills for")
    w("thermal scattering laws. Override explicitly if you need them:")
    if discrepancies:
        for label, derived, table, rel in sorted(discrepancies, key=lambda t: -t[3]):
            w(f"  {label}: derived {derived:.4f} b vs table {table:.4f} b ({100*rel:.1f}%)")
    else:
        w("  (none)")
    w("")
    w("Entries marked ENERGY-DEPENDENT (upstream flag, imaginary b_c, or the")
    w("_SEARS_E_EXTRA list; resonance-region values;")
    w("carried with energy_dependent=True and refused as silent prefills):")
    if energy_dependent:
        for label in energy_dependent:
            w(f"  {label}")
    else:
        w("  (none)")
    if skipped:
        w("")
        w(f"Entries without usable neutron data (skipped): {len(skipped)}")
    w('"""')
    w("from typing import NamedTuple, Optional")
    w("")
    w(f"SOURCE = {source_tag!r}")
    w("")
    w("")
    w("class Nuclide(NamedTuple):")
    w("    Z: int")
    w("    A: int                    # 0 = natural element")
    w("    symbol: str")
    w("    awr: float                # atomic mass / neutron mass")
    w("    b_coh_fm: float           # real bound coherent scattering length")
    w("    sigma_inc_b: float        # bound incoherent cross section")
    w("    sigma_bound_b: float      # 4*pi*b_coh^2/100 + sigma_inc")
    w("    abundance_pct: Optional[float]  # isotopes only")
    w("    energy_dependent: bool    # resonance-region value; not a safe prefill")
    w("    source: str = SOURCE")
    w("")
    w("")
    w("NUCLIDES = {")
    for (z, a) in sorted(nuclides):
        r = nuclides[(z, a)]
        ab = "None" if r["abundance_pct"] is None else f"{r['abundance_pct']!r}"
        w(f"    ({z}, {a}): Nuclide({z}, {a}, {r['symbol']!r}, {r['awr']!r}, "
          f"{r['b_coh_fm']!r}, {r['sigma_inc_b']!r}, {r['sigma_bound_b']!r}, "
          f"{ab}, {r['energy_dependent']!r}),")
    w("}")
    w("")
    w("# Most-abundant isotope per Z, from abundances over ALL isotopes (the")
    w("# identity label is independent of which isotopes carry their own")
    w("# scattering constants). Fully synthetic elements have no entry: there")
    w("# is no natural composition to name, so most_abundant_a() raises.")
    w("MOST_ABUNDANT_A = {")
    for z in sorted(most_abundant):
        w(f"    {z}: {most_abundant[z]},")
    w("}")
    w("")
    w("_SPECIAL_ALIASES = {'D': (1, 2), 'T': (1, 3)}")
    w("")
    w("_SYMBOL_TO_Z = {}")
    w("for (_z, _a), _n in NUCLIDES.items():")
    w("    _SYMBOL_TO_Z.setdefault(_n.symbol, _z)")
    w("")
    w("")
    w("def _int_exact(value, what):")
    w("    if isinstance(value, bool) or not isinstance(value, int):")
    w("        if isinstance(value, float) and value.is_integer():")
    w("            raise KeyError(")
    w("                f'{what} must be an integer, got float {value!r}; pass an '")
    w("                f'int to avoid silent misidentification')")
    w("        raise KeyError(f'{what} must be an integer, got {value!r}')")
    w("    return value")
    w("")
    w("")
    w("def lookup(key) -> Nuclide:")
    w('    """Resolve a nuclide: \'C\' (natural), \'13-C\'/\'C-13\', \'D\', or (Z, A).')
    w("")
    w("    Element symbols and (Z, 0) tuples resolve to the natural-element")
    w("    entry, whose constants are abundance-averaged; A=0 entries must not")
    w("    be used for ENDF ZA identity (use most_abundant_a / an explicit")
    w("    isotope). Isotope labels require A >= 1.")
    w('    """')
    w("    if isinstance(key, tuple):")
    w("        if len(key) != 2:")
    w("            raise KeyError(f'nuclide tuple must be (Z, A), got {key!r}')")
    w("        z = _int_exact(key[0], 'Z')")
    w("        a = _int_exact(key[1], 'A')")
    w("        if z < 1 or a < 0:")
    w("            raise KeyError(f'nuclide (Z={z}, A={a}) out of range')")
    w("        try:")
    w("            return NUCLIDES[(z, a)]")
    w("        except KeyError:")
    w("            raise KeyError(")
    w("                f'no neutron data for nuclide (Z={z}, A={a}); pass explicit '")
    w("                f'constants via the species override instead') from None")
    w("    s = str(key).strip()")
    w("    if s in _SPECIAL_ALIASES:")
    w("        return lookup(_SPECIAL_ALIASES[s])")
    w("    if '-' in s:")
    w("        left, right = s.split('-', 1)")
    w("        if left.isdigit():")
    w("            a, sym = int(left), right")
    w("        elif right.isdigit():")
    w("            sym, a = left, int(right)")
    w("        else:")
    w("            raise KeyError(f'cannot parse nuclide label {s!r}')")
    w("        if a < 1:")
    w("            raise KeyError(")
    w("                f'isotope label {s!r} needs A >= 1; use the bare element '")
    w("                f'symbol for natural-composition constants')")
    w("        z = _SYMBOL_TO_Z.get(sym)")
    w("        if z is None:")
    w("            raise KeyError(f'unknown element symbol {sym!r} in {s!r}')")
    w("        return lookup((z, a))")
    w("    z = _SYMBOL_TO_Z.get(s)")
    w("    if z is None:")
    w("        raise KeyError(")
    w("            f'unknown element or nuclide {s!r}; pass explicit constants '")
    w("            f'via the species override instead')")
    w("    return NUCLIDES[(z, 0)]")
    w("")
    w("")
    w("def most_abundant_a(z: int) -> int:")
    w('    """Mass number of the most abundant natural isotope of element Z.')
    w("")
    w("    Raises for fully synthetic elements (no natural composition exists,")
    w("    so no default identity can be chosen; pass an explicit isotope).")
    w('    """')
    w("    z = _int_exact(z, 'Z')")
    w("    try:")
    w("        return MOST_ABUNDANT_A[z]")
    w("    except KeyError:")
    w("        raise KeyError(")
    w("            f'element Z={z} has no natural isotopic composition; choose '")
    w("            f'an explicit isotope for the ZA identity') from None")
    w("")

    out_path.write_text("\n".join(lines))
    print(f"wrote {out_path}: {len(nuclides)} nuclides "
          f"({sum(1 for (_, a) in nuclides if a == 0)} natural, "
          f"{sum(1 for (_, a) in nuclides if a > 0)} isotopes); "
          f"{len(discrepancies)} sigma_bound discrepancies >1%; "
          f"{len(energy_dependent)} energy-dependent; "
          f"{len(most_abundant)} ZA defaults; {len(skipped)} skipped")


if __name__ == "__main__":
    sys.exit(main())
