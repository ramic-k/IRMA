"""Structure-free NCMAT shell (no @CELL): @DYNINFO freegas + @CUSTOM_ENDFTSL.

multi_pack_ncmat writes one @DYNINFO per element (fractions summing to 1) + one `pack` line per principal
scatterer. The plugin overrides every channel, so the placeholder freegas dynamics
never reach the cross section; the per-pack scatterers (inelastic + incoherent
elastic, plus the coherent block on exactly ONE pack) are summed by the C++.
"""
from __future__ import annotations


def multi_pack_ncmat(elements, density_g_cm3, pack_filenames):
    """elements: list of (symbol, fraction), fractions summing to 1.
    pack_filenames: @CUSTOM_ENDFTSL pack paths (one per principal scatterer)."""
    total = sum(f for _, f in elements)
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"element fractions must sum to 1, got {total}")
    lines = ["NCMAT v5", "@DENSITY", f"  {density_g_cm3:.6g} g_per_cm3"]
    for sym, frac in elements:
        lines += ["@DYNINFO", f"  element {sym}",
                  f"  fraction {frac:.10g}", "  type freegas"]
    lines.append("@CUSTOM_ENDFTSL")
    for pf in pack_filenames:
        lines.append(f"  pack {pf}")
    return "\n".join(lines) + "\n"
