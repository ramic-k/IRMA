"""Write a base ``.ncmat`` from the phonopy primitive cell.

The NCrystal IRMA plugin matches the pack's per-site anisotropic Debye-Waller
tensors to the material's atom sites BY FRACTIONAL POSITION (tolerance 1e-6, with
modulo-1 wrapping). So the pack and the ``.ncmat`` MUST describe the same cell in
the same setting — a stock stdlib NCMAT for the same compound generally uses a
different origin and fails to match. The exporter therefore emits the NCMAT
structure straight from the phonopy primitive cell the pack was built from, so
the two match by construction and the anisotropic-DW elastic line works.

The plugin takes over the inelastic channel (and the coherent/incoherent elastic
when the pack owns it), disabling those base channels — so the base ``@DYNINFO``
is a placeholder (``freegas``) that NCrystal needs only to construct a valid
material; it never contributes to the cross section.
"""
from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence

import numpy as np

from irma.core.crystal import lattice_to_cell_params

# SI constants for the high-temperature Debye-Waller MSD <-> Debye-temperature map.
_HBAR_J_S = 1.054571817e-34
_KB_J_PER_K = 1.380649e-23
_AMU_KG = 1.66053906660e-27
_ANG2_M2 = 1.0e-20


def debye_temperature_from_msd(msd_a2: float, mass_amu: float,
                               temperature_K: float) -> float:
    """High-T Debye temperature reproducing a mean-squared displacement.

    From the harmonic high-temperature limit ``<u^2> = 3 hbar^2 T / (m kB
    theta_D^2)`` → ``theta_D = hbar * sqrt(3 T / (m kB <u^2>))``. Used only to give
    NCrystal a valid MSD source so the crystalline material constructs; for an
    elastic export the plugin overrides the base elastic with the pack's
    anisotropic tensors, so the exact value does not reach the cross section.
    Clamped to a sane [1, 1e5] K range.
    """
    u2 = float(msd_a2) * _ANG2_M2
    m = float(mass_amu) * _AMU_KG
    theta = _HBAR_J_S * math.sqrt(3.0 * float(temperature_K) / (m * _KB_J_PER_K * u2))
    return float(min(1.0e5, max(1.0, theta)))


def _fmt(value: float) -> str:
    """Format a float for NCMAT output (12 significant digits)."""
    return f"{float(value):.12g}"


def assemble_material_ncmat(
    *,
    lattice_ang,
    scaled_positions: Sequence[Sequence[float]],
    symbols: Sequence[str],
    pack_filenames: Sequence[str],
    debye_temperatures: dict[str, float],
) -> str:
    """A complete, loadable NCMAT (v5): the phonopy cell plus ``@CUSTOM_IRMA``.

    ``@CELL`` + ``@ATOMPOSITIONS`` carry the exact phonopy geometry (so the
    pack's DW-tensor positions match the material's atom sites). Each element
    gets a ``@DYNINFO type=vdosdebye`` placeholder whose Debye temperature
    (:func:`debye_temperature_from_msd`) gives NCrystal a valid MSD source; the
    plugin overrides it, and the element neutron data come from NCrystal's
    atom database. ``@CUSTOM_IRMA`` lists one ``pack <path>`` line per
    principal pack (the only key the plugin accepts).

    v5 has no ``@TEMPERATURE``, so NCrystal defaults to 293.15 K: load with
    ``;temp=<bakeT>``; the plugin rejects a temperature that differs from the
    pack's and does not interpolate.
    """
    positions = np.asarray(scaled_positions, float).reshape(len(symbols), 3)
    a, b, c, alpha, beta, gamma = lattice_to_cell_params(lattice_ang)

    lines = ["NCMAT v5"]
    lines.append("@CELL")
    lines.append(f"  lengths {_fmt(a)} {_fmt(b)} {_fmt(c)}")
    lines.append(f"  angles {_fmt(alpha)} {_fmt(beta)} {_fmt(gamma)}")
    lines.append("@ATOMPOSITIONS")
    for sym, pos in zip(symbols, positions):
        lines.append(f"  {sym} {_fmt(pos[0])} {_fmt(pos[1])} {_fmt(pos[2])}")

    counts = Counter(symbols)
    ntot = len(symbols)
    for sym in dict.fromkeys(symbols):                 # first-appearance order
        theta = float(debye_temperatures[sym])
        lines.append("@DYNINFO")
        lines.append(f"  element {sym}")
        lines.append(f"  fraction {_fmt(counts[sym] / ntot)}")
        lines.append("  type vdosdebye")
        lines.append(f"  debye_temp {_fmt(theta)}")
    lines += ["", "@CUSTOM_IRMA"] + [f"  pack {fn}" for fn in pack_filenames]
    return "\n".join(lines) + "\n"
