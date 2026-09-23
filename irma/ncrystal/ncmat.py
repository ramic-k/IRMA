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


def build_base_ncmat(
    *,
    lattice_ang,
    scaled_positions: Sequence[Sequence[float]],
    symbols: Sequence[str],
    debye_temperatures: dict[str, float],
) -> str:
    """Build a valid NCMAT (v5) for the phonopy primitive cell.

    ``@CELL`` + ``@ATOMPOSITIONS`` carry the exact phonopy geometry (so the pack's
    DW-tensor positions match the material's atom sites). Each element gets a
    ``@DYNINFO type=vdosdebye`` with a Debye temperature back-derived from the
    phonopy mean-squared displacement (:func:`debye_temperature_from_msd`), which
    gives NCrystal a valid MSD source to construct the crystalline material. Both
    the base inelastic and (for an elastic export) the base elastic are then
    overridden by the plugin, so this placeholder dynamics never reaches the
    cross section. Neutron data per element comes from NCrystal's built-in atom
    database (natural-element b_coh / sigma_inc / sigma_abs); supply an
    ``@ATOMDB`` yourself for non-natural isotopics.

    The NCMAT carries no ``@TEMPERATURE`` (that section is NCMAT v7+; this is v5),
    so NCrystal defaults the material to 293.15 K. The pack's S(alpha,beta) is
    precomputed at a single bake temperature and the plugin does NOT interpolate,
    so load at that temperature with ``;temp=<bakeT>`` (required whenever the pack
    is not baked at 293.15 K). The plugin REJECTS a requested/pack temperature
    mismatch with a clear error rather than silently sampling the wrong-temperature
    law.
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
    return "\n".join(lines) + "\n"


def custom_irma_section(pack_filenames: Sequence[str]) -> str:
    """The ``@CUSTOM_IRMA`` block referencing the per-principal pack files.

    The plugin parser accepts ONLY ``pack <path>`` lines (one per principal
    pack); no other keys.
    """
    lines = ["@CUSTOM_IRMA"]
    for fn in pack_filenames:
        lines.append(f"  pack {fn}")
    return "\n".join(lines) + "\n"


def assemble_material_ncmat(
    *,
    lattice_ang,
    scaled_positions: Sequence[Sequence[float]],
    symbols: Sequence[str],
    pack_filenames: Sequence[str],
    debye_temperatures: dict[str, float],
) -> str:
    """A complete, loadable ``.ncmat``: phonopy structure + ``@CUSTOM_IRMA``."""
    base = build_base_ncmat(
        lattice_ang=lattice_ang, scaled_positions=scaled_positions,
        symbols=symbols, debye_temperatures=debye_temperatures)
    return base + "\n" + custom_irma_section(pack_filenames)
