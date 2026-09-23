"""Write a base ``.ncmat`` from the phonopy primitive cell.

The NCrystal IRMA plugin matches the pack's per-site anisotropic Debye-Waller
tensors to the material's atom sites BY FRACTIONAL POSITION (tolerance 1e-6, with
modulo-1 wrapping). So the pack and the ``.ncmat`` MUST describe the same cell in
the same setting — a stock stdlib NCMAT for the same compound generally uses a
different origin and fails to match. The exporter therefore emits the NCMAT
structure straight from the phonopy primitive cell the pack was built from, so
the two match by construction and the anisotropic-DW elastic line works.

The plugin takes over the inelastic channel (and the coherent/incoherent elastic
when the pack owns it), disabling those base channels. The base ``@DYNINFO`` is a
``vdosdebye`` placeholder that NCrystal needs to construct a valid material. With
``elastic: true`` it never reaches the cross section; with ``elastic: false``
NCrystal's own elastic uses it, so its Debye temperature reproduces the engine's
mean-squared displacement in NCrystal's Debye model.
"""
from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence

import numpy as np

from irma.core.crystal import lattice_to_cell_params

# SI constants for the Debye MSD <-> Debye-temperature map.
_HBAR_J_S = 1.054571817e-34
_KB_J_PER_K = 1.380649e-23
_AMU_KG = 1.66053906660e-27
_ANG2_M2 = 1.0e-20


def debye_msd(theta_K: float, mass_amu: float, temperature_K: float) -> float:
    """Mean-squared displacement [Angstrom^2] of the Debye model, as NCrystal
    computes it (NCDebyeMSD.cc):
    ``<u^2> = 3 hbar^2 / (m kB theta) [1/4 + (T/theta)^2 int_0^{theta/T} x/(e^x-1) dx]``.
    """
    scale = 3.0 * _HBAR_J_S ** 2 / (float(mass_amu) * _AMU_KG * _KB_J_PER_K
                                    * float(theta_K)) / _ANG2_M2
    if temperature_K <= 0.0:
        return scale * 0.25
    y = float(theta_K) / float(temperature_K)
    # x/(e^x - 1) is below 1e-24 past x = 60, so the integral stops there
    x = np.linspace(0.0, min(y, 60.0), 4001)
    f = np.ones_like(x)
    f[1:] = x[1:] / np.expm1(x[1:])
    integral = (x[1] - x[0]) / 3.0 * (f[0] + f[-1] + 4.0 * f[1:-1:2].sum()
                                      + 2.0 * f[2:-1:2].sum())
    return scale * (0.25 + integral / (y * y))


def debye_temperature_from_msd(msd_a2: float, mass_amu: float,
                               temperature_K: float) -> float:
    """Debye temperature whose :func:`debye_msd` equals ``msd_a2``.

    Bisection on the full Debye formula (the MSD falls as theta rises), so
    NCrystal's ``vdosdebye`` reproduces the engine's MSD. Clamped to
    [1, 1e5] K.
    """
    lo, hi = 1.0, 1.0e5
    if debye_msd(lo, mass_amu, temperature_K) <= msd_a2:
        return lo
    if debye_msd(hi, mass_amu, temperature_K) >= msd_a2:
        return hi
    for _ in range(100):
        mid = math.sqrt(lo * hi)
        if debye_msd(mid, mass_amu, temperature_K) > msd_a2:
            lo = mid
        else:
            hi = mid
        if hi / lo - 1.0 < 1e-13:
            break
    return float(math.sqrt(lo * hi))


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
