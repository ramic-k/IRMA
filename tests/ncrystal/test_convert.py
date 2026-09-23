"""Convention bridge unit tests (SP1): pack_from_irma_sab.

Pure/fast. Verifies the documented transforms are the ONLY thing convert does:
  - alpha mapped by AWR
  - scaled-symmetric storage S_scaled = S_downscatter * exp(-beta/2)
  - column-major (alpha fastest) flattening into sab_values
  - negative-fringe clip with a table-scale guard
"""
from __future__ import annotations

import math

import pytest

from irma.ncrystal.convert import pack_from_irma_sab
from irma.ncrystal.pack import write_pack


def _pack(alpha, beta, S, ratio=12.0):
    return pack_from_irma_sab(
        material_id="m__X", temperature_K=296.0, bound_xs_barn=5.0,
        element_mass_amu=12.0, alpha_mass_ratio=ratio,
        alpha_grid=alpha, beta_downscatter_abs=beta, sab_asym_downscatter=S)


def test_alpha_scaled_by_mass_ratio():
    alpha = [0.1, 0.2, 0.3]
    pack = _pack(alpha, [0.0, 0.5], [[1.0, 0.5], [2.0, 1.0], [3.0, 1.5]],
                 ratio=11.9)
    assert pack.alpha_grid == pytest.approx([a * 11.9 for a in alpha])
    assert pack.element_mass_amu == pytest.approx(12.0)
    assert pack.metadata["alpha_mass_ratio"] == f"{11.9:.17g}"


def test_scaled_sym_storage_and_ordering():
    # alpha is the fast index within each beta block: sab_values layout is
    # [ (all alpha at beta0), (all alpha at beta1), ... ].
    beta = [0.0, 0.8]
    S = [[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]]   # S[ialpha][ibeta]
    pack = _pack([0.1, 0.2, 0.3], beta, S)
    assert pack.beta_grid == pytest.approx(beta)
    e0 = math.exp(-0.5 * 0.0)
    e1 = math.exp(-0.5 * 0.8)
    expected = [
        1.0 * e0, 2.0 * e0, 3.0 * e0,    # beta 0 block, alpha fast
        10.0 * e1, 20.0 * e1, 30.0 * e1,  # beta 0.8 block
    ]
    assert pack.sab_values == pytest.approx(expected)


def test_negative_fringe_clipped_within_guard():
    # a tiny negative cancellation bin (<1% of max) is clipped to 0 + counted.
    pack = _pack([0.1, 0.2], [0.0, 0.5], [[100.0, -0.5], [50.0, 25.0]])
    assert pack.metadata["negative_sab_clipped_count"] == "1"
    # the clipped cell (ialpha=0, ibeta=1) -> 0 in the beta1 block (index 2)
    assert pack.sab_values[2] == 0.0


def test_large_negative_rejected():
    # -50 exceeds the 1%-of-max guard; the message names the remedy
    with pytest.raises(ValueError, match="num_directions"):
        _pack([0.1, 0.2], [0.0, 0.5], [[100.0, -50.0], [50.0, 25.0]])


def test_beta_must_start_at_zero(tmp_path):
    """The scaled-symmetric half-table needs beta starting at 0 (checked on write)."""
    pack = _pack([0.1, 0.2], [0.5, 1.0], [[1.0, 1.0], [1.0, 1.0]])
    with pytest.raises(ValueError, match="start at zero"):
        write_pack(pack, tmp_path / "p.irmapack")
