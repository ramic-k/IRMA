"""MEF (LTHR=3) applies the directional Debye-Waller. On a synthetic
anisotropic crystal the plane-by-plane attenuation
W_s(Ĝ) = (Ĝ·F_s·Ĝ)/(awr_s·kT) must reproduce a hand-evaluated value and
differ between c-axis and basal-plane edges. That SEF and MEF share this
arithmetic is tested in test_elastic_dw.py
(test_sef_and_mef_edge_arithmetic_differ_only_by_scale).
"""
import numpy as np
import pytest

from irma.core.constants import BK
from irma.core.endf_writer import _build_mef_elastic


def _crystal_info(F):
    """Single-species crystal with two single-plane Bragg edges:
    one G along c (anisotropy-sensitive), one in the basal plane."""
    D = np.array([[1.0]])                      # |F_hkl|^2-like plane weight
    return {
        'atom_types': [{
            'Z': 6, 'A': 12, 'awr': 11.898, 'b_coh': 6.646,
            'sigma_inc': 0.001, 'dwpix': [0.5],
        }],
        'principal_atom_idx': 0,
        'species_corr': np.array([[[1.0]], [[1.0]], [[1.0]]]),
        'F_species_per_temp': [[F]],
        'bragg_dir_terms': [
            [(np.array([0.0, 0.0, 1.0]), D)],   # edge 0: G ∥ c
            [(np.array([1.0, 0.0, 0.0]), D)],   # edge 1: G ∥ a
        ],
    }


def test_mef_directional_attenuation_matches_hand_value():
    F = np.diag([0.5, 0.5, 3.0])               # strongly anisotropic
    ci = _crystal_info(F)
    bragg = [(0.002, 1.0), (0.004, 1.0)]
    tempr = [296.0]

    mef = _build_mef_elastic(1, 6012.0, 11.898, bragg, 2, 1, tempr,
                             ci, [0.5])
    assert mef['LTHR'] == 3

    # Hand evaluation of the edge-0 (G ∥ c) coherent increment:
    # W = F_cc/(awr·kT); delta = b_sqb^2 · exp(-2(W+W)·E) · D
    kT = 296.0 * BK
    b_sqb = 6.646 / 10.0
    W_c = 3.0 / (11.898 * kT)
    d0_hand = b_sqb * b_sqb * np.exp(-4.0 * W_c * 0.002)
    W_a = 0.5 / (11.898 * kT)
    d1_hand = b_sqb * b_sqb * np.exp(-4.0 * W_a * 0.004)

    # S is cumulative over edges; recover the per-edge increments by
    # differencing.
    S0 = np.array(mef['S_T0_table']['S'][:2])
    d0, d1 = S0[0], S0[1] - S0[0]
    assert d0 == pytest.approx(d0_hand, rel=1e-6)
    assert d1 == pytest.approx(d1_hand, rel=1e-6)
    # anisotropy visible: same |F|^2 weight, but different attenuation slopes
    assert d0 != pytest.approx(d1, rel=1e-3)

    # And the isotropic fallback (no F matrices) gives a DIFFERENT result,
    # proving the directional branch actually engaged.
    ci_iso = _crystal_info(F)
    ci_iso['F_species_per_temp'] = None
    mef_iso = _build_mef_elastic(1, 6012.0, 11.898, bragg, 2, 1, tempr,
                                 ci_iso, [0.5])
    S0_iso = np.array(mef_iso['S_T0_table']['S'][:2])
    assert S0_iso[0] != pytest.approx(d0, rel=1e-6)
