"""Shared S(α,β)⇄(Q,E) grid convention (irma.core.sab_grids). Pure/fast.

Pins the inverse against the forward map so the ENDF/standalone path and the
NCrystal exporter cannot drift.
"""
from __future__ import annotations

import numpy as np
import pytest

from irma.core.sab_grids import (
    irma_grid_to_physical_qe,
    physical_qe_to_irma_grid,
    auto_sab_grid,
    make_uniform_qe_grid,
)


def test_inverse_recovers_qe():
    q = np.array([0.5, 1.0, 2.0, 4.0, 8.0])
    e = np.array([0.0, 1.0, 5.0, 25.0, 100.0])
    alpha, beta = physical_qe_to_irma_grid(
        q, e, lat=1, temperature_k=296.0, mass_ratio=11.898)
    q2, e2, _aabs, _babs = irma_grid_to_physical_qe(
        alpha, beta, 1, 296.0, 11.898)
    assert np.allclose(q2, q, rtol=1e-12, atol=1e-12)
    assert np.allclose(e2, e, rtol=1e-12, atol=1e-12)


def test_inverse_recovers_alpha_beta():
    alpha = np.array([0.05, 0.2, 0.8, 3.0, 12.0])
    beta = np.array([0.0, 0.5, 1.6, 5.0, 13.0])
    q, e, _aabs, _babs = irma_grid_to_physical_qe(alpha, beta, 1, 500.0, 8.93)
    a2, b2 = physical_qe_to_irma_grid(q, e, lat=1, temperature_k=500.0,
                                      mass_ratio=8.93)
    assert np.allclose(a2, alpha, rtol=1e-12, atol=1e-12)
    assert np.allclose(b2, beta, rtol=1e-12, atol=1e-12)


def test_lat0_no_rescale():
    # lat=0: alpha/beta are absolute, sc=1, so the inverse is a pure unit map.
    q = np.array([1.0, 3.0])
    e = np.array([0.0, 10.0])
    a1, b1 = physical_qe_to_irma_grid(q, e, lat=0, temperature_k=296.0,
                                      mass_ratio=1.0)
    a2, b2 = physical_qe_to_irma_grid(q, e, lat=1, temperature_k=296.0,
                                      mass_ratio=1.0)
    # lat=1 applies the THERM/kT rescale, so the two differ (unless T=293.6K-ish)
    assert not np.allclose(a1, a2)


def test_auto_grid_properties():
    alpha, beta = auto_sab_grid(
        q_min=0.25, q_max=12.0, dq=0.25, e_min=0.0, e_max=80.0, de=1.0,
        lat=1, temperature_k=296.0, mass_ratio=11.898)
    assert alpha[0] > 0.0                                  # strictly positive
    assert np.all(np.diff(alpha) > 0)                      # strictly increasing
    assert beta[0] == 0.0                                  # downscatter origin
    assert np.all(np.diff(beta) > 0)


def test_auto_grid_alpha_is_awr_dependent_beta_is_not():
    a_light, b_light = auto_sab_grid(
        q_min=0.25, q_max=12.0, dq=0.25, e_min=0.0, e_max=80.0, de=1.0,
        lat=1, temperature_k=296.0, mass_ratio=1.0)
    a_heavy, b_heavy = auto_sab_grid(
        q_min=0.25, q_max=12.0, dq=0.25, e_min=0.0, e_max=80.0, de=1.0,
        lat=1, temperature_k=296.0, mass_ratio=16.0)
    assert np.allclose(b_light, b_heavy)                   # β mass-independent
    assert not np.allclose(a_light, a_heavy)               # α scales with 1/A
    assert np.allclose(a_light, a_heavy * 16.0)            # α ∝ 1/mass_ratio


def test_auto_grid_rejects_bad_ranges():
    with pytest.raises(ValueError):
        auto_sab_grid(q_min=1.0, q_max=0.5, dq=0.1, e_min=0.0, e_max=10.0,
                      de=1.0, lat=1, temperature_k=296.0, mass_ratio=1.0)
    with pytest.raises(ValueError):
        make_uniform_qe_grid(0.0, 1.0, 0.0, 0.0, 1.0, 0.1)   # dq=0
