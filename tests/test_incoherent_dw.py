"""Unit tests for irma.core.incoherent_dw (orientation-averaged incoherent
elastic Debye-Waller factor).

The closed forms are validated against an independent brute-force spherical
average (golden-spiral directions), not just against each other, and the
directional sigma(E) is pinned to the standard ENDF LTHR=2 form in the cubic
limit.
"""
import math

import numpy as np
import pytest

from irma.core.incoherent_dw import (
    dawsn,
    dw_orientation_average,
    dw_orientation_average_tsq,
    sigma_elinc_directional,
    u_eigenvalues,
)


def _golden_spiral_average(u_eig, Q, n=200_000):
    """Brute-force <exp(-Q^2 uhat.diag(u).uhat)> over golden-spiral directions."""
    i = np.arange(n) + 0.5
    cos_theta = 1.0 - 2.0 * i / n
    sin_theta = np.sqrt(1.0 - cos_theta**2)
    golden = math.pi * (3.0 - math.sqrt(5.0))
    phi = golden * i
    x, y, z = sin_theta * np.cos(phi), sin_theta * np.sin(phi), cos_theta
    u1, u2, u3 = u_eig
    p = u1 * x * x + u2 * y * y + u3 * z * z
    Q = np.atleast_1d(np.asarray(Q, float))
    return np.exp(-Q[:, None] ** 2 * p[None, :]).mean(axis=1)


GRAPHITE_LIKE = (0.00226, 0.00226, 0.0149)   # basal, basal, c  [Ang^2]
PROLATE = (0.002, 0.011, 0.011)
TRIAXIAL = (0.003, 0.007, 0.012)
Q_GRID = np.array([0.5, 2.0, 5.0, 10.0, 20.0, 40.0])


def test_u_eigenvalues_sorted_and_clipped():
    U = np.diag([0.01, -1e-16, 0.005])
    eig = u_eigenvalues(U)
    assert eig[0] == 0.0
    assert np.all(np.diff(eig) >= 0.0)


def test_u_eigenvalues_rejects_materially_negative():
    """A genuinely indefinite tensor is an input error, not something to
    silently clip (Codex review D3)."""
    with pytest.raises(ValueError, match="positive semidefinite"):
        u_eigenvalues(np.diag([-0.01, 0.02, 0.03]))


def test_isotropic_branch_is_exact_exponential():
    u = 0.006
    f = dw_orientation_average((u, u, u), Q_GRID)
    assert np.allclose(f, np.exp(-Q_GRID**2 * u), rtol=1e-14, atol=0.0)


def test_zero_tensor_gives_unity():
    f = dw_orientation_average((0.0, 0.0, 0.0), Q_GRID)
    assert np.all(f == 1.0)


@pytest.mark.parametrize("eig", [GRAPHITE_LIKE, PROLATE, TRIAXIAL])
def test_closed_forms_match_bruteforce_average(eig):
    f = dw_orientation_average(eig, Q_GRID)
    ref = _golden_spiral_average(eig, Q_GRID)
    assert np.allclose(f, ref, rtol=2e-4)


@pytest.mark.parametrize("eig", [GRAPHITE_LIKE, PROLATE])
def test_uniaxial_closed_form_matches_quadrature(eig):
    """Perturbing the degenerate pair by 1e-6 forces the triaxial quadrature;
    the answers must agree to the perturbation size."""
    u1, u2, u3 = eig
    if u2 == u1:            # oblate: split the low pair
        pert = (u1, u2 * (1.0 + 1e-6), u3)
    else:                   # prolate: split the high pair
        pert = (u1, u2, u3 * (1.0 + 1e-6))
    f_closed = dw_orientation_average(eig, Q_GRID)
    f_quad = dw_orientation_average(pert, Q_GRID)
    assert np.allclose(f_closed, f_quad, rtol=1e-4)


def test_jensen_bound_and_high_q_dominance():
    """f >= exp(-Q^2 tr/3) always, strictly so at high Q for anisotropy."""
    for eig in (GRAPHITE_LIKE, PROLATE, TRIAXIAL):
        ubar = sum(eig) / 3.0
        f = dw_orientation_average(eig, Q_GRID)
        iso = np.exp(-Q_GRID**2 * ubar)
        assert np.all(f >= iso * (1.0 - 1e-12))
        assert f[-1] > 2.0 * iso[-1]


def test_q_zero_limit_is_one_every_branch():
    for eig in ((0.006, 0.006, 0.006), GRAPHITE_LIKE, PROLATE, TRIAXIAL):
        f = dw_orientation_average(eig, np.array([0.0]))
        assert f[0] == pytest.approx(1.0, abs=1e-12)


def test_monotone_decreasing_in_q():
    q = np.linspace(0.0, 30.0, 400)
    for eig in (GRAPHITE_LIKE, PROLATE, TRIAXIAL):
        f = dw_orientation_average(eig, q)
        assert np.all(np.diff(f) <= 1e-15)


def test_dawsn_against_scipy():
    scipy_special = pytest.importorskip("scipy.special")
    x = np.concatenate([np.linspace(0.01, 0.19, 7),
                        np.linspace(0.2, 15.0, 200)])
    ours = dawsn(x)
    ref = scipy_special.dawsn(x)
    assert np.allclose(ours, ref, rtol=5e-7, atol=1e-9)
    assert np.allclose(dawsn(-x), -ref, rtol=5e-7, atol=1e-9)


def test_sigma_cubic_limit_matches_endf_form():
    """Directional sigma(E) with an isotropic tensor == the standard ENDF
    LTHR=2 incoherent-elastic result (sigma_b/2)(1-exp(-4EW'))/(2EW')."""
    u = 0.0052           # Ang^2
    sb = 80.0            # barn
    ksq = np.array([0.01, 0.1, 1.0, 10.0, 100.0, 1000.0])  # 1/Ang^2
    got = sigma_elinc_directional(ksq, [(sb, (u, u, u))])
    a = 2.0 * ksq * u    # = 2 E W' in these variables (4k^2 u / 2)
    ref = (sb / 2.0) * (1.0 - np.exp(-2.0 * a)) / a
    assert np.allclose(got, ref, rtol=1e-6)


def test_sigma_accurate_when_kinematic_range_dwarfs_dw_scale():
    """The log grid must resolve the DW decay onset even when 4k^2 >> 1/u
    (Codex review D2): isotropic exact result to 1e-6 at extreme k^2."""
    for u in (1.0, 0.1, 0.01):
        sb = 1.0
        ksq = np.array([1.0e6, 1.0e8])
        got = sigma_elinc_directional(ksq, [(sb, (u, u, u))])
        a = 2.0 * ksq * u
        ref = (sb / 2.0) * (1.0 - np.exp(-2.0 * a)) / a
        assert np.allclose(got, ref, rtol=1e-6), (u, got, ref)


def test_sigma_zero_energy_limit_is_bound_sum():
    channels = [(2.0, GRAPHITE_LIKE), (3.0, TRIAXIAL)]
    got = sigma_elinc_directional(np.array([0.0]), channels)
    assert got[0] == pytest.approx(5.0, rel=1e-12)


def test_sigma_directional_exceeds_isotropic_at_high_energy():
    """Jensen: the directional sigma must exceed the trace/3 isotropic sigma."""
    sb = 1.0
    ubar = sum(GRAPHITE_LIKE) / 3.0
    ksq = np.array([50.0, 200.0, 1000.0])
    directional = sigma_elinc_directional(ksq, [(sb, GRAPHITE_LIKE)])
    isotropic = sigma_elinc_directional(ksq, [(sb, (ubar, ubar, ubar))])
    assert np.all(directional > isotropic)


def test_tsq_and_q_entry_points_agree():
    f_q = dw_orientation_average(TRIAXIAL, Q_GRID)
    f_t = dw_orientation_average_tsq(TRIAXIAL, Q_GRID**2)
    assert np.array_equal(f_q, f_t)
