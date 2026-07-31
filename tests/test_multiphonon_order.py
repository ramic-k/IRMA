"""Regression tests for derive_required_multiphonon_order — the policy that sizes
the multiphonon Poisson(2W) sum from the anisotropic Debye-Waller tensor:

    2W_max = Q_max^2 * U_max          (U_max = largest thermal-displacement eigenvalue)
    required = ceil(2W_max + 6*sqrt(2W_max)) + 2
    effective = max(requested, min(required, hard_cap))
"""
import math
import numpy as np

from irma.core.noncubic_engine import derive_required_multiphonon_order


def _iso(u):
    """One atom, isotropic thermal-displacement tensor u*I -> shape (1,3,3)."""
    return np.array([np.eye(3) * u])


def _expected_required(q, u, margin=6.0):
    two_w = q * q * u
    return int(math.ceil(two_w + margin * math.sqrt(two_w))) + 2


def test_isotropic_formula():
    q, u = 12.0, 0.02
    eff, req, two_w, u_max = derive_required_multiphonon_order(q, _iso(u), requested_order=2)
    assert u_max == u
    assert math.isclose(two_w, q * q * u)
    assert req == _expected_required(q, u)
    assert eff == max(2, req)


def test_effective_honors_higher_requested():
    q, u = 12.0, 0.02
    req_formula = _expected_required(q, u)
    eff, req, _, _ = derive_required_multiphonon_order(q, _iso(u), requested_order=req_formula + 50)
    assert req == req_formula
    assert eff == req_formula + 50            # a deliberately high deck order is kept


def test_anisotropic_uses_largest_eigenvalue():
    """U_max is the soft-axis (largest) eigenvalue, e.g. graphite c-axis."""
    U = np.array([np.diag([0.005, 0.005, 0.03])])   # u_max = 0.03
    q = 10.0
    eff, req, two_w, u_max = derive_required_multiphonon_order(q, U, requested_order=2)
    assert math.isclose(u_max, 0.03)
    assert math.isclose(two_w, q * q * 0.03)
    assert req == _expected_required(q, 0.03)


def test_higher_Q_needs_higher_order():
    U = _iso(0.02)
    _, req_lo, _, _ = derive_required_multiphonon_order(6.0, U, requested_order=2)
    _, req_hi, _, _ = derive_required_multiphonon_order(24.0, U, requested_order=2)
    assert req_hi > req_lo


def test_zero_displacement_returns_requested():
    """u_max <= 0 or Q <= 0 -> no multiphonon growth; required == requested."""
    eff, req, two_w, u_max = derive_required_multiphonon_order(12.0, _iso(0.0), requested_order=7)
    assert (eff, req, two_w) == (7, 7, 0.0)
    eff0, req0, _, _ = derive_required_multiphonon_order(0.0, _iso(0.02), requested_order=7)
    assert eff0 == req0 == 7


def test_requested_floored_at_two():
    """requested_order is floored at 2 (one-phonon-only is order 1, no Poisson sum)."""
    eff, req, _, _ = derive_required_multiphonon_order(0.0, _iso(0.02), requested_order=1)
    assert eff == 2


def test_hard_cap():
    """A huge Q is capped at the 2000 safety limit in the effective order."""
    U = _iso(0.05)
    eff, req, _, _ = derive_required_multiphonon_order(500.0, U, requested_order=2, hard_cap=2000)
    assert req > 2000
    assert eff == 2000
