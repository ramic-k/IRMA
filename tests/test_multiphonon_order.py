"""Regression tests for derive_required_multiphonon_order — the policy that sizes
the multiphonon Poisson(2W) sum from the anisotropic Debye-Waller tensor:

    2W_max = Q_max^2 * U_max          (U_max = largest thermal-displacement eigenvalue)
    required = ceil(2W_max + 6*sqrt(2W_max)) + 2
    effective = max(requested, min(required, hard_cap))
"""
import math

import numpy as np
import pytest

from irma.core.noncubic_engine import derive_required_multiphonon_order


def _iso(u):
    """One atom, isotropic thermal-displacement tensor u*I -> shape (1,3,3)."""
    return np.array([np.eye(3) * u])


def _expected_required(q, u, margin=6.0):
    two_w = q * q * u
    return int(math.ceil(two_w + margin * math.sqrt(two_w))) + 2


_ANISO = np.array([np.diag([0.005, 0.005, 0.03])])   # soft c axis, u_max = 0.03
_REQ = _expected_required(12.0, 0.02)


@pytest.mark.parametrize("q, U, u_max, requested, cap, want_eff, want_req", [
    (12.0, _iso(0.02), 0.02, 2, 2000, max(2, _REQ), _REQ),        # isotropic formula
    (12.0, _iso(0.02), 0.02, _REQ + 50, 2000, _REQ + 50, _REQ),   # high deck order kept
    (10.0, _ANISO, 0.03, 2, 2000,                                  # largest eigenvalue
     max(2, _expected_required(10.0, 0.03)), _expected_required(10.0, 0.03)),
    (12.0, _iso(0.0), 0.0, 7, 2000, 7, 7),                         # zero u: no growth
    (0.0, _iso(0.02), 0.02, 7, 2000, 7, 7),                        # zero Q: no growth
    (0.0, _iso(0.02), 0.02, 1, 2000, 2, 2),                        # requested floored at 2
    (500.0, _iso(0.05), 0.05, 2, 2000, 2000,                       # hard cap on effective
     _expected_required(500.0, 0.05)),
])
def test_required_order(q, U, u_max, requested, cap, want_eff, want_req):
    eff, req, two_w, got_u_max = derive_required_multiphonon_order(
        q, U, requested_order=requested, hard_cap=cap)
    assert (eff, req) == (want_eff, want_req)
    assert math.isclose(got_u_max, u_max, abs_tol=0.0)
    assert math.isclose(two_w, q * q * u_max)


# --- energy-reach guard (multiphonon_energy_reach) --------------------------
from irma.core.constants import AMASSN, HBAR2_OVER_2MN_MEV_A2    # noqa: E402
from irma.core.noncubic_helpers import (                         # noqa: E402
    KB_MEV_PER_K, multiphonon_energy_reach)

# The 2 meV-cutoff vanadium evaluation that raised the old grid-top warning:
# 293.6 K, a 5 eV energy grid, Q_max = 98.2 1/Angstrom, highest phonon
# energy 31.686 meV, auto order 138.
_V = dict(max_mode_energy_mev=31.686, max_q_ang_inv=98.2, temperature_k=293.6,
          grid_top_mev=5000.0)


def _u_from_spectrum(energies_mev, weights, mass_amu, temperature_k):
    """Mean-square displacement along one axis (Angstrom^2) of an atom whose
    normalised phonon spectrum is the weighted set of mode energies."""
    kt = KB_MEV_PER_K * temperature_k
    w = np.asarray(weights, float) / np.sum(weights)
    e = np.asarray(energies_mev, float)
    hbar2_over_2m = HBAR2_OVER_2MN_MEV_A2 * AMASSN / mass_amu
    return hbar2_over_2m * float(np.sum(w / e / np.tanh(e / (2.0 * kt))))


def test_reach_guard_never_fires_at_the_required_order():
    """Any order meeting the Poisson rule reaches the recoil ridge plus the
    margin, for every atom. U is built from each atom's own spectrum, since
    a U drawn independently of the phonon energies can break the physics the
    guarantee rests on; half the trials use a single mode at E_max, the case
    where the width bound is tight."""
    rng = np.random.default_rng(20260922)
    for trial in range(400):
        e_max = rng.uniform(5.0, 500.0)
        temperature = rng.uniform(1.0, 2000.0)
        q = rng.uniform(1.0, 150.0)
        masses = rng.uniform(1.0, 250.0, int(rng.integers(1, 4)))
        tensors = []
        for mass in masses:
            if trial % 2:
                energies, weights = np.array([e_max]), np.array([1.0])
            else:
                energies = np.append(rng.uniform(0.02, 1.0, 20) * e_max, e_max)
                weights = rng.uniform(0.01, 1.0, energies.size)
            tensors.append(np.eye(3) * _u_from_spectrum(
                energies, weights, mass, temperature))
        _, required, _, _ = derive_required_multiphonon_order(
            q, np.array(tensors), requested_order=2)
        reach = multiphonon_energy_reach(required, e_max, q, masses,
                                         temperature, grid_top_mev=1.0e12)
        assert not reach.short, (trial, reach)


def test_vanadium_auto_order_passes_where_the_grid_top_check_fired():
    reach = multiphonon_energy_reach(138, masses_amu=[50.9415], **_V)
    assert not reach.short
    assert reach.reach_mev < _V["grid_top_mev"]   # the old comparison fired here
    # recoil ridge hbar^2 Q^2 / 2M and width sqrt(2 E_R kT_eff), with
    # kT_eff bounded by (x/2) coth(x/2) kT, x = E_max / kT, written out
    ridge = 2.0721248551 * 98.2 ** 2 * 1.00866491595 / 50.9415
    kt = 0.08617333262 * 293.6
    x = 31.686 / kt
    width = math.sqrt(2.0 * ridge * (x / 2.0) / math.tanh(x / 2.0) * kt)
    assert math.isclose(reach.ridge_mev, ridge, rel_tol=1e-6)
    assert math.isclose(reach.width_mev, width, rel_tol=1e-6)
    assert math.isclose(reach.needed_mev, ridge + 6.0 * width, rel_tol=1e-6)
    assert not reach.capped


def test_order_below_the_requirement_falls_short():
    reach = multiphonon_energy_reach(40, masses_amu=[50.9415], **_V)
    assert math.isclose(reach.reach_mev, 40 * 31.686)
    assert reach.short and not reach.capped


def test_lightest_atom_sets_the_need_and_the_grid_top_caps_it():
    """Hydrogen's ridge at Q_max = 98.2 1/Angstrom is near 20 eV, above a
    5 eV grid, so the need becomes the grid top: the old behaviour."""
    reach = multiphonon_energy_reach(10, masses_amu=[50.9415, 1.00794], **_V)
    assert reach.atom_index == 1 and reach.capped
    assert reach.needed_mev == _V["grid_top_mev"]
    assert reach.short


def test_array_check_ignores_zeros_past_the_needed_reach():
    """The array check measures against the needed reach, not the grid top:
    the vanadium law ends near beta 185 on a grid to 197.6 and needs only
    about beta 51, so it stays silent; a law ending at beta 30 is short."""
    from irma.core.standalone_sab import _truncation_warning
    beta = np.linspace(0.0, 197.6, 400)
    law = np.where(beta <= 185.0, 1.0, 0.0)[None, :]
    assert _truncation_warning("sab", beta, law, 51.3) is None
    short = np.where(beta <= 30.0, 1.0, 0.0)[None, :]
    text = _truncation_warning("sab", beta, short, 51.3)
    assert text is not None and "short of beta=51.3" in text
    # no needed reach recorded: nothing to compare against
    assert _truncation_warning("sab", beta, short, None) is None


def test_array_check_tolerates_a_grid_step_past_the_reach():
    """On a coarse grid the law's last nonzero point may be the grid point
    just below the needed reach; that is not a truncation."""
    from irma.core.standalone_sab import _truncation_warning
    beta = np.array([0.0, 10.0, 40.0, 100.0, 197.6])
    law = np.array([[1.0, 1.0, 1.0, 0.0, 0.0]])
    assert _truncation_warning("sab", beta, law, 51.3) is None
    law_short = np.array([[1.0, 1.0, 0.0, 0.0, 0.0]])
    assert _truncation_warning("sab", beta, law_short, 51.3) is not None
