"""irma.spectra.build_locus_support (P2) -- pure geometry, no engine/phonopy.

The forward orchestrator computes S(Q,E) only on this locus support, so the
support MUST envelope every bank locus Q(E) over the full energy range (else the
sqe interpolator silently zeros the high-E wing). These checks pin that coverage
contract and the uniform loss-side energy grid -- fast, data-free, CI-safe. The
engine-fed end-to-end gate lives in neutron_scattering_analysis/gate_p2_forward.py.
"""
import os

import numpy as np
import pytest

from irma.spectra.forward import build_locus_support, _resolve_jobs
from irma.spectra.sqe import Q_indirect, Q_direct

DE, DQ, EMAX = 0.5, 0.05, 250.0


def test_energy_support_is_uniform_loss_side():
    _, E = build_locus_support("vision", 3.5, [45.0, 135.0], DE, EMAX, DQ)
    assert E[0] == 0.0
    assert E[-1] >= EMAX
    assert np.allclose(np.diff(E), DE)


@pytest.mark.parametrize("geometry,e_fixed,angles,Qof", [
    ("vision", 3.5, [45.0, 135.0], Q_indirect),
    ("indirect", 3.5, [30.0, 90.0, 150.0], Q_indirect),
    ("direct", 250.0, [10.0, 60.0, 120.0], Q_direct),
])
def test_support_envelopes_every_bank_locus(geometry, e_fixed, angles, Qof):
    Q, E = build_locus_support(geometry, e_fixed, angles, DE, EMAX, DQ)
    assert np.all(np.diff(Q) > 0)              # strictly increasing shell grid
    assert Q[0] >= 0.05 - 1e-12                # q_floor respected
    # EVERY finite, positive locus point falls inside [Qmin, Qmax] -> no
    # out-of-range zeros when the sqe interpolator samples along the locus.
    for tt in angles:
        q = np.asarray(Qof(E, e_fixed, tt), float)
        q = q[np.isfinite(q) & (q > 0)]
        assert q.min() >= Q.min() - 1e-9
        assert q.max() <= Q.max() + 1e-9


def test_pad_widens_support_beyond_bare_locus_span():
    angles = [45.0, 135.0]
    Q0, E = build_locus_support("vision", 3.5, angles, DE, EMAX, DQ, q_pad=0.0)
    Q1, _ = build_locus_support("vision", 3.5, angles, DE, EMAX, DQ, q_pad=0.5)
    assert Q1.min() <= Q0.min() + 1e-12 and Q1.max() >= Q0.max() - 1e-12


def test_q_support_covers_energy_gain_locus_when_emin_negative():
    """With e_min<0 (energy-gain measurement) the gain locus Q(E<0) reaches |Q|
    OUTSIDE the loss-side envelope; the Q-support must still cover it, else the
    fill_value=0 interpolator silently zeros the energy-gain wing. The engine
    loss grid E_support stays [0, e_max] (gain rebuilt by detailed balance)."""
    Q, E = build_locus_support("direct", 50.0, [135.0], dE=1.0, e_max=40.0,
                               dQ=0.05, e_min=-20.0, q_pad=0.0)
    assert E[0] == 0.0                          # engine grid unchanged (loss side)
    gain = np.asarray(Q_direct(np.arange(-20.0, 40.5, 1.0), 50.0, 135.0), float)
    gain = gain[np.isfinite(gain) & (gain > 0)]
    assert gain.max() <= Q.max() + 1e-9        # the gain peak Q is now in support
    # e_min=0 default is unchanged (no regression)
    Q0, _ = build_locus_support("direct", 50.0, [135.0], dE=1.0, e_max=40.0,
                                dQ=0.05, e_min=0.0, q_pad=0.0)
    Qneg = Q.copy()
    assert Q0.max() < Qneg.max()               # negative e_min widened the support


def test_kinematic_mask_marks_exactly_the_accessible_band():
    from irma.spectra.forward import kinematic_envelope, kinematic_mask
    Ei = 250.0
    E = np.linspace(0.0, 240.0, 25)
    q_lo, q_hi = kinematic_envelope("direct", Ei, 30.0, 120.0, E)
    Q = np.linspace(0.5, 22.0, 60)
    m = kinematic_mask(Q, q_lo, q_hi)
    assert m.shape == (Q.size, E.size)
    for j in range(E.size):                       # accessible == within [q_lo, q_hi]
        assert np.array_equal(m[:, j], (Q >= q_lo[j]) & (Q <= q_hi[j]))


def test_kinematic_mask_forbidden_energy_is_all_inaccessible():
    """Energy transfer beyond Ei (Ef<0 -> NaN envelope) masks to all-False, not a
    crash -- so blanking the map never errors on the forbidden region."""
    from irma.spectra.forward import kinematic_envelope, kinematic_mask
    Eforbidden = np.array([260.0])                # > Ei=250
    q_lo, q_hi = kinematic_envelope("direct", 250.0, 30.0, 120.0, Eforbidden)
    m = kinematic_mask(np.linspace(0.5, 22.0, 60), q_lo, q_hi)
    assert not m.any()


# ---- parallelism: the forward model must NOT default to serial --------------
def test_resolve_jobs():
    assert _resolve_jobs(None) == max(1, os.cpu_count() or 1)   # auto: every core
    assert _resolve_jobs(8) == 8          # explicit beats auto
    assert _resolve_jobs(1) == 1          # explicit serial still allowed
