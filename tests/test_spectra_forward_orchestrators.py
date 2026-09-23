"""compute_spectrum and compute_sqe_map end to end through the mode-0 DOS
path (no phonopy, no engine pool): shapes, finiteness, the total = inelastic +
elastic split, the gain/loss balance and the response to the DOS.
"""
import numpy as np
import pytest

from irma.spectra.forward import (compute_spectrum, compute_sqe_map,
                                   SpectrumResult, SQEMap)

T_K = 296.0


def _dos(w_max_meV=40.0, opt_meV=90.0, n=400, delta_ev=0.0005):
    """A Debye acoustic band + a Gaussian optical peak on a uniform omega grid (eV)."""
    omega = np.arange(n) * delta_ev
    w = omega * 1000.0                                   # meV
    rho = np.where(w <= w_max_meV, (w / w_max_meV) ** 2, 0.0)
    rho += 0.5 * np.exp(-0.5 * ((w - opt_meV) / 5.0) ** 2)
    return omega, rho


def _carbon(**extra):
    """A graphite-like single species for the DOS path (values not load-bearing)."""
    omega, rho = _dos()
    sp = {"symbol": "C", "omega_ev": omega, "rho": rho, "awr": 11.898,
          "sigma_bound_b": 5.551, "sigma_inc_b": 0.001, "multiplicity": 1}
    sp.update(extra)
    return sp


# mode-0 ignores phonopy_yaml/mesh/sab_* (the DOS path supplies its own sigma_b);
# they are still required kwargs, so pass harmless placeholders. nphon pinned to
# an explicit 40 (auto-order OFF) so these structural tests stay deterministic
# and sub-second; the mode-0 auto-order behavior itself is covered in
# test_spectra_mode0_auto_order.py.
BASE = dict(phonopy_yaml=None, temperature_k=T_K, mesh=None,
            sab_mass_ratio=11.898, sab_sigma_barn=5.551,
            e_max=120.0, dE=1.0, dQ=0.1, multiphonon_max_order=40,
            auto_multiphonon_order=False,
            progress=lambda *a, **k: None)

# the same placeholders for compute_sqe_map, which takes its own grid arguments
MAP_BASE = dict(phonopy_yaml=None, temperature_k=T_K, mesh=None,
                sab_mass_ratio=11.898, sab_sigma_barn=5.551,
                multiphonon_max_order=40, auto_multiphonon_order=False,
                progress=lambda *a, **k: None)


def _assert_spectrum_invariants(r, expect_inelastic=True):
    assert isinstance(r, SpectrumResult)
    # bank-combined curves are 1-D over the output energy axis
    assert r.I_total.shape == r.E.shape
    assert r.I_inelastic.shape == r.E.shape
    assert r.I_elastic.shape == r.E.shape
    assert np.all(np.isfinite(r.I_total))
    # intensities are physical (>= 0 up to float noise from the resolution conv)
    assert np.all(r.I_inelastic >= -1e-9)
    assert np.all(r.I_elastic >= -1e-9)
    # the defining decomposition of the total spectrum
    assert np.allclose(r.I_total, r.I_inelastic + r.I_elastic, rtol=0, atol=1e-12)
    if expect_inelastic:
        assert r.I_inelastic.max() > 0.0
    # per-angle arrays agree with the combined curve in shape + decomposition
    assert r.I_inelastic_per_angle.shape[1] == r.E.size
    assert r.I_elastic_per_angle.shape == r.I_inelastic_per_angle.shape
    assert np.allclose(r.I_total_per_angle,
                       r.I_inelastic_per_angle + r.I_elastic_per_angle)


def test_compute_spectrum_vision_mode0():
    """VISION preset (supplies its own banks): full inelastic-only spectrum."""
    r = compute_spectrum(geometry="vision", dos_species=[_carbon()], **BASE)
    _assert_spectrum_invariants(r)
    assert r.geometry == "vision"
    assert r.metadata["engine_metadata"].get("mode") == 0   # mode-0 DOS path marker
    # no elastic model was supplied -> the elastic channel is identically zero
    assert np.all(r.I_elastic == 0.0)
    assert r.angles_deg and len(r.angles_deg) == r.I_inelastic_per_angle.shape[0]


@pytest.mark.parametrize("geometry,e_fixed,angles", [
    ("indirect", 3.5, [45.0, 135.0]),
    ("direct", 250.0, [30.0, 90.0]),
])
def test_compute_spectrum_indirect_and_direct(geometry, e_fixed, angles):
    """Both fixed-Ef and fixed-Ei geometries project the same S(Q,E) sanely."""
    r = compute_spectrum(geometry=geometry, e_fixed_meV=e_fixed, angles_deg=angles,
                         dos_species=[_carbon()], **BASE)
    _assert_spectrum_invariants(r)
    assert r.geometry == geometry
    assert len(r.angles_deg) == len(angles)
    # each bank locus Q(E) is finite/positive on the loss side it can reach
    Q = np.asarray(r.Q, float)
    assert Q.shape[0] == len(angles)
    assert np.all(Q[np.isfinite(Q)] >= 0.0)


def test_compute_spectrum_gain_side_requires_include_gain():
    """The energy-gain wing must come from detailed-balance reconstruction, not
    resolution leakage: with include_gain=True the negative-E side carries real
    (Boltzmann-suppressed) intensity, and turning include_gain OFF must collapse
    it. A path that ignored include_gain would fail here (it passed the old
    'gain < loss' check on leakage alone)."""
    base = dict(geometry="direct", e_fixed_meV=250.0, angles_deg=[60.0],
                dos_species=[_carbon()], e_min=-60.0)
    r_on = compute_spectrum(include_gain=True, **base, **BASE)
    r_off = compute_spectrum(include_gain=False, **base, **BASE)
    _assert_spectrum_invariants(r_on)
    assert r_on.E.min() < 0.0 < r_on.E.max()
    neg = r_on.E < -2.0                              # the energy-gain wing
    assert neg.any()
    gain_on = float(r_on.I_inelastic[neg].sum())
    gain_off = float(r_off.I_inelastic[neg].sum())
    assert gain_on > 0.0                             # gain side reconstructed
    assert gain_on > 5.0 * max(gain_off, 1.0e-30)    # and it needs include_gain
    # the gain side is Boltzmann-SUPPRESSED relative to the loss side
    loss = r_on.E > 2.0
    assert r_on.I_inelastic[loss].sum() > gain_on


def test_compute_spectrum_coherent_elastic_line():
    """elastic=True + a crystal builds Bragg peaks that add a real elastic
    channel; the total stays the sum of the two channels."""
    sp = _carbon(b_coh_fm=6.6460,
                 positions=[(0.0, 0.0, 0.0), (0.0, 0.0, 0.5),
                            (1.0 / 3, 2.0 / 3, 0.0), (2.0 / 3, 1.0 / 3, 0.5)])
    r = compute_spectrum(geometry="vision", dos_species=[sp], elastic=True,
                         elastic_kind="both",
                         dos_crystal=(2.464, 2.464, 6.711, 90.0, 90.0, 120.0),
                         **BASE)
    _assert_spectrum_invariants(r)
    assert r.metadata["elastic"] is True
    assert r.metadata["n_bragg_edges"] > 0
    assert r.I_elastic.max() > 0.0           # the elastic line is actually present


def test_compute_spectrum_incoherent_elastic_lattice_free():
    """elastic_kind='incoherent' needs no crystal: a lattice-free DW elastic line."""
    r = compute_spectrum(geometry="indirect", e_fixed_meV=3.5, angles_deg=[90.0],
                         dos_species=[_carbon(sigma_inc_b=1.2)], elastic=True,
                         elastic_kind="incoherent", **BASE)
    _assert_spectrum_invariants(r)
    assert r.metadata["elastic"] is True
    assert r.I_elastic.max() > 0.0


def test_compute_spectrum_responds_to_dos():
    """End-to-end, non-tautological: two species differing ONLY in their optical
    phonon-peak position must yield materially different spectra. An orchestrator
    that ignored its DOS input (or returned a canned/zero result) would pass the
    shape/finite asserts but fail here."""
    omega, rho_a = _dos(opt_meV=70.0)
    _, rho_b = _dos(opt_meV=120.0)
    spA = {"symbol": "C", "omega_ev": omega, "rho": rho_a, "awr": 11.898,
           "sigma_bound_b": 5.551}
    spB = {"symbol": "C", "omega_ev": omega, "rho": rho_b, "awr": 11.898,
           "sigma_bound_b": 5.551}
    rA = compute_spectrum(geometry="direct", e_fixed_meV=250.0, angles_deg=[90.0],
                          dos_species=[spA], **BASE)
    rB = compute_spectrum(geometry="direct", e_fixed_meV=250.0, angles_deg=[90.0],
                          dos_species=[spB], **BASE)
    assert rA.E.shape == rB.E.shape
    # the spectra must differ by more than float noise somewhere on the axis
    denom = np.maximum(np.abs(rA.I_inelastic), np.abs(rB.I_inelastic)).max()
    assert denom > 0.0
    assert np.max(np.abs(rA.I_inelastic - rB.I_inelastic)) / denom > 0.05


def test_compute_spectrum_constant_q_cuts():
    """q_cuts produce per-|Q| vertical slices with the right shape + finiteness."""
    r = compute_spectrum(geometry="vision", dos_species=[_carbon()],
                         q_cuts=[3.0, 6.0], cut_dq=0.5, **BASE)
    assert len(r.q_cut_values) == 2
    assert r.I_inelastic_per_q.shape == (2, r.E.size)
    assert np.all(np.isfinite(r.I_inelastic_per_q))
    assert r.I_inelastic_per_q.max() > 0.0
    # the per-Q total decomposition mirrors the per-angle one
    assert np.allclose(r.I_total_per_q, r.I_inelastic_per_q + r.I_elastic_per_q)


def test_compute_sqe_map_mode0():
    """Dense 2-D S(Q,E) map: correct grid shape, finite, non-negative, non-trivial."""
    m = compute_sqe_map(geometry="direct", e_fixed_meV=250.0,
                        dos_species=[_carbon()], q_min=0.5, q_max=10.0, dQ_map=0.25,
                        e_max=120.0, dE=2.0, **MAP_BASE)
    assert isinstance(m, SQEMap)
    assert m.S.shape == (m.Q.size, m.E.size)
    assert np.all(np.isfinite(m.S))
    assert np.all(m.S >= -1e-9)
    assert m.S.max() > 0.0
    assert m.metadata["broadened"] is True
    # S(Q,E) must genuinely vary along BOTH axes (catches a broadcast/transpose
    # bug that fills rows or columns with a single repeated profile)
    assert np.ptp(m.S, axis=0).max() > 0.0          # varies across Q at fixed E
    assert np.ptp(m.S, axis=1).max() > 0.0          # varies across E at fixed Q


def test_compute_sqe_map_kinematic_envelope():
    """An angle range + fixed energy attaches the accessible-Q envelope, aligned
    to the map's own energy axis."""
    m = compute_sqe_map(geometry="direct", e_fixed_meV=250.0,
                        angle_range_deg=(20.0, 120.0), dos_species=[_carbon()],
                        q_min=0.5, q_max=12.0, dQ_map=0.5, e_max=120.0, dE=2.0,
                        **MAP_BASE)
    assert m.envelope is not None
    env_E, q_lo, q_hi = m.envelope
    assert env_E.shape == m.E.shape == q_lo.shape == q_hi.shape
    # the high-angle edge reaches larger |Q| than the low-angle edge where defined
    fin = np.isfinite(q_lo) & np.isfinite(q_hi)
    assert np.all(q_hi[fin] >= q_lo[fin] - 1e-9)


def test_compute_sqe_map_kinematic_factor_weights_each_energy_column():
    """kinematic_factor=True multiplies the map by kf/ki(E) per column, as
    compute_spectrum does (direct geometry: sqrt((Ei - E)/Ei))."""
    common = dict(geometry="direct", e_fixed_meV=250.0, dos_species=[_carbon()],
                  q_min=0.5, q_max=10.0, dQ_map=0.5, e_max=120.0, dE=2.0,
                  broaden=False, **MAP_BASE)
    plain = compute_sqe_map(**common)
    weighted = compute_sqe_map(kinematic_factor=True, **common)
    assert weighted.metadata["kinematic_factor"] is True
    np.testing.assert_allclose(
        weighted.S, plain.S * np.sqrt((250.0 - plain.E) / 250.0)[None, :],
        rtol=1e-12, atol=0.0)


def test_compute_sqe_map_broadening_changes_the_map():
    """broaden=True must actually apply the resolution kernel: the raw and
    broadened maps of the SAME input must differ. A no-op that only flips the
    'broadened' metadata flag would fail here."""
    common = dict(geometry="vision", dos_species=[_carbon()], q_min=1.0, q_max=8.0,
                  dQ_map=0.5, e_max=100.0, dE=2.0, **MAP_BASE)
    m_raw = compute_sqe_map(broaden=False, **common)
    m_brd = compute_sqe_map(broaden=True, **common)
    assert m_raw.metadata["broadened"] is False
    assert m_brd.metadata["broadened"] is True
    denom = float(np.maximum(np.abs(m_raw.S), np.abs(m_brd.S)).max())
    assert denom > 0.0
    assert np.max(np.abs(m_raw.S - m_brd.S)) / denom > 1.0e-3   # kernel applied
