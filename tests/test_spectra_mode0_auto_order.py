"""Mode-0 honors ``auto_multiphonon_order``.

``max_phonon_order='auto'`` -- the production default through config, CLI and
GUI -- must size the DOS-path phonon order from the physics, never resolve
to a FIXED nphon=100: a fixed order means ~10x wasted
ladder time for thermal cases (the contin ladder cost grows ~quadratically
with the order) and silent truncation for high-alpha cases (H at high Q)
where 100 is too FEW. Now 'auto' sizes the ladder to the converged
Poisson(f0*alpha_max) order -- the exact rule modes 1/2 use
(``derive_required_multiphonon_order``) with the isotropic
``U_eq = f0*C_E/(awr*kT)`` standing in for the largest thermal-displacement
eigenvalue, since ``Q^2 U_eq == lambda*alpha``.

These tests pin the EFFECTIVE ORDER (result fields / forward metadata), the
derivation math against a hand-rolled ``start()`` + Poisson-tail bound, and
the convergence claim (auto == fixed-100 to ~1e-6) -- not just runtime.
"""
import math

import numpy as np
import pytest

from irma.core.constants import BK
from irma.core.kernels import start
from irma.spectra.dos_mode0 import compute_mode0_sqe, derive_mode0_phonon_order
from irma.spectra.forward import compute_spectrum, compute_sqe_map
from irma.spectra.sqe import C_E, KB

T_K = 296.0


def _dos(w_max_meV=40.0, opt_meV=90.0, n=400, delta_ev=0.0005):
    """A Debye acoustic band + a Gaussian optical peak on a uniform omega grid (eV)."""
    omega = np.arange(n) * delta_ev
    w = omega * 1000.0
    rho = np.where(w <= w_max_meV, (w / w_max_meV) ** 2, 0.0)
    rho += 0.5 * np.exp(-0.5 * ((w - opt_meV) / 5.0) ** 2)
    return omega, rho


def _species(symbol="C", awr=11.898, sigma=5.551, **extra):
    omega, rho = _dos()
    sp = {"symbol": symbol, "omega_ev": omega, "rho": rho, "awr": awr,
          "sigma_bound_b": sigma}
    sp.update(extra)
    return sp


def _expected_order(species_list, T, q_max, margin=6.0, cap=2000):
    """Hand-rolled reference for the auto rule: the worst species'
    ``lam = f0 * C_E*Q_max^2/(awr*kT)`` Poisson mean, captured to the
    ``margin``-sigma tail: ``n = ceil(lam + margin*sqrt(lam)) + 2``,
    clamped to ``[2, cap]``. Returns ``(effective, required)``."""
    lam_max = 0.0
    for sp in species_list:
        omega = np.asarray(sp["omega_ev"], float)
        rho = np.asarray(sp["rho"], float).copy()
        rho[0] = 0.0
        _, f0, _, _ = start(rho, omega.size, float(omega[1] - omega[0]),
                            BK * T, 1.0)
        lam_max = max(lam_max,
                      f0 * C_E * q_max ** 2 / (float(sp["awr"]) * KB * T))
    req = max(2, int(math.ceil(lam_max + margin * math.sqrt(lam_max))) + 2)
    return min(req, cap), req


def test_derived_order_matches_poisson_rule():
    """The derivation is exactly the engine rule with U_eq = f0*C_E/(awr*kT)."""
    sp = _species()
    eff, req, lam = derive_mode0_phonon_order(
        species=[sp], temperature_k=T_K, q_max_ang_inv=12.0)
    exp_eff, exp_req = _expected_order([sp], T_K, 12.0)
    assert (eff, req) == (exp_eff, exp_req)
    assert lam == pytest.approx(
        C_E * 12.0 ** 2 / (11.898 * KB * T_K) *
        start(np.where(np.arange(400) == 0, 0.0, sp["rho"]), 400,
              0.0005, BK * T_K, 1.0)[1], rel=1e-12)
    # thermal carbon-like case: far below a fixed order of 100
    assert 2 <= eff < 100


def test_worst_species_governs_multi_species_order():
    """H (awr=1) needs a much higher order than C at the same Q; the
    multi-species derivation must size to the WORST species."""
    c, h = _species(), _species(symbol="H", awr=1.008, sigma=82.0)
    eff_c, _, _ = derive_mode0_phonon_order(
        species=[c], temperature_k=T_K, q_max_ang_inv=12.0)
    eff_both, _, _ = derive_mode0_phonon_order(
        species=[c, h], temperature_k=T_K, q_max_ang_inv=12.0)
    exp_eff, _ = _expected_order([c, h], T_K, 12.0)
    assert eff_both == exp_eff
    assert eff_both > eff_c


def test_compute_mode0_sqe_auto_shrinks_and_stays_converged():
    """nphon='auto' runs the derived order and reproduces the fixed-100
    S(Q,E) to well within convergence noise (<< 0.1% on the integral)."""
    sp = _species()
    Q = np.linspace(0.5, 12.0, 60)
    E = np.linspace(0.0, 150.0, 150)
    auto = compute_mode0_sqe(species=[sp], temperature_k=T_K,
                             q_ang_inv=Q, e_mev=E, nphon="auto")
    ref = compute_mode0_sqe(species=[sp], temperature_k=T_K,
                            q_ang_inv=Q, e_mev=E, nphon=100)
    exp_eff, exp_req = _expected_order([sp], T_K, float(Q.max()))
    assert auto["nphon_effective"] == exp_eff
    assert auto["nphon_required"] == exp_req
    assert 2 <= auto["nphon_effective"] < 100
    assert ref["nphon_effective"] == 100          # explicit int honored verbatim
    Sa, Sr = auto["sqe_barn_per_meV"], ref["sqe_barn_per_meV"]
    Ia = np.trapezoid(np.trapezoid(Sa, E, axis=1), Q)
    Ir = np.trapezoid(np.trapezoid(Sr, E, axis=1), Q)
    assert abs(Ia - Ir) / Ir < 1e-6
    assert np.max(np.abs(Sa - Sr)) < 1e-6 * Sr.max()


def test_hydrogen_auto_order_full_output_is_converged():
    """The risky case auto-sizing exists for: H at direct-geometry Q derives
    an order above 100. Derivation alone does not prove the
    derived order suffices -- pin the FULL S(Q,E) against a deliberately
    higher explicit order (+40): the integrals must agree to convergence
    noise, demonstrating the 6-sigma+2 margin rule converges hydrogen too."""
    h = _species(symbol="H", awr=1.008, sigma=82.0)
    Q = np.linspace(0.5, 30.0, 45)
    E = np.linspace(0.0, 400.0, 120)
    auto = compute_mode0_sqe(species=[h], temperature_k=T_K,
                             q_ang_inv=Q, e_mev=E, nphon="auto")
    assert auto["nphon_effective"] > 100          # beyond a fixed order of 100
    ref = compute_mode0_sqe(species=[h], temperature_k=T_K,
                            q_ang_inv=Q, e_mev=E,
                            nphon=int(auto["nphon_effective"]) + 40)
    Sa, Sr = auto["sqe_barn_per_meV"], ref["sqe_barn_per_meV"]
    Ia = np.trapezoid(np.trapezoid(Sa, E, axis=1), Q)
    Ir = np.trapezoid(np.trapezoid(Sr, E, axis=1), Q)
    assert abs(Ia - Ir) / Ir < 1e-3
    assert np.max(np.abs(Sa - Sr)) < 1e-3 * Sr.max()


# placeholders required by the orchestrators; mode-0 ignores them
_FWD = dict(phonopy_yaml=None, mesh=None, sab_mass_ratio=11.898,
            sab_sigma_barn=5.551, temperature_k=T_K)


def test_compute_spectrum_mode0_honors_auto_order():
    """The forward orchestrator derives the order when auto (metadata pins the
    EFFECTIVE order), and that order is genuinely what the ladder ran: an
    explicit run at the same order is bit-identical, and the fixed-100 run
    matches to convergence noise."""
    msgs = []
    common = dict(geometry="vision", dos_species=[_species()],
                  e_max=120.0, dE=1.0, dQ=0.1, **_FWD)
    r_auto = compute_spectrum(multiphonon_max_order=100,
                              auto_multiphonon_order=True,
                              progress=msgs.append, **common)
    eff = r_auto.metadata["effective_multiphonon_order"]
    assert 2 <= eff < 100                          # derived, not the fixed 100
    assert any("(auto-sized" in m for m in msgs)
    r_fix = compute_spectrum(multiphonon_max_order=eff,
                             auto_multiphonon_order=False,
                             progress=lambda *a, **k: None, **common)
    assert r_fix.metadata["effective_multiphonon_order"] == eff
    assert np.array_equal(r_auto.I_inelastic, r_fix.I_inelastic)
    r_100 = compute_spectrum(multiphonon_max_order=100,
                             auto_multiphonon_order=False,
                             progress=lambda *a, **k: None, **common)
    assert r_100.metadata["effective_multiphonon_order"] == 100
    assert np.max(np.abs(r_auto.I_inelastic - r_100.I_inelastic)) \
        < 1e-6 * r_100.I_inelastic.max()


def test_compute_sqe_map_mode0_honors_auto_order():
    """Same contract for the 2-D map orchestrator (it had no auto either)."""
    common = dict(geometry="direct", e_fixed_meV=250.0,
                  dos_species=[_species()], q_min=0.5, q_max=10.0, dQ_map=0.5,
                  e_max=120.0, dE=2.0, broaden=False,
                  progress=lambda *a, **k: None, **_FWD)
    m_auto = compute_sqe_map(multiphonon_max_order=100,
                             auto_multiphonon_order=True, **common)
    eff = m_auto.metadata["effective_multiphonon_order"]
    assert 2 <= eff < 100
    m_fix = compute_sqe_map(multiphonon_max_order=eff,
                            auto_multiphonon_order=False, **common)
    assert m_fix.metadata["effective_multiphonon_order"] == eff
    assert np.array_equal(m_auto.S, m_fix.S)
