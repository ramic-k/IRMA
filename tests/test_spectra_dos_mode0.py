"""DOS-based mode-0 S(Q,E) producer (irma.spectra.dos_mode0).

Pure-physics checks, no engine/phonopy. Pins the per-species incoherent-approx
phonon expansion + the validated asym_downscatter SAB convention: a single
species reproduces a direct contin()+_law_to_sqe build; multi-species is an
exact cross-section/multiplicity-weighted sum; one-phonon follows the DOS.
"""
import numpy as np
import pytest

from irma.core.constants import BK
from irma.core.kernels import contin
from irma.spectra import sqe as si
from irma.spectra.dos_mode0 import compute_mode0_sqe


def _dos(w_max_meV=40.0, opt_meV=100.0, np1=320, delta1=0.0005):
    """A Debye acoustic band + a Gaussian optical peak, on a uniform omega grid (eV)."""
    omega = np.arange(np1) * delta1
    w = omega * 1000.0
    rho = np.where(w <= w_max_meV, (w / w_max_meV) ** 2, 0.0)
    rho += 0.6 * np.exp(-0.5 * ((w - opt_meV) / 6.0) ** 2)
    return omega, rho


T_K = 300.0
Q = np.linspace(0.5, 12.0, 80)
E = np.linspace(0.0, 150.0, 200)


def test_single_species_matches_direct_kernel_build():
    """One species == a hand-rolled contin() + _law_to_sqe."""
    omega, rho = _dos()
    awr, sig = 0.9991673, 80.27
    out = compute_mode0_sqe(species=[{"symbol": "H", "omega_ev": omega, "rho": rho,
                                      "awr": awr, "sigma_bound_b": sig}],
                            temperature_k=T_K, q_ang_inv=Q, e_mev=E, nphon=100)
    # reference: drive the kernel directly, same convention
    kT = si.KB * T_K
    beta = E / kT
    alpha = si.C_E * Q ** 2 / (awr * kT)
    ssm = np.zeros((E.size, Q.size))
    contin(ssm, alpha, beta, Q.size, E.size, 0, 1.0, BK * T_K, rho.copy(),
           omega.size, float(omega[1] - omega[0]), 1.0, 100)
    ref = si._law_to_sqe(ssm, T_K, sig)
    assert np.allclose(out["sqe_barn_per_meV"], ref, rtol=1e-12)
    assert out["sigma_b_total"] == pytest.approx(sig)
    assert out["per_species"][0]["dw_lambda"] > 0


def test_multispecies_is_atom_weighted_average_of_singles():
    """PER-ATOM: total S(Q,E) is the atom-weighted average of the per-species
    single runs, (sum_d mult_d S_d)/N -- NOT a plain sum (that was the old
    per-cell convention). A single-species run is already that species' per-atom
    spectrum, so tot = (mult_H sH + mult_X sX)/(mult_H+mult_X)."""
    oH, rH = _dos(w_max_meV=40, opt_meV=100)
    oX, rX = _dos(w_max_meV=30, opt_meV=70)
    H = {"symbol": "H", "omega_ev": oH, "rho": rH, "awr": 0.999, "sigma_bound_b": 80.0, "multiplicity": 2}
    X = {"symbol": "C", "omega_ev": oX, "rho": rX, "awr": 11.9, "sigma_bound_b": 5.55, "multiplicity": 1}
    tot = compute_mode0_sqe(species=[H, X], temperature_k=T_K, q_ang_inv=Q, e_mev=E)
    sH = compute_mode0_sqe(species=[H], temperature_k=T_K, q_ang_inv=Q, e_mev=E)
    sX = compute_mode0_sqe(species=[X], temperature_k=T_K, q_ang_inv=Q, e_mev=E)
    assert np.allclose(tot["sqe_barn_per_meV"],
                       (2 * sH["sqe_barn_per_meV"] + 1 * sX["sqe_barn_per_meV"]) / 3.0,
                       rtol=1e-12)
    assert tot["sigma_b_total"] == pytest.approx((2 * 80.0 + 5.55) / 3.0)


def test_single_species_spectrum_is_multiplicity_independent():
    """PER-ATOM: one species' spectrum does not depend on its multiplicity
    (the per-cell sum and the /N atom count cancel)."""
    omega, rho = _dos()
    base = {"symbol": "H", "omega_ev": omega, "rho": rho, "awr": 0.999, "sigma_bound_b": 80.0}
    one = compute_mode0_sqe(species=[dict(base, multiplicity=1)], temperature_k=T_K, q_ang_inv=Q, e_mev=E)
    three = compute_mode0_sqe(species=[dict(base, multiplicity=3)], temperature_k=T_K, q_ang_inv=Q, e_mev=E)
    assert np.allclose(three["sqe_barn_per_meV"], one["sqe_barn_per_meV"], rtol=1e-12)


def test_one_phonon_peaks_at_the_dos():
    """nphon=1 -> S(Q,E) peaks exactly at the input DOS frequencies (40, 100)."""
    omega, rho = _dos(w_max_meV=40, opt_meV=100)
    out = compute_mode0_sqe(species=[{"symbol": "H", "omega_ev": omega, "rho": rho,
                                      "awr": 0.999, "sigma_bound_b": 80.0}],
                            temperature_k=T_K, q_ang_inv=Q, e_mev=E, nphon=1)
    S = out["sqe_barn_per_meV"]
    sE = S[np.argmin(np.abs(Q - 5.0))]
    peaks = [E[i] for i in range(1, E.size - 1)
             if sE[i] > sE[i - 1] and sE[i] > sE[i + 1] and sE[i] > 0.05 * sE.max()]
    assert any(abs(p - 40) < 3 for p in peaks) and any(abs(p - 100) < 3 for p in peaks)
    assert np.all(np.isfinite(S)) and S.max() > 0


def test_builds_a_finite_powdersqe_through_the_forward_model():
    """The mode-0 S(Q,E) wraps as a PowderSQE and projects to a finite INS spectrum."""
    from irma.spectra import instruments as ins
    omega, rho = _dos()
    out = compute_mode0_sqe(species=[{"symbol": "H", "omega_ev": omega, "rho": rho,
                                      "awr": 0.999, "sigma_bound_b": 80.0}],
                            temperature_k=T_K, q_ang_inv=Q, e_mev=E)
    p = si.from_noncubic_arrays(out["q_ang_inv"], out["e_mev"],
                                out["sqe_barn_per_meV"], T_K=T_K,
                                sigma_b=out["sigma_b_total"])
    sim = ins.simulate(p, ins.VISION(), np.linspace(0, 150, 300),
                       elastic_model=None, per_angle=False)
    assert np.all(np.isfinite(sim["I_inelastic"])) and sim["I_inelastic"].max() > 0


@pytest.mark.parametrize("bad", [
    {"omega_ev": [0.0], "rho": [0.0]},                                  # < 2 points
    {"omega_ev": [0.0, 0.001, 0.003], "rho": [0.0, 1.0, 1.0]},          # non-uniform
    {"omega_ev": [0.0, 0.001], "rho": [0.0, -1.0]},                     # negative rho
    {"omega_ev": [0.0, 0.001], "rho": [0.0, 0.0]},                      # all-zero
])
def test_bad_dos_grids_rejected(bad):
    sp = {"symbol": "H", "awr": 1.0, "sigma_bound_b": 80.0, **bad}
    with pytest.raises(ValueError):
        compute_mode0_sqe(species=[sp], temperature_k=T_K, q_ang_inv=Q, e_mev=E)


def test_empty_species_and_bad_temperature_rejected():
    with pytest.raises(ValueError):
        compute_mode0_sqe(species=[], temperature_k=T_K, q_ang_inv=Q, e_mev=E)
    omega, rho = _dos()
    with pytest.raises(ValueError):
        compute_mode0_sqe(species=[{"symbol": "H", "omega_ev": omega, "rho": rho,
                                    "awr": 1.0, "sigma_bound_b": 80.0}],
                          temperature_k=0.0, q_ang_inv=Q, e_mev=E)


def test_neutron_weighted_gdos_is_sigma_over_mass_weighted():
    """The bonus GDOS weights each AREA-NORMALIZED partial by m_d*sigma_d/M_d.

    QA4 F10: input DOS files carry arbitrary intensity units (the S(Q,E)
    kernel renormalizes internally), so each partial must be normalized to
    unit area before the neutron weighting — otherwise the files' relative
    scales would silently skew the GDOS."""
    oH, rH = _dos(w_max_meV=40, opt_meV=100)
    oX, rX = _dos(w_max_meV=30, opt_meV=70)
    H = {"symbol": "H", "omega_ev": oH, "rho": rH, "awr": 1.0, "sigma_bound_b": 80.0, "multiplicity": 1}
    # 1000x the intensity scale of X's DOS file: must NOT change the GDOS
    X = {"symbol": "C", "omega_ev": oX, "rho": 1000.0 * rX, "awr": 12.0, "sigma_bound_b": 5.5, "multiplicity": 1}
    out = compute_mode0_sqe(species=[H, X], temperature_k=T_K, q_ang_inv=Q, e_mev=E)
    wH, wX = 80.0 / 1.0, 5.5 / 12.0

    def _unit_area(o, r):
        r2 = np.asarray(r, float).copy()
        r2[0] = 0.0                       # the kernel zeroes the omega=0 point
        g = np.interp(E, o * 1000.0, r2, left=0, right=0)
        return g / np.trapezoid(r2, o * 1000.0)

    expect = (wH * _unit_area(oH, rH) + wX * _unit_area(oX, rX)) / (wH + wX)
    assert np.allclose(out["gdos"], expect, rtol=1e-12)


def test_mode0_rejects_translational_tbeta_below_one():
    """mode-0 is solid-state only: a tbeta<1 (diffusive remainder) must be
    rejected loudly, not silently dropped (#32)."""
    omega, rho = _dos()
    with pytest.raises(ValueError, match="solid-state only"):
        compute_mode0_sqe(
            species=[{"symbol": "C", "omega_ev": omega, "rho": rho,
                      "awr": 11.9, "sigma_bound_b": 5.55}],
            temperature_k=300.0, q_ang_inv=np.array([1.0, 2.0]),
            e_mev=np.array([0.0, 10.0]), tbeta=0.6)
