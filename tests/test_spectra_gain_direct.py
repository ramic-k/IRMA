"""Direct energy-gain evaluation (explicit Bose factors) vs detailed balance.

physics.gain_side='direct' computes the E<0 side of the neutron-scattering
forward model with explicit occupation factors -- n(omega) phonon-annihilation
weights, the phonon orders summed on a signed energy grid
(irma.spectra.dos_mode0.compute_mode0_gain_direct) -- with NO detailed-balance
mirror anywhere in the evaluation. For the equilibrium harmonic model the two
are the same physics ((n+1)e^{-E/kT} = n identically), so their agreement --
~1e-6 comparing the kernel to the mirror (gain-grid interpolation), ~1e-9
end-to-end through the orchestrators -- is BOTH the regression contract for the
new path and a standing cross-validation of the mirror: two independent
implementations (the contin ladder + mirror vs the FFT ladder with explicit
Bose factors) that must coincide.

CI-safe: mode-0 DOS path only, pure numpy, no phonopy/engine.
"""
import numpy as np
import pytest

from irma.spectra import sqe as _sqe
from irma.spectra.dos_mode0 import (compute_mode0_gain_direct, compute_mode0_sqe,
                                    GainGridTooLargeError)
from irma.spectra.forward import compute_spectrum, compute_sqe_map

T_K = 296.0
KT = _sqe.KB * T_K


def _dos():
    omega = np.arange(400) * 0.0005
    w = omega * 1000.0
    rho = np.where(w <= 40.0, (w / 40.0) ** 2, 0.0)
    rho += 0.5 * np.exp(-0.5 * ((w - 90.0) / 5.0) ** 2)
    return omega, rho


def _carbon(**extra):
    omega, rho = _dos()
    sp = {"symbol": "C", "omega_ev": omega, "rho": rho, "awr": 11.898,
          "sigma_bound_b": 5.551, "sigma_inc_b": 0.001, "multiplicity": 1}
    sp.update(extra)
    return sp


BASE = dict(phonopy_yaml=None, temperature_k=T_K, mesh=None,
            sab_mass_ratio=11.898, sab_sigma_barn=5.551,
            multiphonon_max_order=60, auto_multiphonon_order=False,
            progress=lambda *a, **k: None)


def test_direct_kernel_matches_mirror_quantitative():
    """The independently-computed gain side equals the detailed-balance mirror
    of the contin loss side to ~1e-9 -- two implementations, one physics."""
    sp = _carbon()
    E = np.arange(0.0, 120.75, 1.5)
    Q = np.array([2.0, 7.0, 17.0])
    m = compute_mode0_sqe(species=[sp], temperature_k=T_K, q_ang_inv=Q,
                          e_mev=E, nphon=100)
    S_loss = np.asarray(m["sqe_barn_per_meV"], float)
    mirror = S_loss[:, 1:] * np.exp(-E[1:] / KT)
    g = compute_mode0_gain_direct(species=[sp], temperature_k=T_K,
                                  q_ang_inv=Q, e_gain_mev=-E[1:][::-1])
    direct = np.asarray(g["sqe_barn_per_meV"], float)[:, ::-1]
    msk = mirror > 1e-12
    assert msk.sum() > 100
    assert np.max(np.abs(direct[msk] / mirror[msk] - 1.0)) < 1e-6
    # the direct kernel's Debye-Waller integral matches contin's f0
    lam_direct = g["per_species"][0]["dw_lambda"]
    lam_contin = m["per_species"][0]["dw_lambda"]
    assert lam_direct == pytest.approx(lam_contin, rel=1e-5)


def test_direct_kernel_truncates_to_explicit_order():
    """An explicit nphon caps the gain ladder at that order (so the gain side is
    never MORE complete than a finite-order loss side), and the truncation is a
    clean monotone order sum that converges to the closed form:
      * direct@1 == the mirror of the one-phonon loss (contin's order-1 is a
        clean one-phonon term), validating the order-1 truncation;
      * direct@N rises monotonically with N toward direct@auto;
      * by a converged order direct@N == direct@auto to round-off.
    (contin's own explicit-nphon truncation is NOT a clean order sum -- its
    loss is identical for nphon 2..5 -- so this pins the FFT ladder's order
    semantics, not bitwise agreement with contin at low order.)"""
    sp = _carbon()
    E = np.arange(0.0, 120.75, 1.5)
    Q = np.array([17.0])                      # high Q -> ladder needs many orders
    sums = {}
    for N in (1, 2, 3, 5, 10, 40):
        g = compute_mode0_gain_direct(species=[sp], temperature_k=T_K,
                                      q_ang_inv=Q, e_gain_mev=-E[1:][::-1], nphon=N)
        sums[N] = float(np.asarray(g["sqe_barn_per_meV"], float).sum())
    g_auto = compute_mode0_gain_direct(species=[sp], temperature_k=T_K,
                                       q_ang_inv=Q, e_gain_mev=-E[1:][::-1],
                                       nphon="auto")
    s_auto = float(np.asarray(g_auto["sqe_barn_per_meV"], float).sum())
    # monotone increasing in N, strictly below the all-order sum at low N
    for a, b in zip((1, 2, 3, 5, 10), (2, 3, 5, 10, 40)):
        assert sums[a] < sums[b] * (1.0 + 1e-12)
    assert sums[1] < 0.9 * s_auto              # order-1 is materially incomplete
    assert sums[40] == pytest.approx(s_auto, rel=1e-6)   # converged

    # order-1 gain == mirror of contin's one-phonon loss
    m1 = compute_mode0_sqe(species=[sp], temperature_k=T_K, q_ang_inv=Q,
                           e_mev=E, nphon=1)
    mirror1 = np.asarray(m1["sqe_barn_per_meV"], float)[:, 1:] * np.exp(-E[1:] / KT)
    g1 = compute_mode0_gain_direct(species=[sp], temperature_k=T_K, q_ang_inv=Q,
                                   e_gain_mev=-E[1:][::-1], nphon=1)
    direct1 = np.asarray(g1["sqe_barn_per_meV"], float)[:, ::-1]
    msk = mirror1 > 1e-10 * mirror1.max()
    assert np.max(np.abs(direct1[msk] / mirror1[msk] - 1.0)) < 1e-5


def test_direct_kernel_grid_cap_raises():
    """An extreme light-mass / very-high-Q / very-low-T input whose converged
    ladder overflows the FFT budget raises GainGridTooLargeError (the regime
    where the gain side is negligible and the caller mirrors instead)."""
    omega, rho = _dos()
    h = {"symbol": "H", "omega_ev": omega, "rho": rho, "awr": 0.9991,
         "sigma_bound_b": 80.27}
    with pytest.raises(GainGridTooLargeError):
        compute_mode0_gain_direct(species=[h], temperature_k=5.0,
                                  q_ang_inv=np.array([100.0]),
                                  e_gain_mev=-np.arange(1.0, 60.0, 1.5)[::-1],
                                  nphon="auto", max_nfft=1 << 20)


def test_orchestrator_attaches_direct_gain_not_mirror(monkeypatch):
    """Pin that the forward path actually ATTACHES the direct gain (vs silently
    mirroring while reporting gain_side_used='direct'): a sentinel kernel
    returning 2x the physical gain must make the spectrum's gain side exactly
    2x the detailed-balance mirror."""
    import irma.spectra.dos_mode0 as _m0
    real = _m0.compute_mode0_gain_direct

    def doubled(**kw):
        g = real(**kw)
        g["sqe_barn_per_meV"] = 2.0 * g["sqe_barn_per_meV"]
        return g

    # unbroadened map: the E<0 side is purely the deposited gain (no resolution
    # blend with the loss side), so the doubled sentinel must show up as an
    # exact 2x vs the detailed-balance mirror at every gain bin.
    common = dict(geometry="direct", e_fixed_meV=250.0, dos_species=[_carbon()],
                  q_min=1.0, q_max=8.0, dQ_map=0.25, e_min=-40.0, e_max=100.0,
                  dE=1.5, broaden=False)
    m_db = compute_sqe_map(gain_side="detailed_balance", **common, **BASE)
    monkeypatch.setattr(_m0, "compute_mode0_gain_direct", doubled)
    m_dir = compute_sqe_map(gain_side="direct", **common, **BASE)
    # exclude the bin adjacent to E=0: the interpolator blends it with the
    # (undoubled) zero column, so only the deep gain side is a clean 2x test
    gn = m_dir.E < -2.0
    db = np.nan_to_num(m_db.S[:, gn])
    di = np.nan_to_num(m_dir.S[:, gn])
    msk = db > 1e-10 * db.max()
    assert np.allclose(di[msk] / db[msk], 2.0, rtol=1e-9)   # attached, not mirrored
    # the loss side (E>0) is unaffected by the gain sentinel
    lo = m_dir.E > 1e-9
    assert np.allclose(np.nan_to_num(m_dir.S[:, lo]),
                       np.nan_to_num(m_db.S[:, lo]), rtol=1e-12, atol=0.0)


def test_orchestrator_falls_back_to_mirror_on_grid_cap(monkeypatch):
    """When the direct kernel raises GainGridTooLargeError, the forward path
    must downgrade gain_side_used to 'detailed_balance' (with a NOTE) and the
    map must equal the mirror -- never a silent inelastic-only or aliased gain
    side while metadata still claims 'direct'."""
    def boom(**kw):
        raise GainGridTooLargeError("forced for test")

    import irma.spectra.dos_mode0 as _m0
    monkeypatch.setattr(_m0, "compute_mode0_gain_direct", boom)
    notes = []
    common = dict(geometry="direct", e_fixed_meV=250.0, dos_species=[_carbon()],
                  q_min=1.0, q_max=8.0, dQ_map=0.25, e_min=-40.0, e_max=100.0,
                  dE=1.5, broaden=False)
    m = compute_sqe_map(gain_side="direct", **common,
                        **{k: v for k, v in BASE.items() if k != "progress"},
                        progress=notes.append)
    m_db = compute_sqe_map(gain_side="detailed_balance", **common, **BASE)
    assert m.metadata["gain_side"] == "direct"
    assert m.metadata["gain_side_used"] == "detailed_balance"     # downgraded
    assert any("falls back to the detailed-balance mirror" in str(n) for n in notes)
    assert np.allclose(np.nan_to_num(m.S), np.nan_to_num(m_db.S),
                       rtol=1e-12, atol=0.0)


def test_spectrum_gain_side_direct_vs_mirror_agree():
    """End-to-end 1-D: the two gain_side settings agree to round-off on BOTH
    sides (the gain values themselves and the resolution leakage into loss)."""
    common = dict(geometry="direct", e_fixed_meV=250.0, angles_deg=[60.0],
                  dos_species=[_carbon()], e_min=-60.0, e_max=120.0,
                  dE=1.0, dQ=0.1)
    r_dir = compute_spectrum(gain_side="direct", **common, **BASE)
    r_db = compute_spectrum(gain_side="detailed_balance", **common, **BASE)
    assert r_dir.metadata["gain_side"] == "direct"
    assert r_dir.metadata["gain_side_used"] == "direct"
    assert r_db.metadata["gain_side_used"] == "detailed_balance"
    neg = r_dir.E < -2.0
    assert r_db.I_inelastic[neg].max() > 0.0
    assert np.allclose(r_dir.I_inelastic[neg], r_db.I_inelastic[neg],
                       rtol=1e-9, atol=1e-20)
    assert np.allclose(r_dir.I_inelastic, r_db.I_inelastic,
                       rtol=1e-9, atol=1e-16)


def test_map_gain_side_direct_vs_mirror_agree():
    common = dict(geometry="direct", e_fixed_meV=250.0, dos_species=[_carbon()],
                  q_min=1.0, q_max=8.0, dQ_map=0.25, e_min=-40.0, e_max=100.0,
                  dE=1.5, broaden=False)
    m_dir = compute_sqe_map(gain_side="direct", **common, **BASE)
    m_db = compute_sqe_map(gain_side="detailed_balance", **common, **BASE)
    assert m_dir.metadata["gain_side_used"] == "direct"
    gn = m_dir.E < -2.0
    assert np.nanmax(m_db.S[:, gn]) > 0.0
    assert np.allclose(np.nan_to_num(m_dir.S), np.nan_to_num(m_db.S),
                       rtol=1e-9, atol=1e-16)


def test_config_validates_gain_side(tmp_path):
    from irma.spectra.config import SpectraConfig, SpectraConfigError, validate
    rows = "".join(f"{w} {((w / 40.0) ** 2 if w <= 40.0 else 0.0)}\n"
                   for w in range(0, 121))
    dos = tmp_path / "c.dos"
    dos.write_text("# freq_meV dos\n" + rows)
    d = {"material": {"temperature_K": T_K,
                      "scatterers": [{"symbol": "C", "dos_file": str(dos),
                                      "dos_unit": "meV", "awr": 11.898,
                                      "sigma_bound_b": 5.551}]},
         "physics": {"inelastic_mode": 0, "elastic": False,
                     "max_phonon_order": 40},
         "grid": {"e_min_meV": -20.0, "e_max_meV": 60.0, "de_meV": 1.5,
                  "dq_max_invA": 0.1},
         "instrument": {"geometry": "direct", "e_fixed_meV": 250.0,
                        "angles_deg": [30.0, 120.0]}}
    cfg = SpectraConfig.from_dict(d)
    assert cfg.physics.gain_side == "direct"                    # the default
    validate(cfg)
    d["physics"]["gain_side"] = "mirror-ish"
    with pytest.raises(SpectraConfigError):
        validate(SpectraConfig.from_dict(d))


def test_direct_gain_low_temperature_sane():
    """5 K: the gain side must collapse toward zero (almost no thermal phonons
    to annihilate) and stay finite/non-negative under the FFT ladder."""
    sp = _carbon()
    E = np.arange(0.0, 60.75, 1.5)
    Q = np.array([3.0, 8.0])
    g = compute_mode0_gain_direct(species=[sp], temperature_k=5.0,
                                  q_ang_inv=Q, e_gain_mev=-E[1:][::-1])
    Sg = np.asarray(g["sqe_barn_per_meV"], float)
    assert np.all(np.isfinite(Sg)) and np.all(Sg >= 0.0)
    m = compute_mode0_sqe(species=[sp], temperature_k=5.0, q_ang_inv=Q,
                          e_mev=E, nphon=60)
    S_loss = np.asarray(m["sqe_barn_per_meV"], float)
    # strongly Boltzmann-suppressed: the integral is dominated by the first
    # gain bin at -1.5 meV (beta ~ 3.5 at 5 K -> e^-beta ~ 0.03); everything
    # beyond a few meV is gone
    assert Sg.sum() < 1e-3 * S_loss.sum()
    deep = np.abs(np.asarray(g["e_gain_mev"])) > 10.0
    assert Sg[:, deep].max() < 1e-9 * S_loss.max()
    # and the direct evaluation still equals the mirror at 5 K where the
    # values are numerically meaningful (deep tails fall below FFT precision)
    mirror = (S_loss[:, 1:] * np.exp(-E[1:] / (_sqe.KB * 5.0)))[:, ::-1]
    msk = mirror > 1e-12 * mirror.max()
    assert np.allclose(Sg[msk], mirror[msk], rtol=1e-4)
