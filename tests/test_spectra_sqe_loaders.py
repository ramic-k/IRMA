"""Public irma.spectra.sqe loaders / helpers (previously zero CI coverage).

``sqe_interpolator``, ``from_oclimax`` and ``instrument_spectrum`` are exported
but were exercised only through the skip-gated forward gate. Tiny hand-built
inputs pin the interpolation contract, the OCLIMAX npz format + units convention,
and the single-trajectory projection.
"""
import numpy as np
import pytest

from irma.spectra import sqe as si


def test_sqe_interpolator_interpolates_and_zeros_out_of_range():
    q = np.array([1.0, 2.0, 3.0])
    E = np.array([-10.0, 0.0, 10.0])            # signed energy axis
    S = np.array([[1.0, 2.0, 3.0],
                  [4.0, 5.0, 6.0],
                  [7.0, 8.0, 9.0]])             # (nq, nE)
    interp = si.sqe_interpolator(q, E, S)
    assert interp([[2.0, 0.0]])[0] == pytest.approx(5.0)          # exact grid node
    assert interp([[1.5, 5.0]])[0] == pytest.approx((2 + 3 + 5 + 6) / 4)  # bilinear
    assert interp([[10.0, 0.0]])[0] == 0.0       # |Q| out of range -> fill 0
    assert interp([[2.0, 100.0]])[0] == 0.0      # E out of range -> fill 0


def test_from_oclimax_format_and_units(tmp_path):
    q = np.array([1.0, 2.0, 4.0])
    beta = np.array([0.5, 1.0, 2.0])            # |beta| downscatter, > 0
    nbeta, nq = beta.size, q.size
    asy = np.arange(1.0, nbeta * nq + 1.0).reshape(nbeta, nq)  # (nbeta, nq)
    path = tmp_path / "ocl.npz"
    np.savez(path, q_centers_ang_inv=q, beta_downscatter_abs=beta,
             ssm_internal_beta_alpha=asy)
    T, sigma_b = 300.0, 5.0
    p = si.from_oclimax(str(path), sigma_b, T, label="ocl")
    assert p.q.shape == (nq,) and p.E.shape == (nbeta,)
    assert p.S.shape == (nq, nbeta)              # PowderSQE convention (nq, nE)
    kT = si.KB * T
    assert np.allclose(p.E, beta * kT)           # E = beta * kT
    # OCLIMAX ssm is ALREADY asym_downscatter -> S = sigma_b/(4 pi kT) * asy (no exp)
    assert np.allclose(p.S, (sigma_b / (4 * np.pi * kT)) * asy.T)
    assert p.T_K == T and p.sigma_b == sigma_b and p.label == "ocl"


def test_instrument_spectrum_shapes_and_elastic_channel():
    q = np.linspace(0.5, 8.0, 30)
    E = np.linspace(0.0, 100.0, 60)
    S = (q ** 2)[:, None] * np.exp(-0.5 * ((E - 30.0) / 8.0) ** 2)[None, :] + 1e-6
    p = si.from_noncubic_arrays(q, E, S, T_K=300.0, sigma_b=1.0)
    E_out = np.linspace(0.0, 90.0, 91)
    def Q_of_E(Etr):
        return si.Q_indirect(Etr, 3.5, 90.0)

    r = si.instrument_spectrum(p, Q_of_E, E_out, sigma_coeffs=[1.0, 0.0, 0.0])
    for k in ("E", "Q", "I_inelastic", "I_elastic", "I_total"):
        assert np.asarray(r[k]).shape == E_out.shape
    assert np.all(np.isfinite(r["I_total"]))
    assert np.allclose(r["I_total"], r["I_inelastic"] + r["I_elastic"])
    assert np.all(r["I_elastic"] == 0.0) and r["I_inelastic"].max() > 0.0

    r2 = si.instrument_spectrum(p, Q_of_E, E_out, sigma_coeffs=[1.0, 0.0, 0.0],
                                elastic_area=5.0)
    assert r2["I_elastic"].max() > 0.0           # elastic line now contributes
    assert np.allclose(r2["I_total"], r2["I_inelastic"] + r2["I_elastic"])
