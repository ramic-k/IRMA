"""Resolution line-shape backend gate (Gaussian + Lorentzian).

Pure-math checks on ``irma.spectra.sqe.resolution_convolve`` / ``elastic_line``
-- no external validation data needed. The Gaussian path
is the OCLIMAX-equivalent (OCLIMAX applies only a Gaussian resolution function);
the Lorentzian is the extra heavier-tailed option. These pin the normalization,
the heavier Lorentzian tails and the independence from the energy window.
"""
import numpy as np
import pytest

from irma.spectra import sqe as si


# ---- normalization / convolution --------------------------------------------
def _trapz(y, x):
    return float(np.trapezoid(y, x))


@pytest.mark.parametrize("shape,tol", [("gaussian", 1e-6), ("lorentzian", 1e-3)])
@pytest.mark.parametrize("step,width", [(0.5, 2.0),     # fine grid
                                        (1.0, 0.31)])   # step 3x the width
def test_flat_input_preserved_up_to_the_window_edges(shape, tol, step, width):
    """A constant input sampled on the padded grid convolves to itself on the
    whole window, edges included: the pad reaches 6 sigma for a Gaussian and
    leaves at most 1e-3 of a Lorentzian's area outside. On the coarse grid the
    sampled Gaussian alone adds up to 1.34; the grid-sum scaling makes it 1."""
    E = np.arange(-200.0, 200.0 + 0.5 * step, step)
    E_in = si.padded_grid(E, (width, 0.0, 0.0), shape=shape)
    out = si.resolution_convolve(E, np.ones_like(E_in), (width, 0.0, 0.0),
                                 shape=shape, E_in=E_in)
    assert np.allclose(out, 1.0, atol=tol)


@pytest.mark.parametrize("shape", ["gaussian", "lorentzian"])
@pytest.mark.parametrize("coeffs", [(2.0, 0.0, 0.0),      # constant width
                                    (0.5, 0.05, 0.0),     # varying width
                                    (0.31, 0.005, 8.1e-7)])  # VISION
def test_padded_convolution_does_not_depend_on_the_window(shape, coeffs):
    """The convolved values on [0, 150] equal those of the same input
    convolved on the wider [-50, 200]: the result is the restriction of the
    full convolution, not renormalized to the window."""
    def f(E):
        return (np.exp(-0.5 * ((E - 80.0) / 10.0) ** 2)
                + np.exp(-0.5 * ((E - 2.0) / 3.0) ** 2))  # a peak at the edge
    wide = np.linspace(-50.0, 200.0, 1001)
    win = wide[(wide >= 0.0) & (wide <= 150.0)]
    out = {}
    for E in (wide, win):
        E_in = si.padded_grid(E, coeffs, shape=shape)
        out[E.size] = si.resolution_convolve(E, f(E_in), coeffs, shape=shape, E_in=E_in)
    keep = (wide >= 0.0) & (wide <= 150.0)
    tol = 1e-8 if shape == "gaussian" else 2e-3
    assert np.allclose(out[win.size], out[wide.size][keep], rtol=0.0, atol=tol)


@pytest.mark.parametrize("shape", ["gaussian", "lorentzian"])
def test_instrument_spectrum_does_not_depend_on_the_window(shape):
    """The 1-D spectrum samples S past the window edges: its values on
    [0, 90] are those of the [-5, 90] spectrum, near E=0 included."""
    q = np.linspace(0.5, 8.0, 30)
    E = np.linspace(0.0, 100.0, 201)
    bands = (np.exp(-0.5 * ((E - 30.0) / 8.0) ** 2)
             + np.exp(-0.5 * ((E - 3.0) / 1.0) ** 2))
    p = si.from_noncubic_arrays(q, E, (q ** 2)[:, None] * bands[None, :],
                                T_K=300.0, sigma_b=1.0)
    def Q_of_E(Etr):
        return si.Q_indirect(Etr, 3.5, 45.0)
    wide = np.arange(-5.0, 90.0 + 1e-9, 0.25)
    keep = wide >= 0.0
    a = si.instrument_spectrum(p, Q_of_E, wide, si.VISION_SIGMA_COEFFS,
                               elastic_area=2.0, shape=shape)
    b = si.instrument_spectrum(p, Q_of_E, wide[keep], si.VISION_SIGMA_COEFFS,
                               elastic_area=2.0, shape=shape)
    assert np.allclose(b["I_total"], a["I_total"][keep], rtol=1e-9, atol=0.0)


def test_resolution_response_is_unbiased_under_varying_width():
    """A true delta at E0 must come back CENTRED on E0 with std = w(E0), even for
    a steeply energy-dependent width. (Evaluating the width on the OUTPUT energy
    instead pulls the centroid toward where the kernel is wider.)"""
    E = np.linspace(0.0, 300.0, 3001)
    E0 = 150.0
    coeffs = (0.5, 0.2, 0.0)                               # w(150) = 30.5 meV, steep
    delta = np.exp(-0.5 * ((E - E0) / 0.2) ** 2)          # ~delta at E0
    out = si.resolution_convolve(E, delta, coeffs)
    area = _trapz(out, E)
    centroid = _trapz(E * out, E) / area
    std = np.sqrt(_trapz((E - centroid) ** 2 * out, E) / area)
    assert centroid == pytest.approx(E0, abs=0.05)        # unbiased (was ~162.9)
    assert std == pytest.approx(0.5 + 0.2 * E0, rel=0.02)  # width at the TRUE energy


def test_kf_ki_indirect_finite_at_forbidden_boundary():
    """Indirect kf/ki = sqrt(Ef/(Ef+Etr)): at/below Etr=-Ef the incident energy
    is non-positive (forbidden energy gain), so the factor must be finite (0) --
    never inf/nan that would poison the spectrum -- matching kf_ki_direct."""
    Ef = 3.5
    r = si.kf_ki_indirect(Ef)
    val = r(np.array([-Ef - 1.0, -Ef, 0.0, 2.0]))
    assert np.all(np.isfinite(val)) and np.all(val >= 0.0)
    assert val[0] == 0.0 and val[1] == 0.0           # forbidden region -> 0
    assert val[2] == pytest.approx(1.0)              # Etr=0 -> kf/ki = 1


def test_lorentzian_has_heavier_tails_than_gaussian():
    """At equal width, the Lorentzian puts more weight far from a single spike."""
    E = np.linspace(-200.0, 200.0, 4001)
    I_in = np.zeros_like(E)
    I_in[np.argmin(np.abs(E))] = 1.0 / (E[1] - E[0])  # unit-area spike at 0
    g = si.resolution_convolve(E, I_in, (3.0, 0.0, 0.0), shape="gaussian")
    lo = si.resolution_convolve(E, I_in, (3.0, 0.0, 0.0), shape="lorentzian")
    far = np.argmin(np.abs(E - 40.0))   # ~13 widths out
    assert lo[far] > 10.0 * g[far]
    # both are peaked at zero
    assert g.argmax() == lo.argmax() == np.argmin(np.abs(E))


# ---- elastic line -----------------------------------------------------------
@pytest.mark.parametrize("shape", ["gaussian", "lorentzian"])
def test_elastic_line_integrates_to_area(shape):
    E = np.linspace(-300.0, 300.0, 6001)
    area = 7.5
    line = si.elastic_line(E, area, (2.0, 0.0, 0.0), shape=shape)
    rel = 1e-3 if shape == "gaussian" else 3e-2
    assert _trapz(line, E) == pytest.approx(area, rel=rel)
    assert line.argmax() == np.argmin(np.abs(E))      # peak at E=0


# ---- shape errors -------------------------------------------------------------
def test_unknown_shape_rejected():
    E = np.linspace(-10.0, 10.0, 21)
    I_in = np.ones_like(E)
    with pytest.raises(ValueError):
        si.resolution_convolve(E, I_in, (1.0, 0, 0), shape="voigt")


# ---- elastic_line: the line restricted to the window ---------------------------
@pytest.mark.parametrize("e_min,n,visible", [
    (0.5, 2001, None),    # window excludes E=0: only the analytic tail
    (-5.0, 4001, 1.0),    # window includes the peak: the full area
    (0.0, 4001, 0.5),     # axis starts at the peak centre: the half line
    (0.0, 51, 0.5),       # the same on a 2 meV grid, coarse against sigma
])
def test_elastic_line_window(e_min, n, visible):
    area = 10.0
    E = np.linspace(e_min, 100.0, n)
    line = si.elastic_line(E, area, [0.31])     # VISION-like sigma (meV)
    integral = float(np.trapezoid(line, E))
    if visible is None:
        assert integral < 0.2 * area
    else:
        assert integral == pytest.approx(visible * area, rel=1e-6)
    if e_min == 0.0 and n > 51:                 # the peak value is not doubled
        assert line[0] == pytest.approx(area / (np.sqrt(2 * np.pi) * 0.31))
