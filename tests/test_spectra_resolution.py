"""Resolution line-shape backend gate (Gaussian + Lorentzian).

Pure-math checks on ``irma.spectra.sqe.resolution_convolve`` / ``elastic_line``
-- no external validation data needed. The Gaussian path
is the OCLIMAX-equivalent (OCLIMAX applies only a Gaussian resolution function);
the Lorentzian is the extra heavier-tailed option. These pin the normalization,
the heavier Lorentzian tails and the elastic-line window.
"""
import numpy as np
import pytest

from irma.spectra import sqe as si


# ---- normalization / convolution --------------------------------------------
def _trapz(y, x):
    return float(np.trapezoid(y, x))


@pytest.mark.parametrize("shape", ["gaussian", "lorentzian"])
def test_flat_input_preserved(shape):
    """A constant input convolves to itself (normalized kernel) in the interior."""
    E = np.linspace(-200.0, 200.0, 801)
    I_in = np.ones_like(E)
    out = si.resolution_convolve(E, I_in, (2.0, 0.0, 0.0), shape=shape)
    interior = np.abs(E) < 120.0
    assert np.allclose(out[interior], 1.0, atol=2e-2)


@pytest.mark.parametrize("shape", ["gaussian", "lorentzian"])
@pytest.mark.parametrize("coeffs", [(2.0, 0.0, 0.0),      # constant width
                                    (0.5, 0.2, 0.0),      # strongly varying width
                                    (0.31, 0.005, 8.1e-7)])  # VISION
def test_resolution_convolve_conserves_flux(shape, coeffs):
    """The column-normalized kernel conserves flux: integral(out) == integral(in)
    for any input AND for an energy-DEPENDENT width (a flux leak there was the
    HIGH finding). Row-normalizing would instead leak up to several % under a
    varying width."""
    E = np.linspace(0.0, 300.0, 3001)
    I_in = np.exp(-0.5 * ((E - 150.0) / 20.0) ** 2)       # localized interior peak
    out = si.resolution_convolve(E, I_in, coeffs, shape=shape)
    assert _trapz(out, E) == pytest.approx(_trapz(I_in, E), rel=1e-9)


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


# ---- elastic_line: renormalize only when the peak is inside the window -------
@pytest.mark.parametrize("e_min,n,keeps_full_area", [
    (0.5, 2001, False),   # window excludes E=0: only the analytic tail (~5%)
    (-5.0, 4001, True),   # window includes the peak: the area is conserved
    (0.0, 4001, True),    # axis starts at the peak centre: the half line
                          # renormalizes to the full area
])
def test_elastic_line_window(e_min, n, keeps_full_area):
    area = 10.0
    E = np.linspace(e_min, 100.0, n)
    line = si.elastic_line(E, area, [0.31])     # VISION-like sigma (meV)
    integral = float(np.trapezoid(line, E))
    if keeps_full_area:
        assert integral == pytest.approx(area, rel=1e-6)
    else:
        assert integral < 0.2 * area
