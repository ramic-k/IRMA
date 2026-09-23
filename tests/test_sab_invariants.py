"""Physics invariants of the classic continuous-spectrum S(alpha,beta) path.

No reference data needed — these pin properties any correct LEAPR-style
implementation must satisfy:

* ``ssm`` stores the asymmetric DOWNSCATTER law S(alpha,-beta) on the beta>0
  grid (LAT=1: grid values are at T0=0.0253 eV and scale by 0.0253/kT).
* Phonon-expansion sum rule: integrating the inelastic law over all energy
  transfers gives  integral S dbeta = 1 - exp(-f0*alpha)  (the Debye-Waller
  complement), since each expansion term T_n is normalized.
* tbar = T_eff/T >= 1 for a harmonic solid; the Debye-Waller lambda grows
  with temperature.
* Discrete oscillators (``discre``) increase both the Debye-Waller factor and
  the effective temperature.
* A translational component (``trans``) re-normalizes the law so the full
  integral approaches 1 (the formerly elastic weight becomes quasi-elastic).
"""
import numpy as np
import pytest

from irma.core.engine import contin, discre, trans

THERM = 0.0253
BK = 8.617333262e-5


def _debye_dos(ni=41, delta1=0.001):
    """Debye-like rho ~ eps^2 cutting off at 0.04 eV."""
    eps = np.arange(ni) * delta1
    p1 = eps**2
    p1[-1] = 0.0
    return p1, ni, delta1


def _grids():
    beta = np.concatenate(([0.0], np.geomspace(1e-3, 40.0, 400)))
    alpha = np.array([0.01, 0.05, 0.2, 1.0])
    return alpha, beta


def _run_contin(temperature_K, tbeta=1.0, nphon=30):
    p1, ni, delta1 = _debye_dos()
    alpha, beta = _grids()
    tev = BK * temperature_K
    ssm = np.zeros((len(beta), len(alpha)))
    f0, tbar, deltab = contin(ssm, alpha, beta, len(alpha), len(beta),
                              1, 1.0, tev, p1, ni, delta1, tbeta, nphon)
    return ssm, alpha, beta, tev, f0, tbar, deltab


@pytest.fixture(scope="module")
def law_300K():
    return _run_contin(300.0)


@pytest.fixture(scope="module")
def law_600K():
    return _run_contin(600.0)


def _downscatter_integral(ssm, beta, tev):
    """integral over ALL beta of the asymmetric law, from the stored
    downscatter side: S(+b)=S(-b)*exp(-b) by detailed balance."""
    b_act = beta * THERM / tev          # LAT=1 grid scaling
    return np.trapezoid(ssm * (1.0 + np.exp(-b_act))[:, None], b_act, axis=0)


def test_law_is_nonnegative_and_finite(law_300K):
    ssm = law_300K[0]
    assert np.all(np.isfinite(ssm))
    assert np.all(ssm >= 0.0)


def test_phonon_expansion_sum_rule(law_300K):
    """integral S dbeta = 1 - exp(-f0*alpha) — normalization of every T_n."""
    ssm, alpha, beta, tev, f0, _, _ = law_300K
    integrals = _downscatter_integral(ssm, beta, tev)
    a_act = alpha * THERM / tev
    expected = 1.0 - np.exp(-f0 * a_act)
    # Observed deviation on this grid is 7.1e-5; 3e-4 keeps ~4x headroom
    # while still failing on any real normalization regression (the old
    # 1e-2 tolerance was ~140x looser than reality).
    np.testing.assert_allclose(integrals, expected, rtol=3e-4)


def test_effective_temperature_at_least_T(law_300K, law_600K):
    assert law_300K[5] >= 1.0          # tbar = T_eff/T
    assert law_600K[5] >= 1.0
    # T_eff itself must also increase with T
    assert 600.0 * law_600K[5] > 300.0 * law_300K[5]


def test_debye_waller_lambda_grows_with_temperature(law_300K, law_600K):
    f0_300, f0_600 = law_300K[4], law_600K[4]
    assert f0_300 > 0.0
    assert f0_600 > f0_300


def test_small_alpha_limit_vanishes(law_300K):
    """S(alpha->0, beta>0) -> 0: weakest binding transfers no energy."""
    ssm = law_300K[0]
    assert ssm[1:, 0].max() < ssm[1:, -1].max()
    assert ssm[1:, 0].max() < 0.2


@pytest.mark.filterwarnings("ignore:divide by zero encountered in log")
def test_discrete_oscillators_raise_dw_and_teff(law_300K):
    """discre with an Einstein oscillator must raise both DW and T_eff
    (the oscillator adds zero-point motion and stiffens the spectrum).

    (The log-of-zero warnings come from discre's np.where-guarded
    interpolation on empty SAB regions — pre-existing, values are guarded.)"""
    ssm0, alpha, beta, tev, f0, tbar, deltab = _run_contin(300.0, tbeta=0.7)
    ssm = ssm0.copy()
    teff0 = tbar * 300.0
    bdel = np.array([0.2])             # 0.2 eV oscillator
    adel = np.array([0.3])             # weight 0.3 (tbeta=0.7)
    dw_new, teff_new = discre(ssm, alpha, beta, len(alpha), len(beta),
                              1, 1.0, tev, 0.0, 0.7, 1, bdel, adel,
                              f0, teff0, 300.0)
    assert dw_new > f0
    assert teff_new > teff0
    assert np.all(np.isfinite(ssm)) and np.all(ssm >= 0.0)


def test_translational_component_restores_full_normalization():
    """With a free-gas translational part (twt>0) the quasi-elastic component
    absorbs the elastic weight: the full integral approaches 1 at every alpha
    (vs 1-exp(-f0*alpha) for the solid-only law)."""
    ssm0, alpha, beta, tev, f0, tbar, deltab = _run_contin(300.0, tbeta=0.7)
    ssm = ssm0.copy()
    trans(ssm, alpha, beta, len(alpha), len(beta), 1, 1.0, tev,
          0.3, 0.0, 0.7, f0, deltab, tbar)
    integrals = _downscatter_integral(ssm, beta, tev)
    # quasi-elastic peak is hard to resolve at the smallest alpha; check the
    # moderate/large-alpha points where the grid resolves it.
    np.testing.assert_allclose(integrals[2:], 1.0, rtol=5e-2)
    # and the integral must exceed the solid-only sum rule everywhere
    solid_only = 1.0 - np.exp(-f0 * alpha * THERM / tev)
    assert np.all(integrals > solid_only - 1e-9)


@pytest.mark.filterwarnings("ignore:divide by zero encountered in log")
def test_discre_convolves_every_in_range_delta_line():
    """NJOY2016's idone reuse adds only the FIRST in-range negative delta
    line; IRMA deliberately convolves them all. With two Einstein oscillators
    (twt=0) the law must gain localized delta spikes at BOTH oscillator
    betas, not just the first."""
    # small Debye-like spectrum for the continuum part
    ne = 25
    p1 = np.linspace(0.0, 1.0, ne) ** 2
    delta1 = 0.002                              # eV
    alpha = np.array([0.1, 0.5, 1.0, 2.0])
    beta = np.linspace(0.0, 25.0, 126)
    T = 296.0
    tev = BK * T
    tbeta, twt = 0.5, 0.0
    ssm = np.zeros((len(beta), len(alpha)))
    f0, tbar, _ = contin(ssm, alpha, beta, len(alpha), len(beta),
                         1, 1.0, tev, p1, ne, delta1, tbeta, 30)
    bdel = np.array([0.137, 0.250])             # two oscillators, both in range
    adel = np.array([0.3, 0.2])
    ssm_d = ssm.copy()
    discre(ssm_d, alpha, beta, len(alpha), len(beta), 1, 1.0, tev,
           twt, tbeta, 2, bdel, adel, f0, tbar * T, T)
    assert np.all(np.isfinite(ssm_d)) and np.all(ssm_d >= 0.0)

    # each oscillator's delta line lands on the beta row nearest bdel/tev
    # (verified empirically: 0.137 eV -> beta=5.4, 0.250 eV -> beta=9.8)
    ial = 1                                     # alpha = 0.5
    row = ssm_d[:, ial]
    rows_hit = []
    for e_osc in bdel:
        jj = int(np.argmin(np.abs(e_osc / tev - beta)))
        rows_hit.append(jj)
        background = 0.5 * (row[jj - 1] + row[jj + 1])
        assert row[jj] > 3.0 * max(background, 1e-30), (
            f"no delta spike at oscillator {e_osc} eV (row {jj}): "
            f"{row[jj]} vs background {background}")
    assert rows_hit[0] != rows_hit[1], "oscillators must map to distinct rows"


@pytest.mark.parametrize("which", ["vec", "batch"])
def test_sct_tail_is_a_normalized_gaussian(which):
    """Beyond beta_max, sint returns the short-collision-time Gaussian in |x|
    (centre wt*alpha, variance 2*wt*alpha*tbar), which must integrate to 1.
    NJOY omits the square root of the normalization (leapr.f90:1892)."""
    from irma.core.kernels import _sint_batch_exact, sint_vec
    wt, alph, tbart = 1.0, 50.0, 1.0              # sigma = 10, centre 50
    bex = np.array([-1.0, 0.0, 1.0])
    sex = np.full(3, 1e-3)
    x = -np.linspace(1.5, 150.0, 20001)           # energy loss, all in the tail
    if which == "vec":
        s = sint_vec(x, bex, np.ones(3), sex, 3, alph, wt, tbart,
                     np.array([0.0, 1.0]), 2)
    else:
        s = _sint_batch_exact(x, bex, np.ones(3), sex, np.log(sex), 3, alph,
                              wt, tbart, 1.0)
    assert np.trapezoid(s, -x) == pytest.approx(1.0, abs=1e-4)
