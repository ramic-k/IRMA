"""Unit tests for irma.core.extinction (the Sabine + Becker-Coppens models).

Pure/fast: no engine, no plugin. Cross-validation against the CrysXT NCrystal
plugin lives in a separate (plugin-gated) harness; here we check the recipe
limits, monotonicity and model dispatch.
"""
import math

import numpy as np
import pytest

from irma.core import extinction as ext
from irma.core.extinction import _sabine_secondary_factors


# ----- BC2025 'std' recipe limits & monotonicity -----
@pytest.mark.parametrize("kind", [0, 1, 2, 3])
def test_recipe_limits_and_monotonic(kind):
    def recipe_fn(x, sint):
        return ext.bc2025_y(kind, x, sint)
    # y -> 1 as x -> 0 (no extinction), for any Bragg angle
    for sint in (0.05, 0.5, 0.999):
        assert recipe_fn(1e-10, sint) == pytest.approx(1.0, abs=1e-7)
    # y decreases monotonically with extinction strength x, and stays in (0, 1]
    for sint in (0.1, 0.5, 0.95):
        ys = [recipe_fn(x, sint) for x in (0.01, 0.1, 1.0, 10.0, 100.0, 900.0)]
        assert all(0.0 < y <= 1.0 + 1e-9 for y in ys)
        assert all(a > b for a, b in zip(ys, ys[1:]))


def test_recipe_large_x_tail():
    # the x>1000 asymptotic branch is continuous and small
    for kind in (0, 1, 2, 3):
        y_at = ext.bc2025_y(kind, 1000.0, 0.5)
        y_above = ext.bc2025_y(kind, 5000.0, 0.5)
        assert 0.0 < y_above < y_at < 0.2


# ----- model dispatcher -----
_BE = dict(Nc=1.0 / 16.225, wl=3.0, F_hkl=math.sqrt(0.574) * 1e-4, d_hkl=1.79)
_PARAMS = dict(l=8550.0, g=170.0, L=75750.0)


@pytest.mark.parametrize("model", ext.EXTINCTION_MODELS)
def test_every_model_returns_physical_y(model):
    dist = "rect" if model.startswith("Sabine") else "Gauss"
    y = ext.extinction_factor(model, **_BE, **_PARAMS, dist=dist)
    assert 0.0 < y <= 1.0


# BC_mix/BC_mod require g>0,L>0 (secondary), so the all-zero "no extinction"
# limit only applies to the models that accept zero sizes.
@pytest.mark.parametrize("model", ["Sabine_uncorr", "Sabine_corr", "BC_pure"])
def test_no_extinction_when_sizes_zero(model):
    # l=g=L=0 -> no primary, no secondary -> y == 1 (no reduction)
    y = ext.extinction_factor(model, **_BE, l=0.0, g=0.0, L=0.0)
    assert y == pytest.approx(1.0, abs=1e-9)




def test_bc_pure_primary_is_single_parameter():
    # the one-knob case: BC_pure with only l set (g=L=0) -> pure primary
    y = ext.extinction_factor("BC_pure", **_BE, l=8550.0, g=0.0, L=0.0)
    assert 0.0 < y < 1.0
    # bigger crystallite -> stronger primary extinction -> smaller y
    y_big = ext.extinction_factor("BC_pure", **_BE, l=20000.0, g=0.0, L=0.0)
    assert y_big < y


def test_cls_and_std_agree_at_moderate_extinction():
    # the BC1974 'cls' closed form and BC2025 'std' agree where 'cls' is valid
    for model in ("BC_pure", "BC_mix", "BC_mod"):
        y_cls = ext.extinction_factor(model, **_BE, **_PARAMS, dist="Gauss", recipe="cls")
        y_std = ext.extinction_factor(model, **_BE, **_PARAMS, dist="Gauss", recipe="std")
        assert y_cls == pytest.approx(y_std, rel=0.05)






def test_below_bragg_threshold_is_unity():
    # a plane with 2d < wl does not diffract -> y defined as 1 (no contribution)
    y = ext.extinction_factor("BC_mix", Nc=1.0 / 16.225, wl=5.0,
                              F_hkl=math.sqrt(0.574) * 1e-4, d_hkl=1.79,
                              **_PARAMS, dist="Gauss")
    assert y == 1.0


# ---- numerical stability of the Sabine secondary factors ------------------

def test_sabine_triangular_small_x_window_is_stable():
    """The textbook 1-(1-exp(-2x))/(2x) loses all significant digits for x
    just above the 1e-9 branch threshold; the expm1 form must stay in [0, 1]
    through the whole window."""
    for x in np.geomspace(1e-12, 1e3, 4001):
        for tilt in (0, 1):
            el, eb = _sabine_secondary_factors(float(x), tilt)
            assert math.isfinite(el) and math.isfinite(eb)
            # within the 1e-6 rounding band extinction_factor clamps
            assert 0.0 <= el <= 1.0 + 1e-6, (x, tilt, el)
            assert 0.0 <= eb <= 1.0 + 1e-6, (x, tilt, eb)


def test_sabine_stable_forms_match_the_textbook_where_it_is_accurate():
    """E_L against the textbook form on x in [1e-2, 50]; the E_B series used
    below x = 1e-4 against 2/x^2 (x - log1p(x)) on [1e-5, 1e-4), where that
    form is still accurate to about 4e-11 and the series truncation error is
    below 1e-12."""
    for x in np.geomspace(1e-2, 50.0, 200):
        x = float(x)
        el, _ = _sabine_secondary_factors(x, 1)
        el_textbook = 1.0 / x * (1.0 - (1.0 - math.exp(-2.0 * x)) / (2.0 * x))
        assert el == pytest.approx(el_textbook, rel=1e-9)
    for x in np.geomspace(1e-5, 1e-4, 50, endpoint=False):
        x = float(x)
        _, eb = _sabine_secondary_factors(x, 1)
        eb_textbook = 2.0 / x / x * (x - math.log1p(x))
        assert eb == pytest.approx(eb_textbook, rel=1e-9)
