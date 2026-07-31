"""Unit tests for irma.core.extinction (the Sabine + Becker-Coppens models).

Pure/fast: no engine, no plugin. Cross-validation against the CrysXT NCrystal
plugin lives in a separate (plugin-gated) harness; here we check the recipe
limits, monotonicity, model dispatch and guards.
"""
import math

import pytest

from irma.core import extinction as ext


# ----- BC2025 'std' recipe limits & monotonicity -----
@pytest.mark.parametrize("recipe_fn", [
    ext.bc2025_y_primary, ext.bc2025_y_scndgauss,
    ext.bc2025_y_scndlorentz, ext.bc2025_y_scndfresnel])
def test_recipe_limits_and_monotonic(recipe_fn):
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
    for fn in (ext.bc2025_y_primary, ext.bc2025_y_scndgauss,
               ext.bc2025_y_scndlorentz, ext.bc2025_y_scndfresnel):
        y_at = fn(1000.0, 0.5)
        y_above = fn(5000.0, 0.5)
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


@pytest.mark.parametrize("model", ["BC_mix", "BC_mod"])
def test_bc_secondary_models_require_l_g_and_L(model):
    # all three knobs are required: the coupled secondary term is parameterised by
    # the crystallite size l as well as the mosaic g and grain L, so a missing knob
    # would silently collapse the model to y=1 (a no-op stamped as "corrected").
    with pytest.raises(ValueError, match=r"requires l>0, g>0 and L>0"):
        ext.extinction_factor(model, **_BE, l=8550.0, g=0.0, L=0.0)
    with pytest.raises(ValueError, match=r"requires l>0, g>0 and L>0"):
        ext.extinction_factor(model, **_BE, l=0.0, g=170.0, L=75750.0)


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


# ----- guards -----
def test_lux_recipe_is_gated_with_explanation():
    with pytest.raises(NotImplementedError, match="tabulation tolerance"):
        ext.extinction_factor("BC_mix", **_BE, **_PARAMS, dist="Gauss", recipe="lux")


def test_unknown_model_and_recipe_raise():
    with pytest.raises(ValueError, match="unknown extinction model"):
        ext.extinction_factor("Zachariasen", **_BE, **_PARAMS)
    with pytest.raises(ValueError, match="unknown recipe"):
        ext.extinction_factor("BC_mix", **_BE, **_PARAMS, dist="Gauss", recipe="bogus")


def test_below_bragg_threshold_is_unity():
    # a plane with 2d < wl does not diffract -> y defined as 1 (no contribution)
    y = ext.extinction_factor("BC_mix", Nc=1.0 / 16.225, wl=5.0,
                              F_hkl=math.sqrt(0.574) * 1e-4, d_hkl=1.79,
                              **_PARAMS, dist="Gauss")
    assert y == 1.0


# ---- review PH-2: numerical stability of the Sabine secondary factors -----

def test_sabine_triangular_small_x_window_is_stable():
    """The naive 1-(1-exp(-2x))/(2x) lost all significant digits for x just
    above the 1e-9 branch threshold (factors as wrong as -26 at the review's
    reproducer point). The stable expm1 form must stay physical through the
    whole window."""
    import numpy as np
    from irma.core.extinction import _sabine_secondary_factors
    for x in np.geomspace(1e-12, 1e3, 4001):
        for tilt in (0, 1):
            el, eb = _sabine_secondary_factors(float(x), 0.3, tilt)
            assert math.isfinite(el) and math.isfinite(eb)
            assert 0.0 <= el <= 1.0, (x, tilt, el)
            assert 0.0 <= eb <= 1.0 + 1e-12, (x, tilt, eb)


def test_sabine_stable_form_matches_naive_at_moderate_x():
    """Where the textbook expressions are well-conditioned, the stable
    rewrites must agree with them to rounding."""
    import numpy as np
    from irma.core.extinction import _sabine_secondary_factors, _calc_AB_sabine
    y = 0.3
    a, b = _calc_AB_sabine(y)
    for x in np.geomspace(1e-2, 50.0, 200):
        x = float(x)
        el, eb = _sabine_secondary_factors(x, y, 1)
        el_naive = math.exp(-y) / x * (1.0 - (1.0 - math.exp(-2.0 * x))
                                       / (2.0 * x))
        bx = b * x
        eb_naive = 2.0 * a / bx / x * (bx - math.log1p(bx))
        assert el == pytest.approx(el_naive, rel=1e-9)
        assert eb == pytest.approx(eb_naive, rel=1e-9)


def test_sabine_review_reproducer_point_is_physical():
    """The exact PH-2 reproducer: Nc=0.1, F=1e-6 A, d=10 A, l=1000 A, g=1,
    L=1e4 A, triangular. The dense energy sweep used to yield hundreds of
    negative factors (worst -26)."""
    import numpy as np
    for E in np.geomspace(0.005, 0.5, 2001):
        wl = math.sqrt(8.180425e-2 / E)  # WL2EKIN
        if 0.5 * wl / 10.0 > 1.0:
            continue
        y = ext.extinction_factor("Sabine_uncorr", Nc=0.1, wl=wl, F_hkl=1e-6,
                                  d_hkl=10.0, l=1000.0, g=1.0, L=1e4,
                                  dist="tri")
        assert math.isfinite(y)
        assert 0.0 <= y <= 1.0, (E, y)
