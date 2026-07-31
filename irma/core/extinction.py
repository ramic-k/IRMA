"""Crystalline extinction factors for the coherent-elastic Bragg cross section.

Ported (NOT imported) from the NCrystal plugin ``ncplugin-CrysXT``
(https://github.com/dddijulio/ncplugin-CrysXT), which combines the extinction
models of ``ncplugin-CrysExtn`` (S. Xu) with a texture model. This module ports
ONLY the extinction half -- the Sabine (uncorrelated / correlated block) and
Becker-Coppens (pure / mixed / modified) models. Texture is deliberately omitted:
it is a sample-orientation effect with no place in an orientation-averaged
ENDF/TSL evaluation.

The extinction factor ``y(x, theta)`` in (0, 1] multiplies each Bragg plane's
kinematic intensity to account for primary (multiple scattering within one
crystallite) and secondary (crystallite-to-crystallite beam depletion) dynamical
diffraction. It always reduces the coherent-elastic cross section and raises
transmission. Extinction is SAMPLE-SPECIFIC -- the crystallite size ``l``, mosaic
spread ``g`` and grain size ``L`` are properties of the sample, not the material,
and must be supplied by the user.

References (please read these to understand the physics):
  - T. Kittelmann, D. D. DiJulio, S. Xu & J. I. Marquez Damian, "Revisiting
    Becker-Coppens (1974): updated recipes for estimating extinction factors in
    spherical crystallites", Acta Cryst. (2026) A82, 163-178.
    https://doi.org/10.1107/S2053273326001245  -- the BC2025 recipes used here.
  - S. Xu et al., "Impact of extinction effects on neutron transmission in solid
    beryllium metal", J. Appl. Cryst. (2025) 58, 1957-1966.
    https://doi.org/10.1107/S1600576725007939  -- concept & motivation.
  - P. J. Becker & P. Coppens, Acta Cryst. (1974) A30, 129.
  - T. M. Sabine, International Tables for Crystallography (2006), Vol. C, ch. 6.4.

Pure module: ``math`` only, no scipy/numpy required (stays import-safe for the
core). All lengths in angstrom, angles via sin(theta) = lambda / (2 d).
"""
from __future__ import annotations
import math

_INV_PI = 1.0 / math.pi

EXTINCTION_MODELS = ("Sabine_uncorr", "Sabine_corr", "BC_pure", "BC_mix", "BC_mod")
RECIPES = ("cls", "std", "lux")
# tilt/mosaic distribution keyword -> internal code, per model family
_SABINE_DIST = {"rect": 0, "tri": 1}
_BC_DIST = {"Gauss": 1, "Lorentz": 2, "Fresnel": 3}


# --------------------------------------------------------------------------- #
#  BC2025 'std' recipes  (ported from ncplugin-CrysXT/src/NCbc2025.hh)         #
#  y(x, sintheta) = y0(x) + sintheta*sqrt(sintheta)*ydelta(x*sqrt(sintheta))   #
#  Precision guarantee for x<1000: error < 1e-3 * min(y, 1-y).                 #
# --------------------------------------------------------------------------- #
def _nest(coeffs, ops, v):
    """Evaluate  c0 (op0) v*(c1 (op1) v*(c2 ...))  with ops in {+1,-1}."""
    acc = coeffs[-1]
    for i in range(len(coeffs) - 2, -1, -1):
        acc = coeffs[i] + ops[i] * v * acc
    return acc


_M5 = [-1] * 5
_M3 = [-1] * 3


def bc2025_y_primary(x, sintheta):
    """BC2025 'std' primary-extinction factor y(x, sintheta).

    Chebyshev-nested fit to the Becker-Coppens primary integral, with the
    asymptotic sqrt(1e3/x) continuation above x = 1e3.
    """
    if x < 0.1:
        y0 = _nest([1.0, 0.94285714, 0.8204, 0.593, 0.364, 0.19], _M5, x)
    else:
        if x > 1e3:
            return bc2025_y_primary(1e3, sintheta) * math.sqrt(1e3 / x)
        xp = (math.sqrt(x) - 1.0) / (math.sqrt(x) + 1.0)
        y0 = _nest([0.518212, 0.93036, 0.182006, 1.10097, 0.62625, 1.73562,
                    1.08506, 2.19459, 1.40451, 1.62083, 1.1031, 0.49125, 0.357611],
                   [-1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1], xp)
    s = math.sqrt(sintheta)
    u = x * s
    if u < 0.1:
        ydelta = u * u * _nest([0.41021645, 1.187, 2.37, 4.18], _M3, u)
    else:
        up = (math.sqrt(u) - 1.0) / (math.sqrt(u) + 1.0)
        ydelta = _nest([0.05508, 0.1166, 0.2099, 0.5482, 0.5248, 1.402, 1.168,
                        2.096, 2.116, 1.155, 1.952, 0.6046],
                       [1, -1, 1, -1, 1, -1, 1, -1, 1, -1, -1], up)
    return y0 + sintheta * s * ydelta


def bc2025_y_scndgauss(x, sintheta):
    """BC2025 'std' secondary-extinction factor for a Gaussian tilt
    distribution, with the x^-0.933 continuation above x = 1e3."""
    if x < 0.1:
        y0 = _nest([1.0, 1.0606602, 0.9238, 0.667, 0.409, 0.22], _M5, x)
    else:
        if x > 1e3:
            return bc2025_y_scndgauss(1e3, sintheta) * (x * 1e-3) ** (-0.933)
        xp = (math.sqrt(x) - 1.0) / (math.sqrt(x) + 1.0)
        y0 = _nest([0.4588909, 1.038687, 0.2401003, 1.288282, 0.7641972, 1.880246,
                    1.886916, 2.171852, 3.273034, 0.9771599, 2.988445, 0.4993548,
                    1.037121, 0.4353142],
                   [-1, -1, 1, -1, 1, -1, 1, -1, 1, -1, -1, 1, -1], xp)
    s = math.sqrt(sintheta)
    u = x * s
    if u < 0.1:
        ydelta = u * u * _nest([0.46188022, 1.333, 2.66, 4.68], _M3, u)
    else:
        up = (math.sqrt(u) - 1.0) / (math.sqrt(u) + 1.0)
        ydelta = _nest([0.062289443, 0.13177896, 0.240705, 0.61857545, 0.61744404,
                        1.4812474, 1.5419561, 1.9976424, 2.8090858, 0.74297172,
                        2.3120683, 0.8661981],
                       [1, -1, 1, -1, 1, -1, 1, -1, 1, -1, -1], up)
    return y0 + sintheta * s * ydelta


def bc2025_y_scndlorentz(x, sintheta):
    """BC2025 'std' secondary-extinction factor for a Lorentzian tilt
    distribution, with the sqrt(1e3/x) continuation above x = 1e3."""
    if x < 0.1:
        y0 = _nest([1.0, 1.0, 1.0667, 0.988, 0.79, 0.55], _M5, x)
    else:
        if x > 1e3:
            return bc2025_y_scndlorentz(1e3, sintheta) * math.sqrt(1e3 / x)
        xp = (math.sqrt(x) - 1.0) / (math.sqrt(x) + 1.0)
        y0 = _nest([0.53379, 0.84182, 0.16806, 0.65124, 0.67623, 0.47199, 1.0872,
                    0.030142, 0.91361, 0.28313, 0.30078, 0.1507],
                   [-1, -1, 1, -1, 1, -1, 1, -1, -1, 1, -1], xp)
    s = math.sqrt(sintheta)
    u = x * s
    if u < 0.1:
        ydelta = u * u * _nest([0.53333333, 1.9753, 5.14, 11.9], _M3, u)
    else:
        up = (math.sqrt(u) - 1.0) / (math.sqrt(u) + 1.0)
        ydelta = _nest([0.0514714, 0.0863117, 0.191581, 0.266342, 0.504516,
                        0.32195, 0.894662, 0.0162501, 0.708855, 0.33707],
                       [1, -1, 1, -1, 1, -1, -1, 1, -1], up)
    return y0 + sintheta * s * ydelta


def bc2025_y_scndfresnel(x, sintheta):
    """BC2025 'std' secondary-extinction factor for a Fresnel tilt
    distribution."""
    if x < 0.1:
        y0 = _nest([1.0, 1.0, 0.88, 0.639, 0.394, 0.21], _M5, x)
    else:
        if x > 1e3:
            return bc2025_y_scndfresnel(1e3, sintheta) * math.sqrt(1e3 / x)
        xp = (math.sqrt(x) - 1.0) / (math.sqrt(x) + 1.0)
        y0 = _nest([0.493354, 0.963692, 0.235067, 1.18222, 0.672931, 1.78522,
                    1.09976, 2.10882, 1.34721, 1.46841, 1.0054, 0.426279, 0.313436],
                   [-1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1], xp)
    s = math.sqrt(sintheta)
    u = x * s
    if u < 0.1:
        ydelta = u * u * _nest([0.44, 1.278, 2.56, 4.52], _M3, u)
    else:
        up = (math.sqrt(u) - 1.0) / (math.sqrt(u) + 1.0)
        ydelta = _nest([0.05839, 0.12063, 0.233343, 0.578753, 0.584531, 1.42753,
                        1.28278, 1.95436, 2.18561, 0.877761, 1.80505, 0.599956],
                       [1, -1, 1, -1, 1, -1, 1, -1, 1, -1, -1], up)
    return y0 + sintheta * s * ydelta


_BC2025_SECONDARY = {1: bc2025_y_scndgauss, 2: bc2025_y_scndlorentz,
                     3: bc2025_y_scndfresnel}


# --------------------------------------------------------------------------- #
#  BC1974 'cls' closed forms  (theta-dependent A,B; numerically fragile at     #
#  strong extinction -- prefer 'std')                                          #
# --------------------------------------------------------------------------- #
def calc_AB_theta(cos_2theta, opt):
    """A(theta), B(theta); opt 0=primary, 1/2/3=secondary Gauss/Lorentz/Fresnel."""
    if opt == 0:
        return (0.20 + 0.45 * cos_2theta, 0.22 - 0.12 * (0.5 - cos_2theta) ** 2)
    if opt == 1:
        return (0.58 + 0.48 * cos_2theta + 0.24 * cos_2theta ** 2,
                0.02 - 0.025 * cos_2theta)
    if opt == 2:
        a = 0.025 + 0.285 * cos_2theta
        b = (0.15 - 0.2 * (0.75 - cos_2theta) ** 2) if cos_2theta >= 0.0 \
            else (-0.45 * cos_2theta)
        return (a, b)
    return (0.48 + 0.6 * cos_2theta, 0.20 - 0.06 * (0.2 - cos_2theta) ** 2)


def _y_bc1974_primary(x, cos_2theta):
    """Becker-Coppens (1974) analytic primary-extinction factor."""
    a, b = calc_AB_theta(cos_2theta, 0)
    return 1.0 / math.sqrt(1.0 + 2.0 * x + a * x * x / (1.0 + b * x))


def _y_bc1974_secondary(x, cos_2theta, tilt_dist, force_212=False):
    """Becker-Coppens (1974) analytic secondary-extinction factor."""
    # The linear-term prefactor on x differs by Becker-Coppens variant in CrysXT,
    # which we reproduce verbatim (NCPhysicsModel.cc): the PURE secondary type-II
    # path uses 2.12 unconditionally (line 337), whereas the MIXED and MODIFIED
    # paths apply 2.12 only for the Gaussian tilt and 2.0 otherwise (lines 414,
    # 485 -- "the factor 2.12 is only applied in the case of Gaussian"). force_212
    # selects the pure-path (unconditional) behaviour.
    a, b = calc_AB_theta(cos_2theta, tilt_dist)
    t = 2.12 if (force_212 or tilt_dist == 1) else 2.0
    return 1.0 / math.sqrt(1.0 + t * x + a * x * x / (1.0 + b * x))


def _y_secondary(x, sintheta, cos_2theta, tilt_dist, recipe, force_212=False):
    """Secondary-extinction factor for the selected recipe ('cls'/'std')."""
    if recipe == "cls":
        return _y_bc1974_secondary(x, cos_2theta, tilt_dist, force_212)
    return _BC2025_SECONDARY[tilt_dist](x, sintheta)   # 'std' (lux gated upstream)


def _y_primary(x, sintheta, cos_2theta, recipe):
    """Primary-extinction factor for the selected recipe ('cls'/'std')."""
    if recipe == "cls":
        return _y_bc1974_primary(x, cos_2theta)
    return bc2025_y_primary(x, sintheta)               # 'std'


# --------------------------------------------------------------------------- #
#  Sabine's model  (analytic; Int. Tables Vol. C, ch. 6.4)                     #
#  mu (absorption) is kept for fidelity but defaults to 0 as in CrysXT.        #
# --------------------------------------------------------------------------- #
def _calc_AB_sabine(y):
    """Sabine A(y), B(y) absorption-coupling coefficients (y = mu * D)."""
    if y <= 1e-9:
        return (1.0, 1.0)
    return (math.exp(-y) * math.sinh(y) / y,
            1.0 / y - math.exp(-y) / math.sinh(y))


def _sabine_primary_factors(x, y):
    """Sabine primary factors (Bragg/Laue components) at x, with absorption y."""
    """Returns (E_L, E_B): the 2theta=0 and 2theta=pi limiting primary factors."""
    el = math.exp(-y)
    if x <= 1.0:
        el *= (1.0 - x / 2.0 + x * x / 4.0 - 5.0 * x ** 3 / 48.0 + 7.0 * x ** 4 / 192.0)
    else:
        el *= math.sqrt(2.0 * _INV_PI / x)
        el *= (1.0 - 1.0 / (8.0 * x) - 3.0 / (128.0 * x * x) - 15.0 / (1024.0 * x ** 3))
    a, b = _calc_AB_sabine(y)
    return (el, a / math.sqrt(1.0 + b * x))


def _sabine_secondary_factors(x, y, tilt_dist):
    """tilt_dist 0=rectangular, 1=triangular tilt distribution.

    Numerically stable forms (review PH-2): the textbook triangular
    expression ``1 - (1 - exp(-2x))/(2x)`` loses ALL significant digits by
    direct subtraction just above the small-x threshold (relative error
    ~eps/x^2: at x ~ 1e-9 the factor came out anywhere in [-100, +100],
    producing negative extinction factors and negative coherent cross
    sections end to end). Rewritten exactly as
    ``(2x + expm1(-2x)) / (2x)`` the cancellation happens inside expm1 and
    the relative error stays ~eps/x (bounded, ~2e-7 at x=1e-9). The same
    defect exists verbatim in upstream ncplugin-CrysXT (scnd_extn_fact,
    triangular branch). ``bx - log1p(bx)`` (relative error ~2eps/bx) gets a
    series branch below bx=1e-4 for the same reason.
    """
    a, b = _calc_AB_sabine(y)
    bx = b * x
    if tilt_dist == 0:
        el = math.exp(-y) if x < 1e-9 \
            else math.exp(-y) * (-math.expm1(-2.0 * x)) / (2.0 * x)
        return (el, a / (1.0 + bx))
    if x < 1e-9:
        return (math.exp(-y), a * b)
    el = math.exp(-y) / x * (2.0 * x + math.expm1(-2.0 * x)) / (2.0 * x)
    if bx < 1e-4:
        # bx - log1p(bx) = bx^2/2 - bx^3/3 + bx^4/4 - ... ; truncation
        # relative error < bx^3 terms / leading ~ (2/3)bx < 1e-4 * 2/3,
        # and with the bx^3/bx^4 terms kept it is < 1e-12 at the branch.
        series = bx * bx * (0.5 - bx / 3.0 + bx * bx / 4.0)
        eb = 2.0 * a / bx / x * series
    else:
        eb = 2.0 * a / bx / x * (bx - math.log1p(bx))
    return (el, eb)


def _sabine_uncorr(Nc, wl, F, l, d, g, L, tilt_dist, mu):
    """Sabine extinction factor, uncorrelated primary x secondary blocks."""
    sin_t = 0.5 * wl / d
    if sin_t > 1.0:
        return 1.0
    sin2 = sin_t * sin_t
    cos2 = 1.0 - sin2
    cos_t = math.sqrt(cos2)
    y = mu * l
    xp = (Nc * wl * F * l) ** 2
    ep_l, ep_b = _sabine_primary_factors(xp, y)
    ep = ep_l * cos2 + ep_b * sin2
    if sin_t == 0.0 or cos_t == 0.0:
        return 0.0
    q = (Nc * wl * F) ** 2 * wl / (2.0 * sin_t * cos_t)
    xs = ep * q * g * L
    es_l, es_b = _sabine_secondary_factors(xs, y, tilt_dist)
    es = es_l * cos2 + es_b * sin2
    return ep * es


def _sabine_corr(Nc, wl, F, l, d, g, L, mu):
    """Sabine extinction factor, correlated (series-coupled) variant."""
    sin_t = 0.5 * wl / d
    if sin_t > 1.0:
        return 1.0
    sin2 = sin_t * sin_t
    cos2 = 1.0 - sin2
    cos_t = math.sqrt(cos2)
    y = mu * l
    if l > 0.0 and g == 0.0:
        x = (Nc * wl * F * l) ** 2
    else:
        if sin_t == 0.0 or cos_t == 0.0:
            return 0.0
        q = (Nc * wl * F) ** 2 * wl / (2.0 * sin_t * cos_t)
        x = (Nc * wl * F * l + g * q * (L - l)) ** 2
    el, eb = _sabine_primary_factors(x, y)
    return el * cos2 + eb * sin2


# --------------------------------------------------------------------------- #
#  Becker-Coppens model (pure / mixed / modified)                             #
# --------------------------------------------------------------------------- #
def _bc_pure(Nc, wl, F, l, d, g, L, tilt_dist, recipe):
    """Becker-Coppens factor for a PURE case: exactly one of primary
    (l set) or secondary type-I/II (g, L set) is active."""
    sin_t = 0.5 * wl / d
    if sin_t > 1.0:
        return 1.0
    cos_t = math.sqrt(1.0 - sin_t * sin_t)
    sin_2t = 2.0 * sin_t * cos_t
    cos_2t = 1.0 - 2.0 * sin_t * sin_t
    q = (Nc * wl * F) ** 2 * wl        # Q_theta WITHOUT /sin2theta
    if l > 0.0 and g == 0.0 and L == 0.0:                 # pure primary
        x = (2.0 / 3.0) * q * l * l / wl
        return _y_primary(x, sin_t, cos_2t, recipe)
    if l == 0.0 and g > 0.0 and L > 0.0:                  # pure secondary type-I
        if sin_2t == 0.0:
            return 0.0
        return 1.0 / math.sqrt(1.0 + 2.0 * g * (q / sin_2t) * L)
    if l > 0.0 and g == 0.0 and L > 0.0:                  # pure secondary type-II
        x = (2.0 / 3.0) * q * L * l / wl
        return _y_secondary(x, sin_t, cos_2t, tilt_dist, recipe, force_212=True)
    return 1.0


def _bc_secondary_x(q, wl, l, L, g, sin_2t, tilt_dist):
    """xs for BC mixed/modified, BC1974 eq.40b (Gauss/Fresnel) / 41b (Lorentz)."""
    if tilt_dist == 1 or tilt_dist == 3:
        return (2.0 / 3.0) * q * L / math.sqrt((wl / l) ** 2 + sin_2t ** 2 / (2.0 * g * g))
    return (2.0 / 3.0) * q * L / (wl / l + sin_2t * 2.0 / (3.0 * g))


def _bc_mix(Nc, wl, F, l, d, g, L, tilt_dist, recipe, primary):
    """Becker-Coppens factor for the MIXED case (primary and secondary
    both active); ``primary`` selects which factor of the product to return."""
    sin_t = 0.5 * wl / d
    if sin_t > 1.0:
        return 1.0
    cos_t = math.sqrt(1.0 - sin_t * sin_t)
    sin_2t = 2.0 * sin_t * cos_t
    cos_2t = 1.0 - 2.0 * sin_t * sin_t
    q = (Nc * wl * F) ** 2 * wl
    if primary:
        xp = (2.0 / 3.0) * q * l * l / wl
        yp = _y_primary(xp, sin_t, cos_2t, recipe)
    else:
        yp = 1.0                                          # BC_mod: no primary
    if l < 1e-9:
        return yp
    xs = _bc_secondary_x(q, wl, l, L, g, sin_2t, tilt_dist)
    xs *= yp                                              # ys also depends on yp
    ys = _y_secondary(xs, sin_t, cos_2t, tilt_dist, recipe)
    return yp * ys


# --------------------------------------------------------------------------- #
#  Public dispatcher                                                          #
# --------------------------------------------------------------------------- #
def _extinction_factor_raw(model, Nc, wl, F_hkl, d_hkl, *, l=0.0, g=0.0, L=0.0,
                      dist=None, recipe="std", mu=0.0):
    """Per-plane extinction factor y in (0, 1].

    model    : one of EXTINCTION_MODELS.
    Nc       : unit cells per volume = 1/V_cell  [Angstrom^-3].
    wl       : neutron wavelength [Angstrom].
    F_hkl    : |F_hkl|, structure-factor modulus per cell [Angstrom]
               (= sqrt(fsquared[barn]) * 1e-4).
    d_hkl    : interplanar spacing [Angstrom].
    l, g, L  : crystallite size [A], mosaic spread [rad^-1], grain size [A].
    dist     : 'rect'/'tri' (Sabine) or 'Gauss'/'Lorentz'/'Fresnel' (BC).
    recipe   : 'cls' or 'std' (BC models). 'lux' is intentionally unsupported --
               see note below.
    mu       : absorption attenuation [A^-1]; 0 as in CrysXT.
    """
    if model not in EXTINCTION_MODELS:
        raise ValueError(f"unknown extinction model {model!r}; "
                         f"expected one of {EXTINCTION_MODELS}")
    if recipe not in RECIPES:
        raise ValueError(f"unknown recipe {recipe!r}; expected one of {RECIPES}")
    if recipe == "lux":
        # The BC2025 'lux' recipe guarantees error < 1e-6, but the coherent-elastic
        # tape is tabulated to ~0.2% RMSE -- 'lux' is ~1000x tighter than the
        # tabulation tolerance and so cannot change the output. Use 'std'.
        raise NotImplementedError(
            "recipe='lux' is not implemented: its 1e-6 precision is far below the "
            "tape tabulation tolerance (~0.2% RMSE), so it cannot affect the result. "
            "Use recipe='std' (default) or 'cls'.")

    if model.startswith("Sabine"):
        td = _SABINE_DIST.get(dist, 0)
        if model == "Sabine_uncorr":
            return _sabine_uncorr(Nc, wl, F_hkl, l, d_hkl, g, L, td, mu)
        return _sabine_corr(Nc, wl, F_hkl, l, d_hkl, g, L, mu)

    td = _BC_DIST.get(dist, 1)
    if model in ("BC_mix", "BC_mod") and not (l > 0.0 and g > 0.0 and L > 0.0):
        # both run the coupled primary+secondary (BC_mod = secondary-only) path,
        # whose secondary x is parameterised by the crystallite size l as well as
        # the mosaic spread g and grain L -- so all three must be positive, else
        # _bc_mix returns y=1 (a silent no-op stamped as "corrected").
        raise ValueError(
            f"{model} requires l>0, g>0 and L>0 (it couples primary and secondary "
            "extinction). For primary-only extinction use BC_pure with only l set.")
    if model == "BC_pure":
        return _bc_pure(Nc, wl, F_hkl, l, d_hkl, g, L, td, recipe)
    if model == "BC_mix":
        return _bc_mix(Nc, wl, F_hkl, l, d_hkl, g, L, td, recipe, primary=True)
    return _bc_mix(Nc, wl, F_hkl, l, d_hkl, g, L, td, recipe, primary=False)  # BC_mod


# Rounding-scale tolerance for the [0, 1] physical invariant below.
_Y_TOL = 1e-6


def extinction_factor(model, Nc, wl, F_hkl, d_hkl, *, l=0.0, g=0.0, L=0.0,
                      dist="Gauss", recipe="std", mu=0.0):
    """Extinction factor y in [0, 1] -- the invariant-checked public boundary.

    Extinction can only REDUCE the kinematic intensity, so every model must
    return y in [0, 1] (review PH-2). Rounding-scale excursions (composed
    primary x secondary products land ~1e-9 outside on some parameter sets)
    are clamped; a material violation is a numerical-stability or parameter
    bug in a model and raises rather than propagating -- an unchecked
    negative factor reached ENDF tapes as a negative coherent-elastic cross
    section, and a factor > 1 corrupts the extinction scan's deficit bound.
    """
    y = _extinction_factor_raw(model, Nc, wl, F_hkl, d_hkl, l=l, g=g, L=L,
                               dist=dist, recipe=recipe, mu=mu)
    if not math.isfinite(y) or y < -_Y_TOL or y > 1.0 + _Y_TOL:
        raise ValueError(
            f"extinction model {model!r} returned a nonphysical factor "
            f"y={y!r} (must be in [0, 1]) at wl={wl:.6g} A, "
            f"F_hkl={F_hkl:.6g} A, d_hkl={d_hkl:.6g} A, l={l:.6g}, "
            f"g={g:.6g}, L={L:.6g} -- numerical-stability or parameter "
            "problem in the extinction model")
    return min(1.0, max(0.0, y))
