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
RECIPES = ("cls", "std")
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

# kind -> (small-x y0 coefficients, fitted y0 coefficients, their ops,
#          small-u ydelta coefficients, fitted ydelta coefficients, their ops);
# kind 0 is primary extinction, 1/2/3 secondary with a Gauss/Lorentz/Fresnel tilt.
_BC2025 = {
    0: ([1.0, 0.94285714, 0.8204, 0.593, 0.364, 0.19],
        [0.518212, 0.93036, 0.182006, 1.10097, 0.62625, 1.73562,
         1.08506, 2.19459, 1.40451, 1.62083, 1.1031, 0.49125, 0.357611],
        [-1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1],
        [0.41021645, 1.187, 2.37, 4.18],
        [0.05508, 0.1166, 0.2099, 0.5482, 0.5248, 1.402, 1.168,
         2.096, 2.116, 1.155, 1.952, 0.6046],
        [1, -1, 1, -1, 1, -1, 1, -1, 1, -1, -1]),
    1: ([1.0, 1.0606602, 0.9238, 0.667, 0.409, 0.22],
        [0.4588909, 1.038687, 0.2401003, 1.288282, 0.7641972, 1.880246,
         1.886916, 2.171852, 3.273034, 0.9771599, 2.988445, 0.4993548,
         1.037121, 0.4353142],
        [-1, -1, 1, -1, 1, -1, 1, -1, 1, -1, -1, 1, -1],
        [0.46188022, 1.333, 2.66, 4.68],
        [0.062289443, 0.13177896, 0.240705, 0.61857545, 0.61744404,
         1.4812474, 1.5419561, 1.9976424, 2.8090858, 0.74297172,
         2.3120683, 0.8661981],
        [1, -1, 1, -1, 1, -1, 1, -1, 1, -1, -1]),
    2: ([1.0, 1.0, 1.0667, 0.988, 0.79, 0.55],
        [0.53379, 0.84182, 0.16806, 0.65124, 0.67623, 0.47199, 1.0872,
         0.030142, 0.91361, 0.28313, 0.30078, 0.1507],
        [-1, -1, 1, -1, 1, -1, 1, -1, -1, 1, -1],
        [0.53333333, 1.9753, 5.14, 11.9],
        [0.0514714, 0.0863117, 0.191581, 0.266342, 0.504516,
         0.32195, 0.894662, 0.0162501, 0.708855, 0.33707],
        [1, -1, 1, -1, 1, -1, -1, 1, -1]),
    3: ([1.0, 1.0, 0.88, 0.639, 0.394, 0.21],
        [0.493354, 0.963692, 0.235067, 1.18222, 0.672931, 1.78522,
         1.09976, 2.10882, 1.34721, 1.46841, 1.0054, 0.426279, 0.313436],
        [-1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1],
        [0.44, 1.278, 2.56, 4.52],
        [0.05839, 0.12063, 0.233343, 0.578753, 0.584531, 1.42753,
         1.28278, 1.95436, 2.18561, 0.877761, 1.80505, 0.599956],
        [1, -1, 1, -1, 1, -1, 1, -1, 1, -1, -1]),
}


def bc2025_y(kind, x, sintheta):
    """BC2025 'std' extinction factor y(x, sintheta) for ``kind`` (see _BC2025).

    Above x = 1e3 the fit continues as x^-0.933 (Gauss secondary) or
    sqrt(1e3/x) (the others).
    """
    y0_small, y0_c, y0_ops, yd_small, yd_c, yd_ops = _BC2025[kind]
    if x < 0.1:
        y0 = _nest(y0_small, _M5, x)
    else:
        if x > 1e3:
            tail = (x * 1e-3) ** (-0.933) if kind == 1 else math.sqrt(1e3 / x)
            return bc2025_y(kind, 1e3, sintheta) * tail
        xp = (math.sqrt(x) - 1.0) / (math.sqrt(x) + 1.0)
        y0 = _nest(y0_c, y0_ops, xp)
    s = math.sqrt(sintheta)
    u = x * s
    if u < 0.1:
        ydelta = u * u * _nest(yd_small, _M3, u)
    else:
        up = (math.sqrt(u) - 1.0) / (math.sqrt(u) + 1.0)
        ydelta = _nest(yd_c, yd_ops, up)
    return y0 + sintheta * s * ydelta


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
    return bc2025_y(tilt_dist, x, sintheta)


def _y_primary(x, sintheta, cos_2theta, recipe):
    """Primary-extinction factor for the selected recipe ('cls'/'std')."""
    if recipe == "cls":
        return _y_bc1974_primary(x, cos_2theta)
    return bc2025_y(0, x, sintheta)


# --------------------------------------------------------------------------- #
#  Sabine's model  (analytic; Int. Tables Vol. C, ch. 6.4), without absorption #
#  (A = B = 1), as in CrysXT.                                                  #
# --------------------------------------------------------------------------- #
def _sabine_primary_factors(x):
    """Sabine primary factors (E_L, E_B): the 2theta=0 and 2theta=pi limits."""
    if x <= 1.0:
        el = (1.0 - x / 2.0 + x * x / 4.0 - 5.0 * x ** 3 / 48.0 + 7.0 * x ** 4 / 192.0)
    else:
        el = math.sqrt(2.0 * _INV_PI / x)
        el *= (1.0 - 1.0 / (8.0 * x) - 3.0 / (128.0 * x * x) - 15.0 / (1024.0 * x ** 3))
    return (el, 1.0 / math.sqrt(1.0 + x))


def _sabine_secondary_factors(x, tilt_dist):
    """Sabine secondary factors (E_L, E_B); tilt_dist 0=rectangular, 1=triangular.

    Written as ``(2x + expm1(-2x)) / (2x)`` and ``x - log1p(x)`` (with a
    series below 1e-4) instead of the textbook ``1 - (1 - exp(-2x))/(2x)``,
    which loses all digits at small x. The textbook form is also in
    ncplugin-CrysXT.
    """
    if tilt_dist == 0:
        el = 1.0 if x < 1e-9 else (-math.expm1(-2.0 * x)) / (2.0 * x)
        return (el, 1.0 / (1.0 + x))
    if x < 1e-9:
        return (1.0, 1.0)
    el = 1.0 / x * (2.0 * x + math.expm1(-2.0 * x)) / (2.0 * x)
    if x < 1e-4:
        series = x * x * (0.5 - x / 3.0 + x * x / 4.0)
        eb = 2.0 / x / x * series
    else:
        eb = 2.0 / x / x * (x - math.log1p(x))
    return (el, eb)


def _sabine_uncorr(Nc, wl, F, l, d, g, L, tilt_dist):
    """Sabine extinction factor, uncorrelated primary x secondary blocks."""
    sin_t = 0.5 * wl / d
    if sin_t > 1.0:
        return 1.0
    sin2 = sin_t * sin_t
    cos2 = 1.0 - sin2
    cos_t = math.sqrt(cos2)
    xp = (Nc * wl * F * l) ** 2
    ep_l, ep_b = _sabine_primary_factors(xp)
    ep = ep_l * cos2 + ep_b * sin2
    if sin_t == 0.0 or cos_t == 0.0:
        return 0.0
    q = (Nc * wl * F) ** 2 * wl / (2.0 * sin_t * cos_t)
    xs = ep * q * g * L
    es_l, es_b = _sabine_secondary_factors(xs, tilt_dist)
    es = es_l * cos2 + es_b * sin2
    return ep * es


def _sabine_corr(Nc, wl, F, l, d, g, L):
    """Sabine extinction factor, correlated (series-coupled) variant."""
    sin_t = 0.5 * wl / d
    if sin_t > 1.0:
        return 1.0
    sin2 = sin_t * sin_t
    cos2 = 1.0 - sin2
    cos_t = math.sqrt(cos2)
    if l > 0.0 and g == 0.0:
        x = (Nc * wl * F * l) ** 2
    else:
        if sin_t == 0.0 or cos_t == 0.0:
            return 0.0
        q = (Nc * wl * F) ** 2 * wl / (2.0 * sin_t * cos_t)
        x = (Nc * wl * F * l + g * q * (L - l)) ** 2
    el, eb = _sabine_primary_factors(x)
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
def extinction_factor(model, Nc, wl, F_hkl, d_hkl, *, l=0.0, g=0.0, L=0.0,
                      dist="Gauss", recipe="std"):
    """Per-plane extinction factor y in [0, 1].

    model    : one of EXTINCTION_MODELS (the deck parser checks the options).
    Nc       : unit cells per volume = 1/V_cell  [Angstrom^-3].
    wl       : neutron wavelength [Angstrom].
    F_hkl    : |F_hkl|, structure-factor modulus per cell [Angstrom]
               (= sqrt(fsquared[barn]) * 1e-4).
    d_hkl    : interplanar spacing [Angstrom].
    l, g, L  : crystallite size [A], mosaic spread [rad^-1], grain size [A].
    dist     : 'rect'/'tri' (Sabine) or 'Gauss'/'Lorentz'/'Fresnel' (BC).
    recipe   : 'cls' or 'std' (BC models).

    Rounding-scale excursions outside [0, 1] are clamped; a larger one is a
    model error and raises.
    """
    if model.startswith("Sabine"):
        td = _SABINE_DIST.get(dist, 0)
        if model == "Sabine_uncorr":
            y = _sabine_uncorr(Nc, wl, F_hkl, l, d_hkl, g, L, td)
        else:
            y = _sabine_corr(Nc, wl, F_hkl, l, d_hkl, g, L)
    else:
        td = _BC_DIST.get(dist, 1)
        if model == "BC_pure":
            y = _bc_pure(Nc, wl, F_hkl, l, d_hkl, g, L, td, recipe)
        else:   # BC_mix, or BC_mod (no primary factor)
            y = _bc_mix(Nc, wl, F_hkl, l, d_hkl, g, L, td, recipe,
                        primary=(model == "BC_mix"))
    if not (math.isfinite(y) and -1e-6 <= y <= 1.0 + 1e-6):
        raise ValueError(f"extinction model {model!r} returned y={y!r} outside [0, 1]")
    return min(1.0, max(0.0, y))
