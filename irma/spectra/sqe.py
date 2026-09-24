"""Standalone neutron-scattering forward model: powder S(Q,E) -> instrument spectrum.

This module does not depend on the IRMA ENDF package. It consumes a
powder-averaged dynamic structure factor S(Q,E) (from the IRMA engine, an
OCLIMAX map, or any user-supplied (q, E, S) arrays) and produces the
1-D spectrum a spectrometer measures, by:

  1. building the signed-energy S(Q,E): the computed downscatter side plus the
     energy-gain (anti-Stokes) side, taken from the directly computed gain
     arrays when the powder carries them and from detailed balance otherwise,
  2. sampling S(Q,E) along a detector's kinematic Q(E) trajectory
     (indirect / direct geometry, or a fitted curve),
  3. optionally adding a resolution-broadened elastic line at E=0,
  4. convolving the energy resolution: a width polynomial (VISION-style) or the
     direct-geometry chopper model, with a Gaussian or Lorentzian line shape.

This mirrors what OCLIMAX does with INSTR=0/1/2, but fed by an arbitrary
S(Q,E) source.

Sign / unit conventions
-----------------------
* E  (energy transfer, meV) > 0  : neutron energy loss  (Stokes, downscatter,
                                   phonon creation). The loss side is always
                                   computed.
* E < 0                          : neutron energy gain  (anti-Stokes, upscatter,
                                   phonon annihilation): the directly computed
                                   gain side when attached, else the loss side
                                   by detailed balance  S(Q,-E)=e^{-E/kT} S(Q,+E).
* Q in inverse angstrom (1/A); energies in meV.
* S(Q,E) carried internally as the kf/ki-free (sigma/4pi) S(Q,omega) in
  barn / sr / meV. The measured double-differential is
  d2sigma/dOmega/dE' = (kf/ki) (sigma/4pi) S(Q,omega); the kf/ki factor is
  applied only on request (see the kf/ki note above instrument_spectrum).

References for the relations used here: Squires, "Thermal Neutron Scattering";
ENDF-102 (S(alpha,beta) conventions); the IRMA noncubic_engine docstring
(S_asym_down(alpha,|beta|) = (4*pi*kT/sigma_b) * S(Q,E)).
"""
from __future__ import annotations

import dataclasses
import numpy as np
# scipy is imported lazily inside sqe_interpolator so `import irma.spectra`
# works on a bare install (scipy ships in the `spectra` extra).

# Physical constants (neutron) -------------------------------------------------
# Physical constants are sourced from irma.core.constants (CODATA 2018) so the
# spectra subpackage and the ENDF engine never disagree to 8th-digit precision.
from irma.core.constants import HBAR2_OVER_2MN_MEV_A2 as C_E, BK as _BK_EV_PER_K
KB = _BK_EV_PER_K * 1000.0  # Boltzmann constant in meV / K (core BK is eV/K)


# -----------------------------------------------------------------------------
# Powder S(Q,E) container + loaders
# -----------------------------------------------------------------------------
@dataclasses.dataclass
class PowderSQE:
    """Powder-averaged S(Q,E) on a rectilinear (q, E) grid, downscatter side.

    q     : (nq,)      momentum transfer, 1/A, increasing
    E     : (nE,)      energy transfer (loss, >= 0), meV, increasing, E[0] may be 0
    S     : (nq, nE)   kf/ki-free (sigma/4pi) S(Q,E) in barn/sr/meV
    T_K   : float      sample temperature (K)
    sigma_b : float    bound scattering cross section used in the normalization (barn)
    label : str        provenance tag for plots
    """

    q: np.ndarray
    E: np.ndarray
    S: np.ndarray
    T_K: float
    sigma_b: float
    label: str = ""
    # optional directly computed energy-gain side (explicit Bose factors, not
    # mirrored): E_gain is the negative grid -E[E>0][::-1] and S_gain its
    # (nq, nE_gain) intensities. When present, signed_sqe uses these instead of
    # the detailed-balance mirror.
    E_gain: np.ndarray | None = None
    S_gain: np.ndarray | None = None

    @property
    def kT(self) -> float:
        """Thermal energy k_B T in meV (KB is meV/K; kT(300 K) ~ 25.85 meV)."""
        return KB * self.T_K


def _law_to_sqe(law_beta_alpha, T_K, sigma_b):
    """Asymmetric downscatter law S_asym_down(alpha, beta), shape (nbeta, nq),
    -> the physical downscatter S(Q,E) (nq, nE) in barn/sr/meV:
        S(Q,E) = sigma_b/(4*pi*kT) * S_asym_down
    (the inverse of the IRMA noncubic_engine convention
     S_asym_down = (4*pi*kT/sigma_b) * S(Q,E)).
    """
    kT = KB * T_K
    pref = sigma_b / (4.0 * np.pi * kT)
    return np.ascontiguousarray((pref * law_beta_alpha).T)  # (nq, nbeta)


def from_oclimax(path, sigma_b, T_K, label=None):
    """Load an OCLIMAX SAB map (.npz: q_centers_ang_inv, beta_downscatter_abs,
    ssm_internal_beta_alpha).

    OCLIMAX's ``ssm_internal_beta_alpha`` is ALREADY the asymmetric downscatter
    law (= mt4_symmetric * exp(+beta/2), verified from the file), so no further
    symmetrization factor is applied. With beta = E_loss/kT >= 0 (the downscatter
    magnitude used throughout this module), detailed balance makes the
    energy-loss law the larger one, S_asym_down = exp(+beta/2) * S_sym; the
    textbook exp(-beta/2) uses the ENDF sign beta = (E'-E)/kT, which is
    negative for downscatter.
    """
    d = np.load(path)
    q = np.asarray(d["q_centers_ang_inv"], float)
    beta = np.asarray(d["beta_downscatter_abs"], float)
    asy = np.asarray(d["ssm_internal_beta_alpha"], float)  # (nbeta, nq), asym down
    E = beta * (KB * T_K)
    S = _law_to_sqe(asy, T_K, sigma_b)
    return PowderSQE(q=q, E=E, S=S, T_K=T_K, sigma_b=sigma_b,
                     label=label or "OCLIMAX")


def from_noncubic_arrays(q, E, S, T_K, sigma_b, label=None,
                         E_gain=None, S_gain=None):
    """Wrap the IRMA noncubic engine's physical ``sqe_*`` map into a PowderSQE.

    The engine's ``sqe_*_barn_per_meV`` arrays are already the kf/ki-free
    (sigma/4pi) S(Q,E) that ``PowderSQE.S`` holds, on the engine's
    ``(q_ang_inv, e_mev)`` grid, so this bridge applies no SAB inversion and
    no ``exp(+beta/2)``. (The ``4*pi*kT/sigma_b``-scaled
    ``sab_*`` arrays would need them; reading the ``sqe_*`` family avoids that.) This is the canonical engine -> spectra
    bridge used by ``compute_spectrum``. S has shape (nq, nE).
    """
    return PowderSQE(q=np.asarray(q, float), E=np.asarray(E, float),
                     S=np.asarray(S, float), T_K=T_K, sigma_b=sigma_b,
                     label=label or "IRMA noncubic",
                     E_gain=(None if E_gain is None else np.asarray(E_gain, float)),
                     S_gain=(None if S_gain is None else np.asarray(S_gain, float)))


# -----------------------------------------------------------------------------
# Signed-energy S(Q,E): downscatter + energy-gain side
# -----------------------------------------------------------------------------
def signed_sqe(p: PowderSQE, include_gain=True):
    """Return (q, E_signed, S_signed) with the energy-gain side attached.

    When the powder carries a directly computed gain side (``p.S_gain`` from
    the explicit-Bose-factor evaluation), that is used verbatim. Otherwise the
    gain side is built by detailed balance, S(Q,-E) = exp(-E/kT) * S(Q,+E) --
    the closed form of the same physics for the equilibrium harmonic model.

    E_signed is sorted increasing; energy loss is positive.
    """
    E = p.E
    # ensure E[0] == 0 handled: split into zero node + positive nodes
    has_zero = np.isclose(E[0], 0.0)
    Epos = E[1:] if has_zero else E
    Spos = p.S[:, 1:] if has_zero else p.S
    if not include_gain:
        return p.q, E, p.S
    if p.S_gain is not None:
        Eg = np.asarray(p.E_gain, float)
        if Eg.shape != Epos.shape or not np.allclose(Eg, -Epos[::-1]):
            raise ValueError(
                "signed_sqe: the attached direct gain grid E_gain must be the "
                "mirror of the positive loss grid (-E[E>0][::-1])")
        Sgain = np.asarray(p.S_gain, float)[:, ::-1]  # reorder to match -Epos[::-1] slot
    else:
        boltz = np.exp(-Epos / p.kT)  # (nEpos,)
        Sgain = Spos * boltz[None, :]  # (nq, nEpos), placed at -Epos
    E_signed = np.concatenate([-Epos[::-1], [0.0] if has_zero else [], Epos])
    parts = [Sgain[:, ::-1]]
    if has_zero:
        parts.append(p.S[:, :1])
    parts.append(Spos)
    S_signed = np.concatenate(parts, axis=1)
    return p.q, E_signed, S_signed


def sqe_interpolator(q, E_signed, S_signed):
    """RegularGridInterpolator over (q, E); out-of-range -> 0."""
    try:
        from scipy.interpolate import RegularGridInterpolator
    except ImportError as e:  # pragma: no cover - environment-dependent
        raise ImportError(
            "scipy is required for the forward model's S(Q,E) interpolation; "
            "install it with  pip install 'irma[spectra]'") from e
    return RegularGridInterpolator(
        (q, E_signed), S_signed, bounds_error=False, fill_value=0.0)


# -----------------------------------------------------------------------------
# Detector kinematics: Q(E) trajectories
# -----------------------------------------------------------------------------
def k_of_E(E_meV):
    """Neutron wavevector magnitude (1/A) for energy E (meV). NaN for E<0."""
    E = np.asarray(E_meV, float)
    out = np.full_like(E, np.nan)
    pos = E > 0
    out[pos] = np.sqrt(E[pos] / C_E)
    return out


def Q_indirect(Etr, Ef, two_theta_deg):
    """Indirect geometry (fixed final energy Ef): Ei = Ef + Etr.

    Etr = energy transfer (loss > 0). Returns Q (1/A); NaN where kinematically
    forbidden (Ei <= 0, i.e. energy gain beyond Ef).
    """
    Etr = np.asarray(Etr, float)
    Ei = Ef + Etr
    ki = k_of_E(Ei)
    kf = np.sqrt(Ef / C_E)
    c = np.cos(np.deg2rad(two_theta_deg))
    Q2 = ki**2 + kf**2 - 2.0 * ki * kf * c
    return np.sqrt(np.clip(Q2, 0.0, None))


def Q_direct(Etr, Ei, two_theta_deg):
    """Direct geometry (fixed incident energy Ei): Ef = Ei - Etr.

    Returns Q (1/A); NaN where Ef <= 0 (transfer exceeds Ei).
    """
    Etr = np.asarray(Etr, float)
    Ef = Ei - Etr
    kf = k_of_E(Ef)
    ki = np.sqrt(Ei / C_E)
    c = np.cos(np.deg2rad(two_theta_deg))
    Q2 = ki**2 + kf**2 - 2.0 * ki * kf * c
    return np.sqrt(np.clip(Q2, 0.0, None))


# Optional kf/ki kinematic factor. The measured double-differential is
# d2sigma/dOmega/dE' = (kf/ki)(sigma/4pi) S(Q,omega); our stored S(Q,E) is the
# (sigma/4pi) S(Q,omega) part WITHOUT kf/ki. OCLIMAX reports S(Q,omega) along the
# trajectory (no kf/ki), so this is OFF by default; turn it on for a count-rate
# spectrum.
def kf_ki_indirect(Ef):
    """Indirect geometry: kf fixed, ki=sqrt((Ef+Etr)/C). Returns callable Etr->kf/ki.

    The incident energy Ei = Ef + Etr must be positive; at/below the forbidden
    energy-gain boundary (Etr <= -Ef) there is no incident neutron and no flux,
    so kf/ki is 0 there -- guarding it (like :func:`kf_ki_direct`) keeps the
    factor finite instead of returning inf/nan that would poison the spectrum.
    """
    def ratio(Etr):
        """kf/ki at energy transfer ``Etr``, clamped to 0 below threshold."""
        denom = Ef + np.asarray(Etr, float)              # proportional to ki^2
        safe = np.where(denom > 0.0, denom, 1.0)         # avoid 0-division warning
        return np.sqrt(np.where(denom > 0.0, Ef / safe, 0.0))
    return ratio


def kf_ki_direct(Ei):
    """Direct geometry: ki fixed, kf=sqrt((Ei-Etr)/C). Returns callable Etr->kf/ki."""
    return lambda Etr: np.sqrt(np.clip((Ei - np.asarray(Etr, float)) / Ei, 0.0, None))


# VISION presets (inverse geometry). Ef and bank angles reproduce published
# fitted VISION Q(E) trajectories (a*E^b+c) to ~1%.
VISION_EF_MEV = 3.5
VISION_BANKS = {"forward": 45.0, "backward": 135.0}  # 2-theta in degrees
# VISION energy-resolution Gaussian SIGMA polynomial (meV): sigma = c0 + c1*E + c2*E^2
VISION_SIGMA_COEFFS = (0.31, 0.005, 0.81e-6)


# -----------------------------------------------------------------------------
# Trajectory sampling + resolution convolution
# -----------------------------------------------------------------------------
def sample_along(interp, Q_of_E, E_out):
    """Sample S(Q,E) along Q = Q_of_E(E_out). Forbidden points -> 0."""
    E_out = np.asarray(E_out, float)
    Q = Q_of_E(E_out)
    pts = np.column_stack([Q, E_out])
    I = interp(pts)
    I[~np.isfinite(Q)] = 0.0
    return I


def sigma_of_E(E, coeffs):
    """Width polynomial w(|E|) = c0 + c1·|E| + c2·E² (meV).

    The polynomial is defined on the energy-loss axis (E >= 0, the
    OCLIMAX/VISION convention); the energy-gain side mirrors it at |E|, so a
    loss-side fit cannot go negative below E = 0 and collapse the anti-Stokes
    wing to an unbroadened delta (the chopper model clamps its own domain).
    """
    if len(coeffs) > 3:
        # Reject an over-length poly at every entry point, not only in
        # config.validate -- otherwise c[3:] are silently dropped here.
        raise ValueError(
            "sigma_coeffs is a quadratic c0,c1,c2 (at most 3 terms); "
            f"got {len(coeffs)} -- extra terms would be silently ignored")
    c = list(coeffs) + [0.0, 0.0, 0.0]
    E = np.abs(np.asarray(E, float))
    return c[0] + c[1] * E + c[2] * E**2


def _resolve_width(E_out, width):
    """Resolution width at each ``E_out``: ``width`` is a callable ``E -> sigma``
    (the chopper model) or the polynomial coefficients of :func:`sigma_of_E`."""
    E_out = np.asarray(E_out, float)
    if callable(width):
        return np.asarray(width(E_out), float)
    return sigma_of_E(E_out, width)


RESOLUTION_SHAPES = ("gaussian", "lorentzian")


def _normalize_shape(shape):
    """Check a resolution-shape name against RESOLUTION_SHAPES."""
    if shape not in RESOLUTION_SHAPES:
        raise ValueError(
            f"unknown resolution shape {shape!r}; expected one of {RESOLUTION_SHAPES}")
    return shape


# How far past the window the input is sampled, in units of the largest width
# on the window: a Gaussian to 6 sigma; a Lorentzian until at most 1e-3 of its
# area lies outside, x = w / tan(pi/2 * 1e-3), about 637 HWHM.
PAD_WIDTHS = {"gaussian": 6.0, "lorentzian": 1.0 / np.tan(0.5 * np.pi * 1e-3)}


def pad_reach(E_out, width, shape="gaussian"):
    """How far [meV] the input is needed past each end of the window
    ``E_out``: ``PAD_WIDTHS[shape]`` times the largest width on the window."""
    shape = _normalize_shape(shape)
    E_out = np.asarray(E_out, float)
    return PAD_WIDTHS[shape] * float(np.max(np.clip(_resolve_width(E_out, width), 1e-6, None)))


def padded_grid(E_out, width, shape="gaussian", support=None):
    """``E_out`` extended on both sides by the reach of the resolution kernel.

    The convolved value near a window edge needs the input just outside the
    window. Callers sample the input on this grid and pass it to
    :func:`resolution_convolve` as ``E_in``. The extension continues the
    window's end steps out to ``PAD_WIDTHS[shape]`` times the largest width on
    the window. ``support=(lo, hi)`` is the range outside which the input is
    zero; if given, the extension stops there.
    """
    E_out = np.asarray(E_out, float)
    if E_out.size < 2:
        return E_out
    reach = pad_reach(E_out, width, shape)
    lo, hi = E_out[0] - reach, E_out[-1] + reach
    if support is not None:
        lo, hi = max(lo, float(support[0])), min(hi, float(support[1]))
    h_lo, h_hi = E_out[1] - E_out[0], E_out[-1] - E_out[-2]
    n_lo = max(int(np.ceil((E_out[0] - lo) / h_lo - 1e-9)), 0)
    n_hi = max(int(np.ceil((hi - E_out[-1]) / h_hi - 1e-9)), 0)
    return np.concatenate([E_out[0] - h_lo * np.arange(n_lo, 0, -1), E_out,
                           E_out[-1] + h_hi * np.arange(1, n_hi + 1)])


def resolution_convolve(E_out, I_in, width, shape="gaussian", E_in=None):
    """Convolve I_in with an energy-dependent resolution kernel.

    ``I_in`` is sampled on ``E_in`` (default ``E_out``); the result is on
    ``E_out``. The kernel width at energy E is w(E) (meV). ``width`` may be a
    polynomial coeff sequence (w = poly(E) via :func:`sigma_of_E`) or a
    callable ``E -> w`` (e.g. the direct-geometry
    :func:`irma.spectra.chopper_resolution.chopper_sigma_of_E` model). For
    ``shape='gaussian'`` the width is the Gaussian sigma; for
    ``shape='lorentzian'`` it is the Lorentzian HWHM.

    Both shapes are offered because real spectrometers show Gaussian *or*
    Lorentzian-tailed resolution. (OCLIMAX itself applies only a Gaussian
    resolution function -- ERES/QRES are its Gaussian sigma polynomials -- so the
    Gaussian path is the OCLIMAX-equivalent; the Lorentzian is the extra option.)

    Each input intensity is spread over the output energies by the analytic,
    unit-area line shape whose width is evaluated at the *input* (true)
    energy, so a feature at energy E is smeared symmetrically about E with
    width w(E). A true delta at E comes back centred on E (not pulled toward
    where the width is larger). Indexing the width on the output energy
    instead would shift the centroid under a varying-width kernel.

    Each line shape is scaled so that its samples add up to 1 over the
    uniform ``E_out`` grid continued past both ends (:func:`_grid_sum`), so
    the broadening neither creates nor loses intensity, even when the grid
    step is coarse compared with the width. Nothing is renormalized to the
    window: the result on ``E_out`` is the restriction of the full
    convolution and does not depend on where the window starts or ends,
    provided the input covers the kernel's reach beyond ``E_out``. Sample it
    on :func:`padded_grid` and pass that grid as ``E_in``; with
    ``E_in = E_out`` the values near the window edges lack the intensity from
    outside the window.
    """
    R = resolution_kernel(E_out, width, shape=shape, E_in=E_in)
    return apply_resolution_kernel(R, E_out if E_in is None else E_in, I_in)


def _grid_step(E_out):
    """The step of a uniform energy grid (None for a single point)."""
    if E_out.size < 2:
        return None
    h = float(E_out[1] - E_out[0])
    if not np.allclose(np.diff(E_out), h, rtol=1e-6, atol=0.0):
        raise ValueError("resolution broadening needs a uniform energy grid")
    return h


def _grid_sum(offset, width, step, shape):
    """``step * sum_n K(offset + n*step)`` over an infinite grid: what the
    samples of the unit-area line shape K of the given width add up to.

    It is 1 when the step is fine compared with the width, and above or below
    1 on a coarse grid. Lorentzian (HWHM w): the closed form
    sinh(a)/(cosh(a) - cos t) with a = 2 pi w/step, t = 2 pi offset/step.
    Gaussian (sigma, s = sigma/step): the direct sum over the grid points
    within 9 steps of the centre when s < 1, else the Poisson-summation form
    1 + 2 exp(-2 pi^2 s^2) cos t (the next term is below 1e-34).
    """
    offset = np.asarray(offset, float)
    width = np.asarray(width, float)
    t = 2.0 * np.pi * offset / step
    if shape == "lorentzian":
        q = np.exp(-2.0 * np.pi * width / step)
        return (1.0 - q * q) / (1.0 + q * q - 2.0 * q * np.cos(t))
    s = width / step
    total = 1.0 + 2.0 * np.exp(-2.0 * np.pi ** 2 * s ** 2) * np.cos(t)
    fine = s < 1.0
    if np.any(fine):
        u, sf = (offset / step)[fine], s[fine]
        n = np.round(-u)[:, None] + np.arange(-9, 10)[None, :]
        total = np.array(total, float, copy=True)
        total[fine] = (np.exp(-0.5 * ((n + u[:, None]) / sf[:, None]) ** 2).sum(axis=1)
                       / (np.sqrt(2.0 * np.pi) * sf))
    return total


def resolution_kernel(E_out, width, shape="gaussian", E_in=None):
    """The (nE_out, nE_in) kernel :func:`resolution_convolve` uses.

    Column j is the line shape of width w(E_in[j]) centred on E_in[j],
    evaluated at ``E_out`` and divided by its :func:`_grid_sum` on the
    ``E_out`` grid; ``E_in`` defaults to ``E_out``.
    Precompute it once when broadening MANY spectra on the same grids with the
    same width (e.g. every Q row of a 2-D map) and apply each row with
    :func:`apply_resolution_kernel` — identical arithmetic to calling
    ``resolution_convolve`` per row, with the kernel build hoisted out.
    """
    shape = _normalize_shape(shape)
    E_out = np.asarray(E_out, float)
    E_in = E_out if E_in is None else np.asarray(E_in, float)
    w = np.clip(_resolve_width(E_in, width), 1e-6, None)   # width at each INPUT energy
    dE = E_out[:, None] - E_in[None, :]                    # E_out_i - E_in_j
    wj = w[None, :]
    if shape == "gaussian":
        R = np.exp(-0.5 * (dE / wj) ** 2) / (np.sqrt(2 * np.pi) * wj)
    else:  # lorentzian: L(x;w) = (1/pi) * w / (x^2 + w^2), w = HWHM
        R = (wj / np.pi) / (dE ** 2 + wj ** 2)
    h = _grid_step(E_out)
    if h is None:
        return R
    return R / _grid_sum(E_out[0] - E_in, w, h, shape)[None, :]


def apply_resolution_kernel(R, E_in, I_in):
    """Apply a precomputed :func:`resolution_kernel` to one spectrum sampled on ``E_in``."""
    return np.trapezoid(R * np.asarray(I_in, float)[None, :],
                  np.asarray(E_in, float), axis=1)


def elastic_line(E_out, area, width, shape="gaussian"):
    """A resolution-broadened elastic peak at E=0 with given integrated area.

    THERMR keeps the elastic channel as a delta at zero energy transfer
    (E'=E) in a separate MT; the visible peak appears only after the instrument
    resolution is applied. We reproduce that: delta(E)*area -> the line
    shape at 0 with the same width source used for the inelastic kernel
    (poly coeffs or a callable -- see :func:`resolution_convolve`), evaluated
    on ``E_out`` and scaled like a kernel column, so its samples on the
    ``E_out`` grid continued past both ends add up to ``area``. It is not
    renormalized on the window: an axis that starts at E=0 carries half the
    line, and one that excludes E=0 carries its tail.
    """
    shape = _normalize_shape(shape)
    w0 = max(float(_resolve_width(np.array([0.0]), width)[0]), 1e-6)
    E_out = np.asarray(E_out, float)
    if shape == "gaussian":
        line = np.exp(-0.5 * (E_out / w0) ** 2) / (np.sqrt(2 * np.pi) * w0)
    else:
        line = (w0 / np.pi) / (E_out ** 2 + w0 ** 2)
    h = _grid_step(E_out)
    if h is not None:
        line = line / float(_grid_sum(E_out[0], w0, h, shape))
    return area * line


# -----------------------------------------------------------------------------
# Top-level convenience: full instrument spectrum for one detector bank
# -----------------------------------------------------------------------------
def instrument_spectrum(p: PowderSQE, Q_of_E, E_out, sigma_coeffs,
                        include_gain=True, elastic_area=None,
                        kinematic_factor=None, shape="gaussian"):
    """Compute a 1-D instrument spectrum I(E_out) for one detector trajectory.

    p              : PowderSQE source
    Q_of_E         : callable E(meV) -> Q(1/A) for this detector bank
    E_out          : output energy-transfer grid (meV); may include negatives
    sigma_coeffs   : resolution width polynomial coeffs (meV); sigma for a
                     Gaussian, HWHM for a Lorentzian
    include_gain   : add the energy-gain side (see :func:`signed_sqe`)
    elastic_area   : if not None, add a broadened elastic line of this area
    kinematic_factor : optional callable Etr->kf/ki (e.g. kf_ki_indirect(Ef)) for
                       a measured count-rate spectrum; None keeps the S(Q,omega)
                       convention (matches OCLIMAX INSTR output).
    shape          : resolution line shape, 'gaussian' or 'lorentzian'
    Returns dict with 'E', 'I_inelastic', 'I_elastic', 'I_total', 'Q'.
    """
    q, Es, Ss = signed_sqe(p, include_gain=include_gain)
    interp = sqe_interpolator(q, Es, Ss)
    # S is sampled past both window edges so the convolution there sees the
    # intensity outside the window (S is zero beyond the powder's energy range)
    E_in = padded_grid(E_out, sigma_coeffs, shape=shape, support=(Es[0], Es[-1]))
    I_raw = sample_along(interp, Q_of_E, E_in)
    if kinematic_factor is not None:
        I_raw = I_raw * np.nan_to_num(kinematic_factor(E_in), nan=0.0)
    I_inel = resolution_convolve(E_out, I_raw, sigma_coeffs, shape=shape, E_in=E_in)
    I_el = (elastic_line(E_out, elastic_area, sigma_coeffs, shape=shape)
            if elastic_area is not None else np.zeros_like(E_out))
    return {
        "E": np.asarray(E_out, float),
        "Q": Q_of_E(np.asarray(E_out, float)),
        "I_inelastic": I_inel,
        "I_elastic": I_el,
        "I_total": I_inel + I_el,
        "label": p.label,
    }
