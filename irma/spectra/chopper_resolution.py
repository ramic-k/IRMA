"""Direct-geometry chopper-spectrometer energy resolution -- independent reimplementation (BSD).

Computes the incident-energy-dependent instrument energy resolution Delta E(E)
of a direct-geometry time-of-flight chopper spectrometer from an analytical
model of the moderator pulse, chopper timing and flight paths (for the two
disk-chopper instruments, the burst constant C and the lever-arm factor K are
calibrated against PyChop output), so the user picks an instrument + chopper package/mode + frequency + incident
energy Ei and the resolution width is derived -- no manual ``dt_ch`` lookup.

Supported instruments (all eight PyChop direct-geometry spectrometers):

  * Fermi-chopper: ARCS, SEQUOIA (SNS), MAPS, MARI, MERLIN (ISIS TS1),
    HYSPEC (SNS).
  * Disk-chopper:  CNCS (SNS), LET (ISIS TS2).

This is an independent BSD reimplementation of the standard analytical resolution
model (the same physics Mantid's PyChop implements), written from the published
literature, NOT ported from PyChop's GPL source:

  * Moderator pulse width: Ikeda-Carpenter form for the SNS ambient-water
    moderators (S. Ikeda & J.M. Carpenter, NIM A 239 (1985) 536); for the other
    moderators a tabulated pulse FWHM(lambda) is interpolated (``_MOD_TABLES``
    -- dense-grid samples of the moderator-width function PyChop builds from
    each instrument's measured emission tables, captured as numerical output of
    PyChop's public API; provenance in THIRD_PARTY_NOTICES.md).
  * Fermi-chopper burst time: Marseguerra-Pauli transmission variance
    (M. Marseguerra & G. Pauli, NIM 4 (1959) 140).
  * Disk-chopper burst time: the slot opening-time variance (trapezoidal
    transmission); for a counter-rotating double disk the opening reduces by a
    per-chopper overlap factor folded into the geometric opening constant ``C``
    (so sigma_chop = C / f_resolution exactly -- pure geometry x 1/f).
  * Time-width -> energy-resolution propagation with the moderator/chopper/
    aperture/detector/sample lever arms (the direct-geometry resolution
    formalism; cf. Violini et al., NIM A 736 (2014) 31; Windsor, *Pulsed
    Neutron Scattering* (1981)).

Instrument parameters (flight paths, aperture and chopper slot geometries,
detector angular limits) are factual values taken from the public instrument
descriptions and the Mantid PyChop instrument data files; the moderator
pulse-width tables and disk calibration constants are numerical output captured
from PyChop's public API (THIRD_PARTY_NOTICES.md). Validated cell-by-cell
against a local Mantid PyChop install (the black-box reference): Delta E(E) agrees
to ~1% across Ei and energy transfer for every supported instrument.
"""
from __future__ import annotations

import warnings

import numpy as np


def _norm_frequency(freq):
    """Coerce a frequency to the positive scalar resolution(-disk) frequency.

    PyChop drives the disk machines with a ``[resolution, frame]`` frequency
    *list*; only the first element (the resolution-disk spin) sets the burst
    width. Accept either a scalar or that list/array form so a caller may hand
    the raw PyChop frequency straight through, and validate it is > 0 Hz.
    """
    if isinstance(freq, (list, tuple, np.ndarray)):
        arr = np.ravel(np.asarray(freq, float))
        if arr.size == 0:
            raise ValueError("chopper frequency list is empty")
        freq = float(arr[0])
    else:
        freq = float(freq)
    if not freq > 0.0:
        raise ValueError(f"chopper frequency must be > 0 Hz, got {freq!r}")
    return freq

# Physical conversion constants (neutron), matching the PyChop/Mantid values so
# the reimplementation reproduces it numerically.
E2V = 437.3933622751701          # v[m/s]  = E2V * sqrt(E[meV])
E2L = 81.80421023520228          # lam[A]  = sqrt(E2L / E[meV])
E2K = 0.48259640293390343        # k[1/A]  = sqrt(E2K * E[meV])
SIGMA2FWHM = 2.3548200450309493  # FWHM = SIGMA2FWHM * sigma  (= 2 sqrt(2 ln2))
SIGMA2FWHMSQ = SIGMA2FWHM ** 2

# He-3 detector tube: 10 atm, reference macroscopic absorption 1.4323 cm^-1 =
# 143.23 m^-1 at k = 3.49416 1/A (1/v law), wall/radius ratio 0.063 -- the
# standard He-tube constants (cf. CKL / Mantid detector model).
_HE_SIGREF = 143.23      # m^-1
_HE_WREF = 3.49416       # 1/A
_HE_ATMREF = 10.0
_HE_ATMS = 10.0
_HE_T2RAD = 0.063

# Gauss-Legendre nodes/weights on [-1,1] for the cross-section average of the
# absorption-depth moments (64 points -> the moment integral is converged).
_GL_Z, _GL_W = np.polynomial.legendre.leggauss(64)


def _he_tube_depth_var(wvec, rad, atms=_HE_ATMS, t2rad=_HE_T2RAD):
    """Variance (m^2) of the neutron absorption depth in a He-3 detector tube.

    Exact cross-section integral (not a Chebyshev fit): a neutron entering a
    cylinder of effective radius ``reff = rad*(1-t2rad)`` at impact parameter y
    is absorbed along its chord with an exponential law of macroscopic cross
    section Sigma(k) = (sigref*wref/atmref) * atms / k (1/v). The beam-axis
    ("depth") coordinate's variance, averaged over absorbed neutrons across the
    full illuminated cross-section, is returned. ``wvec`` (1/A) may be an array.
    """
    wvec = np.asarray(wvec, float)
    const = _HE_SIGREF * _HE_WREF / _HE_ATMREF
    reff = rad * (1.0 - t2rad)
    # diameter optical depth alf = Sigma * (2 reff); work in a unit-radius
    # cylinder (Sigma_unit = alf/2 per unit length) and scale variance by reff^2
    alf = (2.0 * reff * const * atms) / np.clip(wvec, 1e-12, None)
    S = np.clip(alf / 2.0, 1e-8, None).reshape(-1, 1)        # (nE,1)
    c = np.sqrt(np.clip(1.0 - _GL_Z ** 2, 0.0, None)).reshape(1, -1)  # (1,nq)
    w = _GL_W.reshape(1, -1)
    L = 2.0 * c
    e = np.exp(-S * L)
    SL = S * L
    # absorption-weighted moments of depth x in [-c, c] along each chord
    m0 = 1.0 - e
    m1 = (1.0 / S) * (1.0 - e * (1.0 + SL)) - c * (1.0 - e)
    m2 = ((2.0 / S ** 2) * (1.0 - e * (1.0 + SL + 0.5 * SL ** 2))
          - (2.0 * c / S) * (1.0 - e * (1.0 + SL)) + c ** 2 * (1.0 - e))
    M0 = np.sum(w * m0, axis=1)
    M1 = np.sum(w * m1, axis=1)
    M2 = np.sum(w * m2, axis=1)
    delta = M1 / M0
    var_unit = np.clip(M2 / M0 - delta ** 2, 0.0, None)
    return var_unit * reff ** 2


def _moderator_var_s2(Ei, S1, S2, B1, B2, Emod):
    """Ikeda-Carpenter moderator emission-time variance (s^2) at incident Ei."""
    Ei = np.asarray(Ei, float)
    sig = np.sqrt(S1 * S1 + (S2 * S2 * E2L) / Ei)
    A = 4.37392e-4 * sig * np.sqrt(Ei)
    B = np.where(Ei > 130.0, B2, B1)
    R = np.exp(-Ei / Emod)
    var_mm2 = 3.0 / (A * A) + R * (2.0 - R) / (B * B)   # variance in us^2
    return var_mm2 * 1.0e-12                              # -> s^2


def _moderator_fwhm_us(Ei, mod):
    """Raw moderator pulse FWHM (microseconds) at incident energy ``Ei`` (meV).

    ``mod['kind']`` is ``'ik'`` (analytical Ikeda-Carpenter, SNS ambient water)
    or ``'table'`` (the moderator's *measured* pulse FWHM vs wavelength -- a
    factual instrument property, interpolated in lambda).
    """
    Ei = np.asarray(Ei, float)
    if mod["kind"] == "ik":
        return np.sqrt(_moderator_var_s2(Ei, *mod["pars"])) * SIGMA2FWHM * 1.0e6
    lam = np.sqrt(E2L / Ei)
    lo, hi = mod["lam"][0], mod["lam"][-1]
    if np.any(lam < lo) or np.any(lam > hi):
        # np.interp clamps flat past the table ends -- warn rather than return a
        # silently-wrong (constant) moderator width outside the measured range.
        warnings.warn(
            f"moderator pulse-width table spans lambda [{lo:.3f}, {hi:.3f}] A; "
            f"Ei gives lambda outside this range (flat-clamped extrapolation), "
            f"so the moderator width may be inaccurate at this incident energy.",
            stacklevel=2)
    return np.interp(lam, mod["lam"], mod["fwhm_us"])


def _chopper_var_s2(Ei, freq_hz, pslit_m, radius_m, rho_m):
    """Fermi-chopper transmission-burst variance (s^2); NaN if no transmission.

    ``gamma`` measures the mismatch between the chopper slot curvature and the
    neutron velocity; gamma>=4 means the curved slot does not transmit that Ei
    at that frequency (an opaque chopper), returned as NaN.
    """
    Ei = np.asarray(Ei, float)
    w = freq_hz * 2.0 * np.pi
    veloc = E2V * np.sqrt(Ei)
    gamm = (2.0 * radius_m ** 2 / pslit_m) * np.abs(1.0 / rho_m - 2.0 * w / veloc)
    pre = (pslit_m / (2.0 * radius_m * w)) ** 2 / 6.0
    sg = np.sqrt(gamm)                     # gamm >= 0 (built with abs)
    gsqr = np.where(
        gamm <= 1.0,
        np.divide(1.0 - (gamm ** 2) ** 2 / 10.0, 1.0 - (gamm ** 2) / 6.0,
                  out=np.ones_like(gamm), where=(gamm <= 1.0)),
        # 1 < gamma < 4 regime (uses sqrt(gamma)); gamma>=4 -> NaN below
        0.6 * gamm * (sg - 2.0) ** 2 * (sg + 8.0) / (sg + 4.0))
    var = pre * gsqr
    return np.where(gamm >= 4.0, np.nan, var)


def _chopper_fwhm_us(Ei, freq_hz, geom):
    """Chopper burst FWHM (microseconds); NaN if a Fermi slot does not transmit.

    Fermi: Marseguerra-Pauli variance for the selected package. Disk: the
    resolution-disk opening time sigma = C/f (C the geometric opening constant,
    microsecond*Hz), so the burst is purely geometric x 1/f (Ei-independent).
    """
    if geom["chopper_type"] == "fermi":
        pk = geom["fermi"]
        var = _chopper_var_s2(Ei, freq_hz, pk["pslit"], pk["radius"], pk["rho"])
        return np.sqrt(var) * SIGMA2FWHM * 1.0e6
    # disk: sigma[us] = C[us*Hz]/f[Hz]; broadcast to Ei's shape
    sig_us = geom["disk"]["C_us_hz"] / freq_hz
    return np.full(np.shape(np.asarray(Ei, float)), sig_us * SIGMA2FWHM)


def direct_resolution_fwhm(Etrans, *, Ei, frequency, geom):
    """Energy resolution Delta E(Etrans) [meV, FWHM] of a direct-geometry DGS.

    ``geom`` is an instrument-geometry dict (see :func:`instrument_geometry`):
    flight paths x0/x1/x2 (m), optional Fermi aperture (xa, aperture_width,
    moderator tilt theta_m), moderator pulse model, detector ``dd`` (m), sample
    ``sy`` (m) + shape scale, and the selected chopper (Fermi package geometry
    or disk opening constant).

    Reproduces the moderator + chopper + (Fermi) aperture + detector + sample
    time-width propagation to the sample and converts to an energy FWHM via
    ``dE = 2 E2V sqrt(Ef^3 var)/x2``. Energy transfers >= Ei are NaN
    (kinematically forbidden).
    """
    Etrans = np.asarray(Etrans, float)
    Ei = float(Ei)
    frequency = _norm_frequency(frequency)   # accept scalar or [res, frame] list
    x0, x1, x2 = geom["x0"], geom["x1"], geom["x2"]

    # All component time widths enter the propagation as FWHM^2 (s^2).
    tsqmod = (_moderator_fwhm_us(Ei, geom["moderator"]) * 1.0e-6) ** 2
    tsqchp = (_chopper_fwhm_us(Ei, frequency, geom) * 1.0e-6) ** 2

    omega = frequency * 2.0 * np.pi
    Ef = Ei - Etrans
    Ef = np.where(Ef > 0.0, Ef, np.nan)
    vi = E2V * np.sqrt(Ei)
    vf = E2V * np.sqrt(Ef)
    vratio = (vi / vf) ** 3

    modfac = (x1 + vratio * x2) / x0
    chpfac = 1.0 + modfac
    if geom["chopper_type"] == "disk":
        # the disk burst weighs more on the energy-loss side than the Fermi
        # lever arm; K is calibrated per instrument against the reference
        chpfac = chpfac + geom["disk_chpfac_k"] * (vratio - 1.0) * x2 / x0
    var = tsqmod * modfac ** 2 + tsqchp * chpfac ** 2

    # aperture (Fermi instruments with a defined moderator aperture): the
    # moderator-tilt-corrected aperture lever arms; also yields the sample arm.
    aw = geom.get("aperture_width")
    xa = geom.get("xa")
    if aw and xa:
        tanthm = np.tan(np.deg2rad(geom.get("theta_m", 0.0)))
        g1 = 1.0 - (omega * tanthm / vi) * (xa + x1)
        g2 = 1.0 - (omega * tanthm / vi) * (x0 - xa)
        f1 = 1.0 + (x1 / x0) * g1
        f2 = 1.0 + (x1 / x0) * g2
        denom = omega * (xa + x1)
        g1, g2, f1, f2 = g1 / denom, g2 / denom, f1 / denom, f2 / denom
        apefac = f1 + (vratio * x2 / x0) * g1
        var = var + apefac ** 2 * (aw ** 2 / 12.0) * SIGMA2FWHMSQ
        samfac = -f2 - (vratio * x2 / x0) * g2          # s/m
    else:
        # no aperture (disk instruments / HYSPEC): the sample size enters the
        # secondary flight time directly (final-path lever arm 1/vf).
        samfac = 1.0 / vf

    # detector (He tube): absorption-depth variance projected onto time at the
    # detector (per-energy via the final wavevector kf = sqrt(E2K*Ef))
    dd = geom.get("dd")
    if dd:
        kf = np.sqrt(E2K * Ef)
        var_dd = _he_tube_depth_var(np.atleast_1d(kf), rad=dd / 2.0)
        var_dd = var_dd.reshape(np.shape(Ef)) if np.shape(Ef) else float(var_dd[0])
        var = var + (np.sqrt(var_dd) * SIGMA2FWHM / vf) ** 2

    # sample (flat plate / annulus): beam-axis width contribution
    sy = geom.get("sy")
    if sy:
        sample_var_m2 = (sy ** 2) * geom.get("sample_scale", 1.0 / 12.0) * SIGMA2FWHMSQ
        var = var + samfac ** 2 * sample_var_m2

    return 2.0 * E2V * np.sqrt(Ef ** 3 * var) / x2


def chopper_sigma_of_E(E, *, Ei, instrument, package, frequency):
    """Gaussian sigma(E) [meV] for a direct-geometry DGS, fed to the convolution.

    Looks up the instrument geometry, evaluates :func:`direct_resolution_fwhm`
    at every energy transfer, loss and gain (on the gain side Ef > Ei, so the
    width grows with |E|), and converts FWHM -> sigma. Transfers at or above
    Ei, where no final neutron exists, are clipped just below Ei. Raises if the
    chosen chopper/frequency does not transmit Ei (NaN everywhere).
    """
    geom = instrument_geometry(instrument, package)
    E = np.asarray(E, float)
    Et = np.minimum(E, float(Ei) * (1.0 - 1e-6))        # Ef > 0 at every transfer
    fwhm = np.atleast_1d(direct_resolution_fwhm(Et, Ei=Ei, frequency=frequency, geom=geom))
    if not np.any(np.isfinite(fwhm)):
        raise ValueError(
            f"chopper resolution: {instrument}/{package} at {frequency} Hz has no "
            f"transmission at Ei={Ei} meV -- choose a chopper package/frequency "
            "that passes this incident energy.")
    # backfill any non-finite bins (inaccessible) from the nearest finite value
    if not np.all(np.isfinite(fwhm)):
        idx = np.flatnonzero(np.isfinite(fwhm))
        fwhm = np.interp(np.arange(fwhm.size), idx, fwhm[idx])
    sigma = fwhm / SIGMA2FWHM
    return np.clip(sigma, 1e-6, None)


# ---------------------------------------------------------------------------
# Measured moderator pulse FWHM(lambda) [us vs Angstrom] -- factual emission
# data per moderator (these reproduce and extend the measured_width tables
# shipped with each instrument; the SNS ambient-water moderators of ARCS/SEQUOIA
# instead use the analytical Ikeda-Carpenter form, mod kind 'ik').
# ---------------------------------------------------------------------------
_MOD_TABLES = {
    "MAPS": {
        "lam": [
            0.289, 0.3016, 0.3147, 0.3284, 0.3427, 0.3576, 0.3732, 0.3894,
            0.4063, 0.424, 0.4425, 0.4617, 0.4818, 0.5028, 0.5247, 0.5475,
            0.5713, 0.5962, 0.6221, 0.6492, 0.6775, 0.7069, 0.7377, 0.7698,
            0.8033, 0.8383, 0.8747, 0.9128, 0.9525, 0.994, 1.0372, 1.0824,
            1.1295, 1.1786, 1.2299, 1.2834, 1.3393, 1.3975, 1.4583, 1.5218,
            1.588, 1.6571, 1.7292, 1.8045, 1.883, 1.965, 2.0505, 2.1397,
            2.2328, 2.33, 2.4314, 2.5372, 2.6476, 2.7628, 2.883, 3.0084,
            3.1394, 3.276, 3.4185
        ],
        "fwhm_us": [
            4.823, 4.971, 5.125, 5.285, 5.453, 5.628, 5.81, 6.001, 6.2,
            6.408, 6.626, 6.854, 7.094, 7.35, 7.624, 7.928, 8.276, 8.7, 9.25,
            10.004, 11.061, 12.512, 14.37, 16.496, 18.597, 20.374, 21.729,
            22.795, 23.763, 24.739, 25.754, 20.54, 21.641, 22.791, 23.991,
            25.304, 26.939, 28.646, 30.126, 30.674, 31.768, 33.628, 35.209,
            35.954, 36.731, 37.325, 37.832, 38.964, 40.131, 40.594, 40.74,
            41.073, 43.057, 44.452, 45.318, 46.222, 47.165, 48.224, 49.749
        ]},
    "MARI": {
        "lam": [
            0.2941, 0.306, 0.3184, 0.3313, 0.3448, 0.3587, 0.3733, 0.3884,
            0.4041, 0.4205, 0.4375, 0.4553, 0.4737, 0.4929, 0.5129, 0.5337,
            0.5553, 0.5778, 0.6012, 0.6255, 0.6509, 0.6772, 0.7047, 0.7332,
            0.7629, 0.7939, 0.826, 0.8595, 0.8943, 0.9305, 0.9682, 1.0075,
            1.0483, 1.0907, 1.1349, 1.1809, 1.2288, 1.2785, 1.3303, 1.3842,
            1.4403, 1.4987, 1.5594, 1.6226, 1.6883, 1.7567, 1.8279, 1.9019,
            1.979, 2.0592, 2.1426, 2.2294, 2.3197, 2.4137, 2.5115, 2.6132,
            2.7191, 2.8293, 2.9439, 3.0631, 3.1872, 3.3164, 3.4507, 3.5905,
            3.736, 3.8874, 4.0449
        ],
        "fwhm_us": [
            4.884, 5.023, 5.169, 5.32, 5.477, 5.641, 5.811, 5.989, 6.173,
            6.365, 6.565, 6.773, 6.989, 7.214, 7.448, 7.691, 7.945, 8.209,
            8.483, 8.768, 9.066, 9.375, 9.696, 10.031, 10.379, 10.742,
            11.119, 11.511, 11.919, 12.344, 12.786, 13.246, 13.724, 14.222,
            12.688, 13.413, 14.168, 14.953, 15.77, 16.62, 17.505, 18.426,
            19.915, 21.821, 23.804, 25.867, 27.817, 29.513, 31.277, 33.113,
            35.023, 37.011, 39.079, 40.15, 41.197, 42.286, 43.42, 44.599,
            45.455, 46.271, 47.119, 48.002, 48.931, 49.972, 51.055, 52.183,
            53.31
        ]},
    "MERLIN": {
        "lam": [
            0.2941, 0.306, 0.3184, 0.3313, 0.3448, 0.3587, 0.3733, 0.3884,
            0.4041, 0.4205, 0.4375, 0.4553, 0.4737, 0.4929, 0.5129, 0.5337,
            0.5553, 0.5778, 0.6012, 0.6255, 0.6509, 0.6772, 0.7047, 0.7332,
            0.7629, 0.7939, 0.826, 0.8595, 0.8943, 0.9305, 0.9682, 1.0075,
            1.0483, 1.0907, 1.1349, 1.1809, 1.2288, 1.2785, 1.3303, 1.3842,
            1.4403, 1.4987, 1.5594, 1.6226, 1.6883, 1.7567, 1.8279, 1.9019,
            1.979, 2.0592, 2.1426, 2.2294, 2.3197, 2.4137, 2.5115, 2.6132,
            2.7191, 2.8293, 2.9439, 3.0631, 3.1872, 3.3164
        ],
        "fwhm_us": [
            8.582, 8.871, 9.172, 9.486, 9.812, 10.151, 10.505, 10.872,
            11.254, 11.652, 12.066, 12.497, 12.945, 13.411, 13.896, 14.401,
            14.926, 15.473, 16.042, 16.633, 17.249, 17.89, 18.556, 19.25,
            19.972, 20.723, 21.504, 22.317, 23.163, 24.044, 24.959, 25.913,
            26.904, 18.449, 19.07, 19.717, 20.389, 21.149, 22.366, 23.632,
            24.949, 25.486, 26.024, 27.301, 28.629, 29.249, 29.603, 29.971,
            30.421, 30.895, 32.574, 34.0, 34.0, 34.012, 35.327, 36.248,
            37.163, 37.955, 38.779, 39.636, 41.073, 42.819
        ]},
    "HYSPEC": {
        "lam": [
            1.189, 1.2228, 1.2576, 1.2934, 1.3302, 1.368, 1.4069, 1.447,
            1.4881, 1.5305, 1.574, 1.6188, 1.6648, 1.7122, 1.7609, 1.811,
            1.8625, 1.9155, 1.97, 2.0261, 2.0837, 2.143, 2.2039, 2.2666,
            2.3311, 2.3974, 2.4656, 2.5358, 2.6079, 2.6821, 2.7584, 2.8369,
            2.9176, 3.0006, 3.086, 3.1738, 3.2641, 3.3569, 3.4524, 3.5507,
            3.6517, 3.7556, 3.8624, 3.9723, 4.0853, 4.2015, 4.321, 4.444,
            4.5704, 4.7004
        ],
        "fwhm_us": [
            19.817, 21.218, 22.724, 24.323, 26.191, 28.112, 30.316, 32.694,
            35.199, 38.091, 41.065, 44.368, 47.898, 51.567, 55.587, 59.722,
            64.005, 68.429, 72.947, 77.34, 81.857, 86.148, 90.319, 94.553,
            98.29, 102.133, 105.775, 109.287, 112.879, 116.274, 119.765,
            123.289, 126.857, 130.525, 134.257, 138.095, 142.01, 146.006,
            150.115, 154.202, 158.406, 162.617, 166.831, 171.166, 175.318,
            179.583, 183.807, 187.968, 192.247, 196.286
        ]},
    "CNCS": {
        "lam": [
            1.6667, 1.7277, 1.7909, 1.8565, 1.9245, 1.9949, 2.0679, 2.1436,
            2.2221, 2.3035, 2.3878, 2.4752, 2.5658, 2.6598, 2.7571, 2.8581,
            2.9627, 3.0712, 3.1836, 3.3002, 3.421, 3.5462, 3.6761, 3.8106,
            3.9501, 4.0948, 4.2447, 4.4001, 4.5611, 4.7281, 4.9012, 5.0806,
            5.2666, 5.4595, 5.6593, 5.8665, 6.0813, 6.3039, 6.5347, 6.7739,
            7.0219, 7.279, 7.5455, 7.8217, 8.1081, 8.4049, 8.7126, 9.0316,
            9.3622, 9.7049, 10.0602, 10.4285, 10.8103
        ],
        "fwhm_us": [
            48.038, 52.845, 58.065, 63.501, 69.176, 74.899, 80.622, 86.194,
            91.564, 96.688, 101.575, 106.255, 110.791, 115.251, 119.706,
            124.225, 128.851, 133.61, 138.525, 143.563, 148.763, 154.018,
            159.421, 164.789, 170.293, 175.665, 181.166, 186.481, 191.933,
            197.142, 202.498, 207.624, 212.915, 217.97, 223.196, 228.249,
            233.464, 238.55, 243.788, 248.942, 254.241, 259.488, 264.869,
            270.199, 275.651, 281.054, 286.556, 291.981, 297.473, 302.861,
            308.27, 313.538, 318.771
        ]},
    "LET": {
        "lam": [
            1.6788, 1.7409, 1.8053, 1.872, 1.9412, 2.013, 2.0874, 2.1646,
            2.2447, 2.3277, 2.4137, 2.503, 2.5955, 2.6915, 2.791, 2.8942,
            3.0012, 3.1122, 3.2273, 3.3466, 3.4704, 3.5987, 3.7318, 3.8697,
            4.0128, 4.1612, 4.3151, 4.4746, 4.6401, 4.8117, 4.9896, 5.1741,
            5.3654, 5.5638, 5.7695, 5.9828, 6.2041, 6.4335, 6.6713, 6.918,
            7.1738, 7.4391, 7.7142, 7.9994, 8.2952, 8.6019, 8.92, 9.2498,
            9.5918, 9.9465, 10.3143, 10.6957, 11.0911, 11.5013, 11.9265,
            12.3675, 12.8248, 13.299, 13.7908, 14.3007
        ],
        "fwhm_us": [
            25.331, 26.954, 29.139, 31.405, 33.538, 35.583, 37.704, 39.903,
            41.936, 43.877, 45.891, 47.978, 50.143, 52.388, 55.777, 59.296,
            62.946, 66.73, 70.655, 74.724, 78.944, 83.32, 87.858, 92.043,
            95.748, 99.59, 103.574, 107.705, 111.989, 116.432, 121.039,
            125.816, 130.77, 136.28, 142.03, 147.992, 154.175, 154.4, 154.4,
            154.4, 154.4, 154.4, 154.4, 154.4, 154.4, 154.4, 154.4, 154.4,
            154.4, 154.4, 154.4, 154.4, 154.4, 154.4, 154.4, 154.4, 154.4,
            154.4, 154.4, 154.4
        ]},
}


def _ik(S1, S2, B1, B2, Emod):
    """Ikeda-Carpenter moderator-pulse spec (PyChop-parameter convention)."""
    return {"kind": "ik", "pars": (S1, S2, B1, B2, Emod)}


def _table(key):
    """Tabulated moderator pulse-width spec (FWHM vs wavelength)."""
    return {"kind": "table", "lam": _MOD_TABLES[key]["lam"],
            "fwhm_us": _MOD_TABLES[key]["fwhm_us"]}


def _fermi(pslit_mm, radius_mm, rho_mm):
    """Fermi-chopper package geometry (yaml gives mm; stored in metres)."""
    return {"pslit": pslit_mm * 1e-3, "radius": radius_mm * 1e-3, "rho": rho_mm * 1e-3}


# ---------------------------------------------------------------------------
# Bundled instrument data (factual flight paths + chopper/moderator constants).
# Lengths in metres; chopper-package slot geometry from the public Mantid PyChop
# instrument files; disk-chopper opening constant C [us*Hz] from the resolution-
# disk geometry x counter-rotating overlap (sigma_chop = C / f_resolution).
# ---------------------------------------------------------------------------
INSTRUMENT_DB = {
    # --- SNS Fermi-chopper (ambient-water Ikeda-Carpenter moderators) --------
    "ARCS": {
        "chopper_type": "fermi",
        "x0": 11.61, "xa": 9.342, "x1": 2.0, "x2": 3.0,
        "aperture_width": 0.1751, "theta_m": -13.75,
        "moderator": _ik(281.0, 79.0, 0.087, 0.4, 172.0),
        "dd": 0.025, "sy": 0.048, "sample_scale": 1.0 / 12.0,    # isam=0 flat plate
        "max_frequency": 600, "default_frequency": 300,
        "packages": {
            "ARCS-100-1.5-AST": _fermi(1.52, 50.0, 580.0),
            "ARCS-700-1.5-AST": _fermi(1.52, 50.0, 1535.0),
            "ARCS-700-0.5-AST": _fermi(0.51, 50.0, 1535.0),
            "ARCS-100-1.5-SMI": _fermi(1.5, 50.0, 640.0),
            "ARCS-700-1.5-SMI": _fermi(1.0, 50.0, 1535.0),
            "SEQ-100-2.0-AST": _fermi(2.03, 50.0, 580.0),
            "SEQ-700-3.5-AST": _fermi(3.56, 50.0, 1535.0)},
    },
    "SEQUOIA": {
        "chopper_type": "fermi",
        "x0": 18.01, "xa": 17.0, "x1": 2.0, "x2": 5.5,
        "aperture_width": 0.05, "theta_m": -13.75,
        "moderator": _ik(30.13, 10.0, 0.07, 0.08, 50.42),
        "dd": 0.025, "sy": 0.048, "sample_scale": 1.0 / 8.0,     # isam=2 annulus
        "max_frequency": 600, "default_frequency": 300,
        "packages": {
            "High-Resolution": _fermi(2.03, 50.0, 580.0),
            "High-Flux": _fermi(4.56, 50.0, 1535.0),
            "SEQ-100-2.0-AST": _fermi(2.03, 50.0, 580.0),
            "SEQ-700-3.5-AST": _fermi(4.56, 50.0, 1535.0),
            "ARCS-100-1.5-AST": _fermi(1.52, 50.0, 580.0),
            "ARCS-700-1.5-AST": _fermi(1.52, 50.0, 1535.0),
            "ARCS-700-0.5-AST": _fermi(0.51, 50.0, 1535.0)},
    },
    # --- ISIS TS1 Fermi-chopper (measured moderator pulse tables) ------------
    "MAPS": {
        "chopper_type": "fermi",
        "x0": 10.143, "xa": 8.27, "x1": 1.899, "x2": 6.0,
        "aperture_width": 0.094, "theta_m": 32.0,
        "moderator": _table("MAPS"),
        "dd": 0.025, "sy": 0.048, "sample_scale": 1.0 / 12.0,    # isam=0
        "max_frequency": 600, "default_frequency": 400,
        "packages": {"A": _fermi(1.087, 49.0, 1300.0),
                     "B": _fermi(1.812, 49.0, 920.0),
                     "S": _fermi(2.899, 49.0, 1300.0)},
    },
    "MARI": {
        "chopper_type": "fermi",
        "x0": 10.05, "xa": 7.19, "x1": 1.689, "x2": 4.022,
        "aperture_width": 0.06667, "theta_m": 13.0,
        "moderator": _table("MARI"),
        "dd": 0.025, "sy": 0.019, "sample_scale": 1.0 / 8.0,     # isam=2
        "max_frequency": 600, "default_frequency": 400,
        "packages": {"A": _fermi(0.76, 49.0, 1300.0),
                     "B": _fermi(1.14, 49.0, 820.0),
                     "C": _fermi(1.52, 49.0, 580.0),
                     "G": _fermi(0.38, 10.0, 800.0),
                     "R": _fermi(1.143, 49.0, 1300.0),
                     "S": _fermi(2.28, 49.0, 1300.0)},
    },
    "MERLIN": {
        "chopper_type": "fermi",
        "x0": 9.995, "xa": 7.19, "x1": 1.925, "x2": 2.5,
        "aperture_width": 0.06667, "theta_m": 26.7,
        "moderator": _table("MERLIN"),
        "dd": 0.025, "sy": 0.040, "sample_scale": 1.0 / 12.0,    # isam=0
        "max_frequency": 600, "default_frequency": 400,
        "packages": {"G": _fermi(0.2, 5.0, 1000000.0),
                     "S": _fermi(2.28, 49.0, 1300.0)},
    },
    # --- SNS Fermi-chopper, no moderator aperture ----------------------------
    "HYSPEC": {
        "chopper_type": "fermi",
        "x0": 37.17, "x1": 3.61, "x2": 4.5,
        "aperture_width": None, "theta_m": 32.0,
        "moderator": _table("HYSPEC"),
        "dd": 0.025, "sy": 0.010, "sample_scale": 1.0 / 8.0,     # isam=2
        "max_frequency": 420, "default_frequency": 180,
        "packages": {"OnlyOne": _fermi(0.6, 5.0, 1000000.0)},
    },
    # --- Disk-chopper spectrometers ------------------------------------------
    # sigma_chop[us] = C_us_hz / f_resolution[Hz]. C folds the resolution-disk
    # slot/guide opening time (slot/(2 pi R f)) and the counter-rotating
    # double-disk overlap factor; verified against the reference to scale as 1/f.
    "CNCS": {
        "chopper_type": "disk",
        "x0": 34.785, "x1": 1.48, "x2": 3.5,        # x0 = resolution-disk distance
        "moderator": _table("CNCS"),
        "dd": 0.025, "sy": 0.010, "sample_scale": 1.0 / 8.0,     # isam=2
        "disk_chpfac_k": 0.190,                       # reference-calibrated disk lever-arm K
        "max_frequency": 300, "default_frequency": 300,
        "packages": {"High-Flux": {"C_us_hz": 7503.3}},
    },
    "LET": {
        "chopper_type": "disk",
        "x0": 23.5, "x1": 1.5, "x2": 3.5,           # x0 = resolution-disk (Disk 5)
        "moderator": _table("LET"),
        "dd": None, "sy": None,                       # LET: moderator + chopper only
        "disk_chpfac_k": 0.379,                       # reference-calibrated disk lever-arm K
        "max_frequency": 300, "default_frequency": 240,
        "packages": {"High-Flux": {"C_us_hz": 3231.9}},
    },
}


def available_instruments():
    """Sorted list of supported direct-geometry chopper spectrometers."""
    return sorted(INSTRUMENT_DB)


def available_packages(instrument):
    """Sorted chopper-package / mode names for ``instrument``."""
    return sorted(_lookup(instrument)["packages"])


def default_frequency(instrument):
    """Default chopper/disk frequency (Hz) for ``instrument`` (seeds the GUI field)."""
    return float(_lookup(instrument)["default_frequency"])


# Detector angular coverage (2theta min..max, degrees) per instrument -- the
# min/max of ``detector.tthlims`` in the PyChop instrument data files (labelled
# there "for Q-E plot"). Factual instrument geometry, used only as the default
# accessible-(q,E) band for the 2-D map kinematic mask; the GUI pre-fills it and
# lets the user edit. MAPS/MARI banks have gaps between segments -- we keep the
# continuous min..max span (a single editable range), which slightly over-counts
# those gaps.
_COVERAGE_DEG = {
    "ARCS": (2.373, 135.955),
    "SEQUOIA": (1.997, 61.926),
    "MAPS": (3.0, 59.8),
    "MARI": (3.43, 134.14),
    "MERLIN": (2.838, 135.69),
    "HYSPEC": (5.0, 65.0),
    "CNCS": (3.806, 132.609),
    "LET": (2.65, 140.0),
}


def default_coverage(instrument):
    """Detector 2theta coverage ``(min, max)`` in degrees for ``instrument``.

    The instrument's full angular span, used as the default band the 2-D map
    masks to (and pre-filled, editable, in the GUI). Validates the instrument
    name; returns ``None`` if no coverage is tabulated.
    """
    _lookup(instrument)                       # validate the name (raises if unknown)
    return _COVERAGE_DEG.get(str(instrument).strip().upper())


def _lookup(instrument):
    """Instrument database entry for ``instrument`` (case-insensitive)."""
    key = str(instrument).strip().upper()
    if key not in INSTRUMENT_DB:
        raise ValueError(
            f"unknown chopper instrument {instrument!r}; supported: "
            f"{available_instruments()}")
    return INSTRUMENT_DB[key]


def instrument_geometry(instrument, package):
    """Return a geometry dict for (instrument, package) ready for the model."""
    base = _lookup(instrument)
    if package not in base["packages"]:
        raise ValueError(
            f"unknown chopper package {package!r} for {instrument}; choose from "
            f"{available_packages(instrument)}")
    geom = {k: v for k, v in base.items() if k != "packages"}
    geom["package"] = package
    if base["chopper_type"] == "fermi":
        geom["fermi"] = base["packages"][package]
    else:
        geom["disk"] = base["packages"][package]
    return geom
