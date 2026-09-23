"""Automatic alpha/beta grid generation for IRMA.

Generates physically motivated grids with:
- Logarithmic spacing at low beta (thermal region)
- Linear spacing in the phonon spectrum region
- Logarithmic spacing in the high-energy tail
- Alpha grid linear in momentum transfer Q (alpha quadratic) through the
  thermal scattering window, with a logarithmic high-Q tail
"""

import warnings

import numpy as np

from irma.core.constants import BK, HBAR2_OVER_2MN_MEV_A2, THERM


# Recommended auto-grid defaults shared by the GUI, the NCrystal export config
# and the MLIP deck emitter. generate_beta_grid keeps its own n_lower=50 and
# log-lin n_upper=20 defaults so existing log-lin decks stay byte-identical.
AUTO_GRID_DEFAULTS = {
    "n_lower": 15,           # log low-beta tail points
    "n_phonon": 300,         # linear phonon-region subdivisions
    "n_upper": 80,           # log high-beta tail points
    "beta_max_eV": 5.0,      # upper-tail energy cap [eV]
    "alpha_dq_invA": 0.05,   # linear-in-Q alpha spacing [1/A]
    "alpha_qcut_invA": 12.0, # end of the linear alpha segment [1/A]
    "alpha_nlog": 160,       # log alpha-tail points
}


def grid_reference_temperature_K(lat, first_temperature_K):
    """Reference temperature the generated alpha/beta units are anchored to.

    Card 7 ``lat=1`` stores alpha/beta in fixed THERM = 0.0253 eV units, so a
    grid written into a lat=1 deck must be generated at T_ref = THERM/BK
    (293.6 K) regardless of the deck temperatures: a stored value x then means
    exactly x*0.0253 eV at every temperature, and the designed freq_max /
    beta_max / dQ / q_cut are honored exactly. Generating at temps[0] instead
    rescales the whole layout by kT(T0)/THERM -- exact only at 293.6 K, 3.8x
    too coarse at 77 K, 0.59x truncated at 500 K. ``lat=0`` stores values in
    kT(T0) units, so there temps[0] is the correct anchor.
    """
    return THERM / BK if int(lat) == 1 else float(first_temperature_K)


# Largest beta step below the recoil ridge for lin-lin (INT=2) laws: keeps the
# exp(-beta/2) midpoint error under 1% (cosh(0.125)-1 = 0.78%). Unused for
# log-lin (iint=0), which interpolates the exponential exactly.
DELTA_BETA_MAX_LINLIN = 0.5

# Kernel widths past the back-scatter alpha that keep the fine step;
# calibration in docs/grids.md.
RIDGE_MARGIN_SIGMAS = 2.0

# Deck files carry the grids with six decimals in scientific notation
# (``irma.mlip.emit._array_lines`` and the GUI writer), so two nodes closer
# than that print as one value and the engine rejects the deck as
# non-increasing. The lin-lin seam between the phonon region and the fine
# tail can produce such a pair.
DECK_GRID_DECIMALS = 6


def effective_temperature_bound_ratio(freq_max_eV, temperature_K):
    """Upper bound on T_eff/T for any phonon spectrum ending at ``freq_max_eV``.

    T_eff/T = integral of rho(E) (E/2kT) coth(E/2kT) dE over the normalised
    spectrum, and the integrand increases with E, so no spectrum that ends
    at E_max exceeds a single mode at E_max: (x/2) coth(x/2) with
    x = E_max/kT. The grid generator knows only the spectrum's maximum
    energy, and this bound is the safe direction for the margin below
    whatever the spectrum's shape (a Debye spectrum with the same cutoff
    sits 10 to 13% lower).
    """
    x = float(freq_max_eV) / (BK * float(temperature_K))
    if not np.isfinite(x) or x <= 0.0:
        raise ValueError(f"freq_max_eV/kT must be finite and > 0, got {x}")
    return float((x / 2.0) / np.tanh(x / 2.0))


def linlin_fine_beta_limit(beta_max, awr, freq_max_eV, temperature_K,
                           evaluation_temperatures_K=None):
    """Beta up to which a lin-lin grid keeps the fine step (``DELTA_BETA_MAX_LINLIN``).

    ``beta_max`` and the returned limit are in the grid's own units, kT at
    ``temperature_K`` (the grid temperature, 293.6 K for a lat=1 deck). The
    limit is the back-scatter alpha at the highest incident energy,
    4*beta_max/awr, plus ``RIDGE_MARGIN_SIGMAS`` widths of the
    down-scattering kernel there. The kernel width depends on the
    temperature the law is evaluated at: in stored units it is
    sqrt(2 alpha T_eff(T)/T_grid) with T_eff(T) = T times
    ``effective_temperature_bound_ratio``, and it grows with T, so the limit
    is taken at the hottest of ``evaluation_temperatures_K`` (default: the
    grid temperature itself).
    """
    alpha_max = 4.0 * float(beta_max) / float(awr)
    t_grid = float(temperature_K)
    temps = ([t_grid] if evaluation_temperatures_K is None
             else [float(t) for t in np.atleast_1d(evaluation_temperatures_K)])
    if not temps or not all(np.isfinite(t) and t > 0.0 for t in temps):
        raise ValueError(
            f"evaluation_temperatures_K must be finite and > 0, got {temps}")
    widths = [np.sqrt(2.0 * alpha_max * effective_temperature_bound_ratio(freq_max_eV, t)
                      * t / t_grid) for t in temps]
    return alpha_max + RIDGE_MARGIN_SIGMAS * max(widths)


def _drop_nodes_indistinct_in_a_deck(beta):
    """Remove the earlier of any two neighbours that print as one deck value."""
    beta = np.asarray(beta, dtype=float)
    if beta.size < 2:
        return beta
    text = [f"{x:.{DECK_GRID_DECIMALS}e}" for x in beta]
    keep = np.ones(beta.size, dtype=bool)
    for i in range(beta.size - 1):
        if text[i] == text[i + 1]:
            keep[i] = False
    return beta[keep]


def _upper_beta_tail(beta_lo, beta_hi, n_upper, delta_beta_max, beta_fine):
    """Lin-lin high-beta tail from ``beta_lo`` (exclusive) to ``beta_hi``.

    The geometric step of an ``n_upper``-point log tail, capped at
    ``delta_beta_max`` up to ``beta_fine``; above it the coarse log spacing
    is kept, since the law there is the negligible off-ridge tail.
    """
    bf = float(min(max(beta_fine, beta_lo), beta_hi))
    if bf <= beta_lo:
        # The fine limit is at or below the phonon-region end: pure log tail.
        return np.geomspace(beta_lo, beta_hi, n_upper + 1)[1:]
    ratio = (beta_hi / beta_lo) ** (1.0 / n_upper)
    fine = []
    cur = beta_lo
    while True:
        step = min(cur * (ratio - 1.0), delta_beta_max)
        cur += step
        if cur >= bf:
            break
        fine.append(cur)
    fine = np.asarray(fine)
    if beta_hi <= bf + 1e-9:               # recoil ridge reaches the cap: all fine
        if len(fine) == 0 or fine[-1] < beta_hi:
            return np.append(fine, beta_hi)
        return fine
    # coarse log region above the recoil ridge, at the original log density
    n_coarse = max(1, int(round(n_upper * np.log(beta_hi / bf)
                                / np.log(beta_hi / beta_lo))))
    coarse = np.geomspace(bf, beta_hi, n_coarse + 1)   # includes bf and beta_hi
    return np.concatenate([fine, coarse])


def generate_beta_grid(freq_max_eV, temperature_K,
                       n_lower=50, n_phonon=300, n_upper=None,
                       beta_max_eV=5.0, *, iint=0, awr=None,
                       evaluation_temperatures_K=None):
    """Generate a beta grid with logarithmic tails and linear phonon region.

    Parameters
    ----------
    freq_max_eV : float
        Maximum phonon frequency in eV. Determines the extent of the
        linear region.
    temperature_K : float
        Grid temperature in Kelvin (beta = E/kT); 293.6 K for a lat=1 deck.
    n_lower : int
        Number of points in the logarithmic low-beta region. Default 50.
    n_phonon : int
        Number of subdivisions of the phonon spectrum; the linear region
        contributes ``n_phonon - 1`` interior points at spacing
        ``freq_max / n_phonon``. Default 300.
    n_upper : int or None
        Number of points of the logarithmic high-beta tail, up to
        ``beta_max_eV``. Default 20 for log-lin (byte-stable for existing
        decks) and 80 for lin-lin.
    beta_max_eV : float
        Maximum energy transfer in eV. When it does not exceed the end of the
        linear region the upper tail is dropped with a UserWarning; the
        linear region is never truncated.
    iint : int
        Card 4 interpolation flag. 0 (log-lin, INT=4) keeps the pure log
        tail. 1 (lin-lin, INT=2) caps the tail step at
        ``DELTA_BETA_MAX_LINLIN`` up to ``linlin_fine_beta_limit`` so linear
        interpolation of exp(-beta/2) does not overshoot the free-atom cross
        section at high incident energy.
    awr : float
        Atomic weight ratio; required for iint=1.
    evaluation_temperatures_K : sequence or None
        Temperatures the law is evaluated at (iint=1 only; default the grid
        temperature). A lat=1 grid is written in kT(293.6 K) units, so below
        that temperature the stored cap is scaled by T_lowest/T_grid; the
        fine limit is taken at the hottest temperature.

    Returns
    -------
    beta : ndarray
        Beta grid starting from 0.0, sorted ascending.
    """
    if iint not in (0, 1):
        raise ValueError(f"iint must be 0 (log-lin, INT=4) or 1 (lin-lin, "
                         f"INT=2), got {iint}")
    if n_upper is None:
        n_upper = 80 if iint == 1 else 20
    if n_phonon < 2:
        raise ValueError(f"n_phonon must be >= 2, got {n_phonon}")
    if not np.isfinite(freq_max_eV) or freq_max_eV <= 0.0:
        raise ValueError(f"freq_max_eV must be finite and > 0, got {freq_max_eV}")
    if not np.isfinite(temperature_K) or temperature_K <= 0.0:
        raise ValueError(f"temperature_K must be finite and > 0, got {temperature_K}")
    if not np.isfinite(beta_max_eV) or beta_max_eV <= 0.0:
        raise ValueError(f"beta_max_eV must be finite and > 0, got {beta_max_eV}")
    if iint == 1:
        if awr is None or not (np.isfinite(awr) and awr > 0.0):
            raise ValueError(f"awr must be finite and > 0 for iint=1, got {awr}")
        temps = ([float(temperature_K)] if evaluation_temperatures_K is None
                 else [float(t) for t in np.atleast_1d(evaluation_temperatures_K)])
        if not temps or not all(np.isfinite(t) and t > 0.0 for t in temps):
            raise ValueError(
                f"evaluation_temperatures_K must be finite and > 0, got {temps}")
    kT_eV = BK * temperature_K

    # Linear region covering the phonon spectrum
    delta_e = freq_max_eV / n_phonon
    e_phonon = np.arange(1, n_phonon) * delta_e
    beta_linear = e_phonon / kT_eV

    # Logarithmic lower tail
    beta_min = 1e-9 / kT_eV
    if n_lower > 0 and beta_linear[0] > beta_min:
        beta_lower = np.geomspace(beta_min, beta_linear[0], n_lower + 1)[:-1]
    else:
        beta_lower = np.array([])

    beta_upper_max = beta_max_eV / kT_eV
    if n_upper > 0 and beta_upper_max > beta_linear[-1]:
        if iint == 1:
            cap = DELTA_BETA_MAX_LINLIN * min(1.0, min(temps) / float(temperature_K))
            beta_fine = linlin_fine_beta_limit(beta_upper_max, awr, freq_max_eV,
                                               temperature_K, temps)
            beta_upper = _upper_beta_tail(beta_linear[-1], beta_upper_max,
                                          n_upper, cap, beta_fine)
        else:
            beta_upper = np.geomspace(beta_linear[-1], beta_upper_max, n_upper + 1)[1:]
    else:
        beta_upper = np.array([])
        if n_upper > 0:
            e_linear_end_eV = beta_linear[-1] * kT_eV
            warnings.warn(
                f"generate_beta_grid: upper log tail dropped, beta_max_eV="
                f"{beta_max_eV:g} eV does not exceed the linear phonon region's "
                f"end {e_linear_end_eV:g} eV; raise beta_max_eV or set n_upper=0.",
                UserWarning, stacklevel=2)

    beta = np.concatenate(([0.0], beta_lower, beta_linear, beta_upper))
    # Only the capped lin-lin tail can put a node within deck precision of
    # the phonon-region end.
    return _drop_nodes_indistinct_in_a_deck(beta) if iint == 1 else beta


def describe_beta_grid(beta, temperature_K, iint):
    """One line for the user on the beta grid that was built."""
    tail = "lin-lin, step-capped tail" if iint == 1 else "log-lin tail"
    return (f"beta grid: {len(beta)} points to "
            f"{float(beta[-1]) * BK * float(temperature_K):.3g} eV ({tail})")


def generate_alpha_grid(beta, awr, temperature_K, dq_ang_inv=0.05,
                        q_cut_ang_inv=12.0, n_log=160):
    """Generate an alpha grid linear in momentum transfer Q.

    Two segments: Q = dq, 2*dq, ... up to q_cut (the thermal scattering
    window), then a logarithmic tail of ``n_log`` points up to the Q
    that matches the beta grid's kinematic reach (alpha_max =
    4*beta_max/awr). Alpha follows from Q^2 * (hbar^2/2m) / (awr*kT).

    Linear in Q because the thermal cross section is set by small alpha;
    see docs/grids.md.

    Parameters
    ----------
    beta : ndarray
        Beta grid (including zero). Its maximum sets the alpha extent.
    awr : float
        Atomic weight ratio of the principal scatterer.
    temperature_K : float
        Temperature in Kelvin (converts Q to the dimensionless alpha).
    dq_ang_inv : float
        Q spacing of the linear segment in 1/Angstrom. Default 0.05.
    q_cut_ang_inv : float
        End of the linear segment. Set by neutron kinematics (it must
        cover the thermal upscatter windows), not by the material.
        Default 12.0 1/Angstrom.
    n_log : int
        Sets the logarithmic tail from q_cut to the grid's maximum Q. The
        node coinciding with q_cut is dropped (it duplicates the last linear
        point), so the tail contributes n_log - 1 points. Default 160.

    Returns
    -------
    alpha : ndarray
        Alpha grid, sorted ascending, starting from the smallest value.
    """
    beta_nonzero = beta[beta > 0]
    if len(beta_nonzero) < 1:
        # An all-zero (or empty) beta grid would silently yield a zero-length
        # alpha array, which propagates downstream as nalpha=0. Fail loudly.
        raise ValueError("beta grid has no positive values: cannot build alpha grid")
    if not np.isfinite(temperature_K) or temperature_K <= 0.0:
        raise ValueError(f"temperature_K must be finite and > 0, got {temperature_K}")
    if not (np.isfinite(awr) and awr > 0.0):
        raise ValueError(f"awr must be finite and > 0, got {awr}")
    if not (np.isfinite(dq_ang_inv) and dq_ang_inv > 0.0):
        raise ValueError(f"dq_ang_inv must be finite and > 0, got {dq_ang_inv}")
    if not (np.isfinite(q_cut_ang_inv) and q_cut_ang_inv > 0.0):
        raise ValueError(f"q_cut_ang_inv must be finite and > 0, got {q_cut_ang_inv}")

    kt_mev = BK * temperature_K * 1.0e3
    alpha_max = 4.0 * float(beta_nonzero[-1]) / awr
    q_max = np.sqrt(alpha_max * awr * kt_mev / HBAR2_OVER_2MN_MEV_A2)

    if q_max <= dq_ang_inv:
        # A single Q point yields nalpha=1, which every interpolation /
        # convolution consumer downstream needs at least two of; fail loudly
        # rather than emit a degenerate one-point alpha grid.
        raise ValueError(
            f"alpha grid degenerate: q_max ({q_max:.4g} 1/Ang) <= dq "
            f"({dq_ang_inv:.4g} 1/Ang) gives fewer than two alpha points; "
            "lower dq or raise beta_max")
    elif q_max <= q_cut_ang_inv:
        # short grid: linear all the way, endpoint pinned at q_max
        q = np.arange(1, int(q_max / dq_ang_inv) + 1) * dq_ang_inv
        if q[-1] < q_max * (1.0 - 1e-12):
            q = np.append(q, q_max)
        else:
            q[-1] = q_max
    else:
        # The log tail carries the grid from q_cut to the kinematic q_max.
        # geomspace(...)[1:] with n_log <= 1 is EMPTY, so the grid would end
        # at q_cut (~12 1/A) instead of q_max -- on a graphite-like case that
        # silently cuts alpha_max from ~66 to ~1 with no other symptom (THERMR's
        # SCT extension partially masks it downstream). Fail loudly instead.
        if n_log < 2:
            raise ValueError(
                f"n_log must be >= 2 when the grid extends past q_cut "
                f"(q_max={q_max:.4g} > q_cut={q_cut_ang_inv:.4g} 1/Ang): "
                f"n_log={n_log} would silently drop the log tail and cap the "
                f"alpha grid at q_cut instead of the kinematic alpha_max")
        q_linear = np.arange(1, int(round(q_cut_ang_inv / dq_ang_inv)) + 1) \
            * dq_ang_inv
        q_log = np.geomspace(q_cut_ang_inv, q_max, n_log)[1:]
        q = np.concatenate((q_linear, q_log))

    alpha = q**2 * HBAR2_OVER_2MN_MEV_A2 / (awr * kt_mev)
    return alpha
