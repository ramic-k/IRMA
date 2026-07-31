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


# The RECOMMENDED automatic-grid knob defaults, shared by every input surface
# that offers the converged auto grid: the GUI ENDF-Evaluation grid tab, the
# GUI NCrystal panel (irma.gui.grid_form), and the NCrystal export config
# (irma.ncrystal.config). Change a value HERE and all three follow — the
# surfaces must never carry their own copies — a default that lives in
# more than one place will eventually disagree across the GUI, the export
# config, and the library API.
#
# NOTE the deliberate exception: generate_beta_grid's own n_upper parameter
# default stays 20, byte-stable for existing log-lin decks; 80 here is the
# recommended value for new work (fine enough for lin-lin/INT=2 laws at high
# incident energy). Keys are named as the NCrystal export config spells them.
AUTO_GRID_DEFAULTS = {
    "n_lower": 15,           # log low-beta tail points (author decision
                             # 2026-08-01: 15 vs the previous 50 moves the
                             # NCrystal-reconstructed graphite sigma(E) by
                             # <=0.21%, mean 0.05%, over 1e-4..4.9 eV)
    "n_phonon": 300,         # linear phonon-region subdivisions
    "n_upper": 80,           # log high-beta tail points (see NOTE above)
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


# Cap on the high-beta step (in beta units) that keeps the linear-linear
# (INT=2) interpolation error of the exp(-beta/2) detailed-balance tail below
# ~1% (the exact midpoint overshoot of exp(-beta/2) at d_beta=0.5 is
# cosh(0.5/4)-1 ~ 0.78%). A pure logarithmic upper tail lets the step grow
# without bound (d_beta ~ 30 at beta = 196 for the default n_upper), so a
# lin-lin (iint=1) law overshoots the free-atom cross-section limit by >100% at
# high incident energy. The cap is only needed where the law is still
# appreciable: along the recoil ridge (beta ~ alpha), which for the maximum
# incident energy E=beta_max reaches beta_recoil = 4*beta_max/AWR (back-scatter
# alpha at that energy). Beyond it the law is the off-ridge tail (~exp(-beta/2),
# negligible) and the original coarse log spacing is kept, so the grid stays
# small: capping finely only up to the recoil ridge gives fewer beta points
# than capping the whole tail while removing the overshoot (validated against the
# converged linear-Q grid: sigma(5 eV) within ~1.5% of the reference, versus
# +144% pure log). The cap is irrelevant to log-lin (iint=0) laws, which
# interpolate the exponential tail exactly on any grid, so default-INT
# auto-grids stay byte-identical.
DELTA_BETA_MAX_LINLIN = 0.5


def _upper_beta_tail(beta_lo, beta_hi, n_upper, delta_beta_max=None,
                     beta_fine=None):
    """High-beta tail from ``beta_lo`` (exclusive) to ``beta_hi`` (inclusive).

    With ``delta_beta_max=None`` this is the pure-geometric tail of
    ``n_upper`` points (``np.geomspace(...)[1:]``) — the byte-stable form
    existing log-lin decks depend on. When
    ``delta_beta_max`` is set the geometric step is capped so no interval
    exceeds it, but only up to ``beta_fine`` (the recoil ridge); above
    ``beta_fine`` the original coarse log spacing is kept, since the law there
    is the negligible off-ridge tail. ``beta_fine=None`` caps the whole tail
    (correct for light atoms whose recoil ridge reaches ``beta_hi``).
    """
    if delta_beta_max is None:
        return np.geomspace(beta_lo, beta_hi, n_upper + 1)[1:]
    if delta_beta_max <= 0.0:
        raise ValueError(
            f"delta_beta_max must be > 0, got {delta_beta_max}")
    bf = beta_hi if beta_fine is None else float(min(max(beta_fine, beta_lo),
                                                     beta_hi))
    if bf <= beta_lo:
        # Recoil ridge at or below the phonon-region end: the entire upper tail
        # is the negligible off-ridge tail, where coarse log spacing is fine
        # even for lin-lin. Return the pure-log tail (also avoids a duplicate
        # node at beta_lo, which the coarse geomspace below would re-include).
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
                       n_lower=50, n_phonon=300, n_upper=20,
                       beta_max_eV=5.0, delta_beta_max=None, recoil_awr=None):
    """Generate a beta grid with logarithmic tails and linear phonon region.

    Parameters
    ----------
    freq_max_eV : float
        Maximum phonon frequency in eV. Determines the extent of the
        linear region.
    temperature_K : float
        Temperature in Kelvin. Used to convert eV to beta = E/kT.
    n_lower : int
        Number of points in the logarithmic low-beta region (below the
        phonon spectrum). Default 50.
    n_phonon : int
        Number of subdivisions of the phonon spectrum; the linear region
        contributes ``n_phonon - 1`` interior points at spacing
        ``freq_max / n_phonon``, spanning ``(0, freq_max)`` exclusive
        (``freq_max`` itself is covered by the upper tail in the default
        configuration). Default 300.
    n_upper : int
        Number of points in the logarithmic high-beta tail (above the
        phonon spectrum, up to beta_max_eV). Default 20 (byte-stable for
        log-lin, iint=0). A lin-lin (iint=1) law needs a finer tail -- the
        GUI defaults n_upper to 80, since the coarse 20-point tail grows to
        d_beta ~ 30 near beta_max and lets the lin-lin cross section overshoot
        the free-atom limit above a few eV. Verify grid convergence for the
        material and incident-energy range; ~20 suffices for log-lin or
        thermal-only work, demanding lin-lin cases may want more.
    beta_max_eV : float
        Maximum energy transfer in eV for the upper tail. Default 5.0 eV.
        The cap only takes effect when it lies ABOVE the linear region's
        end ``freq_max_eV * (1 - 1/n_phonon)``; otherwise the upper tail
        is dropped, the grid ends at the linear region's last point, and
        a UserWarning is emitted (the cap cannot truncate the linear
        phonon region without changing every existing auto-grid deck).
    delta_beta_max : float or None
        Cap on the upper-tail beta step (beta units). None (default)
        gives the pure-log tail, the byte-stable form existing log-lin
        (iint=0) decks depend on and the correct one for them. Set to ``DELTA_BETA_MAX_LINLIN`` (0.5) for
        lin-lin (iint=1) laws so the high-beta tail is fine enough that
        lin-lin interpolation of the exponential decay does not overshoot
        the free-atom cross section at high incident energy.
    recoil_awr : float or None
        Atomic weight ratio. When ``delta_beta_max`` is set, the fine cap is
        applied only up to the recoil ridge ``beta = 4*beta_max/recoil_awr``
        and the coarse log tail is kept above it (the law there is the
        negligible off-ridge tail), which keeps the grid small. None caps the
        whole tail (the safe choice when AWR is unknown; needed anyway for
        light atoms whose recoil ridge reaches ``beta_max``).

    Returns
    -------
    beta : ndarray
        Beta grid starting from 0.0, sorted ascending.
    """
    if n_phonon < 2:
        raise ValueError(f"n_phonon must be >= 2, got {n_phonon}")
    # isfinite as well as the sign: inf > 0 is True, so a bare positivity test
    # lets freq_max_eV=inf (a one-character YAML slip upstream) through and the
    # grid comes out full of non-finite values instead of failing here.
    if not np.isfinite(freq_max_eV) or freq_max_eV <= 0.0:
        raise ValueError(f"freq_max_eV must be finite and > 0, got {freq_max_eV}")
    if not np.isfinite(temperature_K) or temperature_K <= 0.0:
        raise ValueError(f"temperature_K must be finite and > 0, got {temperature_K}")
    if not np.isfinite(beta_max_eV) or beta_max_eV <= 0.0:
        raise ValueError(f"beta_max_eV must be finite and > 0, got {beta_max_eV}")
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

    # Logarithmic upper tail (step-capped when delta_beta_max is set, so a
    # lin-lin (iint=1) law does not overshoot its exp(-beta/2) tail)
    beta_upper_max = beta_max_eV / kT_eV
    if n_upper > 0 and beta_upper_max > beta_linear[-1]:
        beta_fine = (4.0 * beta_upper_max / recoil_awr
                     if (delta_beta_max is not None and recoil_awr) else None)
        beta_upper = _upper_beta_tail(beta_linear[-1], beta_upper_max,
                                      n_upper, delta_beta_max, beta_fine)
    else:
        beta_upper = np.array([])
        if n_upper > 0:
            # The grid itself must stay byte-identical for every existing
            # auto-grid deck, so the cap is reported, not enforced: the
            # linear phonon region is never truncated.
            e_linear_end_eV = beta_linear[-1] * kT_eV
            warnings.warn(
                f"generate_beta_grid: upper log tail dropped — beta_max_eV="
                f"{beta_max_eV:g} eV does not exceed the linear phonon "
                f"region's end {e_linear_end_eV:g} eV "
                f"(freq_max_eV={freq_max_eV:g} * (1 - 1/n_phonon={n_phonon})). "
                f"The grid ends at {e_linear_end_eV:g} eV; the beta_max_eV "
                f"cap is NOT applied. Raise beta_max_eV above freq_max_eV "
                f"to get the tail, or set n_upper=0 to silence this warning.",
                UserWarning,
                stacklevel=2,
            )

    # Combine with zero
    beta_nonzero = np.concatenate((beta_lower, beta_linear, beta_upper))
    beta = np.concatenate(([0.0], beta_nonzero))

    return beta


def generate_beta_grid_for_iint(freq_max_eV, temperature_K, *, iint, awr,
                                n_lower=50, n_phonon=300, n_upper=80,
                                beta_max_eV=5.0):
    """Beta grid with the tail treatment matched to the Card 4 ``iint`` flag.

    Single home for the safe pairing of tail treatment and
    interpolation law: a log-lin law (``iint=0``, ENDF INT=4) interpolates
    the exp(-beta/2) tail exactly on the pure-log tail, while a lin-lin law
    (``iint=1``, INT=2) overshoots the free-atom cross section on a coarse
    log tail and needs the step cap (``DELTA_BETA_MAX_LINLIN``) with the
    recoil ridge from ``awr``. Library callers building ``iint=1`` decks
    should use this instead of wiring the cap themselves.

    Note the ``n_upper`` default here is 80 (the GUI / NCrystal-export
    default, fine enough for high incident energies), NOT
    ``generate_beta_grid``'s byte-stable default of 20; pass ``n_upper=20``
    to reproduce an existing log-lin deck's grid exactly.
    """
    if iint not in (0, 1):
        raise ValueError(f"iint must be 0 (log-lin, INT=4) or 1 (lin-lin, "
                         f"INT=2), got {iint}")
    if not (np.isfinite(awr) and awr > 0.0):
        raise ValueError(f"awr must be finite and > 0, got {awr}")
    return generate_beta_grid(
        freq_max_eV, temperature_K, n_lower=n_lower, n_phonon=n_phonon,
        n_upper=n_upper, beta_max_eV=beta_max_eV,
        delta_beta_max=(DELTA_BETA_MAX_LINLIN if iint == 1 else None),
        recoil_awr=awr)


def generate_alpha_grid(beta, awr, temperature_K, dq_ang_inv=0.05,
                        q_cut_ang_inv=12.0, n_log=160, n_extra_low=0):
    """Generate an alpha grid linear in momentum transfer Q.

    Two segments: Q = dq, 2*dq, ... up to q_cut (the thermal scattering
    window), then a logarithmic tail of ``n_log`` points up to the Q
    that matches the beta grid's kinematic reach (alpha_max =
    4*beta_max/awr). Alpha follows from Q^2 * (hbar^2/2m) / (awr*kT).

    Earlier versions mirrored the beta grid through the recoil relation
    alpha = 4*beta/awr, which copies beta's LINEAR layout into alpha.
    The thermal-energy cross section is controlled by S(alpha, beta) at
    alpha ~ 0.01-0.3 (the upscatter windows at Q ~ 1-6 1/A), where a
    constant-Delta-alpha grid is several times coarser than the
    quadratic-in-Q spacing the physics needs: on graphite at 296 K the
    recoil layout under-integrated the thermal inelastic cross section
    by 14-19% against a converged dQ=0.01 reference, while this layout
    lands within 3-5% at the same point count.

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
    n_extra_low : int
        Number of additional logarithmic points below the first alpha
        point (down to 1% of it). Default 0.

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

    if n_extra_low > 0:
        alpha_low = np.geomspace(alpha[0] * 0.01, alpha[0],
                                 n_extra_low + 1)[:-1]
        alpha = np.concatenate((alpha_low, alpha))

    return alpha


def estimate_freq_max_from_dos(energies_eV, dos):
    """Estimate the maximum phonon frequency from a DOS array.

    Finds the energy where the DOS drops below 1% of its maximum,
    searching from the high-energy end.

    Parameters
    ----------
    energies_eV : ndarray
        Energy grid in eV.
    dos : ndarray
        Density of states values.

    Returns
    -------
    freq_max : float
        Estimated maximum phonon frequency in eV.
    """
    dos = np.asarray(dos, dtype=float)
    energies_eV = np.asarray(energies_eV, dtype=float)
    if dos.size == 0 or energies_eV.size == 0:
        raise ValueError("empty DOS array: cannot estimate freq_max")
    # Clamp the peak to its non-negative part so the threshold always means
    # "above 1% of the (non-negative) peak". A physical DOS is non-negative, for
    # which this clamp is an exact no-op; it only guards against an unphysical
    # all-negative DOS where 0.01*max would otherwise be a negative threshold
    # that inverts the "above 1% of peak" test.
    peak = max(float(np.max(dos)), 0.0)
    threshold = 0.01 * peak
    above = np.where(dos > threshold)[0]
    if len(above) > 0:
        return energies_eV[above[-1]]
    return energies_eV[-1]
