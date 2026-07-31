"""LEAPR physics kernels.

The scattering-law computation chain ported from NJOY2016's leapr.f90:
phonon-expansion machinery (start/convol/contin and the cubic trace-DOS
variant), the translational component (trans), discrete oscillators
(discre), cold hydrogen/deuterium (coldh), the Skold correction
(skold_approx), their numerical helpers, and the Fortran-bit-compatible
sigfig rounding used throughout the ENDF writer.

Derived in part from NJOY2016 (Copyright (c) 2016, Los Alamos National
Security, LLC; BSD 3-Clause — notice retained in THIRD_PARTY_NOTICES.md).
This file is a clearly-marked derivative reimplementation, not the version
available from LANL.
"""

import numpy as np
from functools import lru_cache
from math import sqrt, exp, log, pi, sin, inf

from irma.core.constants import BK, EV, AMU, HBAR, AMASSN, THERM


# math.exp raises OverflowError above ~709.78 (exp(x) > DBL_MAX). NJOY's Fortran
# returns +Inf and continues; cold-temperature / wide-beta kernels that pushed an
# argument past this aborted IRMA with a cryptic "math range error". Where the
# overflowing term has a clean finite limit (a discrete oscillator's coth -> 1) we
# clamp the argument so that limit is reproduced; where a clamp would instead
# cascade to NaN (the continuous-spectrum transform) the caller raises up front.
_EXP_MAX_ARG = 709.0


def _safe_exp(x):
    """exp(x) clamped at the float64 overflow ceiling instead of raising.

    Byte-identical to ``math.exp`` for any ``x <= _EXP_MAX_ARG`` (every physical
    deck); only the overflow regime, where the result feeds a finite ratio limit,
    is affected.
    """
    return exp(x if x <= _EXP_MAX_ARG else _EXP_MAX_ARG)


def fsum(n, p, npt, tau, deltab):
    """Compute integrals over the phonon frequency distribution.

    integral 0 to inf of 2*p*beta**n * hyperbolic dbeta
    where 'hyperbolic' is cosh(tau*beta) for n even, sinh(tau*beta) for n odd.
    """
    arg = deltab * tau / 2.0
    edsq = exp(arg)
    v = 1.0
    an = 1.0 - 2.0 * (n % 2)
    be = 0.0
    fs = 0.0
    w = 1.0

    for ij in range(npt):
        if n > 0:
            w = be ** n
        ff = ((p[ij] * v) * v + (p[ij] * an / v) / v) * w
        if ij == 0 or ij == npt - 1:
            ff = ff / 2.0
        fs += ff
        be += deltab
        v *= edsq

    return fs * deltab

def start(p1, np1, delta1, tev, tbeta):
    """Compute integral functions of the phonon frequency distribution.

    Returns: p (modified spectrum), f0 (DW lambda), tbar, deltab
    """
    npt = np1
    deltab = delta1 / tev
    # The t1(beta) transform below evaluates exp(beta/2) up to beta=deltab*(npt-1).
    # Past the float64 exp ceiling this does not have a clean limit -- the fsum
    # running product overflows, f0 -> 0, and line `p*exp(be/2)/f0` becomes 0/0
    # (NaN). A naive clamp would ship a garbage law, so fail clearly instead.
    if npt > 1 and deltab * (npt - 1) / 2.0 > _EXP_MAX_ARG:
        raise ValueError(
            "continuous phonon-spectrum transform overflows: deltab*(npt-1)/2 = "
            f"{deltab * (npt - 1) / 2.0:.1f} exceeds the exp limit "
            f"({_EXP_MAX_ARG:.0f}). The spectrum's beta range is too wide for this "
            f"temperature (kT = {tev:.4g} eV); reduce delta1, shorten the phonon "
            "spectrum, or raise the temperature.")
    p = np.copy(p1[:npt])

    # Transform spectrum
    u = deltab
    v = exp(deltab / 2.0)
    p[0] = p[1] / deltab**2
    vv = v
    for j in range(1, npt):
        p[j] = p[j] / (u * (vv - 1.0 / vv))
        vv = v * vv
        u += deltab

    # Calculate normalizing constant
    tau = 0.5
    an = fsum(1, p, npt, tau, deltab)
    an = an / tbeta
    for i in range(npt):
        p[i] = p[i] / an

    # Calculate Debye-Waller lambda and effective temperature
    f0 = fsum(0, p, npt, tau, deltab)
    tbar = fsum(2, p, npt, tau, deltab) / (2.0 * tbeta)

    # Convert p(beta) into t1(beta)
    for i in range(npt):
        be = deltab * i
        p[i] = p[i] * exp(be / 2.0) / f0

    return p, f0, tbar, deltab

def terpt_vec(tn, ntn, delta, be_arr):
    """Vectorized interpolation in t_n(beta) table for array of beta values."""
    result = np.zeros_like(be_arr)
    i_raw = np.floor(be_arr / delta).astype(int)
    mask = (be_arr >= 0) & (i_raw < ntn - 1)
    if not np.any(mask):
        return result
    be_m = be_arr[mask]
    i = i_raw[mask]
    bt = i * delta
    result[mask] = tn[i] + (be_m - bt) * (tn[i + 1] - tn[i]) / delta
    return result

def convol(t1, tlast, n1, nl, nn, delta):
    """Convolve t1 with tlast to get the next phonon expansion term.

    Returns tnext array and normalization check ckk.
    Uses a vectorized accumulation (loop over the small n1 axis,
    broadcasting each contribution across all k indices). The broadcast
    sum may reassociate relative to NJOY's sequential loop at the
    ~1e-15 level — far below the 6-7 sig figs written to the ENDF tape —
    so the expected tapes still reproduce byte-for-byte. FFT is avoided
    because its larger ~1e-16 relative noise floor *can* perturb the
    written digits and break that reproduction.
    """
    tiny = 1.0e-30

    # Vectorized convolution: loop over j (n1 is typically small, ~48-301)
    # and compute contributions to all k indices simultaneously using numpy.
    # This avoids the O(n1*nn) Python loop overhead while reproducing NJOY
    # to the written ENDF precision (unlike FFT, whose ~1e-16 noise floor
    # can perturb the written digits).
    tnext = np.zeros(nn)
    for j in range(n1):
        if t1[j] <= 0.0:
            continue
        w = 0.5 if (j == 0 or j == n1 - 1) else 1.0
        coeff = w * t1[j]
        exp_be = exp(-j * delta)

        # f1: tnext[k] += coeff * exp(-j*delta) * tlast[k+j]
        #     for k in [0, min(nn, nl-j))
        kmax_f1 = min(nn, nl - j)
        if kmax_f1 > 0:
            tnext[:kmax_f1] += coeff * exp_be * tlast[j:j + kmax_f1]

        # f2 positive (i2 = k-j >= 0): tnext[k] += coeff * tlast[k-j]
        #     for k in [j, min(j+nl, nn))
        klo = j
        khi = min(j + nl, nn)
        if khi > klo:
            tnext[klo:khi] += coeff * tlast[:khi - klo]

        # f2 negative (i2 = k-j < 0): tnext[k] += coeff * exp(-(j-k)*delta) * tlast[j-k]
        #     for k in [max(0, j-nl+1), min(j, nn))
        if j > 0:
            klo2 = max(0, j - nl + 1)
            khi2 = min(j, nn)
            if khi2 > klo2:
                m = j - np.arange(klo2, khi2)  # j-k values (positive)
                tnext[klo2:khi2] += coeff * np.exp(-m * delta) * tlast[m]

    tnext *= delta

    tnext[tnext < tiny] = 0.0

    # Normalization check
    k_arr = np.arange(nn)
    be_k = k_arr * delta
    cc_k = tnext + tnext * np.exp(-be_k)
    cc_k[0] /= 2.0
    if nn > 1:
        cc_k[-1] /= 2.0
    ckk = delta * np.sum(cc_k)

    return tnext, ckk

def _phonon_expand_from_t1(ssm_slice, alpha, beta, nalpha, nbeta, lat, arat,
                           tev, tbeta, nphon, t1, f0, tbar, deltab):
    """Evaluate the phonon expansion from a precomputed t1(beta) kernel.

    Parameters
    ----------
    t1 : ndarray
        First-order phonon kernel on a uniform beta grid.
    f0 : float
        Debye-Waller lambda used in the Poisson weights and envelope.
    tbar : float
        Effective temperature ratio for the SCT approximation.
    deltab : float
        Uniform beta-grid spacing for ``t1``.
    """
    tiny = 1.0e-30
    explim = -250.0

    if f0 <= 1.0e-30:
        ssm_slice[:nbeta, :nalpha] = 0.0
        return

    sc = 1.0
    if lat == 1:
        sc = THERM / tev

    tlast = np.copy(t1)
    npt = len(t1)
    npl = npt

    al_vec = alpha[:nalpha] * sc / arat
    betan = beta[:nbeta] * sc

    st_vec = terpt_vec(t1, npt, deltab, betan)
    xa = np.log(al_vec * f0)
    ex_vec = -f0 * al_vec + xa
    exx_vec = np.where(ex_vec > explim, np.exp(ex_vec), 0.0)
    ssm_slice[:nbeta, :nalpha] = st_vec[:, np.newaxis] * exx_vec[np.newaxis, :]
    ssm_slice[ssm_slice < tiny] = 0.0

    maxt = np.full(nbeta, nalpha + 1, dtype=int)

    for n in range(2, nphon + 1):
        npn = npt + npl - 1
        tnow, ckk = convol(t1, tlast, npt, npl, npn, deltab)

        st_vec = terpt_vec(tnow, npn, deltab, betan)
        xa += np.log(al_vec * f0 / n)
        ex_vec = -f0 * al_vec + xa
        exx_vec = np.where(ex_vec > explim, np.exp(ex_vec), 0.0)
        add_matrix = st_vec[:, np.newaxis] * exx_vec[np.newaxis, :]
        add_matrix[add_matrix < tiny] = 0.0
        ssm_slice[:nbeta, :nalpha] += add_matrix

        if n >= nphon:
            # smallest alpha index j (per beta row) where this order's contribution
            # still exceeds 1/1000 of the accumulated value -- the SCT-start column.
            # Vectorized; byte-identical to the double loop (same float comparisons,
            # first True per row = the minimum satisfying j, integer min into maxt).
            sl = ssm_slice[:nbeta, :nalpha]
            cond = (sl != 0.0) & (add_matrix > sl / 1000.0)
            has = cond.any(axis=1)
            first_j = cond.argmax(axis=1).astype(maxt.dtype)
            maxt = np.where(has, np.minimum(maxt, first_j), maxt)

        tlast = np.copy(tnow[:npn])
        npl = npn

    # DELIBERATE NJOY DIVERGENCE: this monotonicity clamp on the SCT start
    # index lives inside "if (iprint.ne.0)" in NJOY2016 (leapr.f90:566-571),
    # i.e. NJOY's PHYSICS output depends on the print flag. IRMA applies
    # the clamp unconditionally; all NJOY reference tapes in the expected
    # validation were generated with iprint=1 (clamp active), and IRMA
    # reproduces them to ~1e-4.
    for k in range(1, nbeta):
        if maxt[k] > maxt[k - 1]:
            maxt[k] = maxt[k - 1]

    for j in range(nalpha):
        al = al_vec[j]
        alw = al * tbeta
        alp = alw * tbar
        if alp <= 1.0e-30:
            continue
        for k in range(nbeta):
            if j >= maxt[k]:
                be = betan[k]
                ex = -(alw - be)**2 / (4.0 * alp)
                ssct = 0.0
                if ex > explim:
                    ssct = exp(ex) / sqrt(4.0 * pi * alp)
                ssm_slice[k, j] = ssct

def contin(ssm_slice, alpha, beta, nalpha, nbeta, lat, arat, tev,
           p1, np1, delta1, tbeta, nphon, iprint):
    """Main routine for calculating S(alpha,beta) for continuous distributions.

    ssm_slice: 2D array [nbeta, nalpha] to fill with results
    Returns: f0 (Debye-Waller lambda), tbar (effective temp ratio)

    ``iprint`` is accepted for LEAPR signature compatibility (engine.py
    threads the parsed Card-3 value into every kernel) but is unused:
    NJOY's per-alpha print diagnostics are intentionally not ported.
    The same applies to ``stable``/``trans``/``discre``/``coldh``; only
    ``contin_cubic_trace_dos`` honors the flag.
    """
    # Calculate parameters for this temperature
    p, f0, tbar, deltab = start(p1, np1, delta1, tev, tbeta)

    _phonon_expand_from_t1(
        ssm_slice, alpha, beta, nalpha, nbeta, lat, arat, tev, tbeta, nphon,
        p, f0, tbar, deltab)

    return f0, tbar, deltab

def contin_cubic_trace_dos(ssm_slice, alpha, beta, nalpha, nbeta, lat, arat,
                           tev, dos_tensor_d, energy_grid_ev, nphon, tbeta,
                           iprint, expand_ssm=True):
    """Cubicized incoherent-inelastic S(alpha,beta) from the phonopy DOS tensor.

    Used by the mode-1/2 MT4 path to derive the effective-temperature ratio
    (tbar) and beta-grid spacing: each site's tensor DOS is reduced to a scalar
    DOS via rho_iso(eps) = (1/3) Tr[rho(eps)] and fed through start(). With
    ``expand_ssm=True`` the scalar phonon expansion is also evaluated site by
    site (exactly as contin() would) and S(alpha,beta) is averaged over
    equivalent sites into ``ssm_slice``.

    ``expand_ssm=False`` skips the O(nphon x npt^2) expansion and the ssm
    output entirely; ``ssm_slice`` may then be None. The returns are
    bit-identical either way — f0/tbar/deltab all come from start(), BEFORE
    the expansion — so the bookkeeping-only caller (the per-temperature MT4
    scratch pass) avoids seconds-to-minutes of discarded work per site per
    temperature.

    Returns
    -------
    f0 : float
        Mean scalar Debye-Waller lambda derived consistently from start(rho_iso).
    tbar : float
        Mean effective temperature ratio for the SCT approximation.
    deltab : float
        Energy grid spacing in beta units.
    """
    dos_tensor_arr = np.asarray(dos_tensor_d)
    if dos_tensor_arr.ndim == 3:
        dos_tensor_sites = dos_tensor_arr[np.newaxis, ...]
    elif dos_tensor_arr.ndim == 4:
        dos_tensor_sites = dos_tensor_arr
    else:
        raise ValueError("Cubic trace DOS input must have 3 or 4 dimensions")

    n_sites = dos_tensor_sites.shape[0]
    delta_ev = (energy_grid_ev[1] - energy_grid_ev[0]
                if len(energy_grid_ev) > 1 else 1e-4)
    deltab = delta_ev / tev

    rho_iso_sites = np.trace(dos_tensor_sites, axis1=1, axis2=2) / 3.0

    ssm_site_acc = (np.zeros((n_sites, nbeta, nalpha), dtype=float)
                    if expand_ssm else None)
    f0_sites = np.zeros(n_sites, dtype=float)
    tbar_sites = np.zeros(n_sites, dtype=float)

    for site_idx in range(n_sites):
        # identical to contin(): start(), then (optionally) the expansion
        p_site, f0_site, tbar_site, deltab_site = start(
            rho_iso_sites[site_idx], rho_iso_sites.shape[1], delta_ev, tev,
            tbeta)
        if expand_ssm:
            _phonon_expand_from_t1(
                ssm_site_acc[site_idx], alpha, beta, nalpha, nbeta, lat, arat,
                tev, tbeta, nphon, p_site, f0_site, tbar_site, deltab_site)
        f0_sites[site_idx] = f0_site
        tbar_sites[site_idx] = tbar_site

    if expand_ssm:
        ssm_slice[:nbeta, :nalpha] = np.mean(ssm_site_acc, axis=0)
    f0 = float(np.mean(f0_sites))
    tbar = float(np.mean(tbar_sites))

    if iprint >= 1:
        print(f"    Cubicized phonopy DOS: f0={f0:.6f}, tbar={tbar:.4f}, "
              f"nsites={n_sites}")

    return f0, tbar, deltab

def besk1(x):
    """Modified Bessel function K1(x).

    For x <= 1 returns K1(x) directly. For x > 1 returns the EXP-SCALED value
    exp(x) * K1(x) (the exp(-x) asymptotic factor is omitted to avoid underflow at
    large x); a caller that needs the bare K1(x) multiplies by exp(-x). Mirrors
    NJOY's besk1 (leapr.f90)."""
    c_coeffs_small = [
        0.442850424, 0.584115288, 6.070134559, 17.864913364,
        48.858995315, 90.924600045, 113.795967431, 85.331474517,
        32.00008698, 3.999998802
    ]
    c_coeffs_small2 = [
        1.304923514, 1.47785657, 16.402802501, 44.732901977,
        115.837493464, 198.437197312, 222.869709703, 142.216613971,
        40.000262262, 1.999996391
    ]

    if x <= 1.0:
        v = 0.125 * x
        u = v * v
        # Horner evaluation: c1*u^9 + c2*u^8 + ... + c10
        bi1 = c_coeffs_small[0]
        for i in range(1, 10):
            bi1 = bi1 * u + c_coeffs_small[i]
        bi1 *= v

        # Horner evaluation: c11*u^9 + c12*u^8 + ... + c20
        bi3 = c_coeffs_small2[0]
        for i in range(1, 10):
            bi3 = bi3 * u + c_coeffs_small2[i]

        return 1.0 / x + bi1 * (log(0.5 * x) + 0.5772156649) - v * bi3
    else:
        u = 1.0 / x
        c_large = [
            0.0108241775, 0.0788000118, 0.2581303765, 0.5050238576,
            0.663229543, 0.6283380681, 0.4594342117, 0.2847618149,
            0.1736431637, 0.1280426636, 0.1468582957, 0.4699927013,
            1.2533141373
        ]
        # Horner: ((((((((((((-c25*u+c26)*u-c27)*u+c28)*u-c29)*u+c30)
        #   *u-c31)*u+c32)*u-c33)*u+c34)*u-c35)*u+c36)*u+c37)
        bi3 = -c_large[0] * u + c_large[1]
        bi3 = bi3 * u - c_large[2]
        bi3 = bi3 * u + c_large[3]
        bi3 = bi3 * u - c_large[4]
        bi3 = bi3 * u + c_large[5]
        bi3 = bi3 * u - c_large[6]
        bi3 = bi3 * u + c_large[7]
        bi3 = bi3 * u - c_large[8]
        bi3 = bi3 * u + c_large[9]
        bi3 = bi3 * u - c_large[10]
        bi3 = bi3 * u + c_large[11]
        bi3 = bi3 * u + c_large[12]
        return sqrt(u) * bi3

def stable(al, delta, c_diff, twt, iprint, ndmax):
    """Set up table of S-diffusion or S-free.

    Returns sd array and nsd (number of points).
    ``iprint`` is unused (LEAPR signature compatibility; see contin()).
    """
    eps = 1.0e-7
    # Full-size buffer, exactly as NJOY allocates sd(ndmax) in trans
    # (leapr.f90:881). A smaller buffer is unsafe: the loop below has no
    # other bound, and slowly-converging diffusion tables (small c, large
    # alpha) legitimately need several hundred thousand points.
    sd = np.zeros(ndmax)

    if c_diff != 0.0:
        # Diffusion branch
        d = twt * c_diff
        c2 = sqrt(c_diff * c_diff + 0.25)
        c3 = 2.0 * d * al
        c4 = c3 * c3
        c8 = c2 * c3 / pi
        c3_new = 2.0 * d * c_diff * al
        be = 0.0
        j = 0
        idone = False
        while not idone:
            c6 = sqrt(be * be + c4)
            c7 = c6 * c2
            if c7 <= 1.0:
                c5 = c8 * exp(c3_new + be / 2.0)
            else:
                ex = c3_new - c7 + be / 2.0
                c5 = c8 * exp(ex)
            sd[j] = c5 * besk1(c7) / c6
            be += delta
            j += 1
            # NJOY checks after the post-increment: with its 1-based count
            # jN = j + 1, the gates are mod(jN,2)==0 (count odd) and
            # jN >= ndmax, i.e. count >= ndmax - 1 — so the last written
            # index is at most ndmax - 2 and sd(ndmax) is never exceeded.
            if j % 2 == 1:
                if j >= ndmax - 1:
                    idone = True
                if eps * sd[0] >= sd[j - 1]:
                    idone = True
        nsd = j
    else:
        # Free-gas branch
        be = 0.0
        j = 0
        wal = twt * al
        idone = False
        while not idone:
            ex = -(wal - be)**2 / (4.0 * wal)
            sfree = exp(ex) / sqrt(4.0 * pi * wal)
            sd[j] = sfree
            be += delta
            j += 1
            # Same NJOY-exact gates as the diffusion branch above.
            if j % 2 == 1:
                if j >= ndmax - 1:
                    idone = True
                if eps * sd[0] >= sd[j - 1]:
                    idone = True
        nsd = j

    return sd, nsd

def trans(ssm_slice, alpha, beta, nalpha, nbeta, lat, arat, tev,
          twt, c_diff, tbeta, f0, deltab, tbar, iprint):
    """Add translational contribution to S(alpha,beta).

    Modifies ssm_slice in-place.
    ``iprint`` is only forwarded to stable(), which ignores it (LEAPR
    signature compatibility; see contin()).
    """
    tiny = 1.0e-30
    slim = -225.0
    shade = 1.00001
    sc = 1.0
    if lat == 1:
        sc = THERM / tev

    c0, c1_t, c2_t, c3_t, c4_t = 0.4, 1.0, 1.42, 0.2, 10.0
    ndmax = max(nbeta, 1000000)
    betan = beta[:nbeta] * sc
    shade_last = shade * betan[nbeta - 1]

    for ialpha in range(nalpha):
        al = alpha[ialpha] * sc / arat

        # Choose beta interval for convolution
        ded = c0 * (twt * c_diff * al) / sqrt(c1_t + c2_t * (twt * c_diff * al) * c_diff)
        if ded == 0.0:
            ded = c3_t * sqrt(twt * al)
        deb = c4_t * al * deltab
        delta = min(deb, ded)

        # Make table of s-diffusion or s-free
        sd, nsd = stable(al, delta, c_diff, twt, iprint, ndmax)

        if nsd > 1:
            ap = ssm_slice[:nbeta, ialpha].copy()
            nbt = nsd
            n_pts = 2 * nbt - 1

            # Precompute log(ap) and interpolation slopes ONCE per alpha
            log_ap = np.full(nbeta, slim)
            pos_mask = ap > 0.0
            log_ap[pos_mask] = np.log(ap[pos_mask])
            # slope[k] = (log_ap[k] - log_ap[k+1]) / (betan[k] - betan[k+1])
            slope = (log_ap[:-1] - log_ap[1:]) / (betan[:-1] - betan[1:])

            # Simpson's weights * sd
            i_arr = np.arange(nbt)
            f_arr = 2.0 * ((i_arr % 2) + 1).astype(float)
            f_arr[0] = 1.0
            if nbt > 1:
                f_arr[-1] = 1.0
            fsd = f_arr * sd[:nbt]
            fsd_bwd = fsd * np.exp(-i_arr * delta)

            exp_alf0 = exp(-al * f0)
            j_grid = np.arange(n_pts, dtype=np.float64) * delta
            nbt_m1_delta = (nbt - 1) * delta
            delta_nsd = delta * nsd

            # Pre-allocate buffer for inner loop
            sb = np.empty(n_pts)

            # Bracket search and log-interp batched over beta CHUNKS: one
            # searchsorted per chunk instead of one per beta (the per-call
            # dispatch dominated trans). Byte-identity is preserved by
            # construction: searchsorted returns exact integers regardless
            # of batching, the 2-D arithmetic preserves the per-element IEEE
            # op order, and np.exp / np.dot operate per-row (contiguous row
            # views of equal length), so each row's inner loop sees exactly
            # the scalar per-beta operands.
            bmin_arr = -betan - nbt_m1_delta
            chunk = max(1, 2_000_000 // n_pts)
            for i0 in range(0, nbeta, chunk):
                i1 = min(i0 + chunk, nbeta)
                bet2 = bmin_arr[i0:i1, None] + j_grid[None, :]
                b2 = np.abs(bet2)

                j2 = np.searchsorted(betan, b2.ravel(), side='right')
                j2 = j2.reshape(b2.shape)
                out2 = (j2 >= nbeta) & (b2 >= shade_last)
                np.clip(j2, 1, nbeta - 1, out=j2)

                # ((b - betan[j]) * slope[j-1]) + log_ap[j], per element
                sbv2 = np.subtract(b2, betan[j2], out=b2)
                np.multiply(sbv2, slope[j2 - 1], out=sbv2)
                np.add(sbv2, log_ap[j2], out=sbv2)
                np.subtract(sbv2, bet2, out=sbv2, where=bet2 > 0.0)
                good2 = (~out2) & (sbv2 > slim)

                for ibeta in range(i0, i1):
                    be = betan[ibeta]
                    r = ibeta - i0
                    sb[:] = 0.0
                    np.exp(sbv2[r], where=good2[r], out=sb)

                    # Convolution via dot products
                    s = (np.dot(fsd, sb[nbt - 1:2 * nbt - 1])
                         + np.dot(fsd_bwd, sb[nbt - 1::-1]))
                    s *= delta / 3.0
                    if s < tiny:
                        s = 0.0

                    # terps: interpolate in sd table
                    if be <= delta_nsd:
                        i_idx = int(be / delta)
                        if i_idx < nsd - 1:
                            bt = i_idx * delta
                            sd_i = sd[i_idx]
                            sd_ip1 = sd[i_idx + 1]
                            log_sd_i = log(sd_i) if sd_i > 0.0 else slim
                            log_sd_ip1 = log(sd_ip1) if sd_ip1 > 0.0 else slim
                            stt = log_sd_i + (be - bt) * (log_sd_ip1 - log_sd_i) / delta
                            st = exp(stt) if stt > slim else 0.0
                            if st > 0.0:
                                s += exp_alf0 * st
                                # DELIBERATE NJOY DIVERGENCE: NJOY2016 clamps
                                # only the convolution part (leapr.f90:1041,
                                # mirrored at the s *= delta/3 clamp above) and
                                # writes any self-term sum down to smin=1e-75
                                # via endout. IRMA re-clamps the combined
                                # convolution + exp(-al*f0)*st self term below
                                # tiny=1e-30, so a self term landing in
                                # (1e-75, 1e-30) — al*f0 > ~70 at high alpha —
                                # is zeroed where NJOY keeps it. All expected
                                # decks reproduce their NJOY references to
                                # ~1e-4 with the clamp active.
                                if s < tiny:
                                    s = 0.0

                    ssm_slice[ibeta, ialpha] = s

def bfact(x, dwc, betai):
    """Calculate Bessel function terms for discrete oscillators.

    Returns bzero, bplus[50], bminus[50].
    """
    big = 1.0e10
    tiny = 1.0e-30
    imax = 50

    # Compute bessi0
    y = x / 3.75
    if y <= 1.0:
        u = y * y
        bessi0 = (1.0 + u * (3.5156229 + u * (3.0899424 + u * (1.2067492 +
                  u * (0.2659732 + u * (0.0360768 + u * 0.0045813))))))
    else:
        v = 1.0 / y
        bessi0 = (0.39894228 + v * (0.01328592 + v * (0.00225319 +
                  v * (-0.00157565 + v * (0.00916281 + v * (-0.02057706 +
                  v * (0.02635537 + v * (-0.01647633 + v * 0.00392377)))))))) / sqrt(x)

    # Compute bessi1
    if y <= 1.0:
        u = y * y
        bessi1 = (0.5 + u * (0.87890594 + u * (0.51498869 + u * (0.15084934 +
                  u * (0.02658733 + u * (0.00301532 + u * 0.00032411)))))) * x
    else:
        v = 1.0 / y
        bessi1 = 0.02282967 + v * (-0.02895312 + v * (0.01787654 - v * 0.00420059))
        bessi1 = (0.39894228 + v * (-0.03988024 + v * (-0.00362018 +
                  v * (0.00163801 + v * (-0.01031555 + v * bessi1))))) / sqrt(x)

    # Generate higher orders by reverse recursion. Overflow to inf/NaN is
    # expected for extreme x and filtered just below (matching the Fortran
    # propagate-then-compare behavior); errstate only silences the warnings.
    # "divide" is included for the x -> 0 limit (ar underflows to 0 at
    # cryogenic T), where 2.0/x -> inf propagates and is filtered the same way.
    bn = np.zeros(imax)
    bn[imax - 1] = 0.0
    bn[imax - 2] = 1.0
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        i = imax - 2
        while i > 0:
            bn[i - 1] = bn[i + 1] + (i + 1) * (2.0 / x) * bn[i]
            i -= 1
            if bn[i] >= big:
                bn[i:] /= big

        rat = bessi1 / bn[0]
        bn *= rat
    # Clean up inf/NaN from recursion overflow (matches Fortran behavior where
    # such values propagate but are filtered by comparison checks)
    bn[~np.isfinite(bn)] = 0.0
    bn[bn < tiny] = 0.0

    # Apply exponential terms.
    #
    # DELIBERATE NJOY DIVERGENCE: the arg > 709 guards below zero terms whose
    # exponential alone would overflow float64. NJOY2016 has NO guard here —
    # exp(709+) gives Inf, and Inf * an underflowed bn(i)=0 gives NaN in the
    # law. Since bn[i] < 1e-30 is already zeroed (both codes), any term the
    # guard suppresses would have computed as >= exp(709)*1e-30 ~ 1e278 —
    # numerically meaningless for a sum-rule-bounded S(alpha,beta). The
    # condition only arises for high-energy oscillators at cryogenic
    # temperatures (e.g. a 0.2 eV oscillator below ~30 K at order i >~ 18).
    bplus = np.zeros(imax)
    bminus = np.zeros(imax)

    if y <= 1.0:
        bzero = bessi0 * exp(-dwc)
        for i in range(imax):
            if bn[i] != 0.0:
                arg = -dwc - (i + 1) * betai / 2.0
                if arg > 709.0:
                    bplus[i] = 0.0
                else:
                    bplus[i] = exp(arg) * bn[i]
                    if bplus[i] < tiny:
                        bplus[i] = 0.0
                arg = -dwc + (i + 1) * betai / 2.0
                if arg > 709.0:
                    bminus[i] = 0.0
                else:
                    bminus[i] = exp(arg) * bn[i]
                    if bminus[i] < tiny:
                        bminus[i] = 0.0
    else:
        bzero = bessi0 * exp(-dwc + x)
        for i in range(imax):
            if bn[i] != 0.0:
                arg = -dwc - (i + 1) * betai / 2.0 + x
                if arg > 709.0:
                    bplus[i] = 0.0
                else:
                    bplus[i] = exp(arg) * bn[i]
                    if bplus[i] < tiny:
                        bplus[i] = 0.0
                arg = -dwc + (i + 1) * betai / 2.0 + x
                if arg > 709.0:
                    bminus[i] = 0.0
                else:
                    bminus[i] = exp(arg) * bn[i]
                    if bminus[i] < tiny:
                        bminus[i] = 0.0

    return bzero, bplus, bminus

def bfill(betan, nbeta):
    """Set up bex and rdbex arrays for sint.

    Creates the extended beta grid [-beta_max, ..., 0, ..., +beta_max].
    """
    small = 1.0e-9

    bex = []
    # Negative betas (reversed)
    for i in range(nbeta - 1, -1, -1):
        bex.append(-betan[i])

    # Handle beta=0 case
    if betan[0] <= small:
        bex[-1] = 0.0
        k = nbeta
    else:
        bex.append(betan[0])
        k = nbeta + 1

    # Positive betas
    for i in range(1, nbeta):
        bex.append(betan[i])
        k += 1

    bex = np.array(bex)
    nbx = len(bex)
    rdbex = np.zeros(nbx - 1)
    for i in range(nbx - 1):
        rdbex[i] = 1.0 / (bex[i + 1] - bex[i])

    return bex, rdbex, nbx

def exts(sexpb, exb, betan, nbeta):
    """Set up the sex array for sint.

    Extends asymmetric sab to plus and minus beta.
    """
    small = 1.0e-9
    sex = []

    # Negative beta side (reversed)
    for i in range(nbeta - 1, -1, -1):
        sex.append(sexpb[i])

    if betan[0] <= small:
        sex[-1] = sexpb[0]
        k = nbeta
    else:
        sex.append(sexpb[0])
        k = nbeta + 1

    # Positive beta side
    for i in range(1, nbeta):
        sex.append(sexpb[i] * exb[i] * exb[i])
        k += 1

    return np.array(sex)

def sint(x, bex, rdbex, sex, nbx, alph, wt, tbart, betan, nbeta):
    """Interpolate in scattering function or use SCT approximation.

    Semantic REFERENCE for the vectorized forms (sint_vec in discre,
    _sint_batch_exact in coldh); tests/test_sint_batch.py pins bit
    identity against this scalar, so keep them in lockstep.
    """
    slim = -225.0

    # SCT approximation for |x| > beta_max
    if abs(x) > betan[nbeta - 1]:
        if alph <= 0.0:
            return 0.0
        ex = -(wt * alph - abs(x))**2 / (4.0 * wt * alph * tbart)
        if x > 0.0:
            ex -= x
        # DELIBERATE NJOY DIVERGENCE: this SCT prefactor divides by
        # 4*pi*wt*alph*tbart where the Gaussian normalization -- and every
        # sibling SCT/free-gas expression in this module (contin SCT at
        # exp(ex)/sqrt(4*pi*alp), stable free gas at
        # exp(ex)/sqrt(4*pi*wal)) -- divides by the SQUARE ROOT of that
        # quantity. The missing sqrt is NJOY2016's (leapr.f90:1892,
        # sv=exp(ex)/(4*pi*wt*alph*tbart)) and is reproduced deliberately
        # for byte parity with the reference tapes. It affects only the
        # SCT tail reached from discre and coldh when a discrete
        # oscillator or rotational shift pushes |beta| past beta_max.
        # The two vector copies (sint_vec, _sint_batch_exact) carry the
        # same divergence; keep all three in lockstep.
        return exp(ex) / (4.0 * pi * wt * alph * tbart)

    # Binary search for x in bex
    k1 = 0
    k2 = nbeta - 1
    k3 = nbx - 1

    idone = False
    while not idone:
        if x == bex[k2]:
            return sex[k2]
        elif x > bex[k2]:
            k1 = k2
            k2 = (k3 - k2) // 2 + k2
            if k3 - k1 <= 1:
                idone = True
        else:
            k3 = k2
            k2 = (k2 - k1) // 2 + k1
            if k3 - k1 <= 1:
                idone = True

    if sex[k1] <= 0.0:
        ss1 = slim
    else:
        ss1 = log(sex[k1])
    if sex[k3] <= 0.0:
        ss3 = slim
    else:
        ss3 = log(sex[k3])

    ex = ((bex[k3] - x) * ss1 + (x - bex[k1]) * ss3) * rdbex[k1]
    if ex > slim:
        return exp(ex)
    return 0.0

def sint_vec(x_arr, bex, rdbex, sex, nbx, alph, wt, tbart, betan, nbeta):
    """Vectorized interpolation in scattering function or SCT approximation."""
    slim = -225.0
    result = np.zeros(len(x_arr))
    beta_max = betan[nbeta - 1]

    # SCT approximation for |x| > beta_max
    sct_mask = np.abs(x_arr) > beta_max
    if np.any(sct_mask) and alph > 0.0:
        xs = x_arr[sct_mask]
        ex = -(wt * alph - np.abs(xs))**2 / (4.0 * wt * alph * tbart)
        ex = np.where(xs > 0.0, ex - xs, ex)
        # DELIBERATE NJOY DIVERGENCE: no sqrt on the 4*pi*wt*alph*tbart
        # SCT prefactor, faithfully reproducing NJOY2016 leapr.f90:1892
        # for byte parity -- see the full note in sint().
        result[sct_mask] = np.exp(ex) / (4.0 * pi * wt * alph * tbart)

    # Interpolation for |x| <= beta_max
    interp_mask = ~sct_mask
    if np.any(interp_mask):
        xs = x_arr[interp_mask]
        # Use searchsorted to find k3 (upper bracket index in bex)
        k3 = np.searchsorted(bex[:nbx], xs, side='right')
        k3 = np.clip(k3, 1, nbx - 1)
        k1 = k3 - 1

        # log(0) -> -inf is masked to slim by the where(); errstate only
        # silences the divide-by-zero warning the masked branch emits.
        with np.errstate(divide="ignore"):
            ss1 = np.where(sex[k1] > 0.0, np.log(sex[k1]), slim)
            ss3 = np.where(sex[k3] > 0.0, np.log(sex[k3]), slim)

        ex = ((bex[k3] - xs) * ss1 + (xs - bex[k1]) * ss3) * rdbex[k1]
        result[interp_mask] = np.where(ex > slim, np.exp(ex), 0.0)

    return result

def discre(ssm_slice, alpha, beta, nalpha, nbeta, lat, arat, tev,
           twt, tbeta, nd, bdel, adel, dwpix_val, tempf_val, tempr_val,
           iprint):
    """Convolve discrete oscillators with continuous S(alpha,beta).

    Modifies ssm_slice in-place.
    Returns: updated dwpix, updated tempf
    ``iprint`` is unused (LEAPR signature compatibility; see contin()).
    """
    small = 1.0e-8
    vsmall = 1.0e-10
    tiny = 1.0e-20
    maxdd = 500

    sc = 1.0
    if lat == 1:
        sc = THERM / tev

    # Set up oscillator parameters
    bdeln = np.zeros(nd)
    eb = np.zeros(nd)
    ar = np.zeros(nd)
    dbw = np.zeros(nd)
    dist = np.zeros(nd)
    dwt = 0.0

    for i in range(nd):
        bdeln[i] = bdel[i] / tev
        dwt += adel[i]

    tsave = 0.0
    dw0 = dwpix_val

    # errstate: in the cryogenic regime (bdeln/2 clamped at the exp ceiling by
    # _safe_exp) the product sn*bdeln can overflow to inf, and ar = adel/inf
    # = 0.0 is exactly the cold limit -- the values are correct, only the
    # RuntimeWarning is noise. Suppressing warnings changes no bits.
    with np.errstate(over="ignore"):
        for i in range(nd):
            # A high-energy oscillator at cryogenic T gives bdeln/2 > 709; clamp so
            # the cold limit is reproduced (cn/sn = coth(bdeln/2) -> 1) instead of
            # raising OverflowError (bare exp) or yielding 0*Inf = NaN (Fortran Inf).
            eb[i] = _safe_exp(bdeln[i] / 2.0)
            sn = (eb[i] - 1.0 / eb[i]) / 2.0
            cn = (eb[i] + 1.0 / eb[i]) / 2.0
            ar[i] = adel[i] / (sn * bdeln[i])
            dist[i] = adel[i] * bdel[i] * cn / (2.0 * sn)
            tsave += dist[i] / BK
            dbw[i] = ar[i] * cn
            if dwpix_val > 0.0:
                dwpix_val += dbw[i]

    # Prepare functions of beta
    betan = beta[:nbeta] * sc
    exb = np.exp(-betan / 2.0)

    bex, rdbex, nbx = bfill(betan, nbeta)

    # Main alpha loop
    for nal in range(nalpha):
        al = alpha[nal] * sc / arat
        dwf = exp(-al * dw0)
        sex = exts(ssm_slice[:, nal], exb, betan, nbeta)
        sexpb = np.zeros(nbeta)

        # Initialize delta function calculation
        ben = np.zeros(maxdd)
        wtn = np.zeros(maxdd)
        ben[0] = 0.0
        wtn[0] = 1.0
        nn = 1
        n = 0

        bes = np.zeros(maxdd)
        wts = np.zeros(maxdd)

        # DELIBERATE NJOY DIVERGENCE: the SCT effective-temperature ratio is
        # reset for every alpha here. NJOY2016 initialises tbart once before
        # the alpha loop (leapr.f90:1401) and accumulates the oscillator
        # contributions across the WHOLE grid (leapr.f90:1494, never reset),
        # so its tbart grows linearly with the alpha INDEX — by the last
        # alpha of a 40-point grid the SCT tails change support entirely
        # (313/512 tape lines, up to 100% on tail values, in a 2-oscillator
        # probe), and merely adding alpha points changes the answer. That is
        # a latent NJOY bug; T_eff/T is a property of the spectrum, not of
        # the grid position. Expected decks (no SCT-tail exposure) match NJOY
        # references to ~1e-4 either way.
        tbart_local = tempf_val / tempr_val

        # Loop over all oscillators
        for i_osc in range(nd):
            dwc = al * dbw[i_osc]
            x = al * ar[i_osc]
            bzero, bplus, bminus = bfact(x, dwc, bdeln[i_osc])

            n = 0

            # n=0 term
            for m in range(nn):
                besn = ben[m]
                wtsn = wtn[m] * bzero
                if besn <= 0.0 or wtsn >= small:
                    if n < maxdd:
                        bes[n] = besn
                        wts[n] = wtsn
                        n += 1

            # Negative n terms
            k = 0
            idone = False
            while k < 50 and not idone:
                if bminus[k] <= 0.0:
                    idone = True
                else:
                    for m in range(nn):
                        besn = ben[m] - (k + 1) * bdeln[i_osc]
                        wtsn = wtn[m] * bminus[k]
                        if wtsn >= small and n < maxdd:
                            bes[n] = besn
                            wts[n] = wtsn
                            n += 1
                k += 1

            # Positive n terms
            k = 0
            idone = False
            while k < 50 and not idone:
                if bplus[k] <= 0.0:
                    idone = True
                else:
                    for m in range(nn):
                        besn = ben[m] + (k + 1) * bdeln[i_osc]
                        wtsn = wtn[m] * bplus[k]
                        if wtsn >= small and n < maxdd:
                            bes[n] = besn
                            wts[n] = wtsn
                            n += 1
                k += 1

            # Update for next oscillator
            nn = n
            ben[:nn] = bes[:nn].copy()
            wtn[:nn] = wts[:nn].copy()
            n = 0
            tbart_local += dist[i_osc] / BK / tempr_val

        n = nn

        # Sort discrete lines by weight (descending), skip element 0 (zero state)
        # Fortran sorts elements 2..n (1-based), equivalent to 1..n-1 (0-based)
        #
        # TIE-ORDER NOTE: NJOY discre uses a selection-style sort that DOES swap
        # on equal weights, giving a deterministic order for exact ties.
        # np.argsort(-w) (default unstable quicksort) can permute tied-weight
        # entries differently. This is accepted: the sorted WEIGHT sequence is
        # bit-identical to NJOY's (verified over thousands of tie-heavy cases),
        # so the weight-based trim below cuts at the same place and keeps the
        # same set; only the associated `bes` values may be permuted WITHIN an
        # exact-weight group. Surviving lines are summed into sexpb by
        # commutative addition and placed as delta functions by value, so the
        # sole effect of a within-tie permutation is sub-sigfig FP reassociation.
        # Real oscillator decks (e.g. CH4/CH2) produce near-distinct weights, so
        # the byte-exact tapes are unaffected. A plain stable descending sort
        # would NOT reproduce NJOY's swap-on-tie order, so it is not used here.
        if n > 2:
            wts_sub = wts[1:n].copy()
            bes_sub = bes[1:n].copy()
            sort_idx = np.argsort(-wts_sub)
            wts[1:n] = wts_sub[sort_idx]
            bes[1:n] = bes_sub[sort_idx]

        # Trim small entries (Fortran: nn=n-1, loop i=1..nn, n=i)
        # Always drops at least the last (smallest) element
        # When nn=0, the Fortran loop doesn't execute and n stays unchanged
        nn = n - 1
        if nn > 0:
            n_new = nn
            for i in range(nn):  # 0-based, maps to Fortran i=1..nn
                n_new = i + 1
                if wts[i] < 100 * small and (i + 1) > 5:
                    break
            n = n_new

        # Add the continuum part to the scattering law (vectorized over beta)
        for m in range(n):
            be_arr = -betan - bes[m]
            st_arr = sint_vec(be_arr, bex, rdbex, sex, nbx, al, tbeta + twt,
                             tbart_local, betan, nbeta)
            add_arr = wts[m] * st_arr
            add_arr[add_arr < tiny] = 0.0
            sexpb += add_arr

        # Add the delta functions to the scattering law
        # DELIBERATE NJOY DIVERGENCE (reported: njoy/NJOY2016#402):
    # NJOY2016 (leapr.f90:1549-1585) reuses
        # the variable `idone` for both the outer line loop and the inner
        # nearest-grid-point search; the inner search always sets idone=1 for
        # any in-range line, so NJOY adds only the FIRST in-range negative
        # delta line (highest weight after the descending sort) and then
        # terminates the line loop -- almost certainly an unintended
        # restructuring bug, like the THERMR cliq guard (docs/njoy.md). IRMA
        # uses a separate inner flag (`done2`) and convolves EVERY in-range
        # negative line, which is the physically complete treatment. This
        # path is reached only by twt=0 decks with discrete oscillators
        # (none of the byte-exact decks: all nd>0 reference decks have twt>0), so
        # tapes from such decks will NOT match NJOY at the delta-line betas;
        # tests/test_kernels_stable.py pins the all-lines behavior.
        if twt <= 0.0:
            m = 0
            idone = False
            while m < n and not idone:
                if dwf < vsmall:
                    idone = True
                else:
                    if bes[m] < 0.0:
                        be = -bes[m]
                        if be <= betan[nbeta - 2]:
                            # db is the running minimum |be - betan[j]|; the
                            # search exits at the first j whose distance
                            # exceeds it, so the seed must compare greater
                            # than any real distance. NJOY seeds db=1000
                            # (leapr.f90:1562), which a cryogenic beta grid
                            # (beta_max = 5 eV/kT ~ 2900 at 20 K) overruns:
                            # a line at be > 1000 ended the search at j=0,
                            # giving wts[m]/betan[0] (ZeroDivisionError for
                            # betan[0]=0) or a silent deposit via sexpb[-1].
                            # A true infinity makes the first comparison
                            # always record; grids with beta_max <= 1000 are
                            # bit-identical either way. This block is inside
                            # the njoy/NJOY2016#402 divergence above and is
                            # unreachable by the byte-exact reference decks.
                            db = inf
                            done2 = False
                            j = 0
                            jj = 0
                            while j < nbeta and not done2:
                                jj = j
                                if abs(be - betan[j]) > db:
                                    done2 = True
                                else:
                                    db = abs(be - betan[j])
                                j += 1

                            if jj <= 1:
                                add = wts[m] / betan[jj]
                            else:
                                add = 2.0 * wts[m] / (betan[jj] - betan[jj - 2])
                            add *= dwf
                            if add >= tiny:
                                sexpb[jj - 1] += add
                m += 1

        # Record results
        ssm_slice[:, nal] = sexpb

    # Update effective temperature and Debye-Waller
    tempf_new = (tbeta + twt) * tempf_val + tsave

    return dwpix_val, tempf_new

def sjbes(n, x):
    """Spherical Bessel functions for cold hydrogen calculation."""
    if x < 0.0 or n < 0:
        return 0.0

    if n >= 30000 or x > 3.0e4:
        return 0.0

    huge = 1.0e25
    small_val = 2.0e-38

    if x <= 7.0e-4:
        w = 1.0
        if n == 0:
            return w
        elif n > 10:
            return 0.0
        else:
            t1 = 3.0
            t2 = 1.0
            for i in range(n):
                t3 = t2 * x / t1
                t1 += 2.0
                t2 = t3
            return t3

    if x < 0.2:
        y = x * x
        w = 1.0 - y * (1.0 - y / 20.0) / 6.0
    else:
        w = sin(x) / x

    if n == 0:
        return w

    if x >= 100.0:
        l = int(x / 50.0 + 18)
    elif x >= 10.0:
        l = int(x / 10.0 + 10)
    elif x > 1.0:
        l = int(x / 2.0 + 5)
    else:
        l = 5

    iii = int(x)
    kmax = max(n, iii)
    nm = kmax + l
    z = 1.0 / x
    t3 = 0.0
    t2 = small_val
    sj = 0.0

    for i in range(nm, 0, -1):
        k = i - 1
        t1 = (2 * k + 3) * z * t2 - t3
        if n == k:
            sj = t1
        if abs(t1) >= huge:
            t1 /= huge
            t2 /= huge
            sj /= huge
        t3 = t2
        t2 = t1

    return w * sj / t1

@lru_cache(maxsize=None)
def cn_cg(jj, ll, nn):
    """Clebsch-Gordon coefficients for cold hydrogen.

    Pure function of three small integers (a few dozen distinct values per
    run); cached so the log-factorial machinery runs once per triple.
    """
    kdet = (jj + ll + nn) // 2
    kdel = jj + ll + nn - 2 * kdet

    if kdel != 0:
        return 0.0

    ka1 = jj + ll + nn
    ka2 = jj + ll - nn
    ka3 = jj - ll + nn
    ka4 = ll - jj + nn

    # NOTE: NJOY's cn (leapr.f90) has NO out-of-triangle guard here. When any of
    # ka2/ka3/ka4 is negative its Fortran factorial do-loop simply does not run,
    # leaving that factorial factor = 1 (a_x = sqrt(1) = 1, b_x = 1). Our
    # log_factorial returns 0.0 for n<=0 (empty range) and the `if lf > 0.0 else
    # 1.0` guards below already collapse those factors to 1, so the natural
    # computation reproduces NJOY's nonzero value without a short-circuit. The
    # only call pattern (sumh: cn_cg(j,jp,n) with n in [|j-jp|, j+jp], plus
    # cn_cg(0,jp,jp) and cn_cg(j,0,j)) keeps ka2/ka3/ka4 >= 0 by the triangle
    # inequality, so this path is exercised only for hypothetical out-of-triangle
    # arguments and is byte-identical for every reachable triple.
    kb1 = ka1 // 2
    kb2 = ka2 // 2
    kb3 = ka3 // 2
    kb4 = ka4 // 2

    def log_factorial(n):
        """ln(n!) by direct summation, matching LEAPR's accumulation order."""
        s = 0.0
        for i in range(1, n + 1):
            s += log(float(i))
        return s

    lf_ka1 = log_factorial(ka1)
    lf_ka2 = log_factorial(ka2)
    lf_ka3 = log_factorial(ka3)
    lf_ka4 = log_factorial(ka4)
    lf_kb1 = log_factorial(kb1)
    lf_kb2 = log_factorial(kb2)
    lf_kb3 = log_factorial(kb3)
    lf_kb4 = log_factorial(kb4)

    a1 = exp(lf_ka1 / 2.0) if lf_ka1 > 0.0 else 1.0
    a2 = exp(lf_ka2 / 2.0) if lf_ka2 > 0.0 else 1.0
    a3 = exp(lf_ka3 / 2.0) if lf_ka3 > 0.0 else 1.0
    a4 = exp(lf_ka4 / 2.0) if lf_ka4 > 0.0 else 1.0
    b1 = exp(lf_kb1) if lf_kb1 > 0.0 else 1.0
    b2 = exp(lf_kb2) if lf_kb2 > 0.0 else 1.0
    b3 = exp(lf_kb3) if lf_kb3 > 0.0 else 1.0
    b4 = exp(lf_kb4) if lf_kb4 > 0.0 else 1.0

    rat = (2 * nn + 1) / (jj + ll + nn + 1)
    iwign = (jj + ll - nn) // 2
    wign = (-1)**iwign * sqrt(rat) * b1 / a1 * a2 / b2 * a3 / b3 * a4 / b4

    return wign

def sumh(j, jp, y):
    """Sum over Bessel functions and Clebsch-Gordon coefficients."""
    if j == 0:
        return (sjbes(jp, y) * cn_cg(j, jp, jp))**2
    elif jp == 0:
        return (sjbes(j, y) * cn_cg(j, 0, j))**2
    else:
        sum1 = 0.0
        # Bessel orders |j-jp| .. j+jp, truncated to at most 10 terms
        # (orders |j-jp| .. |j-jp|+9), exactly as NJOY's sumh
        # (leapr.f90: imk..ipk with n1=n-1). The truncation branch is
        # unreachable for cold H2/D2 (coldh's jterm=3 keeps
        # min(j,jp) <= 3 while truncation needs >= 5), so the
        # ortho/para-H2 expected tapes are insensitive to it.
        imk = abs(j - jp)
        top = j + jp
        if top - imk > 9:
            top = imk + 9
        for n in range(imk, top + 1):
            sum1 += (sjbes(n, y) * cn_cg(j, jp, n))**2
        return sum1

def bt_stat(j, x):
    """Statistical weight factor for cold hydrogen/deuterium."""
    yy = 0.5 * j * (j + 1)
    a = (2 * j + 1) * exp(-yy * x)
    b = 0.0
    for i in range(10):
        k = 2 * i
        if j % 2 == 1:
            k += 1
        yy = 0.5 * k * (k + 1)
        b += (2 * k + 1) * exp(-yy * x)
    return a / (2.0 * b)

def terpk(ska, nka, delta, be):
    """Interpolate in ska(kappa) table."""
    if be > nka * delta:
        return 1.0
    i = int(be / delta)
    if i < nka - 1:
        bt_val = i * delta
        btp = bt_val + delta
        return ska[i] + (be - bt_val) * (ska[i + 1] - ska[i]) / (btp - bt_val)
    return 1.0

def _sint_batch_exact(x_arr, bex, rdbex, sex, log_sex, nbx, alph, wt,
                      tbart, beta_max):
    """Vector evaluation of sint() with BIT-IDENTICAL scalar semantics.

    Reproduces sint() exactly: the SCT branch beyond beta_max, the raw
    table value on an exact interior grid hit (sint's bisection probes
    every interior index, so equality returns sex[k] untransformed,
    while the two grid ENDPOINTS fall through to the log-interpolation),
    and the log-linear interpolation itself. All transcendentals go
    through math.log/math.exp on the same scalar values (log_sex is the
    precomputed per-point math.log of sex), and the bracket arithmetic
    is element-wise IEEE — so every output float equals the scalar
    sint() result bit for bit.
    """
    slim = -225.0
    n = len(x_arr)
    out = np.zeros(n)
    abs_x = np.abs(x_arr)

    sct = abs_x > beta_max
    if np.any(sct) and alph > 0.0:
        idx = np.flatnonzero(sct)
        # Exact scalar op order: 4.0*wt*alph*tbart and 4.0*pi*wt*alph*tbart
        # DELIBERATE NJOY DIVERGENCE: denom_r carries no sqrt, faithfully
        # reproducing NJOY2016 leapr.f90:1892 for byte parity -- see the
        # full note in sint().
        denom_e = 4.0 * wt * alph * tbart
        denom_r = 4.0 * pi * wt * alph * tbart
        for i in idx:
            ax = abs_x[i]
            ex = -(wt * alph - ax) ** 2 / denom_e
            if x_arr[i] > 0.0:
                ex -= x_arr[i]
            out[i] = exp(ex) / denom_r

    interp = np.flatnonzero(~sct)
    if len(interp) == 0:
        return out
    xs = x_arr[interp]

    k3 = np.searchsorted(bex[:nbx], xs, side='right')
    np.clip(k3, 1, nbx - 1, out=k3)
    k1 = k3 - 1

    # Exact INTERIOR grid hits return the raw table value (matching the
    # bisection's early exit); endpoint hits interpolate like the scalar.
    left = np.searchsorted(bex[:nbx], xs, side='left')
    hit = (left < nbx) & (bex[np.minimum(left, nbx - 1)] == xs)
    hit &= (left >= 1) & (left <= nbx - 2)

    ss1 = log_sex[k1]
    ss3 = log_sex[k3]
    ex_arr = ((bex[k3] - xs) * ss1 + (xs - bex[k1]) * ss3) * rdbex[k1]

    vals = np.zeros(len(xs))
    for j in range(len(xs)):
        if hit[j]:
            vals[j] = sex[left[j]]
        elif ex_arr[j] > slim:
            vals[j] = exp(ex_arr[j])
    out[interp] = vals
    return out


def coldh(ssm_slice, ssp_slice, alpha, beta, nalpha, nbeta, lat, arat, tev,
          twt, tbeta, ncold, ska, nka, dka, tempf_val, tempr_val, iprint):
    """Convolve S(alpha,beta) with rotational modes for cold H2/D2.

    Modifies ssm_slice and ssp_slice in-place.
    ``iprint`` is unused (LEAPR signature compatibility; see contin()).
    """
    pmass = 1.6726231e-24
    dmass = 3.343586e-24
    deh = 0.0147
    ded = 0.0074
    sampch = 0.356
    sampcd = 0.668
    sampih = 2.526
    sampid = 0.403
    angst = 1.0e-8
    jterm = 3

    sc = 1.0
    if lat == 1:
        sc = THERM / tev

    law = ncold + 1
    de = deh
    if law > 3:
        de = ded
    x = de / tev

    if law > 3:
        amassm = 6.69e-24
        sampc = sampcd
        bp = HBAR / 2.0 * sqrt(2.0 / ded / EV / dmass) / angst
        sampi = sampid
    else:
        amassm = 3.3464e-24
        sampc = sampch
        bp = HBAR / 2.0 * sqrt(2.0 / deh / EV / pmass) / angst
        sampi = sampih

    wt = twt + tbeta
    tbart = tempf_val / tempr_val

    # Prepare arrays
    betan = beta[:nbeta].copy()
    if lat == 1:
        betan *= THERM / tev
    exb = np.exp(-betan / 2.0)
    bex, rdbex, nbx = bfill(betan, nbeta)

    for nal in range(nalpha):
        al = alpha[nal] * sc / arat
        # NJOY's alp=wt*al (leapr.f90:2020) feeds only the Young-Koppel
        # free-gas branches gated on ifree==1, which is hardwired to 0 in
        # NJOY2016 and not ported here.
        waven = angst * sqrt(amassm * tev * EV * al) / HBAR
        y = bp * waven

        sk = terpk(ska, nka, dka, waven)

        # Spin-correlation factors
        snorm = sampi**2 + sampc**2
        if law == 2:
            swe = sampi**2 / 3.0
            swo = sk * sampc**2 + 2.0 * sampi**2 / 3.0
        elif law == 3:
            swe = sk * sampc**2
            swo = sampi**2
        elif law == 4:
            swe = sk * sampc**2 + 5.0 * sampi**2 / 8.0
            swo = 3.0 * sampi**2 / 8.0
        elif law == 5:
            swe = 3.0 * sampi**2 / 4.0
            swo = sk * sampc**2 + sampi**2 / 4.0
        swe /= snorm
        swo /= snorm

        sex = exts(ssm_slice[:, nal], exb, betan, nbeta)

        # The rotational machinery — statistical weights bt_stat(j, x), the
        # Bessel/Clebsch-Gordan sums sumh(j, jp, y), and the rotational
        # energy shifts betap — depends only on alpha (through y), NOT on
        # beta. Precompute the (betap, weight) table once per alpha instead
        # of once per (alpha, beta): identical values and identical
        # accumulation order, ~600x fewer sumh evaluations.
        ipo = 1
        if law == 2 or law == 5:
            ipo = 2
        jt1 = 2 * jterm
        if ipo == 2:
            jt1 += 1

        rot_terms = []   # per j: (even-jp terms, odd-jp terms)
        for l in range(ipo, jt1 + 1, 2):
            j = l - 1
            pj = bt_stat(j, x)
            even_terms = []
            for lp in range(1, 11, 2):
                jp = lp - 1
                betap = (-j * (j + 1) + jp * (jp + 1)) * x / 2.0
                tmp = (2 * jp + 1) * pj * swe * 4.0 * sumh(j, jp, y)
                even_terms.append((betap, tmp))
            odd_terms = []
            for lp in range(2, 11, 2):
                jp = lp - 1
                betap = (-j * (j + 1) + jp * (jp + 1)) * x / 2.0
                tmp = (2 * jp + 1) * pj * swo * 4.0 * sumh(j, jp, y)
                odd_terms.append((betap, tmp))
            rot_terms.append((even_terms, odd_terms))

        # Batched evaluation over all beta values (positive and negative).
        # _sint_batch_exact reproduces the scalar sint() bit for bit, and
        # the per-jj accumulation below keeps the exact scalar summation
        # order (element-wise += per term, snlg + snlk per j block).
        jjmax = 2 * nbeta - 1
        jj_idx = np.arange(jjmax)
        k_idx = np.where(jj_idx < nbeta - 1,
                         nbeta - jj_idx - 1, jj_idx - nbeta + 1)
        be_signed = np.where(jj_idx < nbeta - 1,
                             -betan[k_idx], betan[k_idx])
        log_sex = np.array([log(s) if s > 0.0 else -225.0 for s in sex])
        beta_max = betan[nbeta - 1]

        sn_arr = np.zeros(jjmax)
        for even_terms, odd_terms in rot_terms:
            snlg = np.zeros(jjmax)
            for betap, tmp in even_terms:
                snlg += tmp * _sint_batch_exact(
                    be_signed + betap, bex, rdbex, sex, log_sex, nbx,
                    al, wt, tbart, beta_max)
            snlk = np.zeros(jjmax)
            for betap, tmp in odd_terms:
                snlk += tmp * _sint_batch_exact(
                    be_signed + betap, bex, rdbex, sex, log_sex, nbx,
                    al, wt, tbart, beta_max)
            sn_arr += snlg + snlk

        # Store results (jj < nbeta -> ssm, jj >= nbeta-1 -> ssp)
        ssm_slice[k_idx[:nbeta], nal] = sn_arr[:nbeta]
        ssp_slice[k_idx[nbeta - 1:], nal] = sn_arr[nbeta - 1:]

def skold_approx(ssm, alpha, beta, nalpha, nbeta, itemp, ntempr, lat,
                 arat, awr, tev, ska, nka, dka, cfrac):
    """Apply Skold approximation for intermolecular coherence."""
    angst = 1.0e-8
    sc = 1.0
    if lat == 1:
        sc = THERM / tev
    amass = awr * AMASSN * AMU

    for i in range(nbeta):
        scoh_arr = np.zeros(nalpha)
        for j in range(nalpha):
            al = alpha[j] * sc / arat
            waven = angst * sqrt(2.0 * amass * tev * EV * al) / HBAR
            sk = terpk(ska, nka, dka, waven)
            # S(kappa) is a (nonnegative) static structure factor that can dip to
            # ~0 between Bragg peaks. A zero or non-finite value makes alpha/sk
            # blow up to inf and then NaN into the stored law (the bracket search
            # never terminates and log(inf) propagates), so treat the coherent
            # contribution at such a point as its zero limit instead.
            if not np.isfinite(sk) or sk <= 0.0:
                scoh_arr[j] = 0.0
                continue
            ap = alpha[j] / sk

            # Find interpolation bracket
            kk = 0
            for k in range(nalpha):
                kk = k
                if ap < alpha[k]:
                    break
            if kk == 0:
                kk = 1

            if (ssm[i, kk - 1, itemp] == 0.0 or ssm[i, kk, itemp] == 0.0):
                scoh_arr[j] = 0.0
            else:
                # Log-log interpolation (terp1 with flag 5)
                x1, y1 = alpha[kk - 1], ssm[i, kk - 1, itemp]
                x2, y2 = alpha[kk], ssm[i, kk, itemp]
                if y1 > 0 and y2 > 0 and x1 > 0 and x2 > 0 and ap > 0:
                    scoh_arr[j] = exp(log(y1) + (log(ap) - log(x1)) * (log(y2) - log(y1)) / (log(x2) - log(x1)))
                else:
                    scoh_arr[j] = 0.0
            scoh_arr[j] *= sk

        for j in range(nalpha):
            ssm[i, j, itemp] = (1.0 - cfrac) * ssm[i, j, itemp] + cfrac * scoh_arr[j]

def sigfig(x, ndig, idig):
    """Round x to ndig significant figures (matching Fortran sigfig exactly).

    Replicates NJOY's util.f90 sigfig, including its rounding bias and
    multiplicative bias to match Fortran output bit-for-bit.
    """
    bias = 1.0000000000001
    if x == 0.0:
        return 0.0
    aa = np.log10(abs(x))
    ipwr = int(aa)
    if aa < 0:
        ipwr = ipwr - 1
    ipwr = ndig - 1 - ipwr
    # Guard against overflow for extremely small/large values
    if ipwr > 300:
        return 0.0
    if ipwr < -300:
        return x * bias
    # Fortran: ii=nint(x*ten**ipwr+ten**(ndig-11))
    # The ten**(ndig-11) bias pushes values just past 0.5 for tie-breaking
    scaled = x * 10.0**ipwr + 10.0**(ndig - 11)
    ii = int(np.round(scaled))
    # Match NJOY util.f90 sigfig exactly: the digit-carry renormalization uses
    # the SIGNED comparison (ii.ge.10**ndig), not abs(ii). For every IRMA call
    # site x is non-negative (S(a,b), alpha/beta/energy grids, temperatures, DW
    # factors, Bragg edges), so ii>=0 and this is byte-identical to the old
    # abs() form; the signed test only matters for negative x where NJOY does
    # NOT renormalize at the ii ~ -10**ndig boundary.
    if ii >= 10**ndig:
        ii = ii // 10
        ipwr = ipwr - 1
    ii = ii + idig
    xx = ii * 10.0**(-ipwr)
    return xx * bias
