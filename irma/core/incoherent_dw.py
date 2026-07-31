"""Orientation-averaged incoherent-elastic Debye-Waller factor.

For one atom with thermal-displacement tensor ``U`` [Ang^2], the powder
incoherent-elastic differential cross section carries

    f(Q) = < exp(-Q^2 uhat.U.uhat) >_uhat     (uniform average over the sphere)

-- the average of the *exponential*, not the exponential of the averaged
exponent. By Jensen's inequality ``f(Q) >= exp(-Q^2 tr(U)/3)``, with the
softest displacement direction dominating at high Q, so the isotropic
(trace/3) form systematically underestimates the incoherent-elastic tail of
an anisotropic crystal. The ENDF MF7/MT2 incoherent-elastic record stores a
single scalar Debye-Waller integral and cannot represent this; the consumers
of this module are the paths with no format constraint: the NCrystal pack
exporter (``incoherent_elastic_mode = directional``), the plugin's Python
reference oracle, and the spectra elastic line.

Everything here works in eigenvalue space: f depends on U only through its
eigenvalues ``u1 <= u2 <= u3`` [Ang^2].

Method (the C++ plugin mirrors these branches, constants and quadrature
literals exactly, so the reference gate can hold a tight tolerance):

  * isotropic  (u3-u1 <= tol)   f = exp(-Q^2 ubar)
  * uniaxial, unique large axis (u2-u1 <= tol; e.g. graphite u_c > u_ab)
        f = exp(-Q^2 uperp) * (sqrt(pi)/2) erf(s)/s,   s = Q sqrt(u3-uperp)
  * uniaxial, unique small axis (u3-u2 <= tol)
        f = exp(-Q^2 upar) * D(s)/s,                   s = Q sqrt(uperp-upar)
    with D the Dawson function (Numerical-Recipes/Rybicki algorithm, ~2e-7
    relative accuracy -- mirrored bit-for-bit in the C++).
  * triaxial: fixed 24x24 Gauss-Legendre product rule over one octant,
        f = sum_jk w_j w_k exp(-Q^2 p_jk),
        p_jk = u3 c_j^2 + (1-c_j^2)(u1 cos^2(phi_k) + u2 sin^2(phi_k)).

Both closed forms reduce to 1 at Q=0 and are exact; the quadrature weights
sum to 1 by construction.
"""
from __future__ import annotations

import math

import numpy as np


# 24-point Gauss-Legendre nodes/weights transformed to [0, 1], written as
# literals (17 significant digits) so the C++ plugin can embed the SAME
# numbers and the Python/C++ triaxial quadratures agree to rounding.
GL24_NODES01 = (
    0.0024063900014893447, 0.012635722014345263, 0.030862723998633601,
    0.056792236497799464, 0.089999007013048526, 0.12993790421072282,
    0.17595317403151223, 0.22728926430558022, 0.28310324618697746,
    0.3424786601519183, 0.40444056626319186, 0.46797155356869719,
    0.53202844643130276, 0.5955594337368082, 0.6575213398480817,
    0.71689675381302254, 0.77271073569441984, 0.82404682596848777,
    0.87006209578927718, 0.91000099298695147, 0.94320776350220048,
    0.96913727600136634, 0.98736427798565474, 0.99759360999851066,
)
GL24_WEIGHTS01 = (
    0.0061706148999933919, 0.014265694314466842, 0.02213871940870971,
    0.029649292457718267, 0.036673240705540136, 0.043095080765976713,
    0.048809326052057046, 0.053722135057982866, 0.057752834026862855,
    0.060835236463901758, 0.062918728173414248, 0.063969097673376149,
    0.063969097673376149, 0.062918728173414248, 0.060835236463901758,
    0.057752834026862855, 0.053722135057982866, 0.048809326052057046,
    0.043095080765976713, 0.036673240705540136, 0.029649292457718267,
    0.02213871940870971, 0.014265694314466842, 0.0061706148999933919,
)

# Relative eigenvalue-degeneracy tolerance for the branch classification. A
# misclassification is not a physics error (the triaxial quadrature is valid
# for every case, merely slower and quadrature-limited rather than exact), but
# the C++ uses the same constant so both sides always take the same branch.
DEGENERACY_RTOL = 1.0e-8

_SQRT_PI_HALF = 0.88622692545275801  # sqrt(pi)/2, 17g -- mirrored in C++

# Dawson-function constants (Numerical Recipes "dawson", Rybicki's method):
# H = 0.4, NMAX = 6, c_i = exp(-((2i-1)H)^2). Mirrored exactly in the C++.
_DAWSON_H = 0.4
_DAWSON_C = tuple(math.exp(-((2.0 * i - 1.0) * _DAWSON_H) ** 2)
                  for i in range(1, 7))
_DAWSON_INV_SQRT_PI = 0.56418958354775628  # 1/sqrt(pi), 17g


def dawsn(x):
    """Dawson function D(x) = exp(-x^2) * int_0^x exp(t^2) dt, vectorized.

    Numerical Recipes' implementation of Rybicki's sampling-theorem method
    (H=0.4, 6 terms; small-|x| Taylor series below 0.2). Relative accuracy
    ~2e-7 -- ample for a Debye-Waller factor, and mirrored term-for-term in
    the C++ plugin so both sides return identical values.
    """
    x = np.asarray(x, float)
    ax = np.abs(x)
    out = np.empty_like(ax)

    small = ax < 0.2
    if np.any(small):
        x2 = x[small] * x[small]
        out[small] = x[small] * (
            1.0 - (2.0 / 3.0) * x2 * (1.0 - 0.4 * x2 * (1.0 - (2.0 / 7.0) * x2)))

    big = ~small
    if np.any(big):
        xx = ax[big]
        n0 = 2.0 * np.floor(0.5 * xx / _DAWSON_H + 0.5)
        xp = xx - n0 * _DAWSON_H
        e1 = np.exp(2.0 * xp * _DAWSON_H)
        e2 = e1 * e1
        d1 = n0 + 1.0
        d2 = d1 - 2.0
        total = np.zeros_like(xx)
        for ci in _DAWSON_C:
            total += ci * (e1 / d1 + 1.0 / (d2 * e1))
            d1 += 2.0
            d2 -= 2.0
            e1 *= e2
        out[big] = (_DAWSON_INV_SQRT_PI * np.exp(-xp * xp) * total
                    * np.sign(x[big]))
    return out


def u_eigenvalues(U):
    """Ascending eigenvalues [Ang^2] of one 3x3 thermal-displacement tensor.

    Tiny negative eigenvalues from rounding of a positive-semidefinite tensor
    are clipped to zero; a materially negative eigenvalue (beyond a 1e-8
    relative tolerance) means the input is not a displacement tensor and
    raises instead of being silently clipped.
    """
    U = np.asarray(U, float)
    if U.shape != (3, 3):
        raise ValueError(f"U must be a 3x3 tensor, got shape {U.shape}")
    eig = np.linalg.eigvalsh(0.5 * (U + U.T))
    tol = 1.0e-8 * max(1.0e-30, float(np.max(np.abs(eig))))
    if eig[0] < -tol:
        raise ValueError(
            "U tensor is not positive semidefinite (eigenvalues "
            f"{eig.tolist()}); thermal-displacement tensors must be PSD")
    return np.clip(eig, 0.0, None)


def _erf_ratio(s):
    """(sqrt(pi)/2) erf(s)/s, series-stabilized at small s (limit 1)."""
    s = np.asarray(s, float)
    small = s < 1.0e-6
    safe = np.where(small, 1.0, s)
    erf = np.vectorize(math.erf, otypes=[float])
    out = np.where(small,
                   1.0 - s * s / 3.0,
                   _SQRT_PI_HALF * erf(safe) / safe)
    return out


def _dawson_ratio(s):
    """D(s)/s, series-stabilized at small s (limit 1)."""
    s = np.asarray(s, float)
    small = s < 1.0e-6
    safe = np.where(small, 1.0, s)
    out = np.where(small,
                   1.0 - (2.0 / 3.0) * s * s,
                   dawsn(safe) / safe)
    return out


def dw_orientation_average_tsq(u_eig, tsq):
    """f as a function of t = Q^2 [1/Ang^2]: <exp(-t uhat.U.uhat)>_uhat.

    ``u_eig`` is the ascending eigenvalue triple from :func:`u_eigenvalues`;
    ``tsq`` an array of Q^2 values. Dispatches to the branch table in the
    module docstring.
    """
    u1, u2, u3 = (float(v) for v in np.asarray(u_eig, float))
    t = np.atleast_1d(np.asarray(tsq, float))
    if u3 <= 0.0:
        return np.ones_like(t)
    tol = DEGENERACY_RTOL * u3
    if u3 - u1 <= tol:
        # isotropic
        ubar = (u1 + u2 + u3) / 3.0
        return np.exp(-t * ubar)
    if u2 - u1 <= tol:
        # uniaxial, unique LARGE axis (u3): exact erf form
        uperp = 0.5 * (u1 + u2)
        s = np.sqrt(t * (u3 - uperp))
        return np.exp(-t * uperp) * _erf_ratio(s)
    if u3 - u2 <= tol:
        # uniaxial, unique SMALL axis (u1): exact Dawson form
        uperp = 0.5 * (u2 + u3)
        s = np.sqrt(t * (uperp - u1))
        return np.exp(-t * u1) * _dawson_ratio(s)
    # triaxial: fixed Gauss-Legendre product rule over one octant
    c = np.asarray(GL24_NODES01, float)
    wc = np.asarray(GL24_WEIGHTS01, float)
    phi = 0.5 * math.pi * c
    csq = c * c
    cos2 = np.cos(phi) ** 2
    p = (u3 * csq[:, None]
         + (1.0 - csq)[:, None] * (u1 * cos2 + u2 * (1.0 - cos2))[None, :])
    w = wc[:, None] * wc[None, :]
    return np.exp(-t[:, None] * p.reshape(-1)[None, :]) @ w.reshape(-1)


def dw_orientation_average(u_eig, Q):
    """f(Q) = <exp(-Q^2 uhat.U.uhat)>_uhat for Q in 1/Ang."""
    Q = np.atleast_1d(np.asarray(Q, float))
    return dw_orientation_average_tsq(u_eig, Q * Q)


def sigma_elinc_directional(ksq_invA2, channels, *, n_t=16384):
    """Angle-integrated directional incoherent-elastic cross section [barn].

    sigma(E) = sum_d (sigma_d / (4 k^2)) * int_0^{4k^2} f_d(sqrt(t)) dt

    with ``ksq_invA2`` the neutron wavevector squared k^2 [1/Ang^2] (an array;
    k^2 = E_meV / C_E with C_E = hbar^2/2m_n in meV*Ang^2) and ``channels`` a
    sequence of ``(sigma_b_barn, u_eig)`` pairs. The k^2 -> 0 limit is the
    full bound value sum(sigma_d), as for the isotropic ENDF form. For a
    cubic (isotropic) tensor this reproduces the standard ENDF LTHR=2 result
    ``(sigma_b/2)(1 - exp(-4 E W'))/(2 E W')`` identically.

    The t-integral is a cumulative trapezoid on a dense log grid (plus t=0)
    whose low end is anchored at the faster of the kinematic and Debye-Waller
    scales, min(4 k^2_max, 1/u_max) * 1e-8, so the onset of the decay is
    resolved even when the kinematic range vastly exceeds it. Accuracy is
    ~1e-7 relative over that whole domain -- and the discretization is
    INDEPENDENT of the C++ plugin's mixture-of-exponentials representation,
    so the reference gate genuinely cross-checks the plugin instead of
    comparing a table with itself.
    """
    ksq = np.atleast_1d(np.asarray(ksq_invA2, float))
    if np.any(ksq < 0.0):
        raise ValueError("ksq_invA2 must be non-negative")
    t_hi = 4.0 * float(np.max(ksq, initial=0.0))
    out = np.zeros_like(ksq)
    if t_hi <= 0.0:
        return out + sum(float(sb) for sb, _ in channels)
    u_max = max((float(np.max(np.asarray(u_eig, float)))
                 for _, u_eig in channels), default=0.0)
    t_scale = min(t_hi, 1.0 / u_max) if u_max > 0.0 else t_hi
    t = np.concatenate([[0.0],
                        np.geomspace(t_scale * 1.0e-8, t_hi, int(n_t))])
    for sb, u_eig in channels:
        f = dw_orientation_average_tsq(u_eig, t)
        cum = np.concatenate([[0.0],
                              np.cumsum(0.5 * (f[1:] + f[:-1]) * np.diff(t))])
        integral = np.interp(4.0 * ksq, t, cum)
        with np.errstate(divide="ignore", invalid="ignore"):
            out += np.where(ksq > 0.0,
                            float(sb) * integral / (4.0 * ksq),
                            float(sb))
    return out
