"""Invert ENDF storage conventions to physical quantities at one temperature.

The inverse of irma.core.endf_writer (ported, not imported).
"""
from __future__ import annotations
import math
from dataclasses import dataclass
from .constants import T_LAT_K, HBAR2_OVER_2MN_EV_A2


@dataclass
class InelasticLaw:
    alpha_phys: list
    beta_phys: list
    sab_asym_downscatter: list   # [alpha][beta]
    bound_xs_barn: float
    awr: float


def _interp_T(temps, columns, T):
    """lin-lin interpolate a per-temperature column list to T. Raises if T is
    outside the stored grid (spec §10: never silently extrapolate).
    `columns[i]` is the data vector at temperature temps[i]. Returns a vector."""
    ts = [float(t) for t in temps]
    tol = 1e-6 * max(1.0, abs(ts[-1]))
    if T < ts[0] - tol or T > ts[-1] + tol:
        raise ValueError(f"target temperature {T} K outside stored grid [{ts[0]}, {ts[-1]}] K")
    if T <= ts[0]:
        return list(columns[0])
    if T >= ts[-1]:
        return list(columns[-1])
    for i in range(1, len(ts)):
        if T <= ts[i]:
            f = (T - ts[i - 1]) / (ts[i] - ts[i - 1])
            return [a + f * (b - a) for a, b in zip(columns[i - 1], columns[i])]


def coherent_edges(ev, T):
    if ev.coh_edges_ev is None:
        return None
    cumS = _interp_T(ev.coh_temps, ev.coh_cumS, T)   # verbatim values, T-interpolated
    return list(ev.coh_edges_ev), cumS


def incoherent_msd(ev, T):
    if ev.incoh_Wp is None:
        return None
    Wp = _interp_T(ev.incoh_temps, [[w] for w in ev.incoh_Wp], T)[0]
    sb = float(ev.incoh_sb_barn)
    # MF7/MT2 stores SB = sigma_inc_bound * npr (per molecule), parallel to
    # MT4 B(1)=npr*sigma_free; the inelastic bound_xs divides B(1) by npr (per atom),
    # so divide SB by npr too for a consistent per-atom incoherent-elastic xs.
    # LTHR=3 (mixed elastic) follows the SAME molecular convention: the IRMA MEF
    # writer stores SB = per-principal x npr like the classic and SEF/CEF writers
    # (resolved in the IRMA pre-release review), so the division is uniform.
    npr = float(ev.b_array[6]) if len(ev.b_array) > 6 and ev.b_array[6] > 0 else 1.0
    sb /= npr
    return Wp * HBAR2_OVER_2MN_EV_A2, sb


def physical_inelastic(ev, T) -> InelasticLaw:
    if ev.lasym == 1:
        # LEAPR writes LASYM=1 (S for -beta..+beta) for cold H2/D2 (ncold != 0)
        raise NotImplementedError(
            "LASYM=1 (S stored for -beta..+beta, as LEAPR writes for cold H2/D2) "
            "is not supported")
    if ev.lasym in (2, 3):
        raise NotImplementedError(
            "LASYM=2/3 (asymmetric SS, stored with NO e^{±β/2} factor) is deferred (spec §11)")
    if ev.lln != 0:
        # S stored as ln(S)-β/2 (LLN=1); exp() before the e^{+β/2} below.
        def unpack(s, b):
            return math.exp(s) * math.exp(0.5 * b)
    else:
        def unpack(s, b):
            return s * math.exp(0.5 * b)   # physical S = stored · e^{+β/2}
    # The tape stores S(alpha,beta) at one or more discrete temperatures (temps_mt4).
    # Select the column whose temperature EXACTLY matches the request (no interpolation
    # between columns), and LAT-un-scale the grids by THAT column's temperature, not the
    # raw request (a mismatch would pair the column's S-values with a wrong-T grid).
    tindex = None
    for k, tk in enumerate(ev.temps_mt4):
        if abs(T - tk) <= max(0.5, 1e-3 * tk):
            tindex = k
            break
    if tindex is None:
        raise NotImplementedError(
            f"requested temperature {T} K is not one of this tape's stored MF7/MT4 "
            f"temperatures {ev.temps_mt4} K; pick one of them (no interpolation).")
    t_col = float(ev.temps_mt4[tindex])
    lat_scale = (T_LAT_K / t_col) if ev.lat == 1 else 1.0  # stored ref-units -> physical (at T_col)
    beta_phys = [float(b) * lat_scale for b in ev.beta]
    # The alpha grid may LEGALLY vary per beta-block in ENDF MF7/MT4; this converter
    # uses one shared grid (block 0). Assert the other blocks match so a genuinely
    # per-beta-alpha tape fails loudly instead of silently truncating to block 0.
    a0 = ev.alpha[0]
    for bi, ab in enumerate(ev.alpha):
        if len(ab) != len(a0) or any(
                abs(float(a) - float(r)) > 1e-9 + 1e-6 * abs(float(r))
                for a, r in zip(ab, a0)):
            raise NotImplementedError(
                f"ENDF MF7/MT4 alpha grid at beta-block {bi} differs from block 0; "
                "per-beta alpha grids are not supported (the converter assumes one "
                "shared alpha grid).")
    alpha_phys = [float(a) * lat_scale for a in a0]
    na, nb = len(alpha_phys), len(beta_phys)
    sab = [[0.0] * nb for _ in range(na)]
    for bi in range(nb):
        bp = beta_phys[bi]
        for ai in range(na):
            stored = ev.sab[bi][ai][tindex]
            sab[ai][bi] = max(0.0, unpack(stored, bp))
    # ENDF MF7/MT4 B(1) = npr * sigma_FREE of the principal scatterer (NOT bound).
    # NCrystal's SABScatter normalizes to the per-atom BOUND cross section, so
    # convert free -> bound: sigma_bound = sigma_free * ((A+1)/A)^2  (A = AWR).
    # Verified on graphite: B(1)=4.724 * (12.898/11.898)^2 = 5.551 = IRMA pack bound_xs.
    npr = float(ev.b_array[6]) if len(ev.b_array) > 6 and ev.b_array[6] > 0 else 1.0
    awr = float(ev.awr)
    bound_xs = (float(ev.b_array[1]) / npr) * ((awr + 1.0) / awr) ** 2
    return InelasticLaw(alpha_phys, beta_phys, sab, bound_xs, awr)
