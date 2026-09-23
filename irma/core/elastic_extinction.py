"""Energy-dependent extinction-corrected coherent-elastic cross section.

``sigma_coh_ext(E, itemp)`` applies the per-plane extinction factor y of
:mod:`irma.core.extinction` to each Bragg plane's Debye-Waller-attenuated
|F|^2 and sums. Extinction acts only at long wavelength (x ~ lambda^2), so
above a cutoff ``E_active`` sigma_ext equals the kinematic cumulative sum to
within the table tolerance. The upward edge scan finds the last edge whose
correction exceeds the budget and stops only once a per-plane bound on the
deficit of every unseen plane (its weight times its probed max(1-y)) is also
within the budget, so a strong reflection after a run of weak edges cannot
be returned as kinematic. The explicit sum is evaluated only below E_active.
"""
import math
import warnings

import numpy as np

from irma.core.constants import WL2EKIN, BK
from irma.core import extinction as _ext

_MULT = 2.0   # Friedel pair multiplicity, matching compute_bragg_edges_general

# Probe windows for the per-plane deficit bound, as E/E_threshold factors. The
# first (narrow) window catches the common case cheaply; the second (wide)
# window re-checks a plane whose deficit has not clearly decayed. A plane that
# still fails the decay check after the wide window gets the fully conservative
# bound (its entire kinematic weight), which simply keeps the scan going.
_PROBE_WINDOWS = ((16.0, 8), (4096.0, 12))

def _plane_deficit_fraction(model, Nc, f_t, d_t, *, l, g, L, dist, recipe):
    """Worst-case extinction deficit fraction max(1-y) of one plane over all
    incident energies at or above its Bragg threshold.

    The probe evaluates y on a log-energy grid starting at the threshold
    (lambda = 2d, the longest wavelength at which the plane scatters) and
    VERIFIES per model that the probed maximum actually captured the
    supremum: the maximum must not sit at the far edge of the window with
    the deficit still rising. Angular factors in some extinction models
    peak slightly above threshold, so an interior (or threshold) maximum is
    accepted wherever it lands; the decay RATE is irrelevant to the bound.
    Returns ``(fraction, captured)``; when the deficit is still rising at
    the wide window's edge, ``fraction`` is 1.0 (the caller treats the
    plane's full weight as potentially extinguished).
    """
    lam_thr = 2.0 * d_t
    for window, npts in _PROBE_WINDOWS:
        deficits = []
        for fac in np.geomspace(1.0, window, npts):
            lam = lam_thr / math.sqrt(fac)      # E = E_thr * fac
            y = _ext.extinction_factor(model, Nc, lam, f_t, d_t,
                                       l=l, g=g, L=L, dist=dist,
                                       recipe=recipe)
            deficits.append(1.0 - y)
        dmax = max(deficits)
        if dmax <= 0.0:
            return 0.0, True
        if deficits.index(dmax) < npts - 1:     # supremum inside the window
            return dmax, True
    return 1.0, False


def make_sigma_coh_ext(bragg, bragg_dir_terms, species_dw, V, N, scale,
                       ext_cfg, tempr):
    """Build ``sigma_coh_ext(E, itemp)`` -> extinction-corrected coherent sigma.

    bragg          : (nedge, 2) edges, col0 = edge energy E_thr [eV], ascending.
    bragg_dir_terms: per-edge list of (G_hat, D_st_plane[, site_terms]) -- one
                     entry per plane. ``site_terms`` (cos, sin, species index,
                     plane prefactor) is required only for the site-resolved
                     directional path (``species_dw.dir_tensors_uniform`` false).
    species_dw     : resolved SpeciesDW (directional or per-species).
    V, N           : unit-cell volume [A^3], atoms/cell.
    scale          : SEF structure-factor scale (1.0 for MEF).
    ext_cfg        : {'model','l','g','L','dist','recipe',['rmse_tol']}.
    tempr          : temperature list [K].

    Returns ``(sigma_coh_ext, edge_E, E_active)`` where ``E_active`` is the
    highest cutoff energy over all temperatures (above it sigma_ext == sigma_kin
    to within the tolerance, so the ENDF writer can splice in the kinematic edges).
    """
    use_dir = species_dw.use_dir_dw
    # Uniform per-site tensors -> the species-averaged double sum is exact;
    # non-uniform -> site-resolved amplitude sum.
    dir_uniform = (not use_dir) or getattr(
        species_dw, "dir_tensors_uniform", True)
    nsp = species_dw.nsp
    b = species_dw.b_sqb
    awr_sp = species_dw.awr_sp

    xsectfact = 0.5 * WL2EKIN / (V * N)
    Nc = 1.0 / V
    model = ext_cfg["model"]
    l = float(ext_cfg.get("l", 0.0))
    g = float(ext_cfg.get("g", 0.0))
    L = float(ext_cfg.get("L", 0.0))
    dist = ext_cfg.get("dist")
    recipe = ext_cfg.get("recipe", "std")
    # Half the table tolerance goes to the kinematic splice above E_active,
    # half to the tabulation itself.
    rmse_tol = float(ext_cfg.get("rmse_tol", 1e-3))
    tau = 0.5 * rmse_tol

    nedge = len(bragg)                          # bragg may be a list or an ndarray
    edge_E = [float(bragg[j][0]) for j in range(nedge)]
    eE_arr = np.array(edge_E)

    def _explicit(planes, E, kcount):
        """Explicit per-plane extinction sigma at E over the kcount lowest edges."""
        lam = math.sqrt(WL2EKIN / E)
        s = 0.0
        for mm in range(kcount):
            pk = planes[mm]
            if pk is None:
                continue
            d_a, f_a, del_a = pk
            acc = 0.0
            for t in range(len(d_a)):
                acc += del_a[t] * _ext.extinction_factor(
                    model, Nc, lam, f_a[t], d_a[t],
                    l=l, g=g, L=L, dist=dist, recipe=recipe)
            s += scale * acc
        return s / E

    # Per temperature: the kinematic prefix sum and the per-edge plane packs.
    per_temp = []                               # (Kcum, planes_sorted, E_active)    per_temp = []                               # (Kcum, planes_sorted, E_active)
    for itemp in range(len(tempr)):
        if use_dir:
            F_it = species_dw.F_species_per_temp[itemp]
            kT = tempr[itemp] * BK
            F_flat = None if dir_uniform else species_dw.F_sites_per_temp[itemp]
        else:
            W_c = [species_dw.W_ps[si][itemp] for si in range(nsp)]

        K_sorted = np.zeros(nedge)              # scale * sum_planes(delta) per edge
        planes_sorted = [None] * nedge          # (d[], f_ang[], delta[]) per edge
        for m in range(nedge):
            e_thr = edge_E[m]
            d_j = math.sqrt(WL2EKIN / (4.0 * e_thr))
            norm = d_j * _MULT * xsectfact
            ds, fs, dels = [], [], []
            ksum = 0.0
            for plane in bragg_dir_terms[m]:
                G_hat, D_st = plane[0], plane[1]
                if use_dir and not dir_uniform:
                    # Site-resolved: |sum_i b_i e^{-2 W_i E} e^{i phi_i}|^2 * pref,
                    # with pref == norm.
                    cos_a, sin_a, sp_idx, pref = plane[2]
                    acc_re = 0.0
                    acc_im = 0.0
                    for i in range(len(sp_idx)):
                        sp = sp_idx[i]
                        W_i = (float(G_hat @ F_flat[i] @ G_hat)
                               / (awr_sp[sp] * kT))
                        amp = b[sp] * math.exp(-2.0 * W_i * e_thr)
                        acc_re += amp * float(cos_a[i])
                        acc_im += amp * float(sin_a[i])
                    delta = (acc_re * acc_re + acc_im * acc_im) * pref
                else:
                    if use_dir:
                        W = [float(G_hat @ F_it[si] @ G_hat) / (awr_sp[si] * kT)
                             for si in range(nsp)]
                    else:
                        W = W_c
                    delta = 0.0
                    for si in range(nsp):
                        bi, wsi = b[si], W[si]
                        for tj in range(nsp):
                            dw = math.exp(-2.0 * (wsi + W[tj]) * e_thr)
                            delta += bi * b[tj] * dw * float(D_st[si, tj])
                if delta <= 0.0:
                    continue
                ds.append(d_j)
                fs.append(math.sqrt(delta / norm) * 1.0e-4)   # |F| [A]
                dels.append(delta)
                ksum += delta
            K_sorted[m] = scale * ksum
            if ds:
                planes_sorted[m] = (np.array(ds), np.array(fs), np.array(dels))
        Kcum = np.cumsum(K_sorted)

        # ----- per-edge worst-case deficit bound for the scan stop: no unseen
        # plane can contribute more extinction deficit than its kinematic
        # weight times its probed max(1-y).
        B_edge = np.zeros(nedge)
        n_nondecay = 0
        for m in range(nedge):
            pk = planes_sorted[m]
            if pk is None:
                continue
            d_a, f_a, del_a = pk
            bsum = 0.0
            for t in range(len(d_a)):
                frac, decayed = _plane_deficit_fraction(
                    model, Nc, f_a[t], d_a[t],
                    l=l, g=g, L=L, dist=dist, recipe=recipe)
                if not decayed:
                    n_nondecay += 1
                bsum += del_a[t] * frac
            B_edge[m] = scale * bsum
        if n_nondecay:
            warnings.warn(
                f"extinction model {model!r}: the deficit of {n_nondecay} "
                "plane(s) did not decay over the probe window; using the "
                "fully conservative bound (full extinction) for them",
                RuntimeWarning, stacklevel=2)
        # B_tail[m] = summed bound of every edge strictly above m.
        B_tail = np.concatenate(
            (np.cumsum(B_edge[::-1])[::-1][1:], [0.0]))

        # ----- locate E_active: highest energy whose correction still exceeds tau.
        # Scan edges upward (cheap at low E). The consecutive-below run covers the
        # decay of the ALREADY-SEEN planes empirically; the B_tail bound proves the
        # UNSEEN planes cannot re-activate the correction -- a strong reflection
        # after a long run of weak edges keeps the scan alive because its own
        # deficit bound is still in the tail.
        last_active = 0.0
        below = 0
        planes_seen = 0
        for m in range(nedge):
            E = float(eE_arr[m])
            if planes_sorted[m] is not None:
                planes_seen += len(planes_sorted[m][0])
            kin = Kcum[m]
            if kin <= 0.0:
                continue
            corr = 1.0 - _explicit(planes_sorted, E, m + 1) * E / kin
            if corr > tau:
                last_active = E
                below = 0
            else:
                below += 1
            tail_ok = B_tail[m] <= 0.5 * tau * kin
            if (below >= 25 and planes_seen > 500 and tail_ok
                    and E >= 2.0 * max(last_active, float(eE_arr[0]))):
                break
        # 30% margin above the last active edge: the scan is over discrete
        # edges and rounding can move the splice.
        E_active = last_active * 1.3 if last_active > 0.0 else 0.0

        per_temp.append((Kcum, planes_sorted, E_active))

    E_active_max = max((pt[2] for pt in per_temp), default=0.0)

    def sigma_coh_ext(E, itemp):
        """Extinction-corrected coherent-elastic sigma at energy E (eV)."""
        Kcum, planes_sorted, E_active = per_temp[itemp]
        k = int(np.searchsorted(eE_arr, E * (1.0 + 1e-12), side="right"))
        if k == 0:
            return 0.0
        if E >= E_active:                       # extinction negligible -> kinematic
            return float(Kcum[k - 1]) / E
        return _explicit(planes_sorted, E, k)

    return sigma_coh_ext, edge_E, E_active_max
