"""Energy-dependent extinction-corrected coherent-elastic cross section.

Bridges the per-plane Bragg + Debye-Waller data to the pure extinction models in
:mod:`irma.core.extinction`, then exposes ``sigma_coh_ext(E, itemp)`` for the
adaptive **histogram (INT=1)** LTHR=1 tabulation in the ENDF writer.

Per-plane (not per-edge-group): the extinction factor y depends on the plane's
|F|^2, so it is applied to each crystallographic plane separately and summed. The
structure factor used in y is the **Debye-Waller-attenuated** |F|^2 (matching
NCrystal / CrysXT, verified: Be(002) static 2.43 b -> NCrystal 2.27 b), which is
exactly the per-plane DW-weighted kinematic weight / (d.mult.xsectfact).

Works for every Debye-Waller model the iel=10 elastic path produces -- directional
(inelastic_mode=1/2), per-species, or scalar isotropic (inelastic_mode=0) -- using
the always-present per-plane ``bragg_dir_terms`` for the structure factors and the
appropriate DW exponent W_s.

Cost model
----------
The extinction factor y(lambda) is wavelength-dependent, so the cumulative-sum
shortcut that makes the *kinematic* comb O(1) per energy does NOT apply directly:
a naive evaluation re-sums every Bragg plane below E at every tabulation node, and
with ~10^5 planes and ~10^4 adaptive nodes that is hours.

But extinction only *acts* at long wavelength: the dimensionless x ~ lambda^2 (and
higher), so y -> 1 (no extinction) above a material/sample-specific cutoff energy.
Above that cutoff ``sigma_ext == sigma_kin`` to within the tabulation tolerance, and
sigma_kin is the free cumulative prefix sum. We therefore:

  * precompute, per temperature, the kinematic per-edge weight K_j (lambda-free) and
    its cumulative sum -> sigma_kin(E) in O(log nedge);
  * bound, per plane, the worst-case extinction deficit it can ever contribute
    (its kinematic weight times ``max(1-y)`` probed near its own threshold, where
    the wavelength -- and hence the extinction parameter -- is largest for that
    plane; the probe VERIFIES the deficit decays with energy for the selected
    model rather than assuming monotonicity, since the angular factors of some
    models can peak slightly above threshold);
  * locate ``E_active``: the highest energy at which the relative extinction
    correction still exceeds ~0.5 * the table RMSE tolerance, and stop the upward
    scan only when the summed deficit bound of every UNSEEN plane is also inside
    the tolerance budget -- so a strong reflection entering after a long run of
    negligible edges cannot be silently returned as kinematic;
  * evaluate the explicit per-plane extinction sum ONLY for E < E_active (a small
    handful of low-energy edges -> the scalar models are plenty fast there);
  * return the cumulative kinematic sigma for E >= E_active.

This mirrors how the CrysXT plugin bounds its own work (sorted planes with an early
``2 d < lambda`` break) and collapses a ~3 hour build to a couple of seconds. The
edge count written to the tape is controlled separately (tolerance thinning and
the optional Card-6b logarithmic edge grouping in the ENDF writer); the scan
bound changes only where the extinction correction is proven negligible.
"""
import math
import warnings

import numpy as np

from irma.core.constants import WL2EKIN, BK
from irma.core import extinction as _ext
from irma.core.elastic_dw import _flatten_site_tensors

_MULT = 2.0   # Friedel pair multiplicity, matching compute_bragg_edges_general

# Probe windows for the per-plane deficit bound, as E/E_threshold factors. The
# first (narrow) window catches the common case cheaply; the second (wide)
# window re-checks a plane whose deficit has not clearly decayed. A plane that
# still fails the decay check after the wide window gets the fully conservative
# bound (its entire kinematic weight), which simply keeps the scan going.
_PROBE_WINDOWS = ((16.0, 8), (4096.0, 12))

# The [0, 1] physical invariant on extinction factors is enforced at the
# public boundary (irma.core.extinction.extinction_factor, review PH-2), so
# every y consumed in this module is already clamped/validated.

# Extinction-scan work budget (review PH-1). The SOFT cap only permits an
# early stop once the unseen-tail deficit bound is proven within budget; an
# unproven tail keeps the scan running (each additional plane is a few
# closed-form evaluations, so completing the enumeration is cheap for real
# materials). The HARD cap is a fail-closed backstop for pathological
# inputs: the scan RAISES rather than ever returning a kinematic splice the
# bound cannot justify.
_SCAN_SOFT_CAP = 50_000
_SCAN_HARD_CAP = 2_000_000


def _plane_deficit_fraction(model, Nc, f_t, d_t, *, l, g, L, dist, recipe, mu):
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
                                       recipe=recipe, mu=mu)
            deficits.append(1.0 - y)
        dmax = max(deficits)
        if dmax <= 0.0:
            return 0.0, True
        if deficits.index(dmax) < npts - 1:     # supremum inside the window
            return dmax, True
    return 1.0, False


def make_sigma_coh_ext(bragg, bragg_dir_terms, species_dw, dwpix, V, N, scale,
                       ext_cfg, tempr):
    """Build ``sigma_coh_ext(E, itemp)`` -> extinction-corrected coherent sigma.

    bragg          : (nedge, 2) array, col0 = edge energy E_thr [eV].
    bragg_dir_terms: per-edge list of (G_hat, D_st_plane[, site_terms]) -- one
                     entry per plane. ``site_terms`` (cos, sin, species index,
                     plane prefactor) is required only for the site-resolved
                     directional path (``species_dw.dir_tensors_uniform``
                     false, review finding P3).
    species_dw     : resolved SpeciesDW (directional / per-species / None-scalar).
    dwpix          : scalar isotropic DW integral per temperature (fallback).
    V, N           : unit-cell volume [A^3], atoms/cell.
    scale          : SEF structure-factor scale (1.0 for MEF / iel=1-6).
    ext_cfg        : {'model','l','g','L','dist','recipe',['mu'],['rmse_tol']}.
    tempr          : temperature list [K].

    Returns ``(sigma_coh_ext, edge_E, E_active)`` where ``E_active`` is the
    highest cutoff energy over all temperatures (above it sigma_ext == sigma_kin
    to within the tolerance, so the ENDF writer can splice in the kinematic comb).
    """
    use_dir = species_dw is not None and species_dw.use_dir_dw
    use_ps = species_dw is not None and species_dw.use_ps
    # Uniform per-site tensors -> the species-averaged double sum is exact and
    # byte-pinned; non-uniform -> site-resolved amplitude sum (finding P3).
    dir_uniform = (not use_dir) or getattr(
        species_dw, "dir_tensors_uniform", True)
    nsp = species_dw.nsp if species_dw is not None else 1
    b = species_dw.b_sqb if species_dw is not None else None
    awr_sp = species_dw.awr_sp if species_dw is not None else None

    xsectfact = 0.5 * WL2EKIN / (V * N)
    Nc = 1.0 / V
    model = ext_cfg["model"]
    l = float(ext_cfg.get("l", 0.0))
    g = float(ext_cfg.get("g", 0.0))
    L = float(ext_cfg.get("L", 0.0))
    dist = ext_cfg.get("dist")
    recipe = ext_cfg.get("recipe", "std")
    mu = float(ext_cfg.get("mu", 0.0))
    # Cutoff threshold tau: above E_active we approximate sigma_ext by sigma_kin,
    # so that approximation must stay well inside the table's error budget. We
    # spend HALF the table RMSE tolerance on it (factor 0.5), leaving the other
    # half for the tabulation itself -- the two error sources then add to <= the
    # tolerance. rmse_tol defaults to the ENDF-writer table tolerance (and the
    # deck-card default), so the two budgets are the same scale by construction.
    rmse_tol = float(ext_cfg.get("rmse_tol", 1e-3))
    tau = 0.5 * rmse_tol

    nedge = len(bragg)                          # bragg may be a list or an ndarray
    edge_E = [float(bragg[j][0]) for j in range(nedge)]
    edge_d = [math.sqrt(WL2EKIN / (4.0 * e)) if e > 0.0 else 0.0 for e in edge_E]

    def _W_const(itemp):
        """W_s for the species (constant over planes) -- per-species / scalar."""
        if use_ps:
            return [species_dw.W_ps[si][itemp] for si in range(nsp)]
        return [dwpix[itemp]] * nsp             # scalar isotropic: same W for all

    # ----- edges sorted by energy (CrysXT-style; usually already ascending) -----
    eidx = sorted(range(nedge), key=lambda j: edge_E[j])
    eE_arr = np.array([edge_E[j] for j in eidx])

    # ----- per-temperature precompute: kinematic prefix sum + per-edge plane packs.
    # Geometry (E_thr, d) is temperature-independent; the DW-attenuated weights are
    # not, so we rebuild per temperature.  This is ONE pass over all planes per T.
    per_temp = []                               # (Kcum, planes_sorted, E_active)
    for itemp in range(len(tempr)):
        if use_dir:
            F_it = species_dw.F_species_per_temp[itemp]
            kT = tempr[itemp] * BK
            F_flat = (None if dir_uniform else
                      _flatten_site_tensors(species_dw.F_sites_per_temp[itemp]))
        else:
            W_c = _W_const(itemp)

        K_sorted = np.zeros(nedge)              # scale * sum_planes(delta) per edge
        planes_sorted = [None] * nedge          # (d[], f_ang[], delta[]) per edge
        for m, j in enumerate(eidx):
            e_thr = edge_E[j]
            d_j = edge_d[j]
            if d_j <= 0.0:
                continue
            norm = d_j * _MULT * xsectfact
            ds, fs, dels = [], [], []
            ksum = 0.0
            for plane in bragg_dir_terms[j]:
                G_hat, D_st = plane[0], plane[1]
                if use_dir and not dir_uniform:
                    # Site-resolved (finding P3): |sum_i b_i e^{-2 W_i E}
                    # e^{i phi_i}|^2 * pref; pref == norm, so the |F| used by
                    # the extinction factor (sqrt(delta/norm)*1e-4 below) is
                    # unchanged in its relationship to delta.
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
                    l=l, g=g, L=L, dist=dist, recipe=recipe, mu=mu)
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

        def _explicit(E, kcount):
            """Explicit per-plane extinction sigma over the kcount lowest edges."""
            lam = math.sqrt(WL2EKIN / E)
            s = 0.0
            for mm in range(kcount):
                pk = planes_sorted[mm]
                if pk is None:
                    continue
                d_a, f_a, del_a = pk
                acc = 0.0
                for t in range(len(d_a)):
                    acc += del_a[t] * _ext.extinction_factor(
                        model, Nc, lam, f_a[t], d_a[t],
                        l=l, g=g, L=L, dist=dist, recipe=recipe, mu=mu)
                s += scale * acc
            return s / E

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
            corr = 1.0 - _explicit(E, m + 1) * E / kin
            if corr > tau:
                last_active = E
                below = 0
            else:
                below += 1
            tail_ok = B_tail[m] <= 0.5 * tau * kin
            if (below >= 25 and planes_seen > 500 and tail_ok
                    and E >= 2.0 * max(last_active, float(eE_arr[0]))):
                break
            # Review PH-1 (fail closed): past the soft cap the scan may stop
            # ONLY on a proven tail bound; with the bound unproven it keeps
            # scanning -- the enumeration is finite and scanning it to
            # exhaustion makes E_active exact by construction. The hard cap
            # never splices: it raises.
            if planes_seen > _SCAN_SOFT_CAP and tail_ok:
                break
            if planes_seen > _SCAN_HARD_CAP:
                raise RuntimeError(
                    f"extinction scan exceeded the {_SCAN_HARD_CAP}-plane "
                    "hard cap with an unproven tail: the remaining planes' "
                    f"deficit bound is {B_tail[m] / kin:.3e} of the "
                    f"kinematic sum (budget {0.5 * tau:.1e}) at "
                    f"E = {E:.6g} eV. Refusing to emit a tape whose "
                    "extinction correction cannot be bounded; reduce emax, "
                    "raise fsquarecut, or disable the extinction card for "
                    "this material")
        # Push the cutoff 30% above the highest still-active edge as a safety
        # margin: the correction falls ~1/E, the upward scan is over discrete
        # edges, and sigfig rounding can nudge the splice -- 1.3x guarantees the
        # kinematic region we hand to the writer is genuinely sub-tau, never a
        # boundary edge that still carries a visible correction.
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

        lam = math.sqrt(WL2EKIN / E)
        s = 0.0
        for mm in range(k):
            pk = planes_sorted[mm]
            if pk is None:
                continue
            d_a, f_a, del_a = pk
            acc = 0.0
            for t in range(len(d_a)):
                acc += del_a[t] * _ext.extinction_factor(
                    model, Nc, lam, f_a[t], d_a[t],
                    l=l, g=g, L=L, dist=dist, recipe=recipe, mu=mu)
            s += scale * acc
        return s / E

    return sigma_coh_ext, edge_E, E_active_max
