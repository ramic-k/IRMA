"""Crystal structures, Bragg-edge generation, and Debye-Waller helpers.

The generalized coherent-elastic calculator (compute_bragg_edges_general,
NCrystal-style; T. Kittelmann et al., Comp. Phys. Comm. 267 (2021) 108082),
the LEAPR built-in materials (coher: graphite/Be/BeO/Al/Pb/Fe), and the
phonopy-site bookkeeping used by the iel=10 path.

The reciprocal-lattice and Bragg-edge algorithms are derived in part from
NCrystal (NCLatticeUtils.cc; Copyright 2015-2025 NCrystal developers;
Apache License 2.0 — notice retained in THIRD_PARTY_NOTICES.md),
translated to Python and adapted to IRMA's data
structures.
"""

import numpy as np
from math import sqrt, hypot, pi, sin, cos
from dataclasses import dataclass
from typing import List, Tuple

from irma.core.constants import (
    BK, EV, AMU, HBAR, AMASSN, WL2EKIN, _Z_TO_SYMBOL,
)
from irma.core.kernels import start


@dataclass
class AtomSite:
    """One distinct atom species occupying one or more sites in the unit cell.

    Parameters
    ----------
    b_coh_fm : float
        Coherent scattering length [fm].
    positions : list of (x, y, z)
        Fractional coordinates of each atom of this species in the unit cell.
    """
    b_coh_fm: float
    positions: List[Tuple[float, float, float]]

    @property
    def b_coh_sqrtbarn(self) -> float:
        """Coherent scattering length in √barn  (1 barn = 100 fm²)."""
        return self.b_coh_fm / 10.0

@dataclass
class CrystalStructure:
    """Full crystal structure description.

    Parameters
    ----------
    a, b, c : float
        Lattice parameters [Å].
    alpha, beta, gamma : float
        Lattice angles [degrees].
    sites : list of AtomSite
        One entry per distinct atom species.
    """
    a: float
    b: float
    c: float
    alpha: float
    beta:  float
    gamma: float
    sites: List[AtomSite]

    @property
    def n_atoms(self) -> int:
        """Total number of atoms in the unit cell (all sites, all positions)."""
        return sum(len(s.positions) for s in self.sites)

    @property
    def volume(self) -> float:
        """Unit-cell volume [Å³]."""
        ca = np.cos(np.radians(self.alpha))
        cb = np.cos(np.radians(self.beta))
        cg = np.cos(np.radians(self.gamma))
        return (self.a * self.b * self.c *
                np.sqrt(max(0.0, 1.0 - ca**2 - cb**2 - cg**2 + 2.0*ca*cb*cg)))

def lattice_to_cell_params(lattice_ang):
    """Convert a 3×3 row-vector lattice [Å] to (a, b, c, alpha, beta, gamma).

    Rows are the a, b, c cell vectors; the returned angles are in degrees
    (alpha = b^c, beta = a^c, gamma = a^b). The direction cosine is clamped to
    [-1, 1] so floating-point round-off on a near-degenerate cell cannot push
    arccos out of its domain. Shared home for the lattice→cell-params math (the
    NCrystal NCMAT writer and the spectra elastic model both use this).
    """
    L = np.asarray(lattice_ang, float).reshape(3, 3)
    a, b, c = (float(np.linalg.norm(L[i])) for i in range(3))

    def _ang(u, v):
        """Angle between vectors u and v in degrees."""
        cosang = float(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v)))
        return float(np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0))))

    return a, b, c, _ang(L[1], L[2]), _ang(L[0], L[2]), _ang(L[0], L[1])


def _get_reciprocal_lattice_matrix(a, b, c, alpha_deg, beta_deg, gamma_deg):
    """Compute the 3×3 reciprocal lattice matrix G.

    G maps integer Miller indices to Cartesian k-vectors:
        k_vec [Å⁻¹] = G @ [h, k, l]
        d-spacing [Å] = 2π / |k_vec|

    Algorithm mirrors NCrystal NCLatticeUtils.cc (getReciprocalLatticeRot).
    """
    tol  = 1e-10
    k2pi = 2.0 * np.pi
    alpha = np.radians(alpha_deg)
    beta  = np.radians(beta_deg)
    gamma = np.radians(gamma_deg)

    a90  = abs(alpha - np.pi / 2) < tol
    b90  = abs(beta  - np.pi / 2) < tol
    g90  = abs(gamma - np.pi / 2) < tol
    g120 = abs(gamma - 2 * np.pi / 3) < tol

    if a90 and b90 and g90:
        return np.diag([k2pi / a, k2pi / b, k2pi / c])

    if a90 and b90 and g120:
        sq3 = np.sqrt(3.0)
        return np.array([
            [k2pi / a,             0.0,                    0.0    ],
            [k2pi / (a * sq3),  2.0 * k2pi / (b * sq3),   0.0    ],
            [0.0,               0.0,                    k2pi / c  ],
        ])

    if a90 and g90:
        sb   = np.sin(beta)
        cotb = np.cos(beta) / sb
        return np.array([
            [k2pi / a,            0.0,        0.0          ],
            [0.0,                 k2pi / b,   0.0          ],
            [-cotb * k2pi / a,    0.0,        k2pi/(c*sb)  ],
        ])

    # General triclinic. The columns of L are the direct lattice vectors
    # (column norms a, b, c; mutual angles alpha, beta, gamma), so the
    # matrix whose columns are the reciprocal vectors is 2π·L^{-T}: it
    # satisfies a_i · b*_j = 2π δ_ij. Without the transpose the metric
    # (LL^T)^{-1} replaces (L^T L)^{-1} and every oblique d-spacing is wrong.
    ca, cb, cg = np.cos(alpha), np.cos(beta), np.cos(gamma)
    sb, sg     = np.sin(beta),  np.sin(gamma)
    m57 = c * (ca - cb * cg) / sg
    m88 = c * np.sqrt(max(0.0, sb**2 - ((ca - cb * cg) / sg)**2))
    L = np.array([
        [a,    b * cg,  c * cb],
        [0.0,  b * sg,  m57   ],
        [0.0,  0.0,     m88   ],
    ])
    return k2pi * np.linalg.inv(L).T

# Planes whose |F|^2 (or, with several species, whose attenuation-proof bound
# (sum_s |b_s f_s|)^2) is below this are dropped.
_FSQUARECUT = 1e-5


def compute_bragg_edges_general(crystal, emax):
    """Bragg-edge cross-section data for a polycrystalline material up to ``emax`` [eV].

    Returns
    -------
    bragg_data : ndarray, shape (N, 2)
        Edge energy [eV], ascending, and the edge's cross-section contribution
        [barn eV] without Debye-Waller. A flat endpoint (emax, 0) is appended.
    nbe : int
        Number of rows in bragg_data.
    species_corr : ndarray, shape (nbe, nspecies, nspecies)
        Per-species correlation matrix of each edge, for the Debye-Waller factors.
    bragg_dir_terms : list[list[tuple]]
        Per edge, the planes merged into it as ``(G_hat, D_st_plane, site_terms)``,
        with ``site_terms = (cos, sin, species_idx, pref)`` the per-site phases
        (species then position order) and ``pref = d*mult*xsectfact``, for the
        directional and site-resolved Debye-Waller factors.
        ``D_st_plane == pref * (fr fr^T + fi fi^T)``.
    """
    V = crystal.volume
    if V <= 0.0:
        raise ValueError(f"Non-positive unit-cell volume {V:.6g} Å³.")
    xsectfact = 0.5 * WL2EKIN / (V * crystal.n_atoms)
    mult = 2.0      # the half-space loop counts each Friedel pair once

    G = _get_reciprocal_lattice_matrix(
        crystal.a, crystal.b, crystal.c,
        crystal.alpha, crystal.beta, crystal.gamma,
    )
    # Index bounds for d >= dmin: max|h| = a/dmin, etc., for any cell.
    dmin = np.sqrt(WL2EKIN / (4.0 * emax)) * 0.95
    h_max = int(np.ceil(crystal.a / dmin)) + 1
    k_max = int(np.ceil(crystal.b / dmin)) + 1
    l_max = int(np.ceil(crystal.c / dmin)) + 1

    sites_data = [
        (s.b_coh_sqrtbarn, np.asarray(s.positions, dtype=float))
        for s in crystal.sites
    ]
    nspecies = len(crystal.sites)
    # Species index of every site, in the order of the per-plane phase arrays.
    site_species_idx = np.concatenate([
        np.full(len(pos), si, dtype=np.intp)
        for si, (_, pos) in enumerate(sites_data)
    ])

    planes = []
    for h in range(0, h_max + 1):
        k_lo = 0 if h == 0 else -k_max
        for k in range(k_lo, k_max + 1):
            l_lo = 1 if (h == 0 and k == 0) else -l_max
            for l in range(l_lo, l_max + 1):
                k_vec = G @ np.array([h, k, l], dtype=float)
                ksq = float(np.dot(k_vec, k_vec))
                d = 2.0 * np.pi / np.sqrt(ksq)
                E_thr = WL2EKIN / (4.0 * d * d)
                if E_thr > emax:
                    continue

                form_real = np.empty(nspecies)
                form_imag = np.empty(nspecies)
                cos_parts = []
                sin_parts = []
                for si, (_, pos) in enumerate(sites_data):
                    phase = 2.0 * np.pi * (
                        h * pos[:, 0] + k * pos[:, 1] + l * pos[:, 2]
                    )
                    cos_ph = np.cos(phase)
                    sin_ph = np.sin(phase)
                    form_real[si] = float(np.sum(cos_ph))
                    form_imag[si] = float(np.sum(sin_ph))
                    cos_parts.append(cos_ph)
                    sin_parts.append(sin_ph)

                real_part = 0.0
                imag_part = 0.0
                for si in range(nspecies):
                    real_part += sites_data[si][0] * form_real[si]
                    imag_part += sites_data[si][0] * form_imag[si]
                F2 = real_part**2 + imag_part**2

                if F2 < _FSQUARECUT:
                    # Unequal per-species Debye-Waller factors can lift a
                    # cross-species cancellation, so keep the plane unless
                    # even (sum_s |b_s f_s|)^2 is below the cut.
                    if nspecies < 2:
                        continue
                    f2_bound = sum(
                        abs(sites_data[si][0]) * hypot(form_real[si],
                                                       form_imag[si])
                        for si in range(nspecies)) ** 2
                    if f2_bound < _FSQUARECUT:
                        continue

                C_st = (np.outer(form_real, form_real) +
                        np.outer(form_imag, form_imag))
                D_st = d * C_st * mult * xsectfact
                site_terms = (np.concatenate(cos_parts), np.concatenate(sin_parts),
                              site_species_idx, d * mult * xsectfact)
                planes.append((E_thr, d, d * F2 * mult * xsectfact, D_st,
                               k_vec / np.sqrt(ksq), site_terms))

    if not planes:
        return (np.empty((0, 2), dtype=float), 0,
                np.empty((0, nspecies, nspecies), dtype=float), [])

    # Ascending energy; equal energies keep descending d, then enumeration order.
    planes.sort(key=lambda p: (p[0], -p[1]))
    TOLER = 1e-6
    combined = []       # [E, sigma, D_st sum, dir_terms]; merged within TOLER of the first E
    for E, _, sig, D_st, G_hat, site_terms in planes:
        if combined and (E - combined[-1][0]) < TOLER:
            combined[-1][1] += sig
            combined[-1][2] = combined[-1][2] + D_st
            combined[-1][3].append((G_hat, D_st, site_terms))
        else:
            combined.append([E, sig, D_st, [(G_hat, D_st, site_terms)]])

    bragg_data = np.array([[e[0], e[1]] for e in combined], dtype=float)
    species_corr = np.array([e[2] for e in combined], dtype=float)
    bragg_dir_terms = [e[3] for e in combined]
    nbe = len(combined)

    if bragg_data[-1, 0] < emax:
        # Flat extension to emax (ENDF-102 7.2.2): a zero-increment endpoint.
        bragg_data = np.vstack([bragg_data, [emax, 0.0]])
        species_corr = np.concatenate(
            [species_corr, np.zeros((1, nspecies, nspecies), dtype=float)], axis=0)
        bragg_dir_terms.append([])
        nbe += 1

    return bragg_data, nbe, species_corr, bragg_dir_terms


def _frac_pos_matches(pos_a, pos_b, tol=5.0e-4):
    """True if two fractional positions coincide modulo lattice translations.

    The componentwise distance is wrapped into [0, 1) BEFORE taking
    min(d, 1-d): that expression is a periodic distance only for d in
    [0, 1] — for unwrapped coordinates with d > 1 the second term goes
    negative and every position would spuriously "match".
    """
    delta = np.abs(np.asarray(pos_a, dtype=float) - np.asarray(pos_b, dtype=float))
    delta = delta % 1.0
    delta = np.minimum(delta, 1.0 - delta)
    return bool(np.all(delta < tol))


def _build_atom_types_expanded(crystal_info, mesh_data):
    """Build per-phonopy-atom list of atom_type dicts (from crystal_info).

    Returns a list of length N_atoms (phonopy ordering), where each element
    is the matching crystal_info['atom_types'] entry identified by element
    symbol and fractional position modulo lattice translations.
    """

    atom_types_expanded = []
    for sym, pos in zip(mesh_data.atom_symbols, mesh_data.atom_positions):
        symbol_matches = []
        position_matches = []
        for at in crystal_info['atom_types']:
            if _Z_TO_SYMBOL.get(at['Z'], '') != sym:
                continue
            symbol_matches.append(at)
            if any(_frac_pos_matches(pos, at_pos) for at_pos in at['positions']):
                position_matches.append(at)

        matched = None
        if len(position_matches) == 1:
            matched = position_matches[0]
        elif len(position_matches) > 1:
            raise ValueError(
                f"Phonopy atom '{sym}' at fractional position {pos.tolist()} matches "
                f"multiple Card 6d atom types. Refine the structure definition.")
        elif len(symbol_matches) == 1:
            matched = symbol_matches[0]
        if matched is None:
            raise ValueError(
                f"Phonopy atom '{sym}' at fractional position {pos.tolist()} could not be "
                f"matched to any Card 6d atom type by symbol and position. "
                f"Check that Card 6d positions correspond to the phonopy primitive cell.")
        atom_types_expanded.append(matched)
    return atom_types_expanded

def _group_phonopy_atoms_by_type(atom_types, atom_types_expanded):
    """Return phonopy-atom indices grouped by Card 6d atom type."""
    atom_type_site_groups = [
        [i for i, at_d in enumerate(atom_types_expanded) if at_d is at]
        for at in atom_types]

    for si, site_indices in enumerate(atom_type_site_groups):
        if site_indices:
            continue
        at = atom_types[si]
        raise ValueError(
            f"Card 6d atom type {si+1} (Z={at['Z']}, A={at['A']}) has no "
            "matching phonopy atom in the primitive-cell mesh. Check that "
            "the Card 6d positions correspond to the phonopy primitive cell.")

    return atom_type_site_groups

def _wrapped_shift_norm(tau):
    """Norm of a fractional-coordinate shift wrapped into the nearest image."""
    delta = np.asarray(tau, dtype=float) % 1.0
    delta = np.minimum(delta, 1.0 - delta)
    return float(np.sqrt(np.sum(delta * delta)))


def _order_site_groups_by_card6d_positions(atom_types, site_groups,
                                           mesh_positions):
    """Pair every Card 6d position with its phonopy site, per species group.

    ``compute_bragg_edges_general`` flattens its per-plane site phases in
    species-then-Card-6d-position order, so per-site quantities keyed by
    phonopy atom index (e.g. Debye-Waller F-matrices) must be re-paired with
    the Card 6d positions before the site-resolved directional elastic can
    line tensors up with structure-factor phases.

    The deck and the phonopy cell may place the origin differently (e.g. the
    vendored graphite model sits at z=0/0.5 while the deck writes z=1/4/3/4);
    a rigid shift leaves every |F|^2 invariant, so the pairing is done modulo
    ONE shared origin shift tau: candidates come from matching the first
    Card 6d position of species 0 against each phonopy site of its group,
    tried smallest-wrapped-|tau| first (so the shift-free pairing wins when it
    exists, deterministically), and a candidate is accepted only when it pairs
    EVERY group's positions one-to-one.

    Returns a list (per species) of reordered phonopy index lists, or ``None``
    when no single shift produces a complete one-to-one pairing (count
    mismatch, an unmatched position, or an ambiguous double match).
    """
    if any(len(at['positions']) != len(g)
           for at, g in zip(atom_types, site_groups)):
        return None
    ref_pos0 = np.asarray(atom_types[0]['positions'][0], dtype=float)
    candidates = sorted(
        site_groups[0],
        key=lambda gi: (_wrapped_shift_norm(
            ref_pos0 - np.asarray(mesh_positions[gi], dtype=float)), gi))
    for gi0 in candidates:
        tau = ref_pos0 - np.asarray(mesh_positions[gi0], dtype=float)
        ordered_groups = []
        ok = True
        for at, group in zip(atom_types, site_groups):
            remaining = list(group)
            ordered = []
            for pos in at['positions']:
                target = np.asarray(pos, dtype=float) - tau
                matches = [gi for gi in remaining
                           if _frac_pos_matches(mesh_positions[gi], target)]
                if len(matches) != 1:
                    ok = False
                    break
                ordered.append(matches[0])
                remaining.remove(matches[0])
            if not ok:
                break
            ordered_groups.append(ordered)
        if ok:
            return ordered_groups
    return None


# Relative tolerance under which a species group's per-site Debye-Waller
# tensors count as UNIFORM (keeping the byte-pinned species-averaged elastic
# fast path). Symmetry-EQUIVALENT sites produce tensors identical up to
# eigensolver bit noise (~1e-15 relative, living in the symmetry-forbidden
# near-zero elements); crystallographically INEQUIVALENT sites differ at
# >~1e-4 relative (graphite's alpha/beta sublattices: ~1e-3). The ~6 orders
# of magnitude between those scales make the threshold robust.
_TENSOR_UNIFORM_RTOL = 1e-9


def _site_tensors_uniform(site_tensors, site_indices):
    """True when every site tensor in the group is equal up to bit noise."""
    arr = np.asarray(site_tensors)
    F0 = arr[site_indices[0]]
    scale = float(np.max(np.abs(F0)))
    tol = _TENSOR_UNIFORM_RTOL * (scale if scale > 0.0 else 1.0)
    return all(
        np.array_equal(F0, arr[gi])
        or float(np.max(np.abs(arr[gi] - F0))) <= tol
        for gi in site_indices[1:])

# Built-in materials of LEAPR's coher: lat -> (a, c [cm], sigma_coh [b]).
_COHER_MATERIALS = {
    1: (2.4573e-8, 6.700e-8, 5.50),    # graphite
    2: (2.2856e-8, 3.5832e-8, 7.53),   # Be (NJOY's c; the published value is 3.5842e-8)
    3: (2.695e-8, 4.39e-8, 1.0),       # BeO
    4: (4.04e-8, None, 1.495),         # Al (FCC)
    # Pb (FCC): deliberate NJOY divergence, NJOY2016 ships 1.0 b
    # (njoy/NJOY2016#403); 4*pi*b_coh^2 with b_coh = 9.405 fm is 11.115 b.
    5: (4.94e-8, None, 11.115),
    6: (2.86e-8, None, 12.9),          # Fe (BCC)
}


def coher(lat, natom, emax):
    """Compute Bragg energies and structure factors for coherent elastic.

    NJOY's algorithm; its Debye-Waller factor (wint = 0) is left out.
    Returns: bragg array of (energy, structure_factor) pairs, nedge
    """
    if lat not in _COHER_MATERIALS:
        raise ValueError(
            f"coher: invalid built-in material lat={lat} (must be 1-6: "
            f"graphite, Be, BeO, Al, Pb, Fe)")
    twopis = (2.0 * pi)**2
    amne = AMASSN * AMU
    econ = EV * 8.0 * (amne / HBAR) / HBAR
    recon = 1.0 / econ
    tsqx = econ / 20.0
    eps = 0.05
    toler = 1.0e-6

    a, c, sigma_coh = _COHER_MATERIALS[lat]
    scoh = sigma_coh / natom
    if lat < 4:
        c1 = 4.0 / (3.0 * a * a)
        c2 = 1.0 / (c * c)
        sqrt3 = 1.732050808
        scon = scoh * (4.0 * pi)**2 / (2.0 * a * a * c * sqrt3 * econ)
    elif lat <= 5:
        c1 = 3.0 / (a * a)
        scon = scoh * (4.0 * pi)**2 / (16.0 * a * a * a * econ)
    else:
        c1 = 2.0 / (a * a)
        scon = scoh * (4.0 * pi)**2 / (8.0 * a * a * a * econ)
    ulim = econ * emax

    b_tsq = []
    b_f = []
    if lat < 4:
        # Hexagonal: edges with tsq <= tsqx are kept separately; larger ones
        # merge into the first existing edge within a factor 1+eps (NJOY).
        phi = ulim / twopis
        i1m = int(a * sqrt(phi)) + 1
        for i1 in range(1, i1m + 1):
            l1 = i1 - 1
            i2m = int((l1 + sqrt(3.0 * (a * a * phi - l1 * l1))) / 2.0) + 1
            for i2 in range(i1, i2m + 1):
                l2 = i2 - 1
                x = phi - c1 * (l1 * l1 + l2 * l2 - l1 * l2)
                i3m = 0
                if x > 0:
                    i3m = int(c * sqrt(x))
                i3m += 1
                for i3 in range(1, i3m + 1):
                    l3 = i3 - 1
                    w1 = 2.0 if l1 != l2 else 1.0
                    w2 = 2.0
                    if l1 == 0 or l2 == 0:
                        w2 = 1.0
                    if l1 == 0 and l2 == 0:
                        w2 = w2 / 2.0
                    w3 = 2.0 if l3 != 0 else 1.0
                    for sign in (1, -1):
                        tsq = (c1 * (l1 * l1 + l2 * l2 + sign * l1 * l2)
                               + l3 * l3 * c2) * twopis
                        if not (tsq > 0.0 and tsq <= ulim):
                            continue
                        f = w1 * w2 * w3 / sqrt(tsq) * formf(lat, l1, sign * l2, l3)
                        if not b_tsq or tsq <= tsqx:
                            b_tsq.append(tsq)
                            b_f.append(f)
                            continue
                        for ii in range(len(b_tsq)):
                            if tsq >= b_tsq[ii] and tsq < (1 + eps) * b_tsq[ii]:
                                b_f[ii] += f
                                break
                        else:
                            b_tsq.append(tsq)
                            b_f.append(f)
    else:
        # FCC (lat 4-5) or BCC (lat 6), over NJOY's +-15 index box.
        i1m = 15
        twothd = 2.0 / 3.0
        for i1 in range(-i1m, i1m + 1):
            for i2 in range(-i1m, i1m + 1):
                for i3 in range(-i1m, i1m + 1):
                    if lat <= 5:
                        tsq = c1 * (i1*i1 + i2*i2 + i3*i3 + twothd*i1*i2 +
                                    twothd*i1*i3 - twothd*i2*i3) * twopis
                    else:
                        tsq = c1 * (i1*i1 + i2*i2 + i3*i3 + i1*i2 + i2*i3 + i1*i3) * twopis
                    if tsq > 0.0 and tsq <= ulim:
                        b_tsq.append(tsq)
                        b_f.append(1.0 / sqrt(tsq) * formf(lat, i1, i2, i3))

    if not b_tsq:
        return np.array([]), 0

    pairs = sorted(zip(b_tsq, b_f))
    b_tsq = [p[0] for p in pairs] + [ulim]      # final edge at ulim
    b_f = [p[1] for p in pairs]
    b_f.append(b_f[-1])

    # Convert to practical units and combine duplicate Bragg edges
    bragg = []
    bel = -1.0
    for tsq, f in zip(b_tsq, b_f):
        be = tsq * recon
        bs = f * scon
        if be - bel < toler and bragg:
            bragg[-1] = (bragg[-1][0], bragg[-1][1] + bs)
        else:
            bragg.append((be, bs))
            bel = be

    return bragg, len(bragg)

def formf(lat, l1, l2, l3):
    """Compute form factors for the specified lattice."""
    if lat == 1:  # Graphite
        if l3 % 2 != 0:
            return sin(pi * (l1 - l2) / 3.0)**2
        else:
            return (6.0 + 10.0 * cos(2.0 * pi * (l1 - l2) / 3.0)) / 4.0
    elif lat == 2:  # Beryllium
        return 1.0 + cos(2.0 * pi * (2 * l1 + 4 * l2 + 3 * l3) / 6.0)
    elif lat == 3:  # BeO
        return ((1.0 + cos(2.0 * pi * (2 * l1 + 4 * l2 + 3 * l3) / 6.0)) *
                (7.54 + 4.24 + 11.31 * cos(3.0 * pi * l3 / 4.0)))
    elif lat == 4 or lat == 5:  # FCC
        e1 = 2.0 * pi * l1
        e2 = 2.0 * pi * (l1 + l2)
        e3 = 2.0 * pi * (l1 + l3)
        return (1.0 + cos(e1) + cos(e2) + cos(e3))**2 + (sin(e1) + sin(e2) + sin(e3))**2
    elif lat == 6:  # BCC
        e1 = 2.0 * pi * (l1 + l2 + l3)
        return (1.0 + cos(e1))**2 + sin(e1)**2
    return 0.0

def _compute_per_species_msd(crystal_info, tempr_arr, ntempr, dwpix):
    """Per-species Debye-Waller lambda from the Card 6e partial spectra.

    Each spectrum carries its species' full vibrational weight (tbeta = 1).
    A type without a spectrum inherits the principal's lambda ``dwpix``.
    Stored in ``crystal_info['atom_types'][i]['dwpix']``.
    """
    principal = crystal_info.get('principal_atom_idx')
    for iat, at in enumerate(crystal_info['atom_types']):
        if at['spectrum_idx'] is None:
            at['dwpix'] = np.array(dwpix[:ntempr], dtype=float)
            if iat != principal:
                print(f"WARNING: atom type {iat+1} (Z={at['Z']}, A={at['A']}) has "
                      "no Card 6e spectrum; using the principal's Debye-Waller lambda.")
            continue
        sp = crystal_info['partial_spectra'][at['spectrum_idx']]
        at['dwpix'] = np.array([start(sp['rho'], sp['ni'], sp['delta'], BK * T, 1.0)[1]
                                for T in tempr_arr[:ntempr]])
