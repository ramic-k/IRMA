"""compute_bragg_edges_general (coherent-elastic Bragg edges): the flat,
zero-increment endpoint at emax, and correct d-spacings for oblique
(triclinic, rhombohedral) cells, which take the general branch of
_get_reciprocal_lattice_matrix (2π·L⁻ᵀ), checked against a brute-force
enumeration and against the hexagonal special case."""
import numpy as np
from math import sqrt

import pytest

from irma.core.engine import (
    compute_bragg_edges_general, AtomSite, CrystalStructure, WL2EKIN,
)
from irma.core.crystal import _get_reciprocal_lattice_matrix


def _edges(graphite, emax=5.0):
    return compute_bragg_edges_general(graphite, emax=emax)


def _metric_tensor_inv(cr):
    """Inverse direct metric tensor — the textbook triclinic 1/d² formula:
    |tau(hkl)|² = (2π)² · m·M⁻¹·m with M_ij = a_i·a_j."""
    ca, cb, cg = (np.cos(np.radians(x)) for x in (cr.alpha, cr.beta, cr.gamma))
    M = np.array([
        [cr.a * cr.a,      cr.a * cr.b * cg, cr.a * cr.c * cb],
        [cr.a * cr.b * cg, cr.b * cr.b,      cr.b * cr.c * ca],
        [cr.a * cr.c * cb, cr.b * cr.c * ca, cr.c * cr.c     ],
    ])
    return np.linalg.inv(M)


def _brute_force_edges(cr, emax, dcutoff, fsquarecut=1e-5):
    """Independent enumeration: all (h,k,l) in a generous box, |tau| from the
    metric tensor, same F² and σ accumulation, merged at the same tolerance.
    Returns (edge_energies, edge_sigmas) without the appended emax endpoint."""
    Minv = _metric_tensor_inv(cr)
    ksq_max = (2.0 * np.pi / dcutoff) ** 2
    xsectfact = 0.5 * WL2EKIN / (cr.volume * cr.n_atoms)
    n = [int(np.ceil(x / dcutoff)) + 3 for x in (cr.a, cr.b, cr.c)]

    raw = []
    for h in range(-n[0], n[0] + 1):
        for k in range(-n[1], n[1] + 1):
            for l in range(-n[2], n[2] + 1):
                if h == 0 and k == 0 and l == 0:
                    continue
                m = np.array([h, k, l], dtype=float)
                ksq = (2.0 * np.pi) ** 2 * float(m @ Minv @ m)
                if ksq > ksq_max:
                    continue
                d = 2.0 * np.pi / sqrt(ksq)
                E = WL2EKIN / (4.0 * d * d)
                if E > emax:
                    continue
                re = im = 0.0
                for s in cr.sites:
                    pos = np.asarray(s.positions, dtype=float)
                    ph = 2.0 * np.pi * (h * pos[:, 0] + k * pos[:, 1] + l * pos[:, 2])
                    re += s.b_coh_sqrtbarn * float(np.sum(np.cos(ph)))
                    im += s.b_coh_sqrtbarn * float(np.sum(np.sin(ph)))
                F2 = re * re + im * im
                if F2 < fsquarecut:
                    continue
                # full ±(hkl) space: each plane counted once here, twice in
                # the half-space enumeration's mult=2 — identical totals.
                raw.append((E, d * F2 * xsectfact))

    raw.sort()
    E_out, s_out = [], []
    for E, sig in raw:
        if E_out and (E - E_out[-1]) < 1e-6:
            s_out[-1] += sig
        else:
            E_out.append(E)
            s_out.append(sig)
    return np.array(E_out), np.array(s_out)


@pytest.fixture
def triclinic():
    """A genuinely oblique two-species cell (no special-case branch applies)."""
    A = AtomSite(b_coh_fm=6.0, positions=[(0.0, 0.0, 0.0), (0.5, 0.5, 0.5)])
    B = AtomSite(b_coh_fm=4.5, positions=[(0.3, 0.41, 0.17)])
    return CrystalStructure(a=3.1, b=5.7, c=4.3,
                            alpha=72.0, beta=65.0, gamma=100.0, sites=[A, B])


@pytest.fixture
def rhombohedral():
    """alpha=beta=gamma=80° — also routes through the general branch."""
    A = AtomSite(b_coh_fm=5.0, positions=[(0.0, 0.0, 0.0)])
    return CrystalStructure(a=4.0, b=4.0, c=4.0,
                            alpha=80.0, beta=80.0, gamma=80.0, sites=[A])


@pytest.mark.parametrize("cell", ["triclinic", "rhombohedral"])
def test_oblique_edges_complete_and_correct(cell, request):
    """Edge set must match an independent brute-force enumeration over a much
    larger hkl box: same count, energies, per-edge and total σ. Catches both
    a wrong metric and too-tight enumeration bounds."""
    cr = request.getfixturevalue(cell)
    emax, dcut = 2.0, sqrt(WL2EKIN / (4.0 * 2.0)) * 0.95
    bragg, nbe, _, _ = compute_bragg_edges_general(cr, emax=emax)
    # Strip the appended flat emax endpoint
    E_code, s_code = bragg[:-1, 0], bragg[:-1, 1]
    E_ref, s_ref = _brute_force_edges(cr, emax, dcut)
    assert len(E_code) == len(E_ref) > 20
    np.testing.assert_allclose(E_code, E_ref, rtol=1e-9)
    np.testing.assert_allclose(s_code, s_ref, rtol=1e-9)


def test_general_branch_consistent_with_hexagonal_branch(graphite):
    """A gamma infinitesimally off 120° routes through the general triclinic
    branch; |tau| for every hkl must agree with the hard-coded hexagonal
    special case (the matrices differ only by an overall rotation)."""
    g_special = _get_reciprocal_lattice_matrix(2.464, 2.464, 6.711, 90., 90., 120.)
    g_general = _get_reciprocal_lattice_matrix(2.464, 2.464, 6.711, 90., 90., 120. + 1e-7)
    for hkl in [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (1, 0, 2),
                (2, -1, 3), (1, 2, -1)]:
        m = np.array(hkl, dtype=float)
        assert np.linalg.norm(g_special @ m) == pytest.approx(
            np.linalg.norm(g_general @ m), rel=1e-6), hkl


def test_graphite_edges(graphite):
    """Shapes, ascending energies within emax, nonnegative increments, and
    the flat zero-increment endpoint appended at emax (S is a plain 1/E
    extension above the last edge); a lower emax gives fewer edges."""
    bragg, nbe, species_corr, dir_terms = _edges(graphite, emax=5.0)
    E = bragg[:, 0]
    nsp = len(graphite.sites)
    assert nbe > 100                                  # graphite has many edges below 5 eV
    assert bragg.shape == (nbe, 2)
    assert species_corr.shape == (nbe, nsp, nsp) and len(dir_terms) == nbe
    assert E[0] > 0.0 and np.all(np.diff(E) >= 0) and np.isclose(E[-1], 5.0)
    assert np.all(bragg[:, 1] >= 0.0)
    assert bragg[-1, 1] == 0.0                        # zero-increment endpoint
    assert np.allclose(species_corr[-1], 0.0) and dir_terms[-1] == []
    bragg2, nbe2, _, _ = _edges(graphite, emax=2.0)
    assert nbe2 < nbe and bragg2[:, 0].max() <= 2.0 + 1e-9
