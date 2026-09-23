"""Pin the built-in coherent-elastic materials (``coher``, iel=1-6): edge
count, first edge, total strength and edge structure at emax = 5 eV.

Pb deliberately diverges from NJOY: its total is 11.115x NJOY's, because
NJOY's pb4 = 1.0 barn is a placeholder for the physical sigma_coh(Pb) =
11.115 b (see the DELIBERATE NJOY DIVERGENCE note in irma/core/crystal.py).
"""
import numpy as np
import pytest

from irma.core.engine import coher

EMAX = 5.0

#         lat: (name, nedge, E_first (eV), sum of structure factors)
PINNED = {
    1: ("graphite", 345, 4.55581479e-04, 2.74922845e+01),
    2: ("Be",       254, 1.59284518e-03, 3.90427313e+01),
    3: ("BeO",      306, 1.06117406e-03, 6.16546380e+01),
    4: ("Al",       617, 3.75901614e-03, 1.65765018e+00),
    5: ("Pb",       617, 2.51410275e-03, 8.24270944e+00),
    6: ("Fe",       641, 5.00050188e-03, 1.79029412e+01),
}


@pytest.mark.parametrize("lat", sorted(PINNED))
def test_builtin_material_pinned(lat):
    name, nedge_exp, e_first_exp, sum_f_exp = PINNED[lat]
    bragg, nedge = coher(lat, 1, EMAX)
    energies = np.array([bragg[i][0] for i in range(nedge)])
    factors = np.array([bragg[i][1] for i in range(nedge)])

    assert nedge == nedge_exp, f"{name}: edge count changed"
    assert energies[0] == pytest.approx(e_first_exp, rel=1e-6), \
        f"{name}: first Bragg edge moved (lattice constants?)"
    assert factors.sum() == pytest.approx(sum_f_exp, rel=1e-6), \
        f"{name}: total coherent-elastic strength changed"
    assert np.all(np.diff(energies) >= 0.0)
    assert energies[-1] == pytest.approx(EMAX, rel=1e-12)
    assert np.all(factors >= 0.0) and np.all(np.isfinite(factors))


def test_natom_scales_strength_inversely():
    """The per-atom normalization: structure factors scale as 1/natom."""
    bragg1, n1 = coher(1, 1, EMAX)
    bragg2, n2 = coher(1, 2, EMAX)
    assert n1 == n2
    s1 = sum(bragg1[i][1] for i in range(n1))
    s2 = sum(bragg2[i][1] for i in range(n2))
    assert s1 == pytest.approx(2.0 * s2, rel=1e-12)
