"""Pin the built-in coherent-elastic materials (``coher``, iel=1-6).

``coher`` reimplements LEAPR's hard-wired Bragg-edge generators (hexagonal
graphite/Be/BeO, fcc Al/Pb, bcc Fe). The pinned values below are the current
outputs; for graphite (iel=1), aluminum (iel=4) and iron (iel=6) they are
indirectly validated end-to-end: after the writer's degenerate-edge merging
they reproduce the native LEAPR/NJOY reference tapes exactly (221/568/602
written edges, total strength to 5 figures — see
tests/native_LEAPR_NJOY_ENDF_validation/). Be/BeO/Pb are pinned to current
behavior to guard against accidental change. Pb deliberately diverges from
NJOY: its sum-of-structure-factors is 11.115x NJOY's, because NJOY's
pb4 = 1.0 barn is a placeholder for the physical sigma_coh(Pb) = 11.115 b
(see the DELIBERATE NJOY DIVERGENCE note in irma/core/crystal.py).

Raw (pre-merge) edge counts differ from the written counts: e.g. graphite
345 raw -> 221 written.
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


@pytest.mark.parametrize("lat", sorted(PINNED))
def test_builtin_material_structure(lat):
    """Edges ascend to emax and carry non-negative structure factors."""
    bragg, nedge = coher(lat, 1, EMAX)
    energies = np.array([bragg[i][0] for i in range(nedge)])
    factors = np.array([bragg[i][1] for i in range(nedge)])

    assert np.all(np.diff(energies) >= 0.0)
    assert energies[-1] == pytest.approx(EMAX, rel=1e-12)
    assert np.all(factors >= 0.0)
    assert np.all(np.isfinite(factors))


def test_natom_scales_strength_inversely():
    """The per-atom normalization: structure factors scale as 1/natom."""
    bragg1, n1 = coher(1, 1, EMAX)
    bragg2, n2 = coher(1, 2, EMAX)
    assert n1 == n2
    s1 = sum(bragg1[i][1] for i in range(n1))
    s2 = sum(bragg2[i][1] for i in range(n2))
    assert s1 == pytest.approx(2.0 * s2, rel=1e-12)


def test_coher_invalid_lat_is_loud():
    """An out-of-range built-in material index must not die as an opaque
    UnboundLocalError."""
    from irma.core.crystal import coher
    with pytest.raises(ValueError, match="invalid built-in material"):
        coher(7, 1, 5.0)


def test_pb_iel5_uses_physical_sigma_coh():
    """NJOY's pb4=1.0 placeholder made every iel=5 tape ~11.1x low. The
    structure factors scale linearly with scoh, so the total must be 11.115x
    the placeholder value (here on a 1 eV window, distinct from the EMAX=5
    pin above)."""
    bragg, nedge = coher(5, 1, 1.0)
    assert nedge > 0
    sf_total = float(np.sum(np.asarray(bragg).reshape(-1, 2)[:nedge, 1]))
    # frozen pin from the fixed run; NJOY's placeholder gave sf_total/11.115
    assert sf_total == pytest.approx(7.475381153183254, rel=1e-10)
