"""Physical-constant single-sourcing (CODATA 2018 / SI 2019).

The noncubic/phonopy paths used to carry their own literals: hbar^2/2m_n as
2.072146 (~10 ppm high) in three modules, phonopy's old Boltzmann
8.6173383e-5 in the Bose factors, CODATA-1986 e/AMU/hbar, and THzToEv twice.
Everything now derives from the defining constants in irma.core.constants:

  - h = 6.62607015e-34 J s, e = 1.602176634e-19 C, k_B = 1.380649e-23 J/K
    (exact, 2019 SI redefinition);
  - m_n = 1.67492749804e-27 kg, m_u = 1.66053906660e-27 kg (CODATA 2018:
    E. Tiesinga, P. J. Mohr, D. B. Newell, B. N. Taylor, Rev. Mod. Phys. 93,
    025010 (2021); https://physics.nist.gov/constants).

BK (the classic, NJOY-reproducing path) is the exact k_B/e value to 10
significant figures and is intentionally shared by both paths.
"""
from math import pi

import pytest

from irma.core import constants as c


def test_derived_constants_match_defining_expressions():
    assert c.HBAR_J_S == c.PLANCK_J_S / (2.0 * pi)
    assert c.HBAR_EV_S == pytest.approx(6.582119569e-16, rel=1e-9)
    assert c.THZ_TO_EV == pytest.approx(4.135667696e-3, rel=1e-9)
    assert c.HBAR2_OVER_2MN_MEV_A2 == pytest.approx(2.0721248551, rel=1e-9)


def test_bk_is_exact_si_value_to_ten_figures():
    assert c.BK == pytest.approx(c.KB_J_PER_K / c.ECHARGE_C, rel=1e-10)


def test_modules_share_the_constants_object():
    """No module may carry its own literal copy again."""
    from irma.core import phonopy_io, sab_grids, noncubic_engine

    assert phonopy_io.HBAR2_OVER_2MN_MEV_A2 is c.HBAR2_OVER_2MN_MEV_A2
    # standalone_sab's Q<->alpha conversion moved into sab_grids (commit f880397),
    # which now owns the shared hbar^2/2m_n (imported as _HBAR2). Guard it there.
    assert sab_grids._HBAR2 is c.HBAR2_OVER_2MN_MEV_A2
    assert noncubic_engine.HBAR2_OVER_2MN_MEV_A2 is c.HBAR2_OVER_2MN_MEV_A2
    assert noncubic_engine.THzToEv is c.THZ_TO_EV
    assert noncubic_engine.EV is c.ECHARGE_C
    assert noncubic_engine.AMU is c.AMU_KG
    assert noncubic_engine.Hbar is c.HBAR_EV_S
    assert noncubic_engine.NEUTRON_MASS_AMU is c.AMASSN
    assert sab_grids._BK_EV_PER_K is c.BK
    assert noncubic_engine.KB_MEV_PER_K == pytest.approx(c.BK * 1e3, rel=1e-15)


def test_mode_floor_mask_two_tier():
    """Goldstone guard at Gamma (0.1 meV), 1-ueV overflow guard elsewhere."""
    import numpy as np
    from irma.core.phonopy_io import mode_floor_mask

    qpoints = np.array([[0.0, 0.0, 0.0],      # Gamma
                        [0.25, 0.0, 0.0]])    # off-Gamma
    # per q: 3 modes — [ASR noise, soft-but-physical, ordinary]
    energies = np.array([0.02, 0.05, 12.0,    # Gamma modes
                         0.02, 0.05, 12.0])   # off-Gamma modes
    keep = mode_floor_mask(energies, qpoints, n_branches=3)
    # Gamma: sub-0.1-meV noise dropped, ordinary optical kept
    assert list(keep[:3]) == [False, False, True]
    # off-Gamma: everything above 1 ueV is physical and kept
    assert list(keep[3:]) == [True, True, True]
    # exact zero / negative never passes anywhere
    keep0 = mode_floor_mask(np.array([0.0, -1.0, 1e-6, 2e-3] * 1),
                            np.array([[0.25, 0.0, 0.0]]), n_branches=4)
    assert list(keep0) == [False, False, False, True]

