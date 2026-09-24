"""Physical-constant single-sourcing (CODATA 2018 / SI 2019).

Every path derives its constants from the defining constants in
irma.core.constants:

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

