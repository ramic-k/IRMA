"""The (isym, ilog) storage transform of the MF7/MT4 values (``_endf_s``).

Every expectation is hand-computed from the literal values; the row and
mirror selection is covered by the flag tapes and the NJOY minitapes.
"""
import math

import pytest

from irma.core.endf_writer import _endf_s

SMIN = 1e-75
BE = 0.9

CELLS = [
    # isym, ilog, S, be, expected, rel
    (0, 0, 0.22, BE, 0.22 * math.exp(-BE / 2.0), 5e-7),
    (0, 0, 1e-80, BE, 0.0, 0.0),                       # below smin
    (0, 1, 0.13, BE, math.log(0.13) - BE / 2.0, 5e-7),
    (0, 1, 1e-80, BE, math.log(1e-80) - BE / 2.0, 5e-7),  # no smin floor in log form
    (1, 0, 0.04, BE, 0.04 * math.exp(BE / 2.0), 5e-7),
    (1, 1, 0.13, BE, math.log(0.13) + BE / 2.0, 5e-7),  # '+ be/2' in the log form
    (1, 0, 1.2345678e-12, 0.0, 1.23457e-12, 1e-7),     # 6-figure rounding below 1e-9
    (2, 0, 0.013, BE, 0.013, 5e-7),
    (2, 1, 0.22, BE, math.log(0.22), 5e-7),
    (3, 0, 0.065, BE, 0.065, 5e-7),
    (3, 1, 0.11, BE, math.log(0.11), 5e-7),
    (3, 0, 1.2345678e-12, BE, 1.23457e-12, 1e-7),
]


@pytest.mark.parametrize("isym,ilog,s,be,expected,rel", CELLS)
def test_storage_transform(isym, ilog, s, be, expected, rel):
    assert _endf_s(s, be, isym, ilog, SMIN) == pytest.approx(expected, rel=rel)


@pytest.mark.parametrize("isym", [0, 1, 2, 3])
def test_zero_s_is_zero_in_linear_and_minus999_in_log_storage(isym):
    assert _endf_s(0.0, BE, isym, 0, SMIN) == 0.0
    assert _endf_s(0.0, BE, isym, 1, SMIN) == -999.0
