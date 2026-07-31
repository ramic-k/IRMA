"""irma.spectra.elastic._endf_float -- ENDF 11-char Fortran float field parser.

Pins the missing-exponent-'E' insertion AND the leading-sign handling: a leading
+/- is the mantissa sign, NOT an exponent marker, so a cleanup that dropped the
``s[0]`` split would silently misparse negative TSL fields.
"""
import pytest

from irma.spectra.elastic import _endf_float


@pytest.mark.parametrize("s,expected", [
    ("1.234", 1.234),
    (" 2.5 ", 2.5),                # surrounding whitespace stripped
    ("1.23-4", 1.23e-4),          # Fortran exponent with the 'E' omitted
    ("1.23+4", 1.23e4),
    ("-1.23-4", -1.23e-4),        # leading '-' is the SIGN, only the exponent splits
    ("+1.5+2", 1.5e2),            # leading '+' sign + positive exponent
    ("-6.6e-5", -6.6e-5),         # already a valid float (has 'e') -> fast path
    ("6.646-5", 6.646e-5),        # carbon coherent length, ENDF style
    ("", 0.0),                    # blank field
    ("   ", 0.0),
    ("0.0", 0.0),
])
def test_endf_float(s, expected):
    assert _endf_float(s) == pytest.approx(expected, rel=1e-12, abs=1e-30)
