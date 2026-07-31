"""Per-species Debye-Waller from Card 6e partial phonon spectra.

_compute_per_species_msd (iel=10, inelastic_mode=0) computes a DW lambda for
each atom type that has a matching partial spectrum, using the same
transform/normalize/fsum chain as LEAPR's start().  The spectrum branch used
to crash with ``NameError: name 'fsum' is not defined`` (missing import) —
the fallback branch masked it because no validated deck carried Card 6e.
"""
import numpy as np
import pytest

from irma.core.constants import BK
from irma.core.crystal import _compute_per_species_msd
from irma.core.kernels import start

# A smooth Debye-like quadratic-onset spectrum on an equidistant grid.
_NI = 60
_DELTA_E = 0.0015  # eV
_RHO = [(j * _DELTA_E) ** 2 for j in range(_NI)]


def _crystal_info(spectrum_idx):
    return {
        'atom_types': [
            {'Z': 4, 'A': 9, 'awr': 8.9348, 'spectrum_idx': spectrum_idx},
            {'Z': 8, 'A': 16, 'awr': 15.8575, 'spectrum_idx': None},
        ],
        'partial_spectra': [
            {'Z': 4, 'A': 9, 'delta': _DELTA_E, 'ni': _NI, 'rho': list(_RHO)},
        ],
        'principal_atom_idx': 0,
    }


def test_spectrum_branch_runs_and_matches_start():
    """The partial-spectrum branch must reproduce start()'s f0 for the same
    spectrum with full weight (tbeta=1) — they share the identical algorithm."""
    tempr = np.array([296.0, 600.0])
    dwpix = np.array([0.111, 0.222])
    info = _crystal_info(spectrum_idx=0)

    _compute_per_species_msd(info, tempr, 2, dwpix)

    f0 = info['atom_types'][0]['dwpix']
    assert f0.shape == (2,)
    assert np.all(f0 > 0.0)
    for itemp, temp in enumerate(tempr):
        _, f0_ref, _, _ = start(
            np.array(_RHO), _NI, _DELTA_E, BK * temp, 1.0)
        assert f0[itemp] == pytest.approx(f0_ref, rel=1e-12)


def test_no_spectrum_falls_back_to_principal_dw():
    tempr = np.array([296.0, 600.0])
    dwpix = np.array([0.111, 0.222])
    info = _crystal_info(spectrum_idx=0)

    _compute_per_species_msd(info, tempr, 2, dwpix)

    np.testing.assert_allclose(info['atom_types'][1]['dwpix'], dwpix)


def test_fallback_warns_for_non_principal_type(capsys):
    """A NON-principal atom type without a Card 6e spectrum inherits the
    principal's DW lambda — that must be a loud WARNING naming the type,
    what it inherited, and the Card 6e remedy.  Be (awr 8.93) vs O (awr
    15.86) are > 20% apart, so the poor-approximation clause fires too."""
    tempr = np.array([296.0, 600.0])
    dwpix = np.array([0.111, 0.222])
    info = _crystal_info(spectrum_idx=0)

    _compute_per_species_msd(info, tempr, 2, dwpix)

    out = capsys.readouterr().out
    assert "WARNING: Card 6d atom type 2 (Z=8, A=16)" in out
    assert "no matching Card 6e partial spectrum" in out
    assert "INHERITED from the principal scatterer" in out
    assert "more than 20%" in out
    assert "raise Card 6b nspec" in out


def test_fallback_warning_omits_poor_clause_for_similar_mass(capsys):
    """Mass-similar fallback (C-12-like principal, C-13 partner: ~8% apart)
    still warns, but without the >20% poor-approximation escalation."""
    info = _crystal_info(spectrum_idx=0)
    info['atom_types'][0]['awr'] = 11.8980
    info['atom_types'][1].update({'Z': 6, 'A': 13, 'awr': 12.8916})

    _compute_per_species_msd(info, np.array([296.0]), 1, np.array([0.111]))

    out = capsys.readouterr().out
    assert "WARNING: Card 6d atom type 2 (Z=6, A=13)" in out
    assert "more than 20%" not in out


def test_principal_type_fallback_is_quiet(capsys):
    """The PRINCIPAL type inheriting dwpix is exact (dwpix IS its own
    lambda): an info line only, never a WARNING."""
    info = _crystal_info(spectrum_idx=None)
    info['atom_types'][1]['spectrum_idx'] = 0

    _compute_per_species_msd(info, np.array([296.0]), 1, np.array([0.111]))

    out = capsys.readouterr().out
    assert "WARNING" not in out
    assert "using principal DW" in out
    np.testing.assert_allclose(info['atom_types'][0]['dwpix'], [0.111])


def test_dw_lambda_increases_with_temperature():
    """LEAPR's lambda (f0 = integral of rho(eps) coth(eps/2kT)/eps, units
    1/eV) grows with T as the phonon population factor approaches 2kT/eps."""
    tempr = np.array([200.0, 400.0, 800.0])
    dwpix = np.zeros(3)
    info = _crystal_info(spectrum_idx=0)

    _compute_per_species_msd(info, tempr, 3, dwpix)

    f0 = info['atom_types'][0]['dwpix']
    assert 0.0 < f0[0] < f0[1] < f0[2]
