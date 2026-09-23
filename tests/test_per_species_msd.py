"""Per-species Debye-Waller from Card 6e partial phonon spectra.

_compute_per_species_msd (iel=10, inelastic_mode=0) computes a DW lambda for
each atom type that has a matching partial spectrum, using the same
transform/normalize/fsum chain as LEAPR's start(); a type without one
inherits the principal's DW lambda.
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


def test_fallback_warns_only_for_non_principal_types(capsys):
    """A non-principal type without a Card 6e spectrum inherits the
    principal's DW lambda with a WARNING naming the type; the principal
    inheriting its own lambda is exact and quiet."""
    _compute_per_species_msd(_crystal_info(spectrum_idx=0),
                             np.array([296.0, 600.0]), 2, np.array([0.111, 0.222]))
    assert "WARNING: atom type 2 (Z=8, A=16)" in capsys.readouterr().out

    info = _crystal_info(spectrum_idx=None)
    info['atom_types'][1]['spectrum_idx'] = 0
    _compute_per_species_msd(info, np.array([296.0]), 1, np.array([0.111]))
    assert "WARNING" not in capsys.readouterr().out
    np.testing.assert_allclose(info['atom_types'][0]['dwpix'], [0.111])
