"""Per-species Debye-Waller from each temperature's Card 12e partial spectra.

_compute_per_species_msd (iel=10, inelastic_mode=0) computes a DW lambda at
each temperature for each atom type with a partial spectrum in that
temperature's block, using the same transform/normalize/fsum chain as
LEAPR's start(); a type without one inherits the principal's DW lambda.
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


def _spectrum(delta):
    return {'Z': 8, 'A': 16, 'delta': delta, 'ni': _NI, 'rho': list(_RHO)}


def _crystal_info(spectra_by_temp):
    """Principal Be (type 0) and O (type 1); spectra_by_temp[itemp] maps an
    atom type to its spectrum at that temperature."""
    return {
        'atom_types': [
            {'Z': 4, 'A': 9, 'awr': 8.9348},
            {'Z': 8, 'A': 16, 'awr': 15.8575},
        ],
        'partial_spectra_by_temp': spectra_by_temp,
        'principal_atom_idx': 0,
    }


def test_each_temperature_uses_its_own_spectrum():
    """At every temperature the lambda equals start()'s f0 for that
    temperature's spectrum with full weight (tbeta=1)."""
    tempr = np.array([296.0, 600.0])
    by_temp = [{1: _spectrum(_DELTA_E)}, {1: _spectrum(0.8 * _DELTA_E)}]
    info = _crystal_info(by_temp)

    _compute_per_species_msd(info, tempr, 2, np.array([0.111, 0.222]))

    f0 = info['atom_types'][1]['dwpix']
    for itemp, temp in enumerate(tempr):
        sp = by_temp[itemp][1]
        _, f0_ref, _, _ = start(np.array(sp['rho']), sp['ni'], sp['delta'],
                                BK * temp, 1.0)
        assert f0[itemp] == pytest.approx(f0_ref, rel=1e-12)


def test_no_spectrum_falls_back_to_principal_dw():
    """A species without a spectrum takes the principal's lambda at each
    temperature."""
    dwpix = np.array([0.111, 0.222])
    info = _crystal_info([{}, {}])

    _compute_per_species_msd(info, np.array([296.0, 600.0]), 2, dwpix)

    np.testing.assert_allclose(info['atom_types'][1]['dwpix'], dwpix)


def test_fallback_warns_only_for_non_principal_types(capsys):
    """A non-principal type without a Card 12e spectrum inherits the
    principal's DW lambda with a WARNING naming the type; the principal
    inheriting its own lambda is exact and quiet."""
    _compute_per_species_msd(_crystal_info([{}]), np.array([296.0]), 1,
                             np.array([0.111]))
    out = capsys.readouterr().out
    assert "WARNING: atom type 2 (Z=8, A=16)" in out
    assert "atom type 1" not in out

    _compute_per_species_msd(_crystal_info([{1: _spectrum(_DELTA_E)}]),
                             np.array([296.0]), 1, np.array([0.111]))
    assert "WARNING" not in capsys.readouterr().out
