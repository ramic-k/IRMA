"""irma.spectra native-engine bridge (P1) -- self-contained, no external data.

Pins the engine -> spectra contract: (1) ``_pick_sqe_key`` must stay in lockstep
with ``standalone_sab._pick_sab_key`` (same mode/order branch, ``sqe_`` prefix +
``_barn_per_meV`` suffix), and (2) ``from_noncubic_arrays`` wraps the physical
``sqe_*`` map with NO SAB inversion / NO ``exp(+beta/2)`` and auto-orients.
"""
import numpy as np
import pytest

from irma.spectra import from_noncubic_arrays, _pick_sqe_key
from irma.core.standalone_sab import _pick_sab_key


@pytest.mark.parametrize("mode", [1, 2])
@pytest.mark.parametrize("order", [1, 2, 100])
def test_pick_sqe_key_tracks_pick_sab_key(mode, order):
    """The sqe-key selector is the sab-key selector with prefix/suffix swap."""
    sab = _pick_sab_key(None, order, mode)
    sqe = _pick_sqe_key(None, order, mode)
    expected = sab.replace("sab_asym_downscatter_", "sqe_") + "_barn_per_meV"
    assert sqe == expected


def test_pick_sqe_key_rejects_bad_mode():
    with pytest.raises(ValueError):
        _pick_sqe_key(None, 100, 3)


def test_pick_sqe_key_validates_presence():
    # mode-2/order-100 selects the multiphonon-total key; absent -> clear KeyError
    with pytest.raises(KeyError):
        _pick_sqe_key({"some_other_key": None}, 100, 2)
    # present -> returns it
    key = "sqe_one_phonon_total_plus_incoherent_approx_multiphonon_barn_per_meV"
    assert _pick_sqe_key({key: object()}, 100, 2) == key


def test_from_noncubic_arrays_no_inversion_and_orientation():
    q = np.linspace(0.5, 10.0, 5)
    E = np.linspace(0.0, 200.0, 7)
    S = np.arange(35.0).reshape(5, 7)          # (nq, nE)
    p = from_noncubic_arrays(q, E, S, T_K=296.0, sigma_b=5.551)
    # values pass through UNCHANGED (no 4*pi*kT/sigma_b, no exp(+beta/2))
    assert np.array_equal(p.S, S)
    assert p.S.shape == (q.size, E.size)
    # transposed input is auto-oriented to (nq, nE)
    p2 = from_noncubic_arrays(q, E, S.T, T_K=296.0, sigma_b=5.551)
    assert np.array_equal(p2.S, S)


def test_from_noncubic_arrays_bad_shape_raises():
    q = np.linspace(0.5, 10.0, 5)
    E = np.linspace(0.0, 200.0, 7)
    with pytest.raises(ValueError):
        from_noncubic_arrays(q, E, np.zeros((4, 4)), T_K=296.0, sigma_b=5.551)
