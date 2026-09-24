"""irma.spectra native-engine bridge -- self-contained, no external data.

Pins the engine -> spectra contract: (1) ``_pick_sqe_key`` must stay in lockstep
with ``standalone_sab._pick_sab_key`` (same mode/order branch, ``sqe_`` prefix +
``_barn_per_meV`` suffix), and (2) ``from_noncubic_arrays`` wraps the physical
``sqe_*`` map with NO SAB inversion / NO ``exp(+beta/2)``.
"""
import numpy as np
import pytest

from irma.spectra import from_noncubic_arrays, _pick_sqe_key
from irma.core.standalone_sab import _pick_sab_key


@pytest.mark.parametrize("mode", [1, 2])
@pytest.mark.parametrize("order", [1, 2, 100])
def test_pick_sqe_key_tracks_pick_sab_key(mode, order):
    """The sqe-key selector is the sab-key selector with prefix/suffix swap."""
    sab = _pick_sab_key(order, mode)
    sqe = _pick_sqe_key(order, mode)
    expected = sab.replace("sab_asym_downscatter_", "sqe_") + "_barn_per_meV"
    assert sqe == expected


def test_from_noncubic_arrays_no_inversion():
    q = np.linspace(0.5, 10.0, 5)
    E = np.linspace(0.0, 200.0, 7)
    S = np.arange(35.0).reshape(5, 7)          # (nq, nE)
    p = from_noncubic_arrays(q, E, S, T_K=296.0, sigma_b=5.551)
    # values pass through UNCHANGED (no 4*pi*kT/sigma_b, no exp(+beta/2))
    assert np.array_equal(p.S, S)
    assert p.S.shape == (q.size, E.size)
