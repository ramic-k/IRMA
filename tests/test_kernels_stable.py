"""stable() buffer sizing and termination match NJOY: the buffer holds a
weakly-diffusive deck that needs more than 100k points, and the loop stops
by NJOY's post-increment convention (count >= ndmax-1, leapr.f90:1065),
which keeps the last index inside the buffer for either ndmax parity and nsd
odd for Simpson integration on the cap path.
"""
import numpy as np

from irma.core.kernels import stable

NDMAX = 1000000  # trans passes max(nbeta, 1000000)


def test_slow_converging_diffusion_needs_more_than_100k_points():
    # small diffusion constant, large alpha: converges after more than 100k points
    sd, nsd = stable(50.0, 0.03, 0.003, 0.5, NDMAX)
    assert nsd > 100000
    assert nsd % 2 == 1                      # Simpson parity
    assert 1.0e-7 * sd[0] >= sd[nsd - 1]     # terminated by convergence, not cap


def test_cap_termination_matches_njoy_post_increment_rule():
    # NJOY stops when its post-increment 1-based count reaches ndmax,
    # i.e. nsd = ndmax - 1 for even ndmax — never writing past the buffer.
    sd, nsd = stable(50.0, 0.03, 0.003, 0.5, 10)
    assert nsd == 9
    sd, nsd = stable(50.0, 0.03, 0.003, 0.5, 11)   # odd ndmax parity
    assert nsd == 11
    assert np.all(np.isfinite(sd[:nsd]))


def test_free_gas_branch_converges():
    sd, nsd = stable(2.5, 0.05, 0.0, 0.4, NDMAX)
    assert nsd % 2 == 1
    assert 1.0e-7 * sd[0] >= sd[nsd - 1]
    assert sd[0] > 0.0

