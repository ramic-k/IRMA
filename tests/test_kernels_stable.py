"""stable() buffer sizing and termination must match NJOY exactly.

The buffer was previously capped at 100,000 points while the loop's only
bounds are eps-convergence and NJOY's ndmax (>= 1,000,000): a physically
reasonable weakly-diffusive deck (small c, large alpha) needs >100k points
and crashed with IndexError. The termination is also pinned to NJOY's
post-increment convention (stop at count >= ndmax-1, leapr.f90:1065), which
keeps the last written index inside the buffer for either ndmax parity and
keeps nsd odd for Simpson integration on the cap path.
"""
import numpy as np

from irma.core.kernels import stable

NDMAX = 1000000  # trans passes max(nbeta, 1000000)


def test_slow_converging_diffusion_needs_more_than_100k_points():
    # Regression for the 100k-cap IndexError: small diffusion constant,
    # large alpha — converges legitimately above the old cap.
    sd, nsd = stable(50.0, 0.03, 0.003, 0.5, NDMAX)
    assert nsd > 100000                      # the old buffer could not hold this
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


def test_free_gas_branch_unchanged_convergence():
    sd, nsd = stable(2.5, 0.05, 0.0, 0.4, NDMAX)
    assert nsd % 2 == 1
    assert 1.0e-7 * sd[0] >= sd[nsd - 1]
    assert sd[0] > 0.0

