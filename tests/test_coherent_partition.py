"""Pairwise coherent-interference partition (principal_weighted_coherent_partition).

The per-principal coherent one-phonon partial keeps the exact self term and a
share of each pairwise interference term proportional to the participants'
bound coherent cross sections, w_p / (w_p + w_o). Summed over principals every
pair term regains coefficient exactly 1, so the per-species partials rebuild
the exact coherent total for ANY number of site groups. The historical scalar
form (w_p times the whole cross sum) achieved this only for two groups; these
tests pin both the general sum rule and bit-identity with the historical form
in the two-group case.
"""

import numpy as np

from irma.core.noncubic_workers import principal_weighted_coherent_partition


def _random_amplitudes(n_groups, n_modes, seed, n_qsamples=None):
    rng = np.random.default_rng(seed)
    shape = (n_groups, n_modes) if n_qsamples is None else (n_qsamples, n_groups, n_modes)
    return rng.normal(size=shape) + 1j * rng.normal(size=shape)


def _exact_total(amplitudes):
    return np.abs(amplitudes.sum(axis=-2)) ** 2


def _bits_equal(a, b):
    """True bit-pattern equality for float64 arrays (0.0 != -0.0 here,
    unlike np.array_equal)."""
    a = np.asarray(a)
    b = np.asarray(b)
    return (
        a.dtype == np.float64
        and b.dtype == np.float64
        and a.shape == b.shape
        and np.array_equal(a.view(np.uint64), b.view(np.uint64))
    )


def test_sum_over_principals_is_exact_for_three_groups():
    amplitudes = _random_amplitudes(3, 24, seed=1)
    weights = np.array([2.0, 5.0, 1.0])
    summed = sum(
        principal_weighted_coherent_partition(amplitudes, p, weights)[0]
        for p in range(3)
    )
    np.testing.assert_allclose(summed, _exact_total(amplitudes), rtol=1e-12)


def test_sum_over_principals_is_exact_for_five_groups_batched():
    amplitudes = _random_amplitudes(5, 12, seed=2, n_qsamples=7)
    weights = np.array([0.1, 3.0, 0.7, 2.2, 1.5])
    summed = sum(
        principal_weighted_coherent_partition(amplitudes, p, weights)[0]
        for p in range(5)
    )
    np.testing.assert_allclose(summed, _exact_total(amplitudes), rtol=1e-12)


def test_two_groups_bit_identical_to_historical_scalar_share():
    amplitudes = _random_amplitudes(2, 32, seed=3)
    weights = np.array([11.898 * 2, 4.232])  # BeO-like unequal coherent weights
    for p in range(2):
        total, self_term, cross = principal_weighted_coherent_partition(
            amplitudes, p, weights)
        # Scalar reference form: the sigma_coh share of the full cross sum.
        # For exactly two groups the pairwise partition must reduce to this
        # bit-for-bit (see principal_weighted_coherent_partition).
        share = float(weights[p] / float(np.sum(weights)))
        other = amplitudes[1 - p]
        scalar_cross = share * (
            2.0 * np.real(amplitudes[p] * np.conjugate(other)))
        assert _bits_equal(cross, scalar_cross)
        assert _bits_equal(total, self_term + scalar_cross)


def test_single_group_returns_exact_self_term():
    amplitudes = _random_amplitudes(1, 16, seed=4)
    total, self_term, cross = principal_weighted_coherent_partition(
        amplitudes, 0, np.array([3.0]))
    np.testing.assert_array_equal(total, np.abs(amplitudes[0]) ** 2)
    np.testing.assert_array_equal(cross, np.zeros_like(total))
    np.testing.assert_array_equal(self_term, total)


def test_degenerate_all_zero_weights_two_groups_zero_amplitudes():
    # Zero coherent weight forces zero coherent amplitude upstream (the
    # per-atom prefactors scale with b_coh), so the only reachable all-zero-
    # weight case has zero amplitudes. The 0.5 fallback and the historical
    # 1.0 scalar then both multiply an exactly-zero cross term: everything
    # is bit-exact +0.0.
    amplitudes = np.zeros((2, 8), dtype=complex)
    weights = np.array([0.0, 0.0])
    for p in range(2):
        total, self_term, cross = principal_weighted_coherent_partition(
            amplitudes, p, weights)
        zero = np.zeros(8)
        assert _bits_equal(total, zero)
        assert _bits_equal(self_term, zero)
        assert _bits_equal(cross, zero)


def test_zero_weight_pair_is_finite_and_total_stays_exact():
    # Two zero-coherent-weight groups: their mutual pair hits the 0.5
    # fallback; physically their amplitudes would be zero, but the partition
    # must stay finite and the sum rule must still hold for any amplitudes.
    amplitudes = _random_amplitudes(3, 8, seed=5)
    weights = np.array([0.0, 0.0, 4.0])
    partials = [
        principal_weighted_coherent_partition(amplitudes, p, weights)[0]
        for p in range(3)
    ]
    for partial in partials:
        assert np.all(np.isfinite(partial))
    np.testing.assert_allclose(sum(partials), _exact_total(amplitudes), rtol=1e-12)
