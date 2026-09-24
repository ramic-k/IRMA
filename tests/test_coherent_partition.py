"""Coherent one-phonon partition across principals (principal_share_coherent_partition).

Principal p gets |F_p|^2 / sum_g |F_g|^2 of the exact total |sum_g F_g|^2, with
both self-term sums taken over each set of degenerate modes. The shares are
non-negative, add up to the exact total for any number of site groups, and do
not depend on the basis the eigensolver picks inside a degenerate set.
"""

import numpy as np

from irma.core.noncubic_workers import principal_share_coherent_partition


def _random_amplitudes(n_groups, n_modes, seed, n_qsamples=None):
    rng = np.random.default_rng(seed)
    shape = (n_groups, n_modes) if n_qsamples is None else (n_qsamples, n_groups, n_modes)
    return rng.normal(size=shape) + 1j * rng.normal(size=shape)


def test_shares_are_non_negative_and_sum_to_the_exact_total():
    amplitudes = _random_amplitudes(5, 12, seed=2, n_qsamples=7)
    frequencies = np.broadcast_to(np.linspace(1.0, 12.0, 12), (7, 12))
    shares = [principal_share_coherent_partition(amplitudes, p, frequencies)[0]
              for p in range(5)]
    assert min(float(s.min()) for s in shares) >= 0.0
    np.testing.assert_allclose(sum(shares), np.abs(amplitudes.sum(axis=-2)) ** 2,
                               rtol=1e-12)


def test_degenerate_pair_share_does_not_depend_on_the_eigenvector_basis():
    # F_g = <a_g, e> for two orthonormal eigenvectors of one frequency, then
    # the same pair after a unitary rotation within the degenerate plane.
    rng = np.random.default_rng(5)
    a = rng.normal(size=(2, 6)) + 1j * rng.normal(size=(2, 6))
    e, _ = np.linalg.qr(rng.normal(size=(6, 2)) + 1j * rng.normal(size=(6, 2)))
    c, s = np.cos(0.7), np.sin(0.7) * np.exp(0.3j)
    rotated = e @ np.array([[c, -np.conj(s)], [s, c]])

    def pair_shares(vectors, frequencies):
        amplitudes = a @ vectors.conj()                        # (groups, modes)
        return [principal_share_coherent_partition(amplitudes, p, frequencies)[0].sum()
                for p in range(2)]

    degenerate = np.array([3.0, 3.0])
    np.testing.assert_allclose(pair_shares(e, degenerate),
                               pair_shares(rotated, degenerate), rtol=1e-12)
    # treated as two separate modes, the per-mode shares change with the basis
    split = np.array([3.0, 4.0])
    assert not np.allclose(pair_shares(e, split), pair_shares(rotated, split))


def test_single_group_returns_exact_self_term():
    amplitudes = _random_amplitudes(1, 16, seed=4)
    total, self_term, cross = principal_share_coherent_partition(
        amplitudes, 0, np.linspace(1.0, 16.0, 16))
    np.testing.assert_array_equal(total, np.abs(amplitudes[0]) ** 2)
    np.testing.assert_array_equal(cross, np.zeros_like(total))
    np.testing.assert_array_equal(self_term, total)
