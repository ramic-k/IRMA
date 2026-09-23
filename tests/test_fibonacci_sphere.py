"""Regression tests for fibonacci_sphere — the equal-area direction quadrature
used for the coherent and multiphonon powder averages."""
import numpy as np

from irma.core.noncubic_engine import fibonacci_sphere


def test_count_and_unit_vectors():
    for n in (1, 2, 4, 50, 1000):
        dirs = fibonacci_sphere(n)
        assert dirs.shape == (max(n, 1), 3)
        assert np.allclose(np.linalg.norm(dirs, axis=1), 1.0, atol=1e-12)


def test_z_uniformly_spaced():
    """z (= mu = cos theta) is uniform in (-1, 1) -> equal solid angle per point."""
    n = 200
    z = np.sort(fibonacci_sphere(n)[:, 2])
    dz = np.diff(z)
    assert np.allclose(dz, dz[0], atol=1e-9)          # constant spacing in z
    assert -1.0 < z[0] and z[-1] < 1.0


def test_centroid_near_origin():
    """A uniform sphere sampling averages to ~0 (no directional bias)."""
    dirs = fibonacci_sphere(2000)
    assert np.linalg.norm(dirs.mean(axis=0)) < 0.05
