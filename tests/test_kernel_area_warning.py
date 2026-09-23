"""The multiphonon one-phonon seed must carry the Debye-Waller displacement.

multiphonon_seed_area_deficit compares the direction-averaged area of the seed
with Tr(U_a)/3; a mode that falls outside the multiphonon work grid shows up as
a deficit, which the engine reports as a warning.
"""
import numpy as np

from irma.core.noncubic_workers import multiphonon_seed_area_deficit

# One atom, two modes polarized along x and y; U = diag(0.3, 0.3, 0).
_EIGVECS = np.array([[[1.0, 0.0, 0.0]], [[0.0, 1.0, 0.0]]])
_THERMAL = np.diag([0.3, 0.3, 0.0])[None, :, :]
_EMISSION = np.array([0.2, 0.2])
_ABSORPTION = np.array([0.1, 0.1])


def test_seed_matches_displacement_when_all_modes_are_on_the_grid():
    deficit = multiphonon_seed_area_deficit(
        np.array([1.0]), _EIGVECS, _EMISSION, _ABSORPTION,
        np.array([0, 1]), np.array([0, 1]), _THERMAL)
    assert deficit < 1e-12


def test_mode_off_the_grid_is_reported_as_a_deficit():
    deficit = multiphonon_seed_area_deficit(
        np.array([1.0]), _EIGVECS, _EMISSION, _ABSORPTION,
        np.array([0]), np.array([0]), _THERMAL)
    assert abs(deficit - 0.5) < 1e-12
