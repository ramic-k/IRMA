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


def test_a_symmetry_reduced_mesh_is_judged_by_its_star_averaged_tensors():
    """Two equivalent atoms and one irreducible mode whose eigenvector sits on the first: its symmetry copy sits on
    the second, so the star-averaged tensors, which the seed uses, give each atom half. The check takes the area from
    them; from the irreducible eigenvector alone each atom would be off by its whole displacement."""
    eigvecs = np.array([[[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]])
    star = np.zeros((1, 2, 6))
    star[0, :, 0] = 0.5
    thermal = np.tile(np.diag([0.15, 0.0, 0.0]), (2, 1, 1))
    arguments = (np.array([1.0, 1.0]), eigvecs, np.array([0.2]), np.array([0.1]),
                 np.array([0]), np.array([0]), thermal)
    assert multiphonon_seed_area_deficit(*arguments, star) < 1e-12
    assert abs(multiphonon_seed_area_deficit(*arguments) - 1.0) < 1e-12
