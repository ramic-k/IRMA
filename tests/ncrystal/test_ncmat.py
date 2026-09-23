"""Unit tests for the NCMAT writer (SP1). Pure/fast — no engine, no NCrystal."""
from __future__ import annotations

import math

import numpy as np
import pytest

from irma.ncrystal.ncmat import (
    lattice_to_cell_params,
    assemble_material_ncmat,
    check_ncrystal_lattice,
    debye_msd,
    debye_temperature_from_msd,
)


def test_lattice_to_cell_params_hexagonal():
    # graphite-like hexagonal cell: a=b=2.46, c=6.7, gamma=120
    a0, c0 = 2.46, 6.70
    L = np.array([
        [a0, 0.0, 0.0],
        [-a0 / 2.0, a0 * math.sqrt(3) / 2.0, 0.0],
        [0.0, 0.0, c0],
    ])
    a, b, c, alpha, beta, gamma = lattice_to_cell_params(L)
    assert a == pytest.approx(a0)
    assert b == pytest.approx(a0)
    assert c == pytest.approx(c0)
    assert alpha == pytest.approx(90.0)
    assert beta == pytest.approx(90.0)
    assert gamma == pytest.approx(120.0)


def test_debye_temperature_round_trips_through_the_debye_msd():
    for msd, mass, temp in ((0.005, 12.011, 296.0), (0.02, 1.008, 20.0),
                            (0.0008, 207.2, 5.0)):
        theta = debye_temperature_from_msd(msd, mass, temp)
        assert debye_msd(theta, mass, temp) == pytest.approx(msd, rel=1e-10)
    # NCrystal's debyeTempFromIsotropicMSD(0.005 A^2, 296 K, 12.011 amu)
    assert debye_temperature_from_msd(0.005, 12.011, 296.0) == pytest.approx(
        951.2698, rel=1e-6)


def test_assemble_two_species():
    packs = ["/abs/beo__Be.irmapack", "/abs/beo__O.irmapack"]
    txt = assemble_material_ncmat(
        lattice_ang=np.diag([4.0, 4.0, 4.0]),
        scaled_positions=[[0, 0, 0], [0.5, 0.5, 0.5]],
        symbols=["Be", "O"], pack_filenames=packs,
        debye_temperatures={"Be": 900.0, "O": 600.0})
    assert txt.splitlines()[0] == "NCMAT v5"
    assert "@CELL" in txt and "@ATOMPOSITIONS" in txt
    assert txt.count("@DYNINFO") == 2            # one per distinct element
    assert "element Be" in txt and "element O" in txt
    assert "type vdosdebye" in txt and "debye_temp 900" in txt
    assert "@TEMPERATURE" not in txt             # would force NCMAT v7
    # the plugin parser accepts ONLY 'pack <path>' lines after @CUSTOM_IRMA
    custom = txt.split("@CUSTOM_IRMA\n", 1)[1].splitlines()
    assert [line.split() for line in custom] == [["pack", p] for p in packs]


def test_cells_in_ncrystals_buggy_lattice_branch_are_refused():
    # FCC primitive cell (60, 60, 60): cos a - cos b cos g = 0.25, refused
    fcc = 2.025 * np.array([[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]])
    with pytest.raises(ValueError, match="NCLatticeUtils"):
        check_ncrystal_lattice(fcc)
    # orthorhombic, and hexagonal with gamma 120.00285 (the BeO fixture), pass
    check_ncrystal_lattice(np.diag([3.0, 4.0, 5.0]))
    g = math.radians(120.00285)
    check_ncrystal_lattice(np.array([[2.7, 0.0, 0.0],
                                     [2.7 * math.cos(g), 2.7 * math.sin(g), 0.0],
                                     [0.0, 0.0, 4.37]]))
