"""Unit tests for the NCMAT writer (SP1). Pure/fast — no engine, no NCrystal."""
from __future__ import annotations

import math

import numpy as np
import pytest

from irma.ncrystal.ncmat import (
    lattice_to_cell_params,
    assemble_material_ncmat,
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


def test_debye_temperature_reasonable_and_clamped():
    # graphite C, MSD ~0.005 A^2 at 296 K -> ~hundreds-to-1000s K
    theta = debye_temperature_from_msd(0.005, 12.011, 296.0)
    assert 100.0 < theta < 5000.0
    assert debye_temperature_from_msd(1e-12, 12.0, 296.0) <= 1.0e5   # clamped


def _ncmat(symbols, positions, debye, packs=("/abs/graphite__C.irmapack",)):
    return assemble_material_ncmat(
        lattice_ang=np.diag([2.46, 2.46, 6.70]), scaled_positions=positions,
        symbols=symbols, pack_filenames=list(packs), debye_temperatures=debye)


def test_ncmat_structure_and_pack_lines():
    txt = _ncmat(["C", "C"], [[0, 0, 0], [0, 0, 0.5]], {"C": 850.0},
                 packs=["a.irmapack", "b.irmapack"])
    assert txt.splitlines()[0] == "NCMAT v5"
    assert "@CELL" in txt and "@ATOMPOSITIONS" in txt
    assert "type vdosdebye" in txt and "debye_temp 850" in txt
    assert txt.count("@DYNINFO") == 1            # one per distinct element
    assert "@TEMPERATURE" not in txt             # would force NCMAT v7
    # the plugin parser accepts ONLY 'pack <path>' lines after @CUSTOM_IRMA
    custom = txt.split("@CUSTOM_IRMA\n", 1)[1].splitlines()
    assert [line.split() for line in custom] == [["pack", "a.irmapack"],
                                                 ["pack", "b.irmapack"]]


def test_assemble_material_ncmat_full():
    L = np.diag([2.46, 2.46, 6.70])
    txt = assemble_material_ncmat(
        lattice_ang=L,
        scaled_positions=[[0, 0, 0], [0, 0, 0.5]],
        symbols=["C", "C"],
        pack_filenames=["/abs/graphite__C.irmapack"],
        debye_temperatures={"C": 850.0})
    assert txt.startswith("NCMAT v5")
    assert "@CUSTOM_IRMA" in txt
    assert "pack /abs/graphite__C.irmapack" in txt


def test_assemble_two_species():
    L = np.diag([4.0, 4.0, 4.0])
    txt = assemble_material_ncmat(
        lattice_ang=L,
        scaled_positions=[[0, 0, 0], [0.5, 0.5, 0.5]],
        symbols=["Be", "O"], pack_filenames=["be.irmapack"],
        debye_temperatures={"Be": 900.0, "O": 600.0})
    assert txt.count("@DYNINFO") == 2
    assert "element Be" in txt and "element O" in txt
