"""Unit tests for the NCMAT writer (SP1). Pure/fast — no engine, no NCrystal."""
from __future__ import annotations

import math

import numpy as np
import pytest

from irma.ncrystal.ncmat import (
    lattice_to_cell_params,
    build_base_ncmat,
    custom_irma_section,
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
    # degenerate inputs -> fallback / clamp, never raise
    assert debye_temperature_from_msd(0.0, 12.0, 296.0) == 300.0
    assert debye_temperature_from_msd(-1.0, 12.0, 296.0) == 300.0
    assert debye_temperature_from_msd(1e-12, 12.0, 296.0) <= 1.0e5


def test_custom_section_pack_lines_only():
    # the plugin parser accepts ONLY 'pack <path>' lines — no material_id line.
    sec = custom_irma_section(["a.irmapack", "b.irmapack"])
    assert sec.startswith("@CUSTOM_IRMA\n")
    assert "material_id" not in sec
    assert sec.count("pack ") == 2
    for line in sec.splitlines()[1:]:
        toks = line.split()
        assert toks[0] == "pack" and len(toks) == 2


def test_build_base_ncmat_structure():
    L = np.diag([2.46, 2.46, 6.70])
    txt = build_base_ncmat(
        lattice_ang=L,
        scaled_positions=[[0, 0, 0], [0, 0, 0.5]],
        symbols=["C", "C"],
        debye_temperatures={"C": 850.0})
    assert txt.splitlines()[0] == "NCMAT v5"
    assert "@CELL" in txt and "@ATOMPOSITIONS" in txt
    assert "type vdosdebye" in txt
    assert "debye_temp 850" in txt
    # one @DYNINFO per distinct element
    assert txt.count("@DYNINFO") == 1
    assert "@TEMPERATURE" not in txt        # would force NCMAT v7


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
    txt = build_base_ncmat(
        lattice_ang=L,
        scaled_positions=[[0, 0, 0], [0.5, 0.5, 0.5]],
        symbols=["Be", "O"],
        debye_temperatures={"Be": 900.0, "O": 600.0})
    assert txt.count("@DYNINFO") == 2
    assert "element Be" in txt and "element O" in txt
