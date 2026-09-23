"""Core side of the phonopy structure prefill (Tk-free).

Two pieces:

- ``irma.core.phonopy_io.load_phonopy_primitive_structure`` -- the PRIMITIVE
  cell of a phonopy model, read through phonopy itself, in Angstrom.
- ``irma.core.crystal_input`` -- symbol -> (Z, A=0, natural constants) and the
  Card 6c/6d text formatting shared by every prefill path.

UNITS. phonopy keeps cells in the CALCULATOR's native length unit and
``phonopy.load`` never converts them, so a qe/abinit/... model hands back a
cell in bohr. The reader CONVERTS it (``phonopy_io.angstrom_primitive``, the
one conversion point in the package, which the mode-1/2 engine shares), so the
Card 6c/6d it fills describe the same cell the engine will run. The full
cross-calculator equivalence proof lives in ``test_phonopy_units.py``.
"""
import pathlib

import pytest

from irma.core.crystal_input import (
    CrystalSpecies, format_card6d_row, format_lattice_fields,
    resolve_natural_element, species_from_sites,
)

GRAPHITE_YAML = (pathlib.Path(__file__).resolve().parent
                 / "mode2_euphonic_n1_validation" / "graphite" / "phonopy.yaml")


# ---------------------------------------------------------------- reader ----

def test_reader_returns_angstrom_cellpar_symbols_and_positions():
    pytest.importorskip("phonopy")
    from irma.core.phonopy_io import load_phonopy_primitive_structure

    s = load_phonopy_primitive_structure(GRAPHITE_YAML)
    a, b, c, alpha, beta, gamma = s.cellpar
    # The committed model's primitive cell: hexagonal graphite in ANGSTROM.
    assert a == pytest.approx(2.4606, abs=1e-6)
    assert b == pytest.approx(2.4606, abs=1e-6)
    assert c == pytest.approx(6.705, abs=1e-6)
    assert alpha == pytest.approx(90.0, abs=1e-6)
    assert beta == pytest.approx(90.0, abs=1e-6)
    assert gamma == pytest.approx(120.0, abs=1e-6)
    assert s.symbols == ["C", "C", "C", "C"]
    assert s.scaled_positions.shape == (4, 3)
    assert s.scaled_positions[1] == pytest.approx([0.0, 0.0, 0.5])
    assert s.scaled_positions[2] == pytest.approx([1 / 3, 2 / 3, 0.0])
    # Fractional coordinates, not Cartesian.
    assert ((s.scaled_positions >= -1e-9)
            & (s.scaled_positions <= 1 + 1e-9)).all()


def test_reader_rejects_unsafe_yaml(tmp_path):
    pytest.importorskip("phonopy")
    from irma.core.phonopy_io import load_phonopy_primitive_structure

    bad = tmp_path / "phonopy.yaml"
    bad.write_text('phonopy:\n  version: 2.21.0\n'
                   'x: !!python/object/apply:os.system ["true"]\n')
    with pytest.raises(ValueError, match="!!python/"):
        load_phonopy_primitive_structure(bad)


# ---------------------------------------------------------------- units -----

_QE_YAML = """phonopy:
  version: "2.48.0"
  calculator: "qe"
{unit_block}supercell_matrix:
- [ 1, 0, 0 ]
- [ 0, 1, 0 ]
- [ 0, 0, 1 ]
unit_cell:
  lattice:
  - [ 7.5589, 0.0, 0.0 ]
  - [ 0.0, 7.5589, 0.0 ]
  - [ 0.0, 0.0, 7.5589 ]
  points:
  - symbol: C
    coordinates: [ 0.0, 0.0, 0.0 ]
    mass: 12.0107
"""

_UNIT_BLOCK = 'physical_unit:\n  atomic_mass: "AMU"\n  length: "au"\n'


# The cell in _QE_YAML is 7.5589 bohr = 4.00046 Angstrom.
_QE_CELL_ANG = 7.5589 * 0.5291772


def test_non_angstrom_yaml_is_converted_by_recorded_unit(tmp_path):
    """Was a rejection. The reader now converts, because the mode-1/2 engine
    converts at the same boundary -- the two can no longer disagree."""
    pytest.importorskip("phonopy")
    from irma.core.phonopy_io import load_phonopy_primitive_structure
    p = tmp_path / "phonopy.yaml"
    p.write_text(_QE_YAML.format(unit_block=_UNIT_BLOCK))
    s = load_phonopy_primitive_structure(p)
    assert s.cellpar[0] == pytest.approx(_QE_CELL_ANG, abs=1e-4)
    assert s.symbols == ["C"]


def test_non_angstrom_yaml_is_converted_by_calculator(tmp_path):
    """No recorded physical_unit: the calculator table is the authority."""
    pytest.importorskip("phonopy")
    from irma.core.phonopy_io import load_phonopy_primitive_structure
    p = tmp_path / "phonopy.yaml"
    p.write_text(_QE_YAML.format(unit_block=""))
    s = load_phonopy_primitive_structure(p)
    assert s.cellpar[0] == pytest.approx(_QE_CELL_ANG, abs=1e-4)


def test_unconvertible_unit_is_still_rejected(tmp_path):
    """A recorded length unit with no known factor is refused (by phonopy,
    which cross-checks it against its calculator table)."""
    pytest.importorskip("phonopy")
    from irma.core.phonopy_io import load_phonopy_primitive_structure
    p = tmp_path / "phonopy.yaml"
    p.write_text(_QE_YAML.format(
        unit_block='physical_unit:\n  atomic_mass: "AMU"\n  length: "cubit"\n'))
    with pytest.raises(ValueError, match="cubit"):
        load_phonopy_primitive_structure(p)


# -------------------------------------------------------------- resolver ----

def test_resolver_returns_natural_element_with_a_zero():
    from irma.core.nuclear_data import lookup
    z, a, awr, b_coh, sigma_inc = resolve_natural_element("C")
    nat = lookup("C")
    assert (z, a) == (6, 0)                      # ENDF's natural-element code
    assert awr == pytest.approx(nat.awr)
    assert b_coh == pytest.approx(nat.b_coh_fm)
    assert sigma_inc == pytest.approx(nat.sigma_inc_b)
    # ... and NOT the most-abundant isotope's constants
    assert b_coh != pytest.approx(lookup("12-C").b_coh_fm)


def test_resolver_refuses_energy_dependent_nuclide():
    from irma.core.nuclear_data import lookup
    assert lookup("Gd").energy_dependent          # fixture assumption
    with pytest.raises(ValueError, match="ENERGY-DEPENDENT"):
        resolve_natural_element("Gd")


def test_resolver_rejects_unknown_symbol():
    with pytest.raises(KeyError):
        resolve_natural_element("Xx")


def test_species_from_sites_groups_in_first_appearance_order():
    species, warnings = species_from_sites(
        ["Be", "O", "Be", "O"],
        [(0.0, 0.0, 0.0), (0.5, 0.5, 0.5),
         (1 / 3, 2 / 3, 0.25), (2 / 3, 1 / 3, 0.75)])
    assert [s.symbol for s in species] == ["Be", "O"]
    assert [s.npos for s in species] == [2, 2]
    assert [s.za for s in species] == [4000, 8000]
    assert species[0].positions[1] == pytest.approx((1 / 3, 2 / 3, 0.25))
    assert all(s.A == 0 for s in species)
    assert len(warnings) == 2                    # one natural-element note each


def test_species_from_sites_warns_that_h_may_be_deuterium():
    _species, warnings = species_from_sites(["H"], [(0.0, 0.0, 0.0)])
    assert any("deuterium" in w for w in warnings)


def test_resolver_returns_warnings_as_data_not_output(capsys):
    species_from_sites(["C", "H"], [(0.0, 0.0, 0.0), (0.5, 0.5, 0.5)])
    captured = capsys.readouterr()
    assert captured.out == "" and captured.err == ""


# ------------------------------------------------------------ formatting ----

def test_card6d_row_matches_the_widget_format():
    from irma.gui.deck_text import parse_atoms_text
    sp = CrystalSpecies(symbol="C", Z=6, A=0, awr=11.907820,
                        b_coh_fm=6.6472, sigma_inc_b=0.001,
                        positions=((0.0, 0.0, 0.0), (0.0, 0.0, 0.5)))
    row = format_card6d_row(sp)
    assert row.split()[:6] == ["6", "0", "11.907820", "6.647200",
                               "0.001000", "2"]
    parsed = parse_atoms_text(row + "\n")
    assert len(parsed) == 1
    assert parsed[0]["Z"] == 6 and parsed[0]["A"] == 0
    assert parsed[0]["npos"] == 2
    assert parsed[0]["positions"][1] == (0.0, 0.0, 0.5)


def test_lattice_fields_are_six_fixed_point_strings():
    fields = format_lattice_fields((2.4606, 2.4606, 6.705, 90.0, 90.0, 120.0))
    assert fields == ("2.460600", "2.460600", "6.705000",
                      "90.000000", "90.000000", "120.000000")


def test_end_to_end_graphite_rows_are_natural_carbon():
    pytest.importorskip("phonopy")
    from irma.core.phonopy_io import load_phonopy_primitive_structure
    s = load_phonopy_primitive_structure(GRAPHITE_YAML)
    species, _warnings = species_from_sites(s.symbols, s.scaled_positions)
    assert len(species) == 1
    assert species[0].za == 6000          # natural carbon, not 6012
    assert species[0].npos == 4


# ------------------------------------------- principal ZA against the rows ----
def _rows():
    return [
        {"Z": 6, "A": 0, "awr": 11.907819742, "b_coh": 6.6472, "sigma_inc": 0.001,
         "npos": 2, "positions": [(0.0, 0.0, 0.25), (0.0, 0.0, 0.75)]},
        {"Z": 8, "A": 0, "awr": 15.858, "b_coh": 5.803, "sigma_inc": 0.0,
         "npos": 1, "positions": [(0.5, 0.5, 0.5)]},
    ]


def test_principal_row_match_and_messages():
    from irma.core.crystal_input import (
        principal_mismatch_message, principal_row_match)
    rows = _rows()
    assert principal_row_match(6000, rows)["exact"] == [0]
    assert principal_mismatch_message(6000, rows) is None
    match = principal_row_match(6012, rows)
    assert match["exact"] == [] and match["same_z"] == [0]
    message = principal_mismatch_message(6012, rows)
    assert "ZA=6012 (C-12)" in message and "row 1: Z=6, A=0 (C)" in message
    assert "ZA=6000" in message and "No rows were changed" in message
    message = principal_mismatch_message(26056, rows)
    assert "No row has Z=26" in message
    rows.append({"Z": 6, "A": 13, "awr": 12.8916, "b_coh": 6.19, "sigma_inc": 0.52,
                 "npos": 1, "positions": [(0.1, 0.1, 0.1)]})
    message = principal_mismatch_message(6012, rows)
    assert "row 1: A=0 (C)" in message and "row 3: A=13 (C-13)" in message
    assert principal_mismatch_message(6013, rows) is None
    with pytest.raises(ValueError):
        principal_row_match(0, rows)


def test_relabel_row_takes_identity_and_constants_from_one_entry():
    from irma.core.crystal_input import (
        format_atom_row, relabel_row, row_constants_match_table)
    from irma.core.nuclear_data import lookup
    rows = _rows()
    assert row_constants_match_table(rows[0])          # the natural-C prefill
    new, changes = relabel_row(rows[0], 6012)
    c12 = lookup((6, 12))
    assert (new["Z"], new["A"]) == (6, 12)
    assert new["awr"] == c12.awr and new["b_coh"] == c12.b_coh_fm
    assert new["sigma_inc"] == c12.sigma_inc_b
    assert new["positions"] == rows[0]["positions"] and new["npos"] == 2
    assert [c[0] for c in changes][:2] == ["A", "awr"]
    assert row_constants_match_table(new)
    # back to natural is the same operation
    back, _ = relabel_row(new, 6000)
    assert back["A"] == 0 and back["awr"] == pytest.approx(rows[0]["awr"], rel=1e-9)
    # custom constants are recognised as not the table's
    custom = dict(rows[0], awr=11.898, b_coh=6.646)
    assert not row_constants_match_table(custom)
    # wrong element, missing nuclide, energy-dependent nuclide: refused
    with pytest.raises(ValueError):
        relabel_row(rows[1], 6012)
    with pytest.raises(KeyError):
        relabel_row(rows[0], 6014)
    gd = {"Z": 64, "A": 0, "awr": 155.9, "b_coh": 6.5, "sigma_inc": 151.0,
          "npos": 1, "positions": [(0.0, 0.0, 0.0)]}
    with pytest.raises(ValueError, match="ENERGY-DEPENDENT"):
        relabel_row(gd, 64157)
    line = format_atom_row(new)
    assert line.startswith("6  12  ") and "  2  0.000000 0.000000 0.250000  " in line


def test_validator_and_engine_share_the_message():
    from irma.core.crystal_input import principal_mismatch_message
    from irma.mlip.emit import validate_deck_semantics
    staged = {"mat": 28, "za": 6012.0, "iint": 1, "iel": 10,
              "atoms": [{"Z": 6, "A": 0}], "alpha": [0.1, 0.2], "beta": [0.0, 0.1]}
    problems = validate_deck_semantics(staged)
    assert any(principal_mismatch_message(6012, staged["atoms"]) in p for p in problems)
