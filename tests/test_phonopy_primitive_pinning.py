"""Primitive-matrix pinning across phonopy versions.

phonopy 4 changed the meaning of an omitted ``primitive_matrix`` in
``phonopy.load`` from "the yaml's stored matrix, else identity" to
``"auto"`` (symmetry-guessed). ``pinned_primitive_matrix_kwargs`` pins the
2/3-era semantics on every version: pass nothing when the yaml stores a
matrix (stored value wins everywhere; an explicit argument would override
it), pass ``"P"`` when it stores none.
"""
import gzip

import pytest

from irma.core.phonopy_io import pinned_primitive_matrix_kwargs

_BASE = """\
phonopy:
  version: 2.28.0
physical_unit:
  atomic_mass: AMU
"""

_WITH_MATRIX = _BASE + """\
primitive_matrix:
- [1.0, 0.0, 0.0]
- [0.0, 1.0, 0.0]
- [0.0, 0.0, 1.0]
unit_cell:
  lattice: []
"""

# An INDENTED primitive_matrix-like key (e.g. inside a stored settings
# block) is not a top-level key and must not suppress the pin.
_WITH_NESTED_ONLY = _BASE + """\
configuration:
  primitive_matrix: auto
  primitive_axis: auto
unit_cell:
  lattice: []
"""


def _write(tmp_path, name, text, compress=False):
    p = tmp_path / name
    if compress:
        with gzip.open(p, "wt") as f:
            f.write(text)
    else:
        p.write_text(text)
    return p


def test_yaml_without_matrix_pins_identity(tmp_path):
    p = _write(tmp_path, "phonopy.yaml", _BASE + "unit_cell:\n  lattice: []\n")
    assert pinned_primitive_matrix_kwargs(p) == {"primitive_matrix": "P"}


def test_yaml_with_stored_matrix_passes_nothing(tmp_path):
    p = _write(tmp_path, "phonopy.yaml", _WITH_MATRIX)
    assert pinned_primitive_matrix_kwargs(p) == {}


def test_nested_key_is_not_top_level(tmp_path):
    p = _write(tmp_path, "phonopy.yaml", _WITH_NESTED_ONLY)
    assert pinned_primitive_matrix_kwargs(p) == {"primitive_matrix": "P"}


def test_compressed_yaml_is_scanned(tmp_path):
    p = _write(tmp_path, "phonopy.yaml.gz", _WITH_MATRIX, compress=True)
    assert pinned_primitive_matrix_kwargs(p) == {}


def test_all_phonopy_load_sites_use_the_pin():
    """Every phonopy.load call must route through the pinning helper: a file
    has at least as many helper calls as load calls.

    A new load site added without the pin silently reintroduces the
    phonopy-4 auto-primitive drift.
    """
    import pathlib
    root = pathlib.Path(__file__).resolve().parents[1] / "irma"
    offenders = []
    for path in root.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        loads = text.count("phonopy.load(") + text.count("phonopy_load(")
        pins = (text.count("pinned_primitive_matrix_kwargs(")
                - text.count("def pinned_primitive_matrix_kwargs("))
        if loads > pins:
            offenders.append(f"{path}: {loads} load calls, {pins} pinned")
    assert not offenders, (
        f"phonopy.load call sites missing pinned_primitive_matrix_kwargs: "
        f"{offenders}")


def _capture_phonopy_load(monkeypatch):
    """Replace phonopy.load with a recorder returning a minimal fake model."""
    import types

    import numpy as np
    import phonopy

    seen = {}

    def fake_load(phonopy_yaml=None, is_nac=None, produce_fc=None,
                  log_level=None, **kwargs):
        seen.update(kwargs)
        seen["_phonopy_yaml"] = phonopy_yaml
        seen["_is_nac"] = is_nac
        seen["_produce_fc"] = produce_fc
        # Every attribute the reader touches: the cell it converts to
        # Angstrom, plus the fractional positions, symbols and masses it
        # carries through unchanged.
        prim = types.SimpleNamespace(
            cell=np.eye(3) * 3.5,
            scaled_positions=np.zeros((1, 3)),
            symbols=["C"],
            masses=np.array([12.011]),
        )
        return types.SimpleNamespace(primitive=prim, calculator=None)

    monkeypatch.setattr(phonopy, "load", fake_load)
    return seen


def test_structure_reader_pins_identity_when_yaml_stores_no_matrix(
        tmp_path, monkeypatch):
    """The structure prefill is a phonopy.load call site like any other: an
    un-pinned load there would read a phonopy-4 auto-guessed primitive cell
    and prefill Card 6c/6d with positions the engine then rejects."""
    pytest.importorskip("phonopy")
    from irma.core.phonopy_io import load_phonopy_primitive_structure

    seen = _capture_phonopy_load(monkeypatch)
    p = _write(tmp_path, "phonopy.yaml", _BASE + "unit_cell:\n  lattice: []\n")
    load_phonopy_primitive_structure(p)
    assert seen["primitive_matrix"] == "P"
    assert seen["_produce_fc"] is False        # geometry only, never build FC
    assert seen["_is_nac"] is False            # never probe the cwd for BORN


def test_structure_reader_defers_to_a_stored_matrix(tmp_path, monkeypatch):
    pytest.importorskip("phonopy")
    from irma.core.phonopy_io import load_phonopy_primitive_structure

    seen = _capture_phonopy_load(monkeypatch)
    p = _write(tmp_path, "phonopy.yaml", _WITH_MATRIX)
    load_phonopy_primitive_structure(p)
    assert "primitive_matrix" not in seen     # stored value must win
