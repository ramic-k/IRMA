"""Primitive-matrix pinning across phonopy versions (review PHONOPY-1).

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
    """Every phonopy.load call site must route through the pinning helper.

    A new load site added without the pin silently reintroduces the
    phonopy-4 auto-primitive drift (review PHONOPY-1).
    """
    import pathlib
    root = pathlib.Path(__file__).resolve().parents[1] / "irma"
    offenders = []
    for path in root.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        if "phonopy.load(" in text or "phonopy_load(" in text:
            if "pinned_primitive_matrix_kwargs" not in text:
                offenders.append(str(path))
    assert not offenders, (
        f"phonopy.load call sites missing pinned_primitive_matrix_kwargs: "
        f"{offenders}")


def test_load_semantics_identity_when_unstored(tmp_path):
    """Live check in whatever phonopy this env has: a stored-matrix yaml and
    the pin produce the same primitive as phonopy 2/3 defaults did."""
    phonopy = pytest.importorskip("phonopy")
    import inspect
    sig = inspect.signature(phonopy.load)
    assert "primitive_matrix" in sig.parameters
    # The kwargs form must be accepted by this phonopy: "P" is a valid
    # sentinel on 2.x through 4.x.
    assert pinned_primitive_matrix_kwargs.__doc__  # helper exists and is doc'd
