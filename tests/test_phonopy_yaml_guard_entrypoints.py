"""SEC-1: reject_unsafe_phonopy_yaml must fire at every entry point that
hands an untrusted phonopy.yaml to phonopy's unsafe YAML loader — BEFORE
phonopy parses (and thereby executes) anything.

Covered call sites (the GUI and irma.mlip entry points are guarded and
tested elsewhere):

* irma.core.phonopy_io.load_phonopy_mesh — the widest entry point
  (Card 6f / iel=10 decks and irma.spectra.dos_from_phonopy funnel here);
* irma.core.noncubic_inelastic_context.build_model_context;
* irma.ncrystal.build.load_primitive_info (the NCrystal exporter).

Each rejection test feeds a phonopy.yaml carrying a ``!!python/`` tag
whose payload would touch a canary file, and asserts (a) ValueError from
the guard and (b) the canary never appeared — i.e. the file was scanned,
never parsed. Near-miss tests confirm a legitimate phonopy.yaml still
loads through the guarded path (build_model_context's legitimate path is
exercised by the engine integration tests, e.g. test_noncubic_lat0_pin).
"""
import argparse
from pathlib import Path

import pytest

pytest.importorskip("phonopy")

_REPO = Path(__file__).resolve().parents[1]
_GRAPHITE_YAML = (
    _REPO / "tests/mode2_euphonic_n1_validation/graphite/phonopy.yaml")


def _malicious_yaml(tmp_path):
    """A phonopy.yaml whose parse (by phonopy's unsafe loader) would touch
    a canary file. Includes a force_constants key so that, were the guard
    missing, nothing else would reject the file before the parse."""
    canary = tmp_path / "pwned"
    path = tmp_path / "phonopy.yaml"
    path.write_text(
        "phonopy:\n"
        "  version: 2.21.0\n"
        f'extra: !!python/object/apply:os.system ["touch {canary}"]\n'
        "force_constants:\n"
        "  format: full\n")
    return path, canary


def test_load_phonopy_mesh_rejects_python_tag(tmp_path):
    from irma.core.phonopy_io import load_phonopy_mesh
    path, canary = _malicious_yaml(tmp_path)
    with pytest.raises(ValueError, match="!!python/"):
        load_phonopy_mesh(str(path), [1, 1, 1])
    assert not canary.exists()


def test_build_model_context_rejects_python_tag(tmp_path):
    from irma.core.noncubic_inelastic_context import build_model_context
    path, canary = _malicious_yaml(tmp_path)
    # Only the fields read before the guard fires are needed; the guard
    # must raise before any mesh/scattering argument is consulted.
    args = argparse.Namespace(
        phonopy_yaml=str(path), mesh=[1, 1, 1], born=None,
        force_constants=None, force_sets=None)
    with pytest.raises(ValueError, match="!!python/"):
        build_model_context(args)
    assert not canary.exists()


def test_load_primitive_info_rejects_python_tag(tmp_path):
    from irma.ncrystal.build import load_primitive_info
    path, canary = _malicious_yaml(tmp_path)
    with pytest.raises(ValueError, match="!!python/"):
        load_primitive_info(str(path))
    assert not canary.exists()


def test_load_primitive_info_near_miss_legit_yaml_loads():
    from irma.ncrystal.build import load_primitive_info
    symbols, masses, positions, lattice = load_primitive_info(
        str(_GRAPHITE_YAML))
    assert len(symbols) > 0
    assert len(symbols) == len(masses) == positions.shape[0]
    assert lattice.shape == (3, 3)


def test_load_phonopy_mesh_near_miss_legit_yaml_loads():
    from irma.core.phonopy_io import load_phonopy_mesh
    data = load_phonopy_mesh(str(_GRAPHITE_YAML), [1, 1, 1])
    assert data.n_qpoints == 1  # one q-point on the 1x1x1 mesh
    assert data.frequencies_ev.shape == (1, 3 * data.n_atoms)
