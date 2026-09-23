"""reject_unsafe_phonopy_yaml fires before phonopy's unsafe YAML loader at
load_phonopy_mesh, build_model_context and ncrystal load_primitive_info: a
``!!python/`` payload raises ValueError and its canary file never appears.
"""
import argparse

import pytest

pytest.importorskip("phonopy")


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
