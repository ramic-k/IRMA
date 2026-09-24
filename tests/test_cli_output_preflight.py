"""Command-line error handling before any compute starts.

- The shared output-path preflight (irma.cli.validate_output_path): an
  existing-directory target, a missing parent, or a read-only parent are
  caught up front, so a long compute never starts on an unwritable path.
- The spectra output-path resolvers, which must name the file each writer
  opens.
- The NCrystal exporter's routine errors (missing or malformed config, a
  missing optional dependency) exit with a code and a message, not a
  traceback.
"""
import os

import pytest

from irma.cli import validate_output_path


def test_accepts_writable_file_target(tmp_path):
    assert validate_output_path(str(tmp_path / "out.endf")) is None


def test_rejects_existing_directory_target(tmp_path):
    msg = validate_output_path(str(tmp_path))
    assert msg is not None and "directory, not a file" in msg


def test_rejects_missing_parent(tmp_path):
    msg = validate_output_path(str(tmp_path / "nope" / "out.endf"))
    assert msg is not None and "does not exist" in msg


def test_rejects_unwritable_parent(tmp_path):
    ro = tmp_path / "ro"
    ro.mkdir()
    os.chmod(ro, 0o500)
    try:
        if os.access(ro, os.W_OK):          # e.g. running as root -> chmod is moot
            pytest.skip("parent still writable (running as root?)")
        msg = validate_output_path(str(ro / "out.endf"))
        assert msg is not None and "not writable" in msg
    finally:
        os.chmod(ro, 0o700)


def test_deck_cli_fails_fast_before_compute(tmp_path, capsys):
    """A directory output makes the deck path return at the preflight (exit 1),
    WITHOUT importing/running run_leapr."""
    from irma.cli import main
    deck = tmp_path / "in.input"
    deck.write_text("leapr\n")               # input must exist to reach the output check
    rc = main([str(deck), str(tmp_path)])    # output is a directory
    assert rc == 1
    err = capsys.readouterr().err
    assert "directory, not a file" in err


def test_output_resolvers_match_the_writers(tmp_path):
    """One resolver per mode = the exact path its writer opens."""
    from irma.spectra.cli import (resolve_map_output_path,
                                  resolve_spectrum_output_path)

    # map: an unknown or missing suffix gets .npz appended, .json included
    assert resolve_map_output_path(tmp_path / "out") == str(tmp_path / "out.npz")
    assert resolve_map_output_path(tmp_path / "out.json") == str(
        tmp_path / "out.json.npz")
    assert resolve_map_output_path(tmp_path / "OUT.NPZ") == str(
        tmp_path / "OUT.npz")
    # cuts: extensionless writes CSV at the RAW path; known suffixes normalize
    assert resolve_spectrum_output_path(tmp_path / "out") == str(tmp_path / "out")
    assert resolve_spectrum_output_path(tmp_path / "OUT.JSON") == str(
        tmp_path / "OUT.json")


# ---- routine ncrystal exporter errors exit cleanly --------------------------

_CFG = ("material:\n"
        "  phonopy_yaml: x.yaml\n"
        "  mesh: [4, 4, 4]\n"
        "  temperature_K: 296.0\n"
        "  scatterers:\n"
        "    - {symbol: C, sigma_bound_b: 5.551, awr: 11.898,\n"
        "       b_coh_fm: 6.646, sigma_inc_b: 0.001}\n"
        "export:\n"
        "  material_id: x\n")


def test_ncrystal_cli_missing_config_exits_2(tmp_path, capsys):
    from irma.ncrystal.__main__ import main as ncrystal_main

    rc = ncrystal_main([str(tmp_path / "nope.yaml"), "-o", str(tmp_path)])
    assert rc == 2
    err = capsys.readouterr().err
    assert "config file not found" in err and "Traceback" not in err


def test_ncrystal_cli_malformed_yaml_exits_2(tmp_path, capsys):
    pytest.importorskip("yaml")   # bare core install: covered by the PyYAML
    from irma.ncrystal.__main__ import main as ncrystal_main   # -message test

    bad = tmp_path / "bad.yaml"
    bad.write_text("material: [unclosed\n")
    rc = ncrystal_main([str(bad), "-o", str(tmp_path)])
    assert rc == 2
    err = capsys.readouterr().err
    assert "not valid YAML" in err and "Traceback" not in err


def test_ncrystal_cli_config_error_exits_2(tmp_path, capsys):
    pytest.importorskip("yaml")
    from irma.ncrystal.__main__ import main as ncrystal_main

    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(_CFG + "  inelastic_mode: 7\n")
    rc = ncrystal_main([str(cfg), "-o", str(tmp_path)])
    assert rc == 2
    err = capsys.readouterr().err
    assert "inelastic_mode" in err and "Traceback" not in err


def test_ncrystal_cli_missing_phonopy_exits_3(tmp_path, capsys, monkeypatch):
    """ImportError from the optional phonopy dependency must hit the clean
    exit-3 boundary with an install hint, not escape as a traceback."""
    pytest.importorskip("yaml")
    import irma.ncrystal.build as build_mod
    from irma.ncrystal.__main__ import main as ncrystal_main

    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(_CFG)

    def _no_phonopy(*a, **k):
        raise ModuleNotFoundError("No module named 'phonopy'")

    monkeypatch.setattr(build_mod, "write_packs", _no_phonopy)
    rc = ncrystal_main([str(cfg), "-o", str(tmp_path)])
    assert rc == 3
    err = capsys.readouterr().err
    assert "phonopy" in err and "irma[phonopy]" in err
    assert "Traceback" not in err


def test_ncrystal_cli_missing_pyyaml_message(tmp_path, capsys, monkeypatch):
    """On a bare core install the exporter fails with a clean PyYAML hint,
    not a ModuleNotFoundError traceback (caught by the bare-install CI gate)."""
    import builtins

    real_import = builtins.__import__

    def no_yaml(name, *a, **kw):
        if name == "yaml":
            raise ModuleNotFoundError("No module named 'yaml'")
        return real_import(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", no_yaml)
    from irma.ncrystal.__main__ import main as ncrystal_main

    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("material: {}\n")
    rc = ncrystal_main([str(cfg), "-o", str(tmp_path / "out")])
    assert rc == 2
    err = capsys.readouterr().err
    assert "PyYAML is required" in err and "Traceback" not in err
