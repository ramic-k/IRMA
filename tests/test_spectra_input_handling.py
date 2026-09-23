"""Spectra config/CLI input handling (UX hardening).

Engine-free pins for the loud-failure paths:
  (1) a typo'd TOP-LEVEL config section (or a --set override creating one)
      raises naming the bad section and the accepted set, instead of being
      silently dropped (wrong-instrument run, exit 0);
  (2) --resolution-model chopper with omitted chopper flags fails by FLAG
      NAME (no float(None) TypeError, no "unknown chopper instrument 'None'");
  (3) check_input_files names the responsible FIELD for a nonexistent input
      file, and the run/flag CLI forms preflight through it (exit 2);
  (5) --max-phonon-order errors are self-describing (no "_max_order" leak);
  (6) output-format dispatch is case-insensitive ('-o OUT.NPZ' never gets
      CSV bytes) and unrecognized extensions warn;
  (7) success/progress lines go to stdout, diagnostics to stderr.
(Finding 4 -- output_mode='map' allowed for every geometry -- is pinned in
test_spectra_config.py::test_map_output_mode_allowed_for_all_geometries.)
"""

import numpy as np
import pytest

from irma.spectra import cli as scli
from irma.spectra.config import (
    SpectraConfig, SpectraConfigError, check_input_files, validate,
)
from irma.spectra.forward import SpectrumResult


# ---- (1) unknown top-level sections -----------------------------------------
def test_typod_top_level_section_raises():
    """'instrumnet:' must raise naming the typo and the accepted sections,
    not silently run the default (VISION) instrument with exit 0."""
    with pytest.raises(SpectraConfigError, match="instrumnet") as ei:
        SpectraConfig.from_dict({
            "material": {"phonopy_yaml": "g.yaml"},
            "instrumnet": {"geometry": "direct", "e_fixed_meV": 100.0},
        })
    assert "'instrument'" in str(ei.value)        # tells the user the fix


def test_set_override_typo_section_fails_loudly():
    """apply_overrides setdefault creates the orphan section; from_dict must
    then reject it -- a typo'd `--set instrment.geometry=...` was silently
    ignored before."""
    d = {"material": {"phonopy_yaml": "g.yaml"}}
    scli.apply_overrides(d, ["instrment.geometry=direct"])
    with pytest.raises(SpectraConfigError, match="instrment"):
        SpectraConfig.from_dict(d)


def test_all_known_sections_still_accepted():
    cfg = SpectraConfig.from_dict({
        "material": {"phonopy_yaml": "g.yaml"}, "physics": {}, "grid": {},
        "instrument": {}})
    assert cfg.instrument.geometry == "vision"


# ---- (2) chopper resolution: omitted flags / null values --------------------
def test_flag_form_chopper_missing_flags_named():
    """validate() names every omitted chopper flag."""
    ns = scli.build_parser().parse_args(
        ["direct", "--phonopy-yaml", "g.yaml", "--ei", "250", "--e-max", "200",
         "--angles", "30,60", "--resolution-model", "chopper", "-o", "o.csv"])
    with pytest.raises(SpectraConfigError) as ei:
        validate(scli.config_from_args(ns))
    msg = str(ei.value)
    assert "--chopper-instrument" in msg
    assert "--chopper-package" in msg
    assert "--chopper-frequency" in msg


def test_main_chopper_missing_flags_clean_exit_2(tmp_path, capsys):
    rc = scli.main(
        ["direct", "--phonopy-yaml", "g.yaml", "--ei", "250", "--e-max", "200",
         "--angles", "30,60", "--resolution-model", "chopper",
         "-o", str(tmp_path / "o.csv")])
    assert rc == 2
    err = capsys.readouterr().err
    assert "--chopper-instrument" in err and "Traceback" not in err


def test_chopper_null_values_surface_missing_keys():
    """YAML `null` chopper values are reported as missing keys by name."""
    cfg = SpectraConfig.from_dict({
        "material": {"phonopy_yaml": "g.yaml"},
        "grid": {"e_max_meV": 100.0, "de_meV": 1.0, "dq_max_invA": 0.05},
        "instrument": {"geometry": "direct", "e_fixed_meV": 250.0,
                       "angles_deg": [30.0, 60.0],
                       "resolution_model": "chopper",
                       "chopper_spec": {"instrument": None,
                                        "package": "ARCS-700-1.5-AST",
                                        "frequency": None}},
    })
    with pytest.raises(SpectraConfigError) as ei:
        validate(cfg)
    msg = str(ei.value)
    assert "chopper_spec is missing" in msg
    assert "'instrument'" in msg and "'frequency'" in msg
    assert "None" not in msg


# ---- (3) input-file preflight ------------------------------------------------
def _cfg_with(material_extra=None, physics_extra=None):
    mat = {"phonopy_yaml": "g.yaml"}
    mat.update(material_extra or {})
    return SpectraConfig.from_dict(
        {"material": mat, "physics": dict(physics_extra or {})})


def test_check_input_files_names_phonopy_yaml(tmp_path):
    cfg = _cfg_with({"phonopy_yaml": str(tmp_path / "nothere.yaml")})
    with pytest.raises(SpectraConfigError, match="material.phonopy_yaml") as ei:
        check_input_files(cfg)
    assert "file not found" in str(ei.value)
    assert "nothere.yaml" in str(ei.value)


@pytest.mark.parametrize("field", ["born", "force_constants", "force_sets"])
def test_check_input_files_names_optional_model_files(tmp_path, field):
    yam = tmp_path / "g.yaml"
    yam.write_text("x")
    cfg = _cfg_with({"phonopy_yaml": str(yam), field: str(tmp_path / "missing")})
    with pytest.raises(SpectraConfigError, match=f"material.{field}"):
        check_input_files(cfg)


def test_check_input_files_names_elastic_tape(tmp_path):
    yam = tmp_path / "g.yaml"
    yam.write_text("x")
    cfg = _cfg_with({"phonopy_yaml": str(yam)},
                    {"elastic_from_tape": str(tmp_path / "nothere.endf")})
    with pytest.raises(SpectraConfigError, match="physics.elastic_from_tape"):
        check_input_files(cfg)


def test_check_input_files_names_scatterer_dos_file(tmp_path):
    cfg = SpectraConfig.from_dict({
        "material": {"scatterers": [
            {"symbol": "C", "sigma_bound_b": 5.551, "awr": 11.898,
             "dos_file": str(tmp_path / "missing_dos.txt")}]},
        "physics": {"inelastic_mode": 0, "elastic": False}})
    with pytest.raises(SpectraConfigError, match=r"scatterers\[0\]") as ei:
        check_input_files(cfg)
    msg = str(ei.value)
    assert "dos_file" in msg and "(C)" in msg and "missing_dos.txt" in msg


def test_check_input_files_passes_when_files_exist(tmp_path):
    yam = tmp_path / "g.yaml"
    yam.write_text("x")
    tape = tmp_path / "g.endf"
    tape.write_text("x")
    cfg = _cfg_with({"phonopy_yaml": str(yam)},
                    {"elastic_from_tape": str(tape)})
    assert check_input_files(cfg) is cfg


def test_run_command_preflights_config_paths(tmp_path, capsys):
    """`irma spectra run` must fail at load time naming the FIELD, never a
    mid-run \"[Errno 2] No such file or directory\" with no field name."""
    import json
    cfgfile = tmp_path / "cfg.json"
    cfgfile.write_text(json.dumps(
        {"material": {"phonopy_yaml": str(tmp_path / "nothere.yaml")}}))
    rc = scli.main(["run", str(cfgfile), "-o", str(tmp_path / "o.csv")])
    assert rc == 2
    err = capsys.readouterr().err
    assert "material.phonopy_yaml" in err and "file not found" in err
    assert "Errno" not in err


def test_flag_form_preflights_paths(tmp_path, capsys):
    rc = scli.main(["vision", "--phonopy-yaml", str(tmp_path / "nothere.yaml"),
                    "--scatterer", "C,5.551,11.898",
                    "-o", str(tmp_path / "o.csv")])
    assert rc == 2
    err = capsys.readouterr().err
    assert "material.phonopy_yaml" in err and "file not found" in err


# ---- (5) --max-phonon-order message ------------------------------------------
def test_max_order_error_is_self_describing(capsys):
    with pytest.raises(SystemExit):
        scli.build_parser().parse_args(
            ["vision", "--max-phonon-order", "banana", "-o", "o.csv"])
    err = capsys.readouterr().err
    assert "must be an integer or 'auto'" in err and "'banana'" in err
    assert "_max_order" not in err              # no internal-name leak


def test_max_order_accepts_auto_and_int():
    ns = scli.build_parser().parse_args(
        ["vision", "--max-phonon-order", "auto", "-o", "o.csv"])
    assert ns.max_phonon_order == "auto"
    ns = scli.build_parser().parse_args(
        ["vision", "--max-phonon-order", "50", "-o", "o.csv"])
    assert ns.max_phonon_order == 50


# ---- (6) extensionless spectrum output -----------------------------------------
def _result():
    E = np.linspace(0.0, 200.0, 11)
    I = np.abs(np.sin(E / 20.0))
    return SpectrumResult(
        E=E, Q=np.ones((1, 11)), I_inelastic=I, I_elastic=0.5 * I,
        I_total=1.5 * I, geometry="indirect", angles_deg=[45.0],
        I_inelastic_per_angle=I[None, :], I_elastic_per_angle=0.5 * I[None, :],
        metadata={"inelastic_mode": 2, "elastic": True, "n_bragg_edges": 0,
                  "engine_metadata": {}})


def test_write_spectrum_extensionless_is_silent_csv(tmp_path, capsys):
    """No extension -> the documented CSV default, with no warning noise."""
    scli.write_spectrum(_result(), tmp_path / "out")
    assert (tmp_path / "out").read_text().startswith("# IRMA spectrum")
    assert capsys.readouterr().err == ""


# ---- (7) stream conventions ---------------------------------------------------
class _Ins:
    output_mode = "map"
    map_mask = True
    export_components = False


class _MapCfg:
    instrument = _Ins()


def test_success_lines_on_stdout_errors_on_stderr(tmp_path, monkeypatch, capsys):
    """One rule for the whole `irma` entry point (the deck CLI's): provenance,
    progress and the final 'Wrote ...' go to STDOUT; stderr is for diagnostics
    only -- `irma spectra run cfg -o out.csv > log` must capture the run log."""
    monkeypatch.setattr(scli, "_load_cfg", lambda ns: _MapCfg())
    monkeypatch.setattr(scli, "_provenance", lambda c, o: "irma spectra: provenance")
    monkeypatch.setattr(scli, "run_map",
                        lambda c, **kw: kw["progress"]("progress line") or "SM")
    monkeypatch.setattr(scli, "write_map", lambda sm, out, masked=False: out)
    out_path = tmp_path / "m.npz"
    rc = scli.main(["run", str(tmp_path / "cfg.yaml"), "-o", str(out_path)])
    out, err = capsys.readouterr()
    assert rc == 0
    assert "provenance" in out and "progress line" in out
    assert f"Wrote {out_path}" in out
    assert err == ""                              # nothing routine on stderr


def test_config_error_stays_on_stderr(tmp_path, capsys):
    rc = scli.main(["run", str(tmp_path / "nope.yaml"), "-o",
                    str(tmp_path / "o.csv")])
    assert rc == 2
    out, err = capsys.readouterr()
    assert "No such file" in err and "No such file" not in out
