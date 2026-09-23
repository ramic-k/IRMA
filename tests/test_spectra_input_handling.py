"""Spectra config/CLI input handling: a typo'd config section, omitted
chopper flags, missing input files (named by field, exit 2), the
--max-phonon-order message and the extensionless-output default.
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


# ---- (2) chopper resolution: omitted flags / null values --------------------
def test_main_chopper_missing_flags_clean_exit_2(tmp_path, capsys):
    rc = scli.main(
        ["direct", "--phonopy-yaml", "g.yaml", "--ei", "250", "--e-max", "200",
         "--angles", "30,60", "--resolution-model", "chopper",
         "-o", str(tmp_path / "o.csv")])
    assert rc == 2
    err = capsys.readouterr().err
    assert "--chopper-instrument" in err and "--chopper-package" in err
    assert "--chopper-frequency" in err and "Traceback" not in err


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


@pytest.mark.parametrize("section,field", [
    ("material", "phonopy_yaml"), ("material", "born"),
    ("material", "force_constants"), ("material", "force_sets"),
    ("physics", "elastic_from_tape")])
def test_check_input_files_names_the_missing_file(tmp_path, section, field):
    yam = tmp_path / "g.yaml"
    yam.write_text("x")
    mat, phys = {"phonopy_yaml": str(yam)}, {}
    (mat if section == "material" else phys)[field] = str(tmp_path / "nothere")
    with pytest.raises(SpectraConfigError, match=rf"{section}\.{field}") as ei:
        check_input_files(_cfg_with(mat, phys))
    assert "file not found" in str(ei.value) and "nothere" in str(ei.value)


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
