"""CLI tests: `irma mlip build|emit|validate` end to end on the EMT dev
backend (IRMA_MLIP_DEV_BACKENDS=1), plus exit codes and parsing errors."""
import json
import os

import pytest

ase = pytest.importorskip("ase")
pytest.importorskip("phonopy")
pytest.importorskip("yaml")

from ase.build import bulk                          # noqa: E402
from ase.io import write as ase_write               # noqa: E402

from irma.mlip.cli import (                         # noqa: E402
    _parse_pairs, _parse_species, _parse_supercell, main)

QUIET_ENV = {"IRMA_MLIP_DEV_BACKENDS": "1"}


@pytest.fixture(autouse=True)
def _dev_backend(monkeypatch):
    monkeypatch.setenv("IRMA_MLIP_DEV_BACKENDS", "1")


@pytest.fixture()
def al_poscar(tmp_path):
    path = tmp_path / "Al.vasp"
    ase_write(str(path), bulk("Al", "fcc", a=4.05, cubic=True),
              direct=True, format="vasp")
    return str(path)


def test_parse_helpers():
    assert _parse_pairs(["C=31", "O=32"], "--mat", int) == {"C": 31, "O": 32}
    assert _parse_pairs(["H=2-H"], "--nuclide") == {"H": "2-H"}
    with pytest.raises(ValueError, match="SYMBOL=VALUE"):
        _parse_pairs(["C31"], "--mat", int)
    with pytest.raises(ValueError, match="cannot parse"):
        _parse_pairs(["C=abc"], "--mat", int)
    assert _parse_species(["C:b_coh_fm=6.646,sigma_inc_b=0.001"]) == \
        {"C": {"b_coh_fm": 6.646, "sigma_inc_b": 0.001}}
    with pytest.raises(ValueError, match="needs a number"):
        _parse_species(["C:b_coh_fm=x"])
    assert _parse_supercell("12") == 12
    assert _parse_supercell("2 2 1") == (2, 2, 1)
    with pytest.raises(ValueError):
        _parse_supercell("2 2")


def test_build_validate_emit_end_to_end(al_poscar, tmp_path, capsys):
    outdir = str(tmp_path / "bundle")
    rc = main(["build", al_poscar, "-o", outdir, "--potential", "emt", "--allow-dev-backend",
               "--supercell", "2 2 2", "--mesh", "4 4 4"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "bundle written and validated" in out
    assert "irma mlip emit" in out                 # next-steps block

    assert main(["validate", outdir]) == 0
    out = capsys.readouterr().out
    assert "OK" in out and "fingerprint" in out

    rc = main(["emit", outdir, "--to", "endf,spectra,ncrystal",
               "--mat", "Al=45"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Next steps" in out and "irma spectra run" in out
    for name in ("endf_Al.input", "spectra.yaml", "ncrystal.yaml",
                 "emit_manifest.json"):
        assert os.path.isfile(os.path.join(outdir, name)), name
    rec = json.load(open(os.path.join(outdir, "emit_manifest.json")))
    assert rec["species"][0]["mat"] == 45


def test_build_chained_emit(al_poscar, tmp_path, capsys):
    outdir = str(tmp_path / "bundle")
    rc = main(["build", al_poscar, "-o", outdir, "--potential", "emt", "--allow-dev-backend",
               "--supercell", "2 2 2", "--mesh", "4 4 4",
               "--emit", "endf", "--mat", "Al=45"])
    assert rc == 0
    assert os.path.isfile(os.path.join(outdir, "endf_Al.input"))


def test_unconverged_relaxation_is_exit_3(al_poscar, tmp_path, capsys):
    from ase.io import read as ase_read
    atoms = ase_read(al_poscar)
    atoms.rattle(stdev=0.1, seed=3)
    rattled = str(tmp_path / "rattled.vasp")
    ase_write(rattled, atoms, direct=True, format="vasp")

    rc = main(["build", rattled, "-o", str(tmp_path / "b"),
               "--potential", "emt", "--allow-dev-backend", "--supercell", "2 2 2",
               "--fmax", "1e-9", "--nmax", "2"])
    assert rc == 3
    err = capsys.readouterr().err
    assert "did not converge" in err and "--force" in err

    rc = main(["build", rattled, "-o", str(tmp_path / "b2"),
               "--potential", "emt", "--allow-dev-backend", "--supercell", "2 2 2",
               "--mesh", "4 4 4", "--fmax", "1e-9", "--nmax", "2",
               "--force"])
    assert rc == 0


def test_usage_and_validation_errors_are_exit_2(al_poscar, tmp_path, capsys):
    assert main(["build", "missing.cif", "-o", str(tmp_path)]) == 2
    assert main(["validate", str(tmp_path / "nonexistent")]) == 2
    rc = main(["build", al_poscar, "-o", str(tmp_path / "x"),
               "--potential", "vasp"])
    assert rc == 2
    err = capsys.readouterr().err
    assert "unknown potential" in err and "emt" not in err

    # argparse usage errors surface as exit 2 as well
    assert main(["emit"]) == 2
    assert main([]) == 2


def test_emt_is_hidden_without_the_dev_env(al_poscar, tmp_path, monkeypatch,
                                           capsys):
    monkeypatch.delenv("IRMA_MLIP_DEV_BACKENDS", raising=False)
    rc = main(["build", al_poscar, "-o", str(tmp_path / "b"),
               "--potential", "emt"])
    assert rc == 2
    assert "unknown potential" in capsys.readouterr().err


def test_missing_potential_dependency_is_exit_4(al_poscar, tmp_path,
                                                monkeypatch, capsys):
    import sys as _sys
    monkeypatch.setitem(_sys.modules, "mattersim", None)
    rc = main(["build", al_poscar, "-o", str(tmp_path / "b"),
               "--potential", "mattersim"])
    assert rc == 4
    assert "pip install mattersim" in capsys.readouterr().err


def test_emit_rejects_invalid_bundle(tmp_path, capsys):
    (tmp_path / "manifest.json").write_text('{"kind": "wrong"}')
    rc = main(["emit", str(tmp_path), "--to", "endf", "--mat", "Al=45"])
    assert rc == 2


def test_emit_has_no_principal_option():
    """--principal is gone: the spectra emitter has no principal scatterer.

    Its only effect was to sort one symbol first in the emitted scatterer
    list, which the forward model ignores, so the flag advertised a physics
    choice it could not make. The word stays load-bearing for the endf and
    ncrystal targets, which write one deck/pack per principal species.
    """
    from irma.mlip.cli import _build_parser
    with pytest.raises(SystemExit):
        _build_parser().parse_args(
            ["emit", "bundle", "--to", "spectra", "--principal", "C"])


def test_emit_unknown_target(al_poscar, tmp_path, capsys):
    outdir = str(tmp_path / "bundle")
    assert main(["build", al_poscar, "-o", outdir, "--potential", "emt", "--allow-dev-backend",
                 "--supercell", "2 2 2", "--mesh", "4 4 4"]) == 0
    rc = main(["emit", outdir, "--to", "oclimax"])
    assert rc == 2
    assert "unknown emit target" in capsys.readouterr().err


def test_explicit_supercell_12_is_honored_under_disordered(al_poscar,
                                                           tmp_path, capsys):
    # review finding: '--supercell 12' must mean Lmin=12 even with
    # --disordered (only OMISSION selects the 1x1x1 box default)
    outdir = str(tmp_path / "b")
    rc = main(["build", al_poscar, "-o", outdir, "--potential", "emt",
               "--allow-dev-backend", "--disordered", "--supercell", "12",
               "--mesh", "2 2 2"])
    assert rc == 0
    assert "supercell (3, 3, 3)" in capsys.readouterr().out  # ceil(12/4.05)

    outdir2 = str(tmp_path / "b2")
    rc = main(["build", al_poscar, "-o", outdir2, "--potential", "emt",
               "--allow-dev-backend", "--disordered", "--mesh", "2 2 2"])
    assert rc == 0
    assert "supercell (1, 1, 1)" in capsys.readouterr().out


def test_bad_numeric_options_fail_before_any_compute(al_poscar, tmp_path,
                                                     capsys):
    for extra in (["--mesh", "4 4"], ["--mesh", "0 4 4"],
                  ["--threads", "0"], ["--jobs", "0"],
                  ["--supercell", "2 2"]):
        rc = main(["build", al_poscar, "-o", str(tmp_path / "x"),
                   "--potential", "emt", "--allow-dev-backend"] + extra)
        assert rc == 2, extra
        capsys.readouterr()
    assert not (tmp_path / "x").exists()   # nothing was created


def test_outdir_preflight_fails_fast(al_poscar, tmp_path, capsys):
    target = tmp_path / "occupied"
    target.mkdir()
    (target / "keep.txt").write_text("user data")
    rc = main(["build", al_poscar, "-o", str(target), "--potential", "emt",
               "--allow-dev-backend"])
    assert rc == 2
    assert "not empty" in capsys.readouterr().err
    assert (target / "keep.txt").read_text() == "user data"
    assert not (target / "scratch").exists()  # preflight beat the compute


def test_dev_backend_is_stamped_and_warned_on_emit(al_poscar, tmp_path,
                                                   capsys):
    outdir = str(tmp_path / "b")
    assert main(["build", al_poscar, "-o", outdir, "--potential", "emt",
                 "--allow-dev-backend", "--supercell", "2 2 2",
                 "--mesh", "4 4 4"]) == 0
    m = json.load(open(os.path.join(outdir, "manifest.json")))
    assert m["calculator"]["dev_backend"] is True
    capsys.readouterr()
    assert main(["emit", outdir, "--to", "endf", "--mat", "Al=45"]) == 0
    assert "DEVELOPMENT backend" in capsys.readouterr().out


def test_disordered_ncrystal_conflict_fails_before_writing(al_poscar,
                                                           tmp_path, capsys):
    outdir = str(tmp_path / "b")
    assert main(["build", al_poscar, "-o", outdir, "--potential", "emt",
                 "--allow-dev-backend", "--disordered",
                 "--mesh", "2 2 2"]) == 0
    capsys.readouterr()
    rc = main(["emit", outdir, "--to", "endf,ncrystal", "--mat", "Al=45"])
    assert rc == 2
    assert not os.path.isfile(os.path.join(outdir, "endf_Al.input"))


def test_unknown_mat_symbol_is_rejected(al_poscar, tmp_path, capsys):
    outdir = str(tmp_path / "b")
    assert main(["build", al_poscar, "-o", outdir, "--potential", "emt",
                 "--allow-dev-backend", "--supercell", "2 2 2",
                 "--mesh", "4 4 4"]) == 0
    rc = main(["emit", outdir, "--to", "endf", "--mat", "Al=45",
               "--mat", "Cu=99"])
    assert rc == 2
    assert "not present in the structure" in capsys.readouterr().err


def test_top_level_routing():
    from irma.cli import main as top_main
    assert top_main(["mlip", "--help"]) == 0


def test_cross_target_preflight_blocks_before_any_deck_is_written(
        al_poscar, tmp_path, capsys):
    """CDX-1: a conflict on the LAST target of a --to set must surface
    before the FIRST artifact is published, so a failed command never
    leaves a partial, mixed-generation output set."""
    outdir = str(tmp_path / "bundle")
    assert main(["build", al_poscar, "-o", outdir, "--potential", "emt",
                 "--allow-dev-backend", "--supercell", "2 2 2",
                 "--mesh", "4 4 4"]) == 0
    capsys.readouterr()

    # only ncrystal.yaml pre-exists; endf and spectra come first in write
    # order and must NOT be produced
    with open(os.path.join(outdir, "ncrystal.yaml"), "w") as fh:
        fh.write("# hand-written, must not be clobbered\n")
    rc = main(["emit", outdir, "--to", "endf,spectra,ncrystal",
               "--mat", "Al=45"])
    assert rc == 2
    assert "--overwrite" in capsys.readouterr().err
    assert not os.path.isfile(os.path.join(outdir, "endf_Al.input"))
    assert not os.path.isfile(os.path.join(outdir, "spectra.yaml"))
    assert not os.path.isfile(os.path.join(outdir, "emit_manifest.json"))
    with open(os.path.join(outdir, "ncrystal.yaml")) as fh:
        assert "hand-written" in fh.read()

    # --overwrite makes the same invocation succeed end to end
    assert main(["emit", outdir, "--to", "endf,spectra,ncrystal",
                 "--mat", "Al=45", "--overwrite"]) == 0
    for name in ("endf_Al.input", "spectra.yaml", "ncrystal.yaml"):
        assert os.path.isfile(os.path.join(outdir, name)), name


def test_explicit_elastic_format_rejected_on_a_disordered_bundle(
        al_poscar, tmp_path, capsys):
    """--elastic-format is a crystal-deck selector; an EXPLICITLY typed
    value must be rejected on a disordered bundle even when it names the
    argparse default, exactly like --inelastic-mode."""
    outdir = str(tmp_path / "b")
    assert main(["build", al_poscar, "-o", outdir, "--potential", "emt",
                 "--allow-dev-backend", "--disordered",
                 "--mesh", "2 2 2"]) == 0
    capsys.readouterr()

    for value in ("mef", "sef"):
        rc = main(["emit", outdir, "--to", "endf", "--mat", "Al=45",
                   "--elastic-format", value])
        assert rc == 2, value
        assert "--elastic-format is not applicable" in \
            capsys.readouterr().err
        assert not os.path.isfile(os.path.join(outdir, "endf_Al.input"))

    # omitted: the classic disordered path emits normally
    assert main(["emit", outdir, "--to", "endf", "--mat", "Al=45"]) == 0
    assert os.path.isfile(os.path.join(outdir, "endf_Al.input"))


def test_omitted_elastic_format_still_defaults_to_mef(al_poscar, tmp_path,
                                                      capsys):
    """The argparse default moved to None; the emitted crystal deck must
    still be MEF when the flag is omitted."""
    outdir = str(tmp_path / "b")
    assert main(["build", al_poscar, "-o", outdir, "--potential", "emt",
                 "--allow-dev-backend", "--supercell", "2 2 2",
                 "--mesh", "4 4 4"]) == 0
    assert main(["emit", outdir, "--to", "endf", "--mat", "Al=45"]) == 0
    rec = json.load(open(os.path.join(outdir, "emit_manifest.json")))
    assert rec["elastic_format"] == "mef"


def test_uncovered_element_is_refused_before_relaxation(tmp_path, monkeypatch,
                                                       capsys):
    """A checkpoint that does not cover the structure's elements stops
    the build with a readable exit-2 message, before any force call."""
    from ase.calculators.emt import EMT
    from irma.mlip import calculators

    def fake_make_calculator(spec):
        return EMT(), {"potential": spec.potential, "checkpoint": "w.model",
                       "checkpoint_sha256": None, "dtype": "float64",
                       "checkpoint_elements": ["H", "C"],
                       "checkpoint_model_class": "ScaleShiftMACE",
                       "checkpoint_r_max_A": 6.5,
                       "checkpoint_num_interactions": 2,
                       "checkpoint_stored_dtype": "float32",
                       "dtype_note": "stored weights: float32"}
    monkeypatch.setattr(calculators, "make_calculator", fake_make_calculator)
    cu = tmp_path / "Cu.vasp"
    ase_write(str(cu), bulk("Cu", "fcc", a=3.6, cubic=True), direct=True,
              format="vasp")
    rc = main(["build", str(cu), "-o", str(tmp_path / "b"),
               "--potential", "emt", "--allow-dev-backend"])
    assert rc == 2
    err = capsys.readouterr().err
    assert "covers H C" in err and "contains Cu" in err
    assert not (tmp_path / "b").exists()
