from pathlib import Path

import pytest
import yaml

from ncrystal_plugin_ENDFTSL.__main__ import main

TAPE = Path(__file__).parents[1] / "examples" / "graphite" / "graphite_mef_296K.endf"

# one regex ([A-Z][a-z]?) guards pack IDs, filenames and NCMAT element lines
BAD_SYMBOLS = ["../x", "c", "Xyz"]


def _write_cfg(tmp_path, symbols, mid="poly"):
    cfg = {"material_id": mid, "temperature": 296.0, "density": 2.26,
           "species": [{"tape": str(TAPE), "symbol": s, "mass": 12.0107,
                        "fraction": 1.0 / len(symbols)} for s in symbols]}
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return p


def test_cli_writes_pack_and_ncmat(tmp_path):
    rc = main([str(TAPE), "-o", str(tmp_path), "--material-id", "graphite",
               "--symbol", "C", "--mass", "12.0107", "--density", "2.26",
               "--temperature", "296"])
    assert rc == 0
    pk = tmp_path / "graphite.endftslpack"
    nc = tmp_path / "graphite.ncmat"
    assert pk.exists() and nc.exists()
    assert "ENDFTSLPACK_TEXT_V1" == pk.read_text().splitlines()[0]
    assert "@CUSTOM_ENDFTSL" in nc.read_text()
    # the ncmat references the pack by absolute path
    assert str(pk.resolve()) in nc.read_text()


def test_cli_single_requires_the_identity_flags(tmp_path):
    # the tape does not name its element: no silent carbon default
    with pytest.raises(SystemExit):
        main([str(TAPE), "-o", str(tmp_path / "out"), "--symbol", "C"])
    assert not (tmp_path / "out").exists()


def test_cli_single_rejects_bad_symbol(tmp_path):
    out = tmp_path / "out"
    with pytest.raises(SystemExit, match="species symbol"):
        main([str(TAPE), "-o", str(out), "--material-id", "g", "--symbol", "c",
              "--mass", "12.0107", "--density", "2.26", "--temperature", "296"])
    assert not out.exists(), "rejection must leave the output tree untouched"


@pytest.mark.parametrize("sym", BAD_SYMBOLS)
def test_cli_config_rejects_bad_symbol(tmp_path, sym):
    cfg = _write_cfg(tmp_path, [sym])
    out = tmp_path / "out"
    with pytest.raises(SystemExit, match="species symbol"):
        main(["--config", str(cfg), "-o", str(out)])
    assert not out.exists(), "rejection must leave the output tree untouched"


def test_cli_config_rejects_duplicate_symbols(tmp_path):
    cfg = _write_cfg(tmp_path, ["C", "C"])
    out = tmp_path / "out"
    with pytest.raises(SystemExit, match="duplicate species symbol"):
        main(["--config", str(cfg), "-o", str(out)])
    assert not out.exists(), "rejection must leave the output tree untouched"


def test_cli_config_valid_polyatomic_one_pack_per_species(tmp_path):
    # pseudo-polyatomic: the monatomic graphite tape as two 0.5-fraction species
    # (same layout as ENDF/B-VIII.1 BeO: the per-atom Bragg edges replicated per tape).
    cfg = _write_cfg(tmp_path, ["C", "N"])
    out = tmp_path / "out"
    rc = main(["--config", str(cfg), "-o", str(out)])
    assert rc == 0
    packs = [out / "poly__C.endftslpack", out / "poly__N.endftslpack"]
    for p in packs:
        assert p.exists()
    nc = (out / "poly.ncmat").read_text()
    pack_lines = [ln for ln in nc.splitlines() if ln.strip().startswith("pack ")]
    assert len(pack_lines) == 2                       # one NCMAT pack line per species
    assert len(set(pack_lines)) == 2                  # each references a unique path
    for p, ln in zip(packs, pack_lines):
        assert str(p.resolve()) in ln


# ---- reviews IO-1 + CLI-1c: args validated before work; clean boundaries ---

def test_cli_density_zero_exits_2_before_writing_anything(tmp_path, capsys):
    rc = main([str(TAPE), "-o", str(tmp_path / "out"), "--symbol", "C",
               "--mass", "12.0107", "--density", "0"])
    assert rc == 2
    err = capsys.readouterr().err
    assert "density" in err and "Traceback" not in err
    assert not (tmp_path / "out").exists()      # NOTHING was written


def test_cli_missing_tape_exits_3_cleanly(tmp_path, capsys):
    rc = main([str(tmp_path / "nope.endf"), "-o", str(tmp_path / "out"),
               "--symbol", "C", "--mass", "12.0107", "--density", "2.26"])
    assert rc == 3
    err = capsys.readouterr().err
    assert "ENDFTSL converter failed" in err and "Traceback" not in err


def test_cli_malformed_yaml_config_exits_2(tmp_path, capsys):
    bad = tmp_path / "bad.yaml"
    bad.write_text("species: [unclosed\n", encoding="utf-8")
    rc = main(["--config", str(bad), "-o", str(tmp_path / "out")])
    assert rc == 2
    err = capsys.readouterr().err
    assert "invalid YAML" in err and "Traceback" not in err


def test_cli_config_missing_key_exits_2(tmp_path, capsys):
    import yaml as _yaml
    cfg = tmp_path / "c.yaml"
    cfg.write_text(_yaml.safe_dump({"material_id": "m", "temperature": 296.0,
                                    "density": 1.0}), encoding="utf-8")
    rc = main(["--config", str(cfg), "-o", str(tmp_path / "out")])
    assert rc == 2
    err = capsys.readouterr().err
    assert "species" in err and "Traceback" not in err
