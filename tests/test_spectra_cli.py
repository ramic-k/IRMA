"""irma CLI retrofit (P6) -- dispatch, flag->config sugar, guards, I/O.

Engine-free: no run_spectra execution. Pins (1) the legacy ENDF deck path stays
reachable byte-for-byte (dispatch + exit codes preserved), (2) the spectra flag
form builds the SAME SpectraConfig the config form loads, (3) geometry guards /
angle+override parsing, and (4) the csv/npz/json writers.
"""
import argparse
import json

import numpy as np
import pytest

from irma import cli
from irma.spectra import cli as scli
from irma.spectra.config import SpectraConfig
from irma.spectra.forward import SpectrumResult


# ---- backward-compat dispatch ----------------------------------------------
def test_version(capsys):
    assert cli.main(["--version"]) == 0
    assert "IRMA v" in capsys.readouterr().out


def test_no_args_prints_help():
    assert cli.main([]) == 0


def test_legacy_deck_missing_file():
    assert cli.main(["/no/such/deck.leapr", "out.endf"]) == 1


def test_legacy_deck_one_arg():
    assert cli.main(["onlyone"]) == 1


@pytest.mark.parametrize("prefix", [[], ["evaluate"]])
def test_deck_reaches_run_leapr(tmp_path, monkeypatch, prefix):
    """`irma deck out` and `irma evaluate deck out` both reach run_leapr."""
    deck = tmp_path / "graphite.leapr"
    deck.write_text("dummy")
    called = {}
    import irma.core.engine as eng
    monkeypatch.setattr(eng, "run_leapr",
                        lambda i, o: called.update(inp=i, out=o))
    assert cli.main(prefix + [str(deck), str(tmp_path / "out.endf")]) == 0
    assert called["inp"] == str(deck)


def test_deck_error_exit_code(tmp_path, monkeypatch):
    deck = tmp_path / "g.leapr"
    deck.write_text("x")
    import irma.core.engine as eng

    def _raise(i, o):
        raise eng.DeckError("bad card")
    monkeypatch.setattr(eng, "run_leapr", _raise)
    assert cli.main([str(deck), str(tmp_path / "o.endf")]) == 2


def test_spectra_routes_to_spectra_main(monkeypatch):
    import irma.spectra.cli as s
    monkeypatch.setattr(s, "main", lambda argv: 77)
    assert cli.main(["spectra", "vision", "--whatever"]) == 77


# ---- flag form -> SpectraConfig --------------------------------------------
def _vision_ns(extra=None):
    args = ["vision", "--phonopy-yaml", "g.yaml", "--temperature", "296",
            "--mesh", "40", "40", "40", "--de", "0.5", "--e-max", "250",
            "--dq", "0.05", "--inelastic-mode", "2",
            "--scatterer", "C,5.551,11.898,6.646,0.001", "-o", "v.csv"]
    return scli.build_parser().parse_args(args + (extra or []))


def test_flag_form_builds_expected_config():
    cfg = scli.config_from_args(_vision_ns())
    assert isinstance(cfg, SpectraConfig)
    assert cfg.instrument.geometry == "vision"
    assert cfg.instrument.e_fixed_meV == 3.5         # vision preset
    assert cfg.material.temperature_K == 296.0
    assert cfg.physics.inelastic_mode == 2
    assert cfg.material.scatterers[0].b_coh_fm == 6.646
    assert cfg.physics.elastic and cfg.physics.elastic_kind == "both"
    assert cfg.instrument.resolution_shape == "gaussian"
    flags = scli.config_from_args(_vision_ns(
        ["--elastic-kind", "incoherent", "--resolution-shape", "lorentzian",
         "--min-phonon-energy", "0.5"]))
    assert flags.physics.elastic_kind == "incoherent"
    assert flags.instrument.resolution_shape == "lorentzian"
    assert flags.physics.min_phonon_energy_meV == 0.5
    assert scli.config_from_args(_vision_ns(["--elastic", "off"])).physics.elastic is False


def test_flag_form_equals_config_form(tmp_path):
    """The sugar must build the identical SpectraConfig the config file loads."""
    from irma.spectra.config import dump, load
    cfg_flag = scli.config_from_args(_vision_ns())
    p = dump(cfg_flag, tmp_path / "cfg.yaml")
    assert load(p) == cfg_flag


def test_mode0_flag_form_builds_dos_config(tmp_path):
    """QA4 F2/F8: mode 0 is runnable from the flag form -- --inelastic-mode 0
    plus dos=/mult=/pos= scatterer tokens and --lattice must build a config
    that validates with NO phonopy_yaml."""
    from irma.spectra.config import validate
    dos = tmp_path / "c.txt"
    dos.write_text("0 0\n10 1\n20 2\n100 0\n")
    ns = scli.build_parser().parse_args(
        ["vision", "--inelastic-mode", "0", "--temperature", "300",
         "--scatterer", f"C,5.551,11.898,6.646,0.001,dos={dos},mult=2,"
                        "pos=0:0:0;0:0:0.5",
         "--lattice", "2.46,2.46,6.71,90,90,120", "-o", "o.csv"])
    cfg = scli.config_from_args(ns)
    assert cfg.physics.inelastic_mode == 0
    assert cfg.physics.dos_source == "file"
    assert cfg.material.phonopy_yaml is None
    assert cfg.material.lattice == [2.46, 2.46, 6.71, 90.0, 90.0, 120.0]
    s = cfg.material.scatterers[0]
    assert s.dos_file == str(dos) and s.multiplicity == 2
    assert s.positions == [[0.0, 0.0, 0.0], [0.0, 0.0, 0.5]]
    assert validate(cfg) is cfg


def test_unknown_scatterer_key_fails_loudly():
    """QA4 F19: a typo'd key=value token (dso=...) must be rejected naming the
    accepted keys, never silently dropped."""
    with pytest.raises(argparse.ArgumentTypeError, match="unknown key 'dso'"):
        scli.parse_scatterer("C,5.5,11.9,dso=c.txt")


def test_missing_config_file_is_clean_exit_2(tmp_path, capsys):
    """QA4 F6/F0: a nonexistent config path exits 2 with one clean message,
    not a traceback."""
    rc = scli.main(["run", str(tmp_path / "nope.yaml"), "-o", "o.csv"])
    assert rc == 2
    err = capsys.readouterr().err
    assert "No such file" in err and "Traceback" not in err


def test_malformed_yaml_is_clean_exit_2(tmp_path, capsys):
    bad = tmp_path / "bad.yaml"
    bad.write_text("material: [unclosed")
    rc = scli.main(["run", str(bad), "-o", "o.csv"])
    assert rc == 2
    assert "could not be parsed" in capsys.readouterr().err


def test_provenance_handles_mode0_without_yaml(tmp_path):
    """QA4 F0: _provenance must not crash when phonopy_yaml is None (the
    mode-0 DOS-files case the GUI Run button funnels through)."""
    from irma.spectra.config import SpectraConfig
    cfg = SpectraConfig.from_dict({
        "material": {"temperature_K": 300.0,
                     "scatterers": [{"symbol": "H", "awr": 1.0,
                                     "sigma_bound_b": 80.0,
                                     "dos_file": "h.txt"}]},
        "physics": {"inelastic_mode": 0, "elastic": False}})
    line = scli._provenance(cfg, "out.csv")
    assert "dos=[h.txt]" in line and "None" not in line


# ---- geometry guards (structural via argparse) -----------------------------
def test_direct_rejects_ef():
    with pytest.raises(SystemExit):
        scli.build_parser().parse_args(
            ["direct", "--phonopy-yaml", "x", "--ei", "250", "--ef", "3.5",
             "--angles", "30,90", "-o", "o.csv"])


def test_vision_rejects_ei():
    with pytest.raises(SystemExit):
        scli.build_parser().parse_args(
            ["vision", "--phonopy-yaml", "x", "--ei", "250", "-o", "o.csv"])


def test_direct_builds_with_ei():
    ns = scli.build_parser().parse_args(
        ["direct", "--phonopy-yaml", "x", "--ei", "250", "--angles", "30,90",
         "--scatterer", "Si,2.0,27.8", "--e-max", "200", "-o", "o.csv",
         "--kinematic-factor"])
    cfg = scli.config_from_args(ns)
    assert cfg.instrument.geometry == "direct"
    assert cfg.instrument.e_fixed_meV == 250.0
    assert cfg.physics.kinematic_kf_ki is True


# ---- value parsers ----------------------------------------------------------
def test_parse_angles_comma_and_range():
    assert scli.parse_angles("30,60,90") == [30.0, 60.0, 90.0]
    r = scli.parse_angles("5:135:5")
    assert r[0] == 5.0 and r[-1] == 135.0 and len(r) == 27


def test_parse_scatterer_minimal_and_full():
    assert scli.parse_scatterer("C,5.5,11.9") == {
        "symbol": "C", "sigma_bound_b": 5.5, "awr": 11.9}
    full = scli.parse_scatterer("O,4.2,15.9,5.8,0.0008")
    assert full["b_coh_fm"] == 5.8 and full["sigma_inc_b"] == 0.0008


def test_parse_scatterer_too_few():
    with pytest.raises(Exception):
        scli.parse_scatterer("C,5.5")


# ---- --set overrides --------------------------------------------------------
def test_apply_overrides_dotted_and_coercion():
    d = {"material": {"temperature_K": 296.0}, "physics": {}}
    scli.apply_overrides(d, ["material.temperature_K=500", "physics.elastic=false",
                             "physics.jobs=8", "grid.q_max_invA=null",
                             "physics.kinematic_kf_ki=on", "instrument.map_mask=Off"])
    assert d["material"]["temperature_K"] == 500
    assert d["physics"]["elastic"] is False
    assert d["physics"]["kinematic_kf_ki"] is True
    assert d["instrument"]["map_mask"] is False
    assert d["physics"]["jobs"] == 8
    assert d["grid"]["q_max_invA"] is None


# ---- map-config honoring (audit #12 #13 + sweep-1) -------------------------
class _Ins:
    output_mode = "map"
    map_mask = True
    export_components = False


class _MapCfg:
    instrument = _Ins()


def test_run_dispatches_to_map_when_output_mode_map(tmp_path, monkeypatch):
    """A `run` config with output_mode='map' must produce a 2-D map, not cuts."""
    cfg = _MapCfg()
    seen = {}

    def fake_run_map(c, **kw):
        seen["cfg"] = c
        return "SM"

    monkeypatch.setattr(scli, "_load_cfg", lambda ns: cfg)
    monkeypatch.setattr(scli, "_provenance", lambda c, o: "")
    monkeypatch.setattr(scli, "run_map", fake_run_map)
    monkeypatch.setattr(scli, "write_map",
                        lambda sm, out, masked=False: (seen.update(sm=sm, masked=masked), out)[1])
    rc = scli.main(["run", str(tmp_path / "cfg.yaml"), "-o", str(tmp_path / "m.npz")])
    assert rc == 0
    assert seen["cfg"] is cfg and seen["sm"] == "SM"
    assert seen["masked"] is True                 # honored cfg.instrument.map_mask


def test_map_subcommand_mask_defaults_to_config(tmp_path, monkeypatch):
    """`map` without --mask/--no-mask follows the config's instrument.map_mask."""
    cfg = _MapCfg()
    cfg.instrument.map_mask = False
    seen = {}
    monkeypatch.setattr(scli, "_load_cfg", lambda ns: cfg)
    monkeypatch.setattr(scli, "_provenance", lambda c, o: "")
    monkeypatch.setattr(scli, "run_map", lambda c, **kw: "SM")
    monkeypatch.setattr(scli, "write_map",
                        lambda sm, out, masked=False: (seen.update(masked=masked), out)[1])
    rc = scli.main(["map", str(tmp_path / "cfg.yaml"), "-o", str(tmp_path / "m.npz")])
    assert rc == 0 and seen["masked"] is False     # config drove it, no CLI flag


# ---- output writers (per-angle) --------------------------------------------
def _result(n_angles=2):
    E = np.linspace(0, 200, 11)
    I = np.abs(np.sin(E / 20.0))
    Ipa = np.stack([(k + 1) * I for k in range(n_angles)])      # (na, nE)
    Epa = 0.5 * Ipa
    return SpectrumResult(
        E=E, Q=np.ones((n_angles, 11)), I_inelastic=I, I_elastic=0.5 * I,
        I_total=1.5 * I, geometry="indirect",
        angles_deg=[45.0, 135.0][:n_angles],
        I_inelastic_per_angle=Ipa, I_elastic_per_angle=Epa,
        metadata={"inelastic_mode": 2, "elastic": True,
                  "n_bragg_edges": 42, "engine_metadata": {}})


def _csv_header(path):
    return [ln for ln in path.read_text().splitlines() if not ln.startswith("#")][0]


def _result_with_qcuts():
    r = _result(2)
    E = r.E
    q = np.abs(np.cos(E / 15.0))
    r.q_cut_values = [2.0, 5.0]
    r.I_inelastic_per_q = np.stack([q, 1.5 * q])
    r.I_elastic_per_q = np.stack([0.1 * q, 0.1 * q])
    return r


@pytest.mark.parametrize("ext", [".csv", ".npz", ".json"])
def test_write_spectrum_per_angle_and_q_cuts(tmp_path, ext):
    out = tmp_path / f"spec{ext}"
    scli.write_spectrum(_result_with_qcuts(), out)
    if ext == ".npz":
        d = np.load(out)
        assert d["I_total_per_angle"].shape == (2, 11)
        assert list(d["angles_deg"]) == [45.0, 135.0]
        assert list(d["q_cuts"]) == [2.0, 5.0]
        assert d["I_total_per_q"].shape == (2, 11)
    elif ext == ".json":
        d = json.loads(out.read_text())
        assert len(d["E_meV"]) == 11 and len(d["I_total_per_angle"]) == 2
        assert d["angles_deg"] == [45.0, 135.0]
        assert d["q_cuts"] == [2.0, 5.0] and len(d["I_total_per_q"]) == 2
    else:
        assert out.read_text().startswith("# IRMA spectrum")
        header = _csv_header(out)
        # one (total, inelastic, elastic) triple per bank and per Q cut
        assert "total@45deg" in header and "total@135deg" in header
        assert "total@Q=2" in header and "total@Q=5" in header


def test_write_spectrum_single_angle_csv(tmp_path):
    out = tmp_path / "spec.csv"
    scli.write_spectrum(_result(1), out)
    assert _csv_header(out) == "E_meV,total@45deg,inelastic@45deg,elastic@45deg"


def _result_q_only():
    """A constant-Q-ONLY run (cut_by='q'): no detector-angle spectra."""
    E = np.linspace(0, 200, 11)
    q = np.abs(np.cos(E / 15.0))
    return SpectrumResult(
        E=E, Q=np.zeros(11), I_inelastic=np.zeros(11), I_elastic=np.zeros(11),
        I_total=np.zeros(11), geometry="direct", angles_deg=[],
        I_inelastic_per_angle=np.empty((0, 11)), I_elastic_per_angle=np.empty((0, 11)),
        q_cut_values=[2.0, 5.0], I_inelastic_per_q=np.stack([q, 1.5 * q]),
        I_elastic_per_q=np.stack([0.1 * q, 0.1 * q]),
        metadata={"inelastic_mode": 2, "elastic": True, "n_bragg_edges": 0,
                  "engine_metadata": {}})


@pytest.mark.parametrize("ext", [".csv", ".npz", ".json"])
def test_write_spectrum_totals_only_and_q_only(tmp_path, ext):
    """components=False (the config/GUI default) writes only each cut's total;
    a cut_by='q' result writes no per-angle columns or arrays."""
    lean, q_only = tmp_path / f"lean{ext}", tmp_path / f"q{ext}"
    scli.write_spectrum(_result_with_qcuts(), lean, components=False)
    scli.write_spectrum(_result_q_only(), q_only)
    if ext == ".csv":
        header = _csv_header(lean)
        assert "inelastic@" not in header and "elastic@" not in header
        assert "total@45deg" in header and "total@Q=2" in header
        header = _csv_header(q_only)
        assert "deg" not in header
        assert "total@Q=2" in header and "total@Q=5" in header
        return
    if ext == ".npz":
        d, q = np.load(lean), np.load(q_only)
        keys, q_keys = d.files, q.files
    else:
        d, q = json.loads(lean.read_text()), json.loads(q_only.read_text())
        keys, q_keys = d, q
    assert "I_total_per_angle" in keys and "I_inelastic_per_angle" not in keys
    assert "I_total_per_q" in keys and "I_inelastic_per_q" not in keys
    assert "I_total_per_angle" not in q_keys
    assert len(q["I_total_per_q"]) == 2


def test_spectra_main_fails_fast_on_directory_output(tmp_path, capsys):
    """The spectra CLI preflights the output path before any compute: a
    directory target returns exit 2 with a clear message."""
    (tmp_path / "g.yaml").write_text("{}\n")
    rc = scli.main(["vision", "--phonopy-yaml", str(tmp_path / "g.yaml"),
                    "--scatterer", "C,5.551,11.898", "-o", str(tmp_path)])
    assert rc == 2
    assert "directory, not a file" in capsys.readouterr().err


def test_flag_defaults_are_the_config_defaults():
    """Unset flags keep the SpectraConfig defaults (mode 2, 10000/1000
    directions, ...); the Python entry points default to mode 2 too, and the
    map subcommand defers its Q grid to the config."""
    import inspect

    from irma.spectra.config import MaterialConfig, SpectraConfig
    from irma.spectra.forward import compute_spectrum, compute_sqe_map

    cfg = scli.config_from_args(scli.build_parser().parse_args(["vision", "-o", "v.csv"]))
    assert cfg == SpectraConfig(material=MaterialConfig())
    for fn in (compute_spectrum, compute_sqe_map):
        assert inspect.signature(fn).parameters["inelastic_mode"].default == 2
    mns = scli.build_parser().parse_args(["map", "c.yaml", "-o", "m.png"])
    assert (mns.q_min, mns.dq_map, mns.q_max) == (0.0, None, None)
