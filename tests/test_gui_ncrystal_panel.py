"""irma.gui NCrystalPanel -- widget <-> NCrystalExportConfig assembly.

Requires tkinter (skips without it). Uses a withdrawn root, so no display is
shown. Pins: (1) the panel constructs headlessly; (2) build_config() turns the
populated fields into a valid NCrystalExportConfig (gain_side / multiphonon
order / mode mappings included); (3) invalid input surfaces a clear error; (4)
"Export NCrystal data" drives the runner with the right argv WITHOUT running the
engine (the runner call is monkeypatched); (5) "Open Config..." reads a config
back through the exporter's OWN loader, round-tripping both grid modes, and a
file it rejects mutates nothing. The third top-level tab is asserted in
test_gui_ns_panel.test_app_has_two_top_tabs_and_endf_intact.
"""
import pytest

# importorskip FIRST -- a hard `import tkinter` at module top would error at
# COLLECTION time on a Python built without _tkinter, aborting the whole pytest
# run instead of skipping cleanly.
tk = pytest.importorskip("tkinter")
from irma.gui.runner import ComputationRunner               # noqa: E402
from irma.ncrystal.config import NCrystalExportConfig       # noqa: E402
from irma.spectra.config import SpectraConfigError          # noqa: E402


@pytest.fixture
def root():
    try:
        r = tk.Tk()
    except tk.TclError:
        pytest.skip("no display for tkinter")
    r.withdraw()
    yield r
    r.destroy()


@pytest.fixture
def make_panel(root):
    """Factory for fresh panels.

    The Open Config... tests load into a SECOND, untouched form, so what they
    assert is 'the file filled this in', not 'the values were already there'.
    """
    def _make():
        from irma.gui.ncrystal_panel import NCrystalPanel
        return NCrystalPanel(tk.Frame(root), runner=ComputationRunner())
    return _make


@pytest.fixture
def panel(make_panel):
    return make_panel()


def _declare(panel, symbol="C"):
    """Declare the material the panel deliberately ships BLANK.

    The scatterer table is material identity, so it starts as one empty row;
    typing the symbol fills the nuclear constants from the built-in table.
    """
    r = panel.element_table.rows[0]
    r["_var"]["symbol"].set(symbol)
    panel.element_table.autofill_row(r)
    return r


# ---- construction ----------------------------------------------------------
def test_fresh_panel_ships_no_material_identity(panel):
    """A fresh panel asserts nothing about the user's material: the scatterer
    row is empty (the old prefilled natural-carbon row let another material's
    constants ride into a pack unnoticed)."""
    row = panel.element_table.get_rows()[0]
    assert all(row[k] == "" for k in
               ("symbol", "sigma_bound_b", "awr", "b_coh_fm", "sigma_inc_b"))
    # ...while the export/methodology defaults stay prefilled
    assert panel.mesh.get() == "40 40 40"
    assert panel.temperature.get() == "296"
    assert panel.num_directions.get() == "10000"
    assert panel.multiphonon_num_directions.get() == "1000"
    assert panel.multiphonon_max_order.get() == "auto"
    assert panel.material_id.get() == "material"


# ---- build_config: defaults + populated fields -----------------------------
def test_defaults_build_a_valid_config(panel):
    """A declared carbon row + material_id builds a valid config."""
    panel.phonopy_yaml.set("graphite/phonopy.yaml")
    panel.material_id.set("graphite")
    _declare(panel)
    cfg = panel.build_config()
    assert isinstance(cfg, NCrystalExportConfig)
    assert cfg.material_id == "graphite"
    assert cfg.inelastic_mode == 2                       # default mode
    assert cfg.num_directions == 10000
    assert cfg.multiphonon_num_directions == 1000
    assert cfg.multiphonon_max_order == "auto"           # default
    assert cfg.gain_side == "scaled_sym"                 # default
    assert cfg.elastic is True
    assert cfg.jobs is None            # blank -> unset (see the jobs test)
    m = cfg.material
    assert m.phonopy_yaml == "graphite/phonopy.yaml"
    assert m.mesh == [40, 40, 40] and m.temperature_K == 296.0
    # the symbol autofill comes from the built-in nuclear table, not literals
    from irma.core.nuclear_data import lookup
    nat_c = lookup("C")
    s = m.scatterers[0]
    assert s.symbol == "C"
    assert s.sigma_bound_b == pytest.approx(nat_c.sigma_bound_b, rel=1e-5)
    assert s.awr == pytest.approx(nat_c.awr, rel=1e-5)
    assert s.b_coh_fm == pytest.approx(nat_c.b_coh_fm, rel=1e-5)
    assert s.sigma_inc_b == pytest.approx(nat_c.sigma_inc_b, rel=1e-5)
    # the shared grid form defaults to the converged automatic grid, with
    # freq_max unset (auto-estimated from phonopy)
    assert cfg.grid_mode == "auto"
    assert cfg.alpha_grid is None and cfg.beta_grid is None
    assert cfg.freq_max_eV is None
    assert cfg.n_phonon == 300 and cfg.alpha_dq_invA == 0.05
    assert cfg.alpha_qcut_invA == 12.0 and cfg.n_lower == 15
    # the cached {material, export} mapping the bake path serializes
    # round-trips back to an equal config
    assert "n_phonon" in panel._last_built_dict["export"]
    assert "alpha_grid" not in panel._last_built_dict["export"]
    assert NCrystalExportConfig.from_dict(panel._last_built_dict) == cfg


def test_explicit_grid_flows_into_config(panel):
    """Entering explicit alpha/beta lists -> cfg.grid_mode == 'explicit' with the
    parsed lists, and the automatic Q/E keys are absent from the export dict."""
    panel.phonopy_yaml.set("graphite/phonopy.yaml")
    panel.material_id.set("graphite")
    _declare(panel)
    panel.grid_form.grid_mode.set("explicit (alpha/beta)")
    panel.grid_form.alpha_grid.set("0.1 0.5, 1.0 2.0")
    panel.grid_form.beta_grid.set("0.0 0.5 1.0")
    cfg = panel.build_config()
    assert cfg.grid_mode == "explicit"
    assert cfg.alpha_grid == [0.1, 0.5, 1.0, 2.0]
    assert cfg.beta_grid == [0.0, 0.5, 1.0]
    export = panel._last_built_dict["export"]
    assert export["alpha_grid"] == [0.1, 0.5, 1.0, 2.0]
    assert "n_phonon" not in export and "alpha_dq_invA" not in export


def test_populated_fields_map_onto_config(panel):
    """Every exposed field flows into the config with the right type/mapping
    (a round trip compares two build_config outputs, so it cannot catch a
    dropped or mis-mapped field)."""
    _fill(panel)
    cfg = panel.build_config()
    assert cfg.material_id == "beo"
    assert cfg.inelastic_mode == 1
    assert cfg.num_directions == 4000
    assert cfg.multiphonon_num_directions == 200
    assert cfg.multiphonon_max_order == 120              # int path
    assert cfg.jobs == 8
    assert cfg.incoherent_elastic_mode == "directional"
    m = cfg.material
    assert m.phonopy_yaml == "beo/phonopy.yaml"
    assert m.born == "BORN" and m.force_constants == "FORCE_CONSTANTS"
    assert m.force_sets == "FORCE_SETS"
    assert m.mesh == [30, 30, 24] and m.temperature_K == 500.0
    assert [s.symbol for s in m.scatterers] == ["Be", "O"]


def test_blank_elastic_constants_rejected(panel):
    """Blank b_coh_fm / sigma_inc_b fail loudly at build time: the GUI export
    always bakes the elastic block (the control was deliberately removed), and
    the old silent None -> 0.0 substitution deleted the coherent-Bragg /
    incoherent-elastic channel from the pack (pre-release review S2)."""
    panel.material_id.set("x")
    panel.phonopy_yaml.set("x.yaml")  # required by the export config
    panel.element_table.set_rows([{"symbol": "C", "sigma_bound_b": "5.551",
                                   "awr": "11.898"}])
    with pytest.raises(SpectraConfigError, match="'C' is missing b_coh_fm"):
        panel.build_config()


# ---- invalid input surfaces an error ---------------------------------------
def test_blank_material_id_rejected(panel):
    """material_id is required -- a blank one raises (SpectraConfigError)."""
    panel.phonopy_yaml.set("g.yaml")
    panel.material_id.set("")
    with pytest.raises(SpectraConfigError):
        panel.build_config()


def test_bad_numeric_entry_names_the_field(panel):
    """A typo'd numeric entry fails with the field's name, not a bare
    'could not convert string to float'."""
    panel.material_id.set("g")
    panel.mesh.set("40 xx 40")
    with pytest.raises(ValueError, match=r"mesh"):
        panel.build_config()
    panel.mesh.set("40 40 40")
    panel.temperature.set("")
    with pytest.raises(ValueError, match="temperature"):
        panel.build_config()
    panel.temperature.set("296")
    panel.multiphonon_max_order.set("seven")
    with pytest.raises(ValueError, match="multiphonon order"):
        panel.build_config()


# ---- "Export NCrystal data" drives the runner WITHOUT running the engine ----
def test_export_invokes_runner_with_cli_argv(panel, monkeypatch, tmp_path):
    """Export serializes the config to YAML and calls the runner with the
    `python -m irma.ncrystal <cfg> -o <outdir>` argv -- the engine never runs
    (the runner call is captured)."""
    # The export path serializes/round-trips via PyYAML, an OPTIONAL dependency
    # (the 'spectra' extra). Skip on the core/bare install that omits it; the
    # full pytest job installs .[phonopy,spectra] and exercises this.
    pytest.importorskip("yaml")
    panel.phonopy_yaml.set("g.yaml")
    panel.material_id.set("graphite")
    _declare(panel)
    outdir = tmp_path / "out"
    panel.outdir.set(str(outdir))

    seen = {}

    def fake_run_command(argv, success_msg="", on_log=None, on_done=None,
                         error_label="Input error", **kw):
        seen.update(argv=argv, msg=success_msg, label=error_label)

    monkeypatch.setattr(panel.runner, "run_command", fake_run_command)
    panel._export()

    assert seen["argv"][1:4] == ["-u", "-m", "irma.ncrystal"]
    assert seen["argv"][-2:] == ["-o", str(outdir)]
    cfg_path = seen["argv"][4]
    assert cfg_path.endswith(".yaml")
    assert "NCrystal export config error" == seen["label"]
    # the serialized config reloads into an equal config (faithful round-trip)
    reloaded = NCrystalExportConfig.from_yaml(cfg_path)
    assert reloaded.material_id == "graphite"
    assert reloaded.material.scatterers[0].symbol == "C"
    import os
    os.unlink(cfg_path)


@pytest.mark.parametrize("material_id, outdir", [("g", ""), ("", "out")])
def test_export_error_shows_a_dialog_and_does_not_run(panel, monkeypatch,
                                                     material_id, outdir):
    """No output directory, or a config error (blank material_id) -> a dialog
    error and the runner is never called."""
    panel.phonopy_yaml.set("g.yaml")
    panel.material_id.set(material_id)
    panel.outdir.set(outdir)
    called = {"run": False, "error": None}
    monkeypatch.setattr(panel.runner, "run_command",
                        lambda *a, **k: called.__setitem__("run", True))
    import irma.gui.ncrystal_panel as mod
    monkeypatch.setattr(mod.messagebox, "showerror",
                        lambda *a, **k: called.__setitem__("error", a))
    panel._export()
    assert called["run"] is False and called["error"] is not None


def test_element_table_add_remove(panel):
    t = panel.element_table
    n0 = len(t.rows)
    t.add_row({"symbol": "O", "sigma_bound_b": "4.232", "awr": "15.858"})
    assert len(t.rows) == n0 + 1
    assert t.get_rows()[-1]["symbol"] == "O"
    t._remove(t.rows[-1])
    assert len(t.rows) == n0


# ---- "Open Config..." reads a config back into the form --------------------
BEO_ROWS = [
    {"symbol": "Be", "sigma_bound_b": "7.63", "awr": "8.935",
     "b_coh_fm": "7.79", "sigma_inc_b": "0.0018"},
    {"symbol": "O", "sigma_bound_b": "4.232", "awr": "15.858",
     "b_coh_fm": "5.803", "sigma_inc_b": "0.0008"},
]


def _fill(panel):
    """Populate every widget build_config reads, away from its default."""
    panel.phonopy_yaml.set("beo/phonopy.yaml")
    panel.born.set("BORN")
    panel.force_constants.set("FORCE_CONSTANTS")
    panel.force_sets.set("FORCE_SETS")
    panel.mesh.set("30 30 24")
    panel.temperature.set("500")
    panel.material_id.set("beo")
    panel.inelastic_mode.set(panel._INELASTIC_BY_INT[1])
    panel.num_directions.set("4000")
    panel.multiphonon_num_directions.set("200")
    panel.multiphonon_max_order.set("120")
    panel.jobs.set("8")
    panel.incoherent_elastic_mode.set("directional")
    panel.element_table.set_rows(BEO_ROWS)


def _write(panel, path):
    """Build the config and write the YAML the export CLI would run."""
    cfg = panel.build_config()
    panel._write_config_yaml(panel._last_built_dict, str(path))
    return cfg


def _open(panel, path, monkeypatch):
    """Click Open Config... with the file dialog answering ``path``."""
    import irma.gui.ncrystal_panel as mod
    monkeypatch.setattr(mod.filedialog, "askopenfilename",
                        lambda *a, **k: str(path))
    panel._open_config()


_GRID_WIDGETS = ("freq_max", "n_lower", "n_phonon", "n_upper", "beta_max",
                 "alpha_dq", "alpha_qcut", "alpha_nlog", "alpha_grid",
                 "beta_grid")


def _snapshot(panel):
    """Every value Open Config... could touch, as plain strings."""
    return {
        "material": [panel.phonopy_yaml.get(), panel.born.get(),
                     panel.force_constants.get(), panel.force_sets.get(),
                     panel.mesh.get(), panel.temperature.get()],
        "export": [panel.material_id.get(), panel.inelastic_mode.get(),
                   panel.num_directions.get(),
                   panel.multiphonon_num_directions.get(),
                   panel.multiphonon_max_order.get(), panel.jobs.get(),
                   panel.incoherent_elastic_mode.get()],
        "grid_mode": panel.grid_form.mode(),
        "grid": {k: getattr(panel.grid_form, k).get() for k in _GRID_WIDGETS},
        "rows": panel.element_table.get_rows(),
        "carried": dict(panel._carried_export),
    }


def test_round_trip_automatic_grid(make_panel, tmp_path, monkeypatch):
    """Fill the form, export the dict as YAML, load it into a FRESH panel:
    that panel's build_config() equals the original config, automatic grid
    knobs (and a pinned freq_max) included."""
    pytest.importorskip("yaml")
    src = make_panel()
    _fill(src)
    src.grid_form.freq_max.set("0.18")
    src.grid_form.alpha_dq.set("0.02")
    src.grid_form.n_upper.set("40")
    cfg = _write(src, tmp_path / "auto.yaml")
    assert cfg.grid_mode == "auto"

    dst = make_panel()
    _open(dst, tmp_path / "auto.yaml", monkeypatch)
    assert dst.build_config() == cfg
    # the form SHOWS what it loaded, not just a config that happens to match
    assert dst.grid_form.mode() == "auto"
    assert float(dst.grid_form.freq_max.get()) == 0.18
    assert float(dst.grid_form.alpha_dq.get()) == 0.02
    assert dst.grid_form.n_upper.get() == "40"


def test_round_trip_explicit_grid(make_panel, tmp_path, monkeypatch):
    """Same round trip in explicit mode: the MODE survives, and the automatic
    group is left at its shipped defaults rather than half-filled."""
    pytest.importorskip("yaml")
    src = make_panel()
    _fill(src)
    src.grid_form.grid_mode.set("explicit (alpha/beta)")
    src.grid_form.alpha_grid.set("0.1 0.5, 1.0 2.0")
    src.grid_form.beta_grid.set("0.0 0.5 1.0")
    cfg = _write(src, tmp_path / "explicit.yaml")
    assert cfg.grid_mode == "explicit"

    dst = make_panel()
    _open(dst, tmp_path / "explicit.yaml", monkeypatch)
    assert dst.grid_form.mode() == "explicit"
    assert dst.build_config() == cfg
    assert dst.build_config().alpha_grid == [0.1, 0.5, 1.0, 2.0]
    from irma.gui.grid_form import AUTO_GRID_ENTRY_DEFAULTS
    for name, default in AUTO_GRID_ENTRY_DEFAULTS.items():
        assert getattr(dst.grid_form, name).get() == default
    assert dst.grid_form.freq_max.get() == ""


def test_loading_an_auto_config_clears_a_stale_explicit_grid(
        make_panel, tmp_path, monkeypatch):
    """The inverse direction of the mode switch: explicit lists left on screen
    must not survive into an automatic-grid config (export_fields would ignore
    them, but the form would be lying about what it will run)."""
    pytest.importorskip("yaml")
    src = make_panel()
    _fill(src)
    cfg = _write(src, tmp_path / "auto.yaml")

    dst = make_panel()
    dst.grid_form.grid_mode.set("explicit (alpha/beta)")
    dst.grid_form.alpha_grid.set("9.0 9.5")
    dst.grid_form.beta_grid.set("0.0 1.0")
    _open(dst, tmp_path / "auto.yaml", monkeypatch)
    assert dst.grid_form.mode() == "auto"
    assert dst.grid_form.alpha_grid.get() == ""
    assert dst.grid_form.beta_grid.get() == ""
    assert dst.build_config() == cfg


def test_loaded_rows_carry_the_autofill_provenance(
        make_panel, tmp_path, monkeypatch):
    """A rebuilt row must behave like one the user filled by typing a symbol.

    Constants equal to the built-in table's are machine-filled, so retyping
    the symbol refreshes them for the new element instead of silently keeping
    the old one's; a constant the config sets to something else is user data
    and survives the same edit.
    """
    pytest.importorskip("yaml")
    from irma.core.nuclear_data import lookup

    src = make_panel()
    src.phonopy_yaml.set("g.yaml")
    src.material_id.set("graphite")
    _declare(src)                                # natural-carbon constants
    src.element_table.rows[0]["_var"]["sigma_inc_b"].set("0.5")   # hand-typed
    _write(src, tmp_path / "c.yaml")

    dst = make_panel()
    _open(dst, tmp_path / "c.yaml", monkeypatch)
    row = dst.element_table.rows[0]
    assert row["_autofill"]["symbol"] == "C"
    assert float(row["_var"]["sigma_inc_b"].get()) == 0.5

    row["_var"]["symbol"].set("Be")
    dst.element_table.autofill_row(row)
    be = lookup("Be")
    assert float(row["_var"]["b_coh_fm"].get()) == pytest.approx(be.b_coh_fm,
                                                                rel=1e-5)
    assert float(row["_var"]["awr"].get()) == pytest.approx(be.awr, rel=1e-5)
    assert float(row["_var"]["sigma_inc_b"].get()) == 0.5     # user data stays


def test_omitted_awr_loads_blank_and_stays_omitted(
        make_panel, tmp_path, monkeypatch):
    """A config that leaves awr out (derive it from the phonopy mass) must not
    acquire one from the autofill on the way in."""
    pytest.importorskip("yaml")
    import yaml

    path = tmp_path / "no_awr.yaml"
    path.write_text(yaml.safe_dump({
        "material": {"phonopy_yaml": "g.yaml", "mesh": [40, 40, 40],
                     "temperature_K": 296.0,
                     "scatterers": [{"symbol": "C", "sigma_bound_b": 5.551,
                                     "b_coh_fm": 6.646, "sigma_inc_b": 0.001}]},
        "export": {"material_id": "graphite"}}))
    dst = make_panel()
    _open(dst, path, monkeypatch)
    assert dst.element_table.get_rows()[0]["awr"] == ""
    assert dst.build_config().material.scatterers[0].awr is None


def test_open_emitted_ncrystal_yaml(make_panel, tmp_path, monkeypatch):
    """The config `irma mlip emit` writes loads, and re-exports UNCHANGED.

    The emitted file carries export settings this form has no control for
    (`coherent_partition_mode: auto`, `gain_side`, `elastic`); rebuilding from
    the widgets alone would quietly reset the partition mode to the dataclass
    default, which is a different partition on a single-group material.
    """
    pytest.importorskip("yaml")
    pytest.importorskip("ase")
    from ase.build import bulk
    from ase.io import write as ase_write

    from irma.mlip.bundle import Bundle
    from irma.mlip.emit import emit_ncrystal_yaml

    struct = tmp_path / "relaxed.extxyz"
    ase_write(str(struct), bulk("Al", "fcc", a=4.05, cubic=True))
    bundle = Bundle(path=str(tmp_path), structure=str(struct),
                    phonopy_yaml=str(tmp_path / "phonopy.yaml"), manifest={})
    emitted = emit_ncrystal_yaml(bundle, temperature_k=296.0,
                                 material_id="al",
                                 out_path=str(tmp_path / "ncrystal.yaml"),
                                 progress=lambda *a, **k: None)
    want = NCrystalExportConfig.from_yaml(emitted)

    dst = make_panel()
    _open(dst, emitted, monkeypatch)
    assert [r["symbol"] for r in dst.element_table.get_rows()] == ["Al"]
    assert dst.material_id.get() == "al"
    assert float(dst.temperature.get()) == 296.0
    assert dst.grid_form.mode() == "auto"
    assert dst._carried_export == {"coherent_partition_mode": "auto"}
    assert dst.build_config() == want


_REJECTED_YAML = (
    "material:\n  phonopy_yaml: g.yaml\n  mesh: [40, 40, 40]\n"
    "  temperature_K: 296.0\n"
    "  scatterers:\n  - {symbol: C, sigma_bound_b: 5.551, awr: 11.898}\n"
    "export: {material_id: graphite}\n")


@pytest.mark.parametrize("text, message", [
    ("material: [1, 2\n  export: {\n", ""),          # unparseable YAML
    (_REJECTED_YAML, "b_coh_fm"),                      # the exporter refuses it
])
def test_bad_config_file_reports_and_changes_nothing(panel, tmp_path,
                                                     monkeypatch, text,
                                                     message):
    """Unparseable YAML, or valid YAML the EXPORTER refuses (a scatterer with
    no b_coh_fm): an error dialog, and the form is exactly as it was."""
    pytest.importorskip("yaml")
    import irma.gui.ncrystal_panel as mod

    path = tmp_path / "config.yaml"
    path.write_text(text)
    before = _snapshot(panel)
    errors = []
    monkeypatch.setattr(mod.messagebox, "showerror",
                        lambda *a, **k: errors.append(a))
    _open(panel, path, monkeypatch)
    assert errors and message in str(errors[0])
    assert _snapshot(panel) == before
    assert len(panel.element_table.rows) == 1


def test_min_phonon_energy_round_trips_through_the_export_form(make_panel):
    from irma.ncrystal.config import NCrystalExportConfig
    d = {"material": {"phonopy_yaml": "g.yaml", "mesh": [4, 4, 4],
                      "temperature_K": 296.0,
                      "scatterers": [{"symbol": "C", "sigma_bound_b": 5.551, "awr": 11.898,
                                      "b_coh_fm": 6.646, "sigma_inc_b": 0.001}]},
         "export": {"material_id": "graphite", "inelastic_mode": 2,
                    "min_phonon_energy_meV": 0.5}}
    panel = make_panel()
    panel.load_config(NCrystalExportConfig.from_dict(d))
    assert panel.min_phonon_energy.get() == "0.5"
    assert panel.build_config().min_phonon_energy_meV == 0.5
    blank = make_panel()
    assert blank.min_phonon_energy.get() == ""       # a fresh form has no cutoff
    d["export"].pop("min_phonon_energy_meV")
    panel.load_config(NCrystalExportConfig.from_dict(d))
    assert panel.min_phonon_energy.get() == ""
    assert panel.build_config().min_phonon_energy_meV == 0.0
