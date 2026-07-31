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
from irma.gui.element_table import NUCLEAR                  # noqa: E402
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
def test_panel_constructs_headless(panel):
    """The panel builds against a withdrawn root (no display)."""
    assert len(panel.element_table.rows) == 1
    assert panel.element_table.get_rows()[0]["symbol"] == ""
    # only the nuclear columns are shown -- the export ignores DOS/positions
    from irma.gui.element_table import NUCLEAR
    assert panel.element_table._visible == list(NUCLEAR)


def test_fresh_panel_ships_no_material_identity(panel):
    """A fresh panel asserts nothing about the user's material: the scatterer
    row is empty (the old prefilled natural-carbon row let another material's
    constants ride into a pack unnoticed)."""
    row = panel.element_table.get_rows()[0]
    assert all(row[k] == "" for k in
               ("symbol", "sigma_bound_b", "awr", "b_coh_fm", "sigma_inc_b"))


def test_fresh_panel_keeps_methodology_defaults(panel):
    """...while the export/methodology defaults stay prefilled."""
    assert panel.mesh.get() == "40 40 40"
    assert panel.temperature.get() == "296"
    assert panel.num_directions.get() == "10000"
    assert panel.multiphonon_num_directions.get() == "1000"
    assert panel.multiphonon_max_order.get() == "auto"
    assert panel.material_id.get() == "material"


def test_scatterer_hint_names_the_way_to_fill_the_table():
    """The blank table reads as deliberately required, not merely empty."""
    from irma.gui.ncrystal_panel import IDENTITY_HINT_SCATTERERS
    assert "YOUR material" in IDENTITY_HINT_SCATTERERS
    assert "phonopy.yaml" in IDENTITY_HINT_SCATTERERS


def test_scatterer_help_describes_the_empty_starting_row():
    """The '?' popup must match what construction actually inserts: one
    empty row, not the natural-carbon row the panel used to prefill."""
    from irma.gui.ncrystal_panel import HELP
    assert "EMPTY row" in HELP["scatterers"]
    assert "default row is natural carbon" not in HELP["scatterers"]


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


def test_default_build_config_uses_auto_grid_with_config_defaults(panel):
    """The shared grid form defaults to the converged automatic grid; build_config
    flows the ENDF-style knobs through and reports grid_mode == 'auto' with
    freq_max_eV unset (auto-estimated from phonopy)."""
    panel.phonopy_yaml.set("graphite/phonopy.yaml")
    panel.material_id.set("graphite")
    _declare(panel)
    cfg = panel.build_config()
    assert cfg.grid_mode == "auto"
    assert cfg.alpha_grid is None and cfg.beta_grid is None
    assert cfg.freq_max_eV is None       # blank -> auto-estimate from phonopy
    assert cfg.n_phonon == 300 and cfg.alpha_dq_invA == 0.05
    assert cfg.alpha_qcut_invA == 12.0 and cfg.n_lower == 15
    # the merged grid knobs survive a YAML round-trip via the cached dict
    assert "n_phonon" in panel._last_built_dict["export"]
    assert "alpha_grid" not in panel._last_built_dict["export"]


def test_blank_jobs_resolves_to_every_core(panel, monkeypatch):
    """A blank jobs field means ALL CORES, not serial.

    The panel only records ``jobs=None``; the number that reaches the engine
    comes from the consumer in irma.ncrystal.build, so assert the RESOLVED
    count (the old test stopped at ``cfg.jobs is None`` and called it
    'serial', which is why the stale 'blank=serial' label and popup survived
    a shipped default change). An explicit value still caps it, which is the
    lever for a memory-limited machine.
    """
    from irma.ncrystal import build as ncbuild

    panel.phonopy_yaml.set("graphite/phonopy.yaml")
    panel.material_id.set("graphite")
    _declare(panel)

    monkeypatch.setattr(ncbuild.os, "cpu_count", lambda: 7)
    assert panel.build_config().jobs is None
    assert ncbuild.resolve_jobs(panel.build_config().jobs) == 7

    panel.jobs.set("3")
    cfg = panel.build_config()
    assert cfg.jobs == 3
    assert ncbuild.resolve_jobs(cfg.jobs) == 3

    # a machine that reports no core count still gets a usable worker count
    monkeypatch.setattr(ncbuild.os, "cpu_count", lambda: None)
    assert ncbuild.resolve_jobs(None) == 1


def test_jobs_label_and_help_state_the_shipped_default(panel):
    """Label and popup must say what a blank field actually does."""
    from irma.gui.ncrystal_panel import HELP

    assert "all cores" in panel.jobs.label.cget("text")
    assert "serial" not in panel.jobs.label.cget("text")
    assert "blank = every CPU core" in HELP["jobs"]
    assert "cap it" in HELP["jobs"]


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
    """Every exposed field flows into the config with the right type/mapping."""
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
    # two-species material
    panel.element_table.set_rows([
        {"symbol": "Be", "sigma_bound_b": "7.63", "awr": "8.935",
         "b_coh_fm": "7.79", "sigma_inc_b": "0.0018"},
        {"symbol": "O", "sigma_bound_b": "4.232", "awr": "15.858",
         "b_coh_fm": "5.803", "sigma_inc_b": "0.0008"}])
    cfg = panel.build_config()
    assert cfg.material_id == "beo"
    assert cfg.inelastic_mode == 1
    assert cfg.num_directions == 4000
    assert cfg.multiphonon_num_directions == 200
    assert cfg.multiphonon_max_order == 120              # int path
    assert cfg.gain_side == "scaled_sym"                 # default; GUI control removed
    assert cfg.elastic is True                           # default; GUI control removed
    assert cfg.jobs == 8
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


def test_build_config_caches_yaml_dict(panel):
    """build_config stashes the {material, export} mapping the bake path
    serializes, and it round-trips back through from_dict to an equal config."""
    panel.phonopy_yaml.set("g.yaml")
    panel.material_id.set("g")
    _declare(panel)
    cfg = panel.build_config()
    rebuilt = NCrystalExportConfig.from_dict(panel._last_built_dict)
    assert rebuilt.material_id == cfg.material_id
    assert rebuilt.gain_side == cfg.gain_side
    assert rebuilt.material.scatterers[0].symbol == "C"


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


def test_invalid_inelastic_mode_rejected_by_config(panel):
    """A leading-digit mode the config rejects surfaces its SpectraConfigError.
    (The dropdown only offers 1/2, but build_config must not silently coerce.)"""
    panel.material_id.set("g")
    panel.inelastic_mode.var.set("3 (bogus)")
    with pytest.raises(SpectraConfigError, match="inelastic_mode"):
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


def test_export_without_outdir_errors_and_does_not_run(panel, monkeypatch):
    """No output directory -> a dialog error and the runner is never called."""
    panel.phonopy_yaml.set("g.yaml")
    panel.material_id.set("g")
    panel.outdir.set("")
    called = {"run": False, "error": None}
    monkeypatch.setattr(panel.runner, "run_command",
                        lambda *a, **k: called.__setitem__("run", True))
    import irma.gui.ncrystal_panel as mod
    monkeypatch.setattr(mod.messagebox, "showerror",
                        lambda *a, **k: called.__setitem__("error", a))
    panel._export()
    assert called["run"] is False and called["error"] is not None


def test_export_with_bad_config_errors_and_does_not_run(panel, monkeypatch, tmp_path):
    """A config error (blank material_id) -> a dialog and no run."""
    panel.material_id.set("")
    panel.outdir.set(str(tmp_path))
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


def test_open_config_button_exists_next_to_export(panel):
    """The tab can round-trip its inputs, like the other two."""
    assert panel.open_btn.cget("text") == "Open Config..."


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
    assert dst.mesh.get() == "30 30 24"
    assert float(dst.temperature.get()) == 500.0
    assert dst.material_id.get() == "beo"
    assert dst.inelastic_mode.get() == dst._INELASTIC_BY_INT[1]
    assert dst.jobs.get() == "8"
    assert dst.incoherent_elastic_mode.get() == "directional"
    assert dst.born.get() == "BORN"
    assert dst.force_constants.get() == "FORCE_CONSTANTS"
    assert dst.force_sets.get() == "FORCE_SETS"


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


def test_polyatomic_config_rebuilds_one_row_per_species(
        make_panel, tmp_path, monkeypatch):
    """The scatterer table is rebuilt row by row from the config's species."""
    pytest.importorskip("yaml")
    src = make_panel()
    _fill(src)
    _write(src, tmp_path / "beo.yaml")

    dst = make_panel()
    _open(dst, tmp_path / "beo.yaml", monkeypatch)
    rows = dst.element_table.get_rows()
    assert len(rows) == 2                       # the empty starting row is gone
    assert [r["symbol"] for r in rows] == ["Be", "O"]
    for row, want in zip(rows, BEO_ROWS):
        for key in NUCLEAR[1:]:
            assert float(row[key]) == float(want[key]), key


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


def test_malformed_file_reports_and_changes_nothing(panel, tmp_path,
                                                    monkeypatch):
    """Unparseable YAML: an error dialog, and the form is exactly as it was."""
    import irma.gui.ncrystal_panel as mod

    bad = tmp_path / "broken.yaml"
    bad.write_text("material: [1, 2\n  export: {\n")
    before = _snapshot(panel)
    errors = []
    monkeypatch.setattr(mod.messagebox, "showerror",
                        lambda *a, **k: errors.append(a))
    _open(panel, bad, monkeypatch)
    assert errors
    assert _snapshot(panel) == before
    assert len(panel.element_table.rows) == 1


def test_config_the_exporter_rejects_reports_and_changes_nothing(
        panel, tmp_path, monkeypatch):
    """Valid YAML the EXPORTER refuses (a scatterer with no b_coh_fm) is
    refused here too, by that same loader -- and mutates nothing."""
    pytest.importorskip("yaml")
    import yaml

    import irma.gui.ncrystal_panel as mod

    path = tmp_path / "rejected.yaml"
    path.write_text(yaml.safe_dump({
        "material": {"phonopy_yaml": "g.yaml", "mesh": [40, 40, 40],
                     "temperature_K": 296.0,
                     "scatterers": [{"symbol": "C", "sigma_bound_b": 5.551,
                                     "awr": 11.898}]},
        "export": {"material_id": "graphite"}}))
    before = _snapshot(panel)
    errors = []
    monkeypatch.setattr(mod.messagebox, "showerror",
                        lambda *a, **k: errors.append(a))
    _open(panel, path, monkeypatch)
    assert errors and "b_coh_fm" in str(errors[0])
    assert _snapshot(panel) == before
    assert len(panel.element_table.rows) == 1


def test_cancelled_dialog_keeps_the_empty_starting_row(panel, monkeypatch):
    """Dismissing the dialog leaves the deliberately EMPTY scatterer row (and
    everything else) untouched."""
    import irma.gui.ncrystal_panel as mod

    before = _snapshot(panel)
    monkeypatch.setattr(mod.filedialog, "askopenfilename", lambda *a, **k: "")
    panel._open_config()
    assert len(panel.element_table.rows) == 1
    row = panel.element_table.get_rows()[0]
    assert all(row[k] == "" for k in NUCLEAR)
    assert _snapshot(panel) == before
