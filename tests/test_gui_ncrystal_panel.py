"""irma.gui NCrystalPanel -- widget -> NCrystalExportConfig assembly.

Requires tkinter (skips without it). Uses a withdrawn root, so no display is
shown. Pins: (1) the panel constructs headlessly; (2) build_config() turns the
populated fields into a valid NCrystalExportConfig (gain_side / multiphonon
order / mode mappings included); (3) invalid input surfaces a clear error; (4)
"Export NCrystal data" drives the runner with the right argv WITHOUT running the
engine (the runner call is monkeypatched). The third top-level tab is asserted in
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
def panel(root):
    from irma.gui.ncrystal_panel import NCrystalPanel
    frame = tk.Frame(root)
    return NCrystalPanel(frame, runner=ComputationRunner())


# ---- construction ----------------------------------------------------------
def test_panel_constructs_headless(panel):
    """The panel builds against a withdrawn root (no display)."""
    assert panel.element_table.get_rows()[0]["symbol"] == "C"
    # only the nuclear columns are shown -- the export ignores DOS/positions
    from irma.gui.element_table import NUCLEAR
    assert panel.element_table._visible == list(NUCLEAR)


# ---- build_config: defaults + populated fields -----------------------------
def test_defaults_build_a_valid_config(panel):
    """The pre-filled defaults (carbon row + material_id) build a valid config."""
    panel.phonopy_yaml.set("graphite/phonopy.yaml")
    panel.material_id.set("graphite")
    cfg = panel.build_config()
    assert isinstance(cfg, NCrystalExportConfig)
    assert cfg.material_id == "graphite"
    assert cfg.inelastic_mode == 2                       # default mode
    assert cfg.num_directions == 10000
    assert cfg.multiphonon_num_directions == 1000
    assert cfg.multiphonon_max_order == "auto"           # default
    assert cfg.gain_side == "scaled_sym"                 # default
    assert cfg.elastic is True
    assert cfg.jobs is None                              # blank -> serial
    m = cfg.material
    assert m.phonopy_yaml == "graphite/phonopy.yaml"
    assert m.mesh == [40, 40, 40] and m.temperature_K == 296.0
    # the default row comes from the built-in nuclear table, not literals
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
    cfg = panel.build_config()
    assert cfg.grid_mode == "auto"
    assert cfg.alpha_grid is None and cfg.beta_grid is None
    assert cfg.freq_max_eV is None       # blank -> auto-estimate from phonopy
    assert cfg.n_phonon == 300 and cfg.alpha_dq_invA == 0.05
    assert cfg.alpha_qcut_invA == 12.0 and cfg.n_lower == 50
    # the merged grid knobs survive a YAML round-trip via the cached dict
    assert "n_phonon" in panel._last_built_dict["export"]
    assert "alpha_grid" not in panel._last_built_dict["export"]


def test_explicit_grid_flows_into_config(panel):
    """Entering explicit alpha/beta lists -> cfg.grid_mode == 'explicit' with the
    parsed lists, and the automatic Q/E keys are absent from the export dict."""
    panel.phonopy_yaml.set("graphite/phonopy.yaml")
    panel.material_id.set("graphite")
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
