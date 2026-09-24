"""irma.gui NSPanel -- widget<->SpectraConfig round-trips.

Requires tkinter (skips without it). Uses a withdrawn root, so no display is
shown. Pins: (1) the app's top-level tabs and the five parts of the ENDF form;
(2) the NS panel's build_config/load_config are exact inverses for every
geometry.
"""
import pytest

# importorskip first: on a Python built without _tkinter a plain import would
# fail at collection and stop the whole run instead of skipping this file.
tk = pytest.importorskip("tkinter")
from irma.spectra.config import SpectraConfig, SpectraConfigError  # noqa: E402
from irma.gui.runner import ComputationRunner               # noqa: E402


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
    from irma.gui.ns_panel import NSPanel
    frame = tk.Frame(root)
    return NSPanel(frame, runner=ComputationRunner())


def _declare(panel, symbol="C", **cells):
    """Declare the material the panel deliberately ships BLANK.

    The element table starts as one empty row (the scatterers are material
    identity, which IRMA cannot know); typing the symbol is what fills the
    nuclear constants from the built-in table. Extra ``cells`` set the other
    columns of that row.
    """
    r = panel.element_table.rows[0]
    r["_var"]["symbol"].set(symbol)
    panel.element_table.autofill_row(r)
    for key, value in cells.items():
        r["_var"][key].set(value)
    return r


def _cfg(geometry, **over):
    d = {
        "material": {"phonopy_yaml": "g.yaml", "mesh": [40, 40, 40],
                     "temperature_K": 296.0,
                     "scatterers": [{"symbol": "C", "sigma_bound_b": 5.551,
                                     "awr": 11.898, "b_coh_fm": 6.646,
                                     "sigma_inc_b": 0.001}]},
        "physics": {"inelastic_mode": 2, "max_phonon_order": "auto",
                    "n_directions": 8000, "multiphonon_directions": 800,
                    "elastic": True, "elastic_kind": "coherent"},
        "grid": {"e_min_meV": 0.0, "e_max_meV": 200.0, "de_meV": 0.5,
                 "dq_max_invA": 0.05},
        "instrument": {"geometry": geometry, "bank_halfwidth_deg": 5.0,
                       "combine": "mean"},
    }
    if geometry == "vision":
        d["instrument"]["e_fixed_meV"] = 3.5
    elif geometry == "indirect":
        d["instrument"].update(e_fixed_meV=4.0, angles_deg=[30.0, 90.0, 150.0])
    else:
        d["instrument"].update(e_fixed_meV=250.0, angles_deg=[10.0, 60.0, 120.0])
    d["instrument"].update(over.pop("instrument", {}))
    return SpectraConfig.from_dict(d)


# ---- top-level tabs, ENDF form parts, geometry tabs --------------------------
def test_app_top_tabs_and_endf_parts(root):
    from irma.gui.app import IrmaApp
    app = IrmaApp(root)
    top = [app.top_notebook.tab(i, "text") for i in range(len(app.top_notebook.tabs()))]
    assert top == ["ENDF Evaluation", "Neutron Scattering Experiments",
                   "NCrystal plugin", "MLIP phonon models"]
    assert app.mlip_panel is not None
    # the ENDF form is one scrolling page of five parts in deck order
    assert list(app._endf_parts) == ["Material", "Scattering", "Grids",
                                     "Phonon", "Run"]
    geom = [app.ns_panel.geom_nb.tab(i, "text")
            for i in range(len(app.ns_panel.geom_nb.tabs()))]
    assert geom == ["Indirect (VISION defaults)", "Direct"]


# ---- build_config / load_config round-trips --------------------------------
def _mode0_cfg():
    """A DOS-based (mode 0) config with the elastic crystal -- exercises the
    extended scatterer line (dos=/mult=/pos=), the lattice field and dos_source."""
    return SpectraConfig.from_dict({
        "material": {"temperature_K": 300.0,
                     "lattice": [2.866, 2.866, 2.866, 90.0, 90.0, 90.0],
                     "scatterers": [{"symbol": "Fe", "sigma_bound_b": 12.2,
                                     "awr": 55.34, "b_coh_fm": 9.45,
                                     "sigma_inc_b": 0.4, "dos_file": "fe.txt",
                                     "multiplicity": 2,
                                     "positions": [[0.0, 0.0, 0.0],
                                                   [0.5, 0.5, 0.5]]}]},
        "physics": {"inelastic_mode": 0, "dos_source": "file",
                    "max_phonon_order": "auto", "elastic": True,
                    "elastic_kind": "both"},
        "grid": {"e_min_meV": 0.0, "e_max_meV": 200.0, "de_meV": 0.5,
                 "dq_max_invA": 0.05},
        "instrument": {"geometry": "indirect", "e_fixed_meV": 4.0,
                       "angles_deg": [30.0, 90.0, 150.0]},
    })


def _mode0_incoherent_cfg():
    """Mode-0 elastic_kind='incoherent' still needs the crystal (per-atom
    multiplicity from positions)."""
    cfg = _mode0_cfg()
    cfg.physics.elastic_kind = "incoherent"
    return cfg


def _phonopy_mode0_cfg():
    """Mode 0 with the DOS from phonopy: the phonopy input-source gate."""
    return SpectraConfig.from_dict({
        "material": {"phonopy_yaml": "g.yaml", "mesh": [20, 20, 20],
                     "temperature_K": 300.0,
                     "scatterers": [{"symbol": "C", "sigma_bound_b": 5.551,
                                     "awr": 11.898}]},
        "physics": {"inelastic_mode": 0, "dos_source": "phonopy",
                    "elastic": False},
        "grid": {"e_max_meV": 200.0, "de_meV": 0.5, "dq_max_invA": 0.05},
        "instrument": {"geometry": "indirect", "e_fixed_meV": 4.0,
                       "angles_deg": [30.0, 90.0]},
    })


def _force_sets_cfg():
    cfg = _cfg("indirect")
    cfg.material.force_constants = None
    cfg.material.force_sets = "FORCE_SETS"
    return cfg


def _two_species_cfg():
    return SpectraConfig.from_dict({
        "material": {"phonopy_yaml": "beo.yaml", "mesh": [20, 20, 20],
                     "temperature_K": 296.0,
                     "scatterers": [
                         {"symbol": "Be", "sigma_bound_b": 7.63, "awr": 8.93,
                          "b_coh_fm": 7.79, "sigma_inc_b": 0.0018},
                         {"symbol": "O", "sigma_bound_b": 4.23, "awr": 15.86,
                          "b_coh_fm": 5.803, "sigma_inc_b": 0.0008}]},
        "instrument": {"geometry": "indirect", "e_fixed_meV": 4.0,
                       "angles_deg": [30.0, 90.0, 150.0]},
    })


def _min_phonon_cfg():
    cfg = _cfg("direct")
    cfg.physics.min_phonon_energy_meV = 0.5
    return cfg


def _no_widget_cfg():
    # fields the panel has no control for: carried through build_config
    cfg = _cfg("indirect", instrument={"bank_halfwidth_deg": 3.0,
                                       "combine": "sum"})
    cfg.physics.elastic_from_tape = "graphite.endf"
    cfg.grid.q_pad_invA = 1.0
    return cfg


IDENTITY_CASES = {
    "indirect": lambda: _cfg("indirect"),
    "direct": lambda: _cfg("direct"),
    "mode0_dos_files": _mode0_cfg,
    "mode0_incoherent": _mode0_incoherent_cfg,
    "mode0_phonopy": _phonopy_mode0_cfg,
    "force_sets": _force_sets_cfg,
    "two_species": _two_species_cfg,
    "lorentzian": lambda: _cfg(
        "indirect", instrument={"resolution_shape": "lorentzian"}),
    # build_config reads q_cuts / cut_dq_invA outside the geometry branch
    "direct_q_cuts": lambda: _cfg(
        "direct", instrument={"q_cuts": [3.0, 6.0, 9.0]}),
    "indirect_q_cuts": lambda: _cfg(
        "indirect", instrument={"q_cuts": [2.0, 4.0], "cut_dq_invA": 0.1}),
    "direct_map": lambda: _cfg("direct", instrument={
        "output_mode": "map", "map_coverage_deg": [2.373, 135.955],
        "map_mask": False, "cut_by": "q", "cut_dq_invA": 0.1}),
    "sigma_three_terms": lambda: _cfg(
        "indirect", instrument={"sigma_coeffs": [0.31, 0.005, 8.07e-7]}),
    "sigma_two_terms": lambda: _cfg(
        "direct", instrument={"sigma_coeffs": [0.5, 0.01]}),
    "components": lambda: _cfg(
        "direct", instrument={"export_components": True}),
    "min_phonon_energy": _min_phonon_cfg,
    "no_widget_fields": _no_widget_cfg,
}


@pytest.mark.parametrize("case", list(IDENTITY_CASES))
def test_load_then_build_is_identity(panel, case):
    """load_config then build_config gives the same config back."""
    cfg = IDENTITY_CASES[case]()
    panel.load_config(cfg)
    assert panel.build_config() == cfg


def test_indirect_map_config_is_refused_on_load(panel):
    # the panel runs maps for direct geometry only; nothing is loaded
    before = panel.build_config()
    with pytest.raises(SpectraConfigError, match="irma spectra map"):
        panel.load_config(_cfg("indirect", instrument={"output_mode": "map"}))
    assert panel.build_config() == before


def test_map_run_writes_the_full_map(panel, monkeypatch):
    # the Plot tab masks at view and export time, so the cached map is unmasked
    panel.load_config(IDENTITY_CASES["direct_map"]())
    sent = {}
    monkeypatch.setattr(panel.runner, "run_command",
                        lambda argv, **kw: sent.update(argv=argv))
    panel._run_map()
    panel.cleanup_temp_files()
    assert "--no-mask" in sent["argv"]


def test_loading_a_config_clears_what_the_previous_one_set(panel):
    """Values a previously loaded config set (coverage band, q cuts, width
    polynomial, breakdown, cutoff) must not carry into the next config, where
    they would change its run without warning."""
    full = _cfg("direct", instrument={
        "output_mode": "map", "map_coverage_deg": [3.0, 135.0],
        "q_cuts": [2.0, 4.0], "cut_dq_invA": 0.1,
        "sigma_coeffs": [0.5, 0.01], "export_components": True})
    full.physics.min_phonon_energy_meV = 0.5
    panel.load_config(full)
    plain = _cfg("direct", instrument={"output_mode": "map"})
    panel.load_config(plain)
    assert panel.build_config() == plain


# ---- fresh panel: identity blank, methodology prefilled --------------------
def test_fresh_panel_ships_no_material_identity(panel):
    """The scatterer table is material identity: a fresh panel must assert
    nothing about the user's material, or a BeO evaluation could run on
    another material's constants unnoticed. One empty row is offered, and the
    hint says how to fill it."""
    rows = panel.element_table.get_rows()
    assert len(rows) == 1
    assert all(v == "" for k, v in rows[0].items() if k != "dos_unit")
    assert panel.lattice.get() == ""              # mode-0 cell: also identity


def test_fresh_panel_keeps_methodology_defaults(panel):
    """...while everything the validation campaign settled stays prefilled,
    matching the PhysicsConfig defaults."""
    from irma.spectra.config import PhysicsConfig
    assert panel.mesh.get() == "40 40 40"
    assert panel.temperature.get() == "296"
    assert panel.n_directions.get() == "10000"
    assert panel.mp_directions.get() == "1000"
    p = PhysicsConfig()
    assert p.n_directions == 10000 and p.multiphonon_directions == 1000
    assert panel.max_phonon_order.get() == "auto"
    assert panel.ind_ef.get() == "3.5"            # VISION Ef


def test_element_hint_names_the_ways_to_fill_the_table():
    """The blank table must read as deliberately required, not merely empty:
    one gray line naming the fill routes."""
    from irma.gui.ns_panel import IDENTITY_HINT_ELEMENTS
    assert "YOUR material" in IDENTITY_HINT_ELEMENTS
    assert "Auto-fill elements from phonopy.yaml" in IDENTITY_HINT_ELEMENTS
    assert "examples/spectra" in IDENTITY_HINT_ELEMENTS


def test_loading_a_config_without_scatterers_leaves_the_blank_row(panel):
    """The clear path agrees with fresh: no scatterers -> the same single
    blank row, not a header with nothing under it."""
    cfg = _cfg("indirect")
    cfg.material.scatterers = []
    panel.load_config(cfg)
    assert len(panel.element_table.rows) == 1
    assert panel.element_table.get_rows()[0]["symbol"] == ""


# ---- redesigned mode-0 surface: input-source gate + context show/hide ------
def test_default_is_phonopy_mode2(panel):
    panel.phonopy_yaml.set("g.yaml")
    assert panel.input_source.get() == "phonopy"
    c = panel.build_config()
    assert c.physics.inelastic_mode == 2
    assert bool(panel.eng_only_box.winfo_manager())          # engine knobs visible
    assert panel.dos_frame.winfo_manager() == ""             # DOS branch hidden


def test_dos_files_branch_emits_mode0_and_nulls_phonopy(panel):
    panel.input_source.set("dos_files")
    panel._sync_ns_context()
    row = _declare(panel)
    row["_var"]["dos_file"].set("c.dos")
    row["_var"]["multiplicity"].set("4")
    c = panel.build_config()
    assert c.physics.inelastic_mode == 0 and c.physics.dos_source == "file"
    assert c.material.phonopy_yaml is None                   # no phonopy artifacts
    assert panel.eng_only_box.winfo_manager() == ""          # engine knobs hidden
    s = c.material.scatterers[0]
    assert s.dos_file == "c.dos" and s.multiplicity == 4


def test_element_table_add_remove(panel):
    t = panel.element_table
    n0 = len(t.rows)
    t.add_row({"symbol": "H", "sigma_bound_b": "80.27", "awr": "0.999"})
    assert len(t.rows) == n0 + 1
    assert t.get_rows()[-1]["symbol"] == "H"
    t._remove(t.rows[-1])
    assert len(t.rows) == n0


# ---- hidden fields, positions, vision configs, mode-0 crystal ----------------
def test_dos_files_mode_ignores_hidden_engine_fields(panel):
    """Garbage left in the hidden mesh/jobs widgets must not break a DOS-files
    build (those fields are not parsed in mode 0)."""
    panel.mesh.set("not a mesh")
    panel.jobs.set("xyz")
    panel.n_directions.set("oops")
    panel.input_source.set("dos_files")
    panel._sync_ns_context()
    _declare(panel, dos_file="c.dos")
    c = panel.build_config()                                 # must not raise
    assert c.physics.inelastic_mode == 0 and c.material.mesh == [40, 40, 40]


def test_bad_positions_count_rejected(panel):
    panel.input_source.set("dos_files")
    panel.elastic.set("on")
    panel.elastic_kind.set("both")
    panel._sync_ns_context()
    panel.lattice.set("2.8,2.8,2.8,90,90,90")
    # 4 coords -> not a multiple of 3
    _declare(panel, positions="0 0 0 0.5")
    with pytest.raises(ValueError):
        panel.build_config()


def test_vision_config_maps_onto_indirect_tab(panel):
    """A vision-geometry config has no tab of its own -- it loads onto the
    Indirect tab as the equivalent Ef=3.5, 45/135-bank indirect calculation,
    and keeps its q cuts."""
    cfg = _cfg("vision", instrument={"q_cuts": [2.5]})
    panel.load_config(cfg)
    built = panel.build_config()
    assert built.instrument.geometry == "indirect"
    assert built.instrument.e_fixed_meV == 3.5
    assert built.instrument.angles_deg == [45.0, 135.0]
    assert built.instrument.q_cuts == [2.5]


def test_defaults_build_a_valid_indirect_config(panel):
    from irma.spectra.config import validate
    panel.phonopy_yaml.set("g.yaml")    # material identity: the user's to give
    _declare(panel)                     # ...as is the scatterer list
    cfg = panel.build_config()
    # the GUI opens on the Indirect tab; its defaults reproduce VISION
    assert cfg.instrument.geometry == "indirect"
    assert cfg.instrument.e_fixed_meV == 3.5
    assert cfg.instrument.angles_deg == [45.0, 135.0]
    assert cfg.instrument.resolution_shape == "gaussian"
    assert cfg.material.scatterers[0].symbol == "C"
    validate(cfg)   # default indirect config (with a material) must validate


def test_direct_chopper_hides_shape_width_and_forces_gaussian(panel):
    """Selecting the Direct chopper model swaps the shape/width controls out for
    the chopper params, and the built config carries a Gaussian shape (the
    chopper derives its own width). winfo_manager() == '' iff pack_forget'd
    (robust under a withdrawn root, unlike winfo_ismapped)."""
    cfg = _cfg("direct", instrument={
        "resolution_model": "chopper",
        "chopper_spec": {"instrument": "ARCS", "package": "ARCS-700-1.5-AST",
                         "frequency": 600.0}})
    panel.load_config(cfg)
    # chopper frame packed (shown), poly shape/width frame removed
    assert panel._chop_frame.winfo_manager() == "pack"
    assert panel._dir_poly_frame.winfo_manager() == ""
    built = panel.build_config()
    assert built.instrument.resolution_model == "chopper"
    assert built.instrument.resolution_shape == "gaussian"
    assert built.instrument.sigma_coeffs is None
    assert built == cfg
    # switching back to the width-polynomial model restores the shape/width frame
    panel.dir_res_model.set("width polynomial")
    panel._sync_res_model()
    assert panel._dir_poly_frame.winfo_manager() == "pack"
    assert panel._chop_frame.winfo_manager() == ""


def test_geometry_switch_changes_config(panel):
    panel.geom_nb.select(1)            # direct
    assert panel.build_config().instrument.geometry == "direct"
    panel.geom_nb.select(0)            # indirect
    assert panel.build_config().instrument.geometry == "indirect"


def test_nonempty_q_cuts_stay_visible_in_angles_mode(panel):
    """build_config emits q_cuts in both cut modes, so a non-empty entry must
    stay on screen in angles mode -- a hidden field must never feed the run."""
    panel.geom_nb.select(1)                      # direct
    panel.dir_cut_by.set("constant-Q")
    panel._sync_cut_by()
    panel.q_cuts.set("2.0, 4.0")
    panel.dir_cut_by.set("detector angles")
    panel._sync_cut_by()
    assert panel._cut_q_frame.winfo_manager() == "pack"      # still visible
    assert panel._cut_angles_frame.winfo_manager() == "pack"
    assert panel.build_config().instrument.q_cuts == [2.0, 4.0]
    panel.q_cuts.set("")                          # cleared -> hidden again
    panel._sync_cut_by()
    assert panel._cut_q_frame.winfo_manager() == ""
    assert panel.build_config().instrument.q_cuts is None


def test_map_output_frame_visibility_follows_selector(panel):
    """Selecting '2-D map' shows the map-config frame and hides fixed-cuts."""
    panel.geom_nb.select(1)                      # direct
    panel.dir_output.set("2-D map")
    panel._sync_output()
    assert panel._mapcfg_frame.winfo_manager() == "pack"
    assert panel._cuts_frame.winfo_manager() == ""
    panel.dir_output.set("fixed cuts")
    panel._sync_output()
    assert panel._cuts_frame.winfo_manager() == "pack"
    assert panel._mapcfg_frame.winfo_manager() == ""


# ---- width-polynomial coefficient fields (c0/c1/c2) ------------------------
def test_width_coeffs_all_blank_is_none(panel):
    """Default (all three fields blank) => sigma_coeffs None (VISION preset)."""
    cfg = _cfg("indirect")
    panel.load_config(cfg)
    for w in panel.ind_sigma_coeffs._entries:
        assert w.get() == ""
    assert panel.build_config().instrument.sigma_coeffs is None


def test_width_coeffs_leading_blank_reads_as_zero(panel):
    """A non-blank later term with an earlier blank => the blank reads as 0.0."""
    panel.geom_nb.select(0)                       # indirect
    panel.ind_sigma_coeffs.c0.set("")
    panel.ind_sigma_coeffs.c1.set("0.02")
    panel.ind_sigma_coeffs.c2.set("")
    assert panel.build_config().instrument.sigma_coeffs == [0.0, 0.02]


# ---- export components (breakdown) toggle ----------------------------------
@pytest.mark.parametrize("ext", [".csv", ".npz", ".json"])
def test_read_spectrum_handles_total_only_and_breakdown(panel, tmp_path, ext):
    """_read_spectrum returns I/El=None for a total-only file and real arrays
    when the breakdown was kept -- so the plot adapts to either."""
    import numpy as np
    from irma.spectra.cli import write_spectrum
    from irma.spectra.forward import SpectrumResult
    E = np.linspace(0, 100, 6)
    r = SpectrumResult(
        E=E, Q=np.ones((1, 6)), I_inelastic=np.ones(6), I_elastic=0.2 * np.ones(6),
        I_total=1.2 * np.ones(6), geometry="direct", angles_deg=[30.0],
        I_inelastic_per_angle=np.ones((1, 6)), I_elastic_per_angle=0.2 * np.ones((1, 6)),
        metadata={"inelastic_mode": 2, "elastic": True, "n_bragg_edges": 0,
                  "engine_metadata": {}})
    tot = tmp_path / f"tot{ext}"
    write_spectrum(r, str(tot), components=False)
    _, labels, T, I, El = panel._read_spectrum(str(tot))
    assert labels == ["30 deg"] and T.shape == (1, 6)
    assert I is None and El is None                # total-only -> no breakdown

    brk = tmp_path / f"brk{ext}"
    write_spectrum(r, str(brk), components=True)
    _, _, T2, I2, El2 = panel._read_spectrum(str(brk))
    assert I2 is not None and El2 is not None and I2.shape == (1, 6)


# ---- elastic crystal, defaults, entry errors, file dialogs, close ------------
def test_phonopy_mode0_elastic_multiplicity_from_positions(panel):
    """phonopy-source mode 0 with the elastic crystal emits multiplicity ==
    n(positions) (the mult column is hidden there), so a multi-atom species
    passes validate's mult == len(positions) check and round-trips."""
    from irma.spectra.config import validate
    panel.phonopy_yaml.set("g.yaml")
    panel.inelastic_mode.set(panel._MODE_BY_INT[0])
    panel._sync_ns_context()
    panel.lattice.set("2.464,2.464,6.711,90,90,120")
    _declare(panel,
             positions="0 0 0  0 0 0.5  0.33333 0.66667 0  0.66667 0.33333 0.5")
    cfg = panel.build_config()
    s = cfg.material.scatterers[0]
    assert s.multiplicity == 4 and len(s.positions) == 4
    validate(cfg)
    panel.load_config(cfg)                       # and it round-trips
    assert panel.build_config() == cfg


def test_direct_tab_fresh_defaults_validate(panel):
    """the Direct tab's untouched defaults must validate out of the box
    (the Ei pre-fill sits above the Grid 'E max' default)."""
    from irma.spectra.config import validate
    panel.phonopy_yaml.set("g.yaml")    # the one field with no default
    panel.geom_nb.select(1)             # Direct tab
    cfg = panel.build_config()
    assert cfg.instrument.geometry == "direct"
    assert cfg.grid.e_max_meV < cfg.instrument.e_fixed_meV
    validate(cfg)


def test_mode0_save_keeps_crystal_when_elastic_off(panel):
    """toggling the elastic line 'off' must not erase the lattice and
    positions from a built (saved) mode-0 config."""
    cfg = _mode0_cfg()
    panel.load_config(cfg)
    panel.elastic.set("off")
    panel._sync_ns_context()
    built = panel.build_config()
    assert built.physics.elastic is False
    assert built.material.lattice == [2.866, 2.866, 2.866, 90.0, 90.0, 90.0]
    assert built.material.scatterers[0].positions == [[0.0, 0.0, 0.0],
                                                      [0.5, 0.5, 0.5]]


def test_bad_numeric_entry_names_the_field(panel):
    """a blank or typo'd numeric entry must fail with the field's name,
    not a bare 'could not convert string to float' message."""
    panel.de.set("0,5")
    with pytest.raises(ValueError, match=r"dE \(meV\)"):
        panel.build_config()
    panel.de.set("0.5")
    panel.temperature.set("")
    with pytest.raises(ValueError, match="temperature"):
        panel.build_config()


def test_constant_q_blank_angles_still_validates(panel):
    """'cut by constant-Q' with a blanked (hidden) angles field must build
    a config that passes the angles_deg requirement (constant-Q ignores it)."""
    from irma.spectra.config import validate
    panel.phonopy_yaml.set("g.yaml")
    panel.geom_nb.select(1)
    panel.dir_cut_by.set("constant-Q")
    panel._sync_cut_by()
    panel.dir_angles.set("")
    panel.q_cuts.set("2.0,4.0")
    cfg = panel.build_config()
    assert cfg.instrument.cut_by == "q" and cfg.instrument.angles_deg
    validate(cfg)


def test_file_selector_save_extension_follows_filetypes(root, panel):
    """the save-dialog default extension derives from the widget's
    filetypes (first concrete pattern) instead of a hardcoded '.endf'."""
    from irma.gui.widgets import FileSelector
    assert panel.output.defaultextension == ".csv"        # CSV is listed first
    f = tk.Frame(root)
    endf = FileSelector(f, "out:", mode="save",
                        filetypes=[("ENDF files", "*.endf"), ("All files", "*.*")])
    assert endf.defaultextension == ".endf"
    anyf = FileSelector(f, "out:", mode="save")           # wildcard only
    assert anyf.defaultextension == ""
    explicit = FileSelector(f, "out:", mode="save", defaultextension=".yaml")
    assert explicit.defaultextension == ".yaml"


def test_map_temp_files_unlinked_on_destroy(root, tmp_path):
    """the panel's temp 2-D map .npz files are removed on teardown."""
    from irma.gui.ns_panel import NSPanel
    holder = tk.Frame(root)
    p = NSPanel(holder, runner=ComputationRunner())
    plotted = tmp_path / "plotted.npz"
    plotted.write_bytes(b"x")
    stale = tmp_path / "stale.npz"
    stale.write_bytes(b"y")
    p._map_path = str(plotted)
    p._pending_map = str(stale)
    holder.destroy()
    assert not plotted.exists() and not stale.exists()


def test_close_handler_confirms_and_shuts_down_active_run(root, monkeypatch):
    """WM_DELETE_WINDOW confirms during an active run, then shuts down the
    run (a blocking reap, not the fire-and-forget cancel whose daemon kill
    thread dies with the interpreter) before destroying; when idle it
    destroys without prompting."""
    import irma.gui.app as app_mod
    app = app_mod.IrmaApp(root)
    assert root.protocol("WM_DELETE_WINDOW")              # handler installed
    calls = {"shutdown": 0, "destroy": 0}
    monkeypatch.setattr(app.runner, "shutdown",
                        lambda: calls.__setitem__("shutdown", calls["shutdown"] + 1))
    monkeypatch.setattr(app.root, "destroy",
                        lambda: calls.__setitem__("destroy", calls["destroy"] + 1))
    app.runner._running = True                            # simulate active run
    monkeypatch.setattr(app_mod.messagebox, "askyesno", lambda *a, **k: False)
    app._on_close()                                       # declined -> stays open
    assert calls == {"shutdown": 0, "destroy": 0}
    monkeypatch.setattr(app_mod.messagebox, "askyesno", lambda *a, **k: True)
    app._on_close()
    assert calls == {"shutdown": 1, "destroy": 1}
    app.runner._running = False                           # idle -> no prompt
    monkeypatch.setattr(app_mod.messagebox, "askyesno",
                        lambda *a, **k: pytest.fail("prompted while idle"))
    app._on_close()
    assert calls == {"shutdown": 1, "destroy": 2}


# ---- About dialog attribution list -----------------------------------------
def test_about_dialog_names_the_third_party_projects(root, monkeypatch):
    """The About box names the projects THIRD_PARTY_NOTICES.md covers (a fixed
    list here, to keep in step with the notices file by hand) and points to
    the file; INSPIRED is credited in docs/mlip.md, not in the notices."""
    import irma.gui.app as app_mod
    app = app_mod.IrmaApp(root)
    seen = {}
    monkeypatch.setattr(app_mod.messagebox, "showinfo",
                        lambda title, text: seen.update(text=text))
    app._show_about()
    text = seen["text"]
    for covered in ("NJOY2016", "NCrystal", "ncplugin-CrysXT", "phonopy",
                    "endf-parserpy", "THIRD_PARTY_NOTICES"):
        assert covered in text, covered
    assert "INSPIRED" not in text


# ---- progressive disclosure: elastic / gain-side / map-mode actions --------
def test_elastic_off_hides_kind_and_dw(panel):
    """elastic 'off' leaves nothing for elastic_kind or the incoherent-
    elastic DW selector to steer; kind 'coherent' drops the DW selector
    only. Values are preserved (visibility only)."""
    panel._sync_ns_context()
    assert panel.elastic_kind.winfo_manager() == "pack"
    assert panel.incoherent_elastic_dw.winfo_manager() == "pack"
    panel.elastic.set("off")
    panel._sync_ns_context()
    assert panel.elastic_kind.winfo_manager() == ""
    assert panel.incoherent_elastic_dw.winfo_manager() == ""
    panel.elastic.set("on")
    panel.elastic_kind.set("coherent")
    panel._sync_ns_context()
    assert panel.elastic_kind.winfo_manager() == "pack"
    assert panel.incoherent_elastic_dw.winfo_manager() == ""
    panel.elastic_kind.set("both")
    panel._sync_ns_context()
    assert panel.incoherent_elastic_dw.winfo_manager() == "pack"


def test_gain_side_follows_include_gain(panel):
    """The gain-side method is read only while the gain side is included."""
    assert panel.gain_side.winfo_manager() == "pack"
    panel.include_gain.set(False)
    assert panel.gain_side.winfo_manager() == ""
    panel.include_gain.set(True)
    assert panel.gain_side.winfo_manager() == "pack"


def test_direct_map_hides_output_and_breakdown(panel):
    """Direct + '2-D map' never reads the 1-D output file selector or the
    inelastic/elastic breakdown checkbox (the map goes to a temp file,
    exported via 'Save map...'): both hide in exactly that state."""
    assert panel.output.winfo_manager() == "pack"
    assert panel._export_row.winfo_manager() == "pack"
    panel.geom_nb.select(1)                          # Direct tab
    panel.dir_output.set("2-D map")
    panel._sync_output()
    assert panel.output.winfo_manager() == ""
    assert panel._export_row.winfo_manager() == ""
    panel.dir_output.set("fixed cuts")
    panel._sync_output()
    assert panel.output.winfo_manager() == "pack"
    assert panel._export_row.winfo_manager() == "pack"
    # leaving the Direct tab restores them even with '2-D map' selected
    panel.dir_output.set("2-D map")
    panel._sync_output()
    assert panel.output.winfo_manager() == ""
    panel.geom_nb.select(0)                          # Indirect tab
    panel._sync_actions_rows()
    assert panel.output.winfo_manager() == "pack"
    assert panel._export_row.winfo_manager() == "pack"


def test_fresh_direct_map_is_unmasked_for_the_width_polynomial(panel):
    """A fresh panel's Direct tab uses the 'width polynomial' model, whose
    map is unmasked, as after switching to that model."""
    assert panel.dir_res_model.get() == "width polynomial"
    assert panel.dir_map_mask.get() is False


def test_map_mask_checkbox_starts_disabled_like_save_map(panel):
    """The Plot tab's 'mask to accessible' toggle only redraws a cached
    2-D map, so it shares the Save-map button's disabled-until-a-map-runs
    state."""
    assert str(panel.savemap_btn.cget("state")) == "disabled"
    assert str(panel.map_mask_chk.cget("state")) == "disabled"
