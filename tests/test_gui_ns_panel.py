"""irma.gui NSPanel + restructure (P8) -- widget<->SpectraConfig round-trips.

Requires tkinter (skips without it). Uses a withdrawn root, so no display is
shown. Pins: (1) the restructure keeps the ENDF notebook's five tabs intact
under the new top-level "ENDF Evaluation" tab (so deck generation is unchanged --
the existing test_gui_deck_io suite is the byte-identical-deck gate); (2) the NS
panel's build_config/load_config are exact inverses for every geometry; (3)
Save/Open is config.dump/load.
"""
import pytest

# importorskip FIRST -- a hard `import tkinter` at module top would error at
# COLLECTION time on a Python built without _tkinter (e.g. the CI runner's
# Homebrew python), aborting the whole pytest run instead of skipping cleanly.
tk = pytest.importorskip("tkinter")
from irma.spectra.config import SpectraConfig, dump, load   # noqa: E402
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


# ---- restructure: ENDF single-page form intact + two geometry tabs ---------
def test_app_has_two_top_tabs_and_endf_intact(root):
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
@pytest.mark.parametrize("geometry", ["indirect", "direct"])
def test_load_then_build_is_identity(panel, geometry):
    cfg = _cfg(geometry)
    panel.load_config(cfg)
    assert panel.build_config() == cfg


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


def test_mode0_load_build_identity(panel):
    cfg = _mode0_cfg()
    panel.load_config(cfg)
    built = panel.build_config()
    assert built.physics.inelastic_mode == 0
    assert built.physics.dos_source == "file"
    assert built.material.lattice == [2.866, 2.866, 2.866, 90.0, 90.0, 90.0]
    s = built.material.scatterers[0]
    assert s.dos_file == "fe.txt" and s.multiplicity == 2
    assert s.positions == [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]
    assert built == cfg


# ---- fresh panel: identity blank, methodology prefilled --------------------
def test_fresh_panel_ships_no_material_identity(panel):
    """The scatterer table is material IDENTITY: a fresh panel must assert
    nothing about the user's material (the old prefilled natural-carbon row
    let a BeO evaluation run on graphite's constants unnoticed). One EMPTY
    row is offered, and the hint says how to fill it."""
    rows = panel.element_table.get_rows()
    assert len(rows) == 1
    assert all(v == "" for k, v in rows[0].items() if k != "dos_unit")
    assert panel.lattice.get() == ""              # mode-0 cell: also identity


def test_fresh_panel_keeps_methodology_defaults(panel):
    """...while everything the validation campaign settled stays prefilled."""
    assert panel.mesh.get() == "40 40 40"
    assert panel.temperature.get() == "296"
    assert panel.n_directions.get() == "10000"
    assert panel.mp_directions.get() == "1000"
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


def test_phonopy_mode0_round_trips(panel):
    cfg = SpectraConfig.from_dict({
        "material": {"phonopy_yaml": "g.yaml", "mesh": [20, 20, 20],
                     "temperature_K": 300.0,
                     "scatterers": [{"symbol": "C", "sigma_bound_b": 5.551,
                                     "awr": 11.898}]},
        "physics": {"inelastic_mode": 0, "dos_source": "phonopy", "elastic": False},
        "grid": {"e_max_meV": 200.0, "de_meV": 0.5, "dq_max_invA": 0.05},
        "instrument": {"geometry": "indirect", "e_fixed_meV": 4.0,
                       "angles_deg": [30.0, 90.0]},
    })
    panel.load_config(cfg)
    assert panel.input_source.get() == "phonopy"             # DOS-from-phonopy stays on phonopy gate
    assert panel.build_config() == cfg


def test_element_table_add_remove(panel):
    t = panel.element_table
    n0 = len(t.rows)
    t.add_row({"symbol": "H", "sigma_bound_b": "80.27", "awr": "0.999"})
    assert len(t.rows) == n0 + 1
    assert t.get_rows()[-1]["symbol"] == "H"
    t._remove(t.rows[-1])
    assert len(t.rows) == n0


def test_phonopy_symbols_reads_yaml():
    import os
    from irma.gui.ns_panel import _phonopy_symbols
    yaml = os.path.join(os.path.dirname(__file__),
                        "mode2_euphonic_n1_validation", "graphite", "phonopy.yaml")
    if not os.path.exists(yaml):
        pytest.skip("graphite phonopy fixture not present")
    syms = _phonopy_symbols(yaml)
    assert syms == ["C"]                                     # graphite: one species


# ---- Codex-review fixes -----------------------------------------------------
def test_mode0_incoherent_keeps_the_crystal(panel):
    """Codex HIGH: mode-0 elastic_kind='incoherent' still needs the crystal
    (per-atom mult/N from positions) -- it must round-trip, not be dropped."""
    cfg = _mode0_cfg()
    cfg.physics.elastic_kind = "incoherent"
    panel.load_config(cfg)
    built = panel.build_config()
    assert built.material.lattice == [2.866, 2.866, 2.866, 90.0, 90.0, 90.0]
    assert built.material.scatterers[0].positions == [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]
    assert built == cfg


def test_force_sets_round_trips(panel):
    cfg = _cfg("indirect")
    cfg.material.force_constants = None
    cfg.material.force_sets = "FORCE_SETS"
    panel.load_config(cfg)
    built = panel.build_config()
    assert built.material.force_sets == "FORCE_SETS"
    assert built == cfg


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
    """A legacy 'vision' config has no tab of its own -- it loads onto the
    Indirect tab as the equivalent Ef=3.5, 45/135-bank indirect calculation."""
    cfg = _cfg("vision")
    panel.load_config(cfg)
    built = panel.build_config()
    assert built.instrument.geometry == "indirect"
    assert built.instrument.e_fixed_meV == 3.5
    assert built.instrument.angles_deg == [45.0, 135.0]


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


def test_resolution_shape_round_trips(panel):
    cfg = _cfg("indirect", instrument={"resolution_shape": "lorentzian"})
    panel.load_config(cfg)
    assert panel.ind_resolution_shape.get() == "lorentzian"
    assert panel.build_config().instrument.resolution_shape == "lorentzian"


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


def test_q_cuts_round_trip_every_geometry(panel):
    """SPG-3: build_config reads q_cuts / cut_dq_invA OUTSIDE the geometry
    branch -- a CLI-authored vision/indirect config carrying them must
    survive Run/Save (the CLI attaches --q-cuts to every geometry and
    run_spectra honours them geometry-independently)."""
    # a direct config carrying q_cuts round-trips
    cfg = _cfg("direct", instrument={"q_cuts": [3.0, 6.0, 9.0]})
    panel.load_config(cfg)
    assert panel.build_config() == cfg
    # an indirect config carrying q_cuts + cut_dq keeps BOTH
    cfg = _cfg("indirect", instrument={"q_cuts": [2.0, 4.0],
                                       "cut_dq_invA": 0.1})
    panel.load_config(cfg)
    built = panel.build_config()
    assert built.instrument.q_cuts == [2.0, 4.0]
    assert built.instrument.cut_dq_invA == 0.1
    assert built == cfg
    # a legacy vision config keeps them too (mapped onto the indirect tab)
    cfg = _cfg("vision", instrument={"q_cuts": [2.5]})
    panel.load_config(cfg)
    assert panel.build_config().instrument.q_cuts == [2.5]
    # and a config WITHOUT them still builds none (no stale carry-over)
    panel.load_config(_cfg("indirect"))
    built = panel.build_config().instrument
    assert built.q_cuts is None and built.cut_dq_invA is None


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


def test_direct_output_mode_round_trips(panel):
    """The Direct-tab Output selector + map fields round-trip; the '2-D map' mode
    and the fixed-cuts sub-mode (by constant-Q with a cut dQ) survive load->build."""
    cfg = _cfg("direct", instrument={
        "output_mode": "map", "map_coverage_deg": [2.373, 135.955],
        "map_mask": False, "cut_by": "q", "cut_dq_invA": 0.1})
    panel.load_config(cfg)
    assert panel.dir_output.get() == "2-D map"
    assert panel.dir_cut_by.get() == "constant-Q"
    built = panel.build_config().instrument
    assert built.output_mode == "map"
    assert built.map_coverage_deg == [2.373, 135.955]
    assert built.map_mask is False
    assert built.cut_by == "q" and built.cut_dq_invA == 0.1


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


# ---- Save/Open is config.dump/load -----------------------------------------
@pytest.mark.parametrize("geometry", ["indirect", "direct"])
def test_save_open_round_trip(panel, tmp_path, geometry):
    cfg = _cfg(geometry)
    panel.load_config(cfg)
    p = dump(panel.build_config(), tmp_path / f"{geometry}.yaml")
    reloaded = load(p)
    assert reloaded == cfg
    panel.load_config(reloaded)
    assert panel.build_config() == cfg


def test_multi_species_scatterers_round_trip(panel):
    cfg = SpectraConfig.from_dict({
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
    panel.load_config(cfg)
    assert panel.build_config() == cfg


# ---- width-polynomial coefficient fields (c0/c1/c2) ------------------------
def test_width_coeffs_three_fields_round_trip(panel):
    """An explicit 3-coefficient sigma poly loads into the three fields and
    rebuilds to itself; the fields hold the individual terms."""
    cfg = _cfg("indirect", instrument={"sigma_coeffs": [0.31, 0.005, 8.07e-7]})
    panel.load_config(cfg)
    assert panel.ind_sigma_coeffs.c0.get() == "0.31"
    assert panel.ind_sigma_coeffs.c1.get() == "0.005"
    assert panel.ind_sigma_coeffs.c2.get() == "8.07e-07"
    assert panel.build_config().instrument.sigma_coeffs == [0.31, 0.005, 8.07e-7]


def test_width_coeffs_all_blank_is_none(panel):
    """Default (all three fields blank) => sigma_coeffs None (VISION preset)."""
    cfg = _cfg("indirect")
    panel.load_config(cfg)
    for w in panel.ind_sigma_coeffs._entries:
        assert w.get() == ""
    assert panel.build_config().instrument.sigma_coeffs is None


def test_width_coeffs_trailing_blank_trims_to_short_list(panel):
    """A 2-coefficient list loads c0,c1 (c2 blank) and rebuilds to [c0, c1] --
    trailing blanks are dropped so a short list round-trips exactly."""
    cfg = _cfg("direct", instrument={"sigma_coeffs": [0.5, 0.01]})
    panel.load_config(cfg)
    assert panel.dir_sigma_coeffs.c2.get() == ""
    assert panel.build_config().instrument.sigma_coeffs == [0.5, 0.01]


def test_width_coeffs_leading_blank_reads_as_zero(panel):
    """A non-blank later term with an earlier blank => the blank reads as 0.0."""
    panel.geom_nb.select(0)                       # indirect
    panel.ind_sigma_coeffs.c0.set("")
    panel.ind_sigma_coeffs.c1.set("0.02")
    panel.ind_sigma_coeffs.c2.set("")
    assert panel.build_config().instrument.sigma_coeffs == [0.0, 0.02]


# ---- export components (breakdown) toggle ----------------------------------
def test_export_components_defaults_off_and_round_trips(panel):
    """The breakdown toggle defaults OFF (total-only) and round-trips."""
    assert panel.build_config().instrument.export_components is False
    cfg = _cfg("direct", instrument={"export_components": True})
    panel.load_config(cfg)
    assert panel.export_components.get() is True
    assert panel.build_config().instrument.export_components is True


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


# ---- QA4 fixes ---------------------------------------------------------------
def test_loading_config_without_coverage_clears_stale_band(panel):
    """F4: a config with no map_coverage_deg must not inherit the previously
    loaded config's coverage band (which would silently mask its 2-D map)."""
    a = _cfg("direct", instrument={"output_mode": "map",
                                   "map_coverage_deg": [3.0, 135.0]})
    panel.load_config(a)
    assert panel.map_coverage.get() == "3.0,135.0"
    b = _cfg("direct", instrument={"output_mode": "map"})
    panel.load_config(b)
    assert panel.map_coverage.get() == ""
    assert panel.build_config().instrument.map_coverage_deg is None


def test_phonopy_mode0_elastic_multiplicity_from_positions(panel):
    """F5: phonopy-source mode 0 with the elastic crystal emits multiplicity ==
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
    """F17: the Direct tab's untouched defaults must validate out of the box
    (the Ei pre-fill sits above the Grid 'E max' default)."""
    from irma.spectra.config import validate
    panel.phonopy_yaml.set("g.yaml")    # the one field with no default
    panel.geom_nb.select(1)             # Direct tab
    cfg = panel.build_config()
    assert cfg.instrument.geometry == "direct"
    assert cfg.grid.e_max_meV < cfg.instrument.e_fixed_meV
    validate(cfg)


def test_mode0_save_keeps_crystal_when_elastic_off(panel):
    """F18: toggling the elastic line 'off' must not erase the lattice and
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
    """F29: a blank or typo'd numeric entry must fail with the field's name,
    not a bare 'could not convert string to float' message."""
    panel.de.set("0,5")
    with pytest.raises(ValueError, match=r"dE \(meV\)"):
        panel.build_config()
    panel.de.set("0.5")
    panel.temperature.set("")
    with pytest.raises(ValueError, match="temperature"):
        panel.build_config()


def test_constant_q_blank_angles_still_validates(panel):
    """F47: 'cut by constant-Q' with a blanked (hidden) angles field must build
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
    """F44: the save-dialog default extension derives from the widget's
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
    """F46: the panel's temp 2-D map .npz files are removed on teardown."""
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
    """F27 + S8: WM_DELETE_WINDOW confirms during an active run, then SHUTS
    DOWN (blocking reap, not the fire-and-forget cancel whose daemon kill
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


# ---- direction defaults (author amendment to SPG-1/DOC-1) ------------------
def test_direction_defaults_are_10000_and_1000(panel):
    """The spectra direction defaults were raised to the manual's advertised
    10000/1000 at every site: the GUI widgets, PhysicsConfig, and the CLI
    (the NCrystal exporter was already there)."""
    from irma.spectra.config import PhysicsConfig
    assert panel.n_directions.get() == "10000"
    assert panel.mp_directions.get() == "1000"
    panel.phonopy_yaml.set("g.yaml")
    cfg = panel.build_config()
    assert cfg.physics.n_directions == 10000
    assert cfg.physics.multiphonon_directions == 1000
    p = PhysicsConfig()
    assert p.n_directions == 10000 and p.multiphonon_directions == 1000


# ---- About dialog agrees with THIRD_PARTY_NOTICES.md (REL-8) ---------------
def test_about_dialog_matches_third_party_notices(root, monkeypatch):
    """The About box is the shipped attribution surface: it must list every
    project the notices file covers and nothing the notices file does not
    (INSPIRED is credited in docs/mlip.md, not in the notices)."""
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
    panel._on_geometry_tab()
    assert panel.output.winfo_manager() == "pack"
    assert panel._export_row.winfo_manager() == "pack"


def test_map_mask_checkbox_starts_disabled_like_save_map(panel):
    """The Plot tab's 'mask to accessible' toggle only redraws a cached
    2-D map, so it shares the Save-map button's disabled-until-a-map-runs
    state."""
    assert str(panel.savemap_btn.cget("state")) == "disabled"
    assert str(panel.map_mask_chk.cget("state")) == "disabled"
