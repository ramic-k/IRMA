"""irma.gui SabGridForm -- the reusable S(alpha,beta) grid widget.

Requires tkinter (skips without it). Uses a withdrawn root, so no display is
shown. Pins: (1) the form constructs headlessly; (2) automatic mode returns the
converged ENDF-style grid knobs with the config defaults (freq_max omitted when
blank = auto); (3) switching to explicit + setting alpha/beta text returns the
parsed float lists (space/comma separated); (4) explicit with only one list
raises; (5) a bad number raises a field-named error.
"""
import pytest

# importorskip FIRST -- a hard `import tkinter` at module top would error at
# COLLECTION time on a Python built without _tkinter, aborting the whole pytest
# run instead of skipping cleanly.
tk = pytest.importorskip("tkinter")


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
def form(root):
    from irma.gui.grid_form import SabGridForm
    frame = tk.Frame(root)
    return SabGridForm(frame)


# ---- construction + automatic defaults -------------------------------------
def test_auto_mode_returns_converged_grid_knobs_with_defaults(form):
    """Automatic mode yields the converged ENDF-style grid knobs with the config
    defaults, NO alpha/beta keys, and NO freq_max_eV when blank (auto-estimate)."""
    assert form.export_fields() == {
        "n_lower": 15, "n_phonon": 300, "n_upper": 80, "beta_max_eV": 5.0,
        "alpha_dq_invA": 0.05, "alpha_qcut_invA": 12.0, "alpha_nlog": 160}


def test_auto_mode_picks_up_edited_values(form):
    """Edited automatic fields flow through; a set freq_max appears as a float."""
    form.alpha_dq.set("0.02")
    form.freq_max.set("0.18")
    fields = form.export_fields()
    assert fields["alpha_dq_invA"] == 0.02
    assert fields["freq_max_eV"] == 0.18


# ---- explicit mode ---------------------------------------------------------
def test_explicit_mode_returns_parsed_float_lists(form):
    """Switching to explicit + setting alpha/beta text returns ONLY the two
    parsed float lists (mixed whitespace + comma separators)."""
    form.grid_mode.set("explicit (alpha/beta)")
    form.alpha_grid.set("0.1 0.5, 1.0  2.0")
    form.beta_grid.set("0.0, 0.5 1.0")
    fields = form.export_fields()
    assert set(fields) == {"alpha_grid", "beta_grid"}
    assert fields["alpha_grid"] == [0.1, 0.5, 1.0, 2.0]
    assert fields["beta_grid"] == [0.0, 0.5, 1.0]


def test_explicit_mode_only_one_list_raises(form):
    """Explicit with only alpha (beta blank) -- or vice versa -- raises."""
    form.grid_mode.set("explicit (alpha/beta)")
    form.alpha_grid.set("0.1 0.5 1.0")
    form.beta_grid.set("")
    with pytest.raises(ValueError, match="explicit grid"):
        form.export_fields()
    form.alpha_grid.set("")
    form.beta_grid.set("0.0 0.5 1.0")
    with pytest.raises(ValueError, match="explicit grid"):
        form.export_fields()


@pytest.mark.parametrize("explicit, match", [(True, "alpha_grid"),
                                              (False, "alpha dQ")])
def test_bad_number_names_the_field(form, explicit, match):
    """A typo'd entry fails with the field's name, not a bare 'could not
    convert string to float'."""
    if explicit:
        form.grid_mode.set("explicit (alpha/beta)")
        form.alpha_grid.set("0.1 xx 1.0")
        form.beta_grid.set("0.0 0.5")
    else:
        form.alpha_dq.set("not-a-number")
    with pytest.raises(ValueError, match=match):
        form.export_fields()


# ---- single-source auto-grid defaults (ENDF tab == NCrystal panel == config)
def test_auto_grid_defaults_single_source(root):
    """The seven auto-grid knobs must carry the SAME defaults on the ENDF
    Evaluation grid tab, the NCrystal panel form, and the export config --
    all traced to irma.core.grids.AUTO_GRID_DEFAULTS (the one place to
    change them). Guards against the drift the n_upper 20->80 change had
    to chase across files."""
    import dataclasses

    from irma.core.grids import AUTO_GRID_DEFAULTS
    from irma.gui.app import IrmaApp
    from irma.gui.grid_form import SabGridForm
    from irma.ncrystal.config import NCrystalExportConfig

    # widget-attribute name -> AUTO_GRID_DEFAULTS key
    knob_keys = {
        "n_lower": "n_lower", "n_phonon": "n_phonon", "n_upper": "n_upper",
        "beta_max": "beta_max_eV", "alpha_dq": "alpha_dq_invA",
        "alpha_qcut": "alpha_qcut_invA", "alpha_nlog": "alpha_nlog",
    }

    app = IrmaApp(root)
    ncr = SabGridForm(tk.Frame(root))
    for attr, key in knob_keys.items():
        want = float(AUTO_GRID_DEFAULTS[key])
        assert float(getattr(app, attr).get()) == want, (attr, "ENDF tab")
        assert float(getattr(ncr, attr).get()) == want, (attr, "NCrystal panel")

    cfg_defaults = {f.name: f.default
                    for f in dataclasses.fields(NCrystalExportConfig)}
    for key, want in AUTO_GRID_DEFAULTS.items():
        assert cfg_defaults[key] == want, (key, "NCrystalExportConfig")
