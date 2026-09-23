"""GUI widget behavior.

Covers the help popup sizing -- a fixed 12-line, non-resizable,
scrollbar-less Text would clip the longest help texts (Bragg-edge grouping,
mpdir, ilog, ...) mid-sentence -- and the fresh form's two-part contract:

  * METHODOLOGY defaults must match the documented production settings
    (Card 6g '10000 1000 1', mesh 40^3, Auto-size ON, LAT=1, npr=1, 296 K);
  * MATERIAL IDENTITY (ZA, MAT, AWR, sigma_free, the lattice, the Card 6d
    atom block) must be BLANK -- IRMA cannot know the user's material, and a
    plausible-but-wrong prefill ships a tape for the wrong one. Reset must
    land in that same blank state, an example deck must fill it, and
    generating from it must name the missing cards.

Requires a display (Tk); skipped headless (CI).
"""
import sys

import pytest

tk = pytest.importorskip("tkinter")
from tkinter import ttk  # noqa: E402

from irma.gui.widgets import InfoLabel, _HELP_MAX_VISIBLE_LINES  # noqa: E402


@pytest.fixture(scope="module")
def root():
    try:
        r = tk.Tk()
    except tk.TclError:
        pytest.skip("no display available for Tk")
    r.withdraw()
    yield r
    r.destroy()


def _open_popup(root, message):
    btn = InfoLabel(root, "Test Help", message)
    btn._show()
    top = next(w for w in btn.winfo_children() if isinstance(w, tk.Toplevel))
    top.update_idletasks()
    return top


def _find_all(widget, klass):
    found = []

    def _rec(w):
        for child in w.winfo_children():
            if isinstance(child, klass):
                found.append(child)
            _rec(child)

    _rec(widget)
    return found


def _resizable_flags(top):
    res = top.wm_resizable()
    if isinstance(res, str):
        res = res.split()
    return tuple(int(v) for v in res)


# ---------------- help popup sizing ----------------

@pytest.mark.skipif(sys.platform == "win32", reason="Tk display-line metrics differ on Windows; the popup sizing is cosmetic and pinned on X11/aqua")
def test_short_help_stays_compact(root):
    top = _open_popup(root, "one short line")
    txt = _find_all(top, tk.Text)[0]
    assert int(txt.cget("height")) <= 5
    assert not _find_all(top, ttk.Scrollbar)     # no pointless scrollbar
    top.destroy()


@pytest.mark.skipif(sys.platform == "win32", reason="Tk display-line metrics differ on Windows; the popup sizing is cosmetic and pinned on X11/aqua")
def test_medium_help_sized_to_content_not_12_lines(root):
    # 20 short logical lines: the old fixed height=12 clipped lines 13-20.
    msg = "\n".join(f"line {i}" for i in range(1, 21))
    top = _open_popup(root, msg)
    txt = _find_all(top, tk.Text)[0]
    assert int(txt.cget("height")) == 20
    assert not _find_all(top, ttk.Scrollbar)
    top.destroy()


def test_long_help_caps_height_and_gets_scrollbar(root):
    # Far beyond the cap (and wider than the wrap width, so display lines
    # exceed logical lines): height caps at _HELP_MAX_VISIBLE_LINES and a
    # scrollbar appears so the rest is discoverable.
    msg = "\n".join(f"line {i} " + "word " * 12 for i in range(1, 61))
    top = _open_popup(root, msg)
    txt = _find_all(top, tk.Text)[0]
    assert int(txt.cget("height")) == _HELP_MAX_VISIBLE_LINES
    assert _find_all(top, ttk.Scrollbar), "long help text needs a scrollbar"
    top.destroy()


def test_popup_is_vertically_resizable(root):
    top = _open_popup(root, "anything")
    assert _resizable_flags(top) == (0, 1)       # fixed width, free height
    top.destroy()


# ---------------- fresh-form builder defaults (IrmaApp) ----------------

@pytest.fixture(scope="module")
def fresh_app(root):
    """One genuinely untouched form for the read-only default checks."""
    from irma.gui.app import IrmaApp
    app_root = tk.Toplevel(root)
    app_root.withdraw()
    app = IrmaApp(app_root)
    yield app
    app_root.destroy()


@pytest.fixture
def new_app(root):
    """A fresh form per test for the checks that MUTATE it (fill-from-ZA,
    reset, import); ``fresh_app`` has to stay untouched."""
    from irma.gui.app import IrmaApp
    app_root = tk.Toplevel(root)
    app_root.withdraw()
    app = IrmaApp(app_root)
    yield app
    app_root.destroy()


def _identity_state(app):
    """Every MATERIAL-IDENTITY field of the ENDF form, as strings.

    These are the fields IRMA cannot know: the scatterer (ZA/AWR/sigma_free),
    the evaluation's MAT number, and the crystal (lattice + Card 6d atoms).
    """
    return {
        "za": app.za.get(), "mat": app.mat.get(),
        "awr": app.awr.get(), "spr": app.spr.get(),
        "lattice": tuple(w.get() for w in (app.latt_a, app.latt_b, app.latt_c,
                                           app.latt_alpha, app.latt_beta,
                                           app.latt_gamma)),
        "atoms": app.atoms_text.get("1.0", "end-1c"),
    }


def _example_deck(name):
    import pathlib
    return str(pathlib.Path(__file__).resolve().parents[1]
               / "examples" / "tsl" / name)


def test_fresh_app_sampling_defaults_are_production_recommended(fresh_app):
    """A user who accepts every mode-1/2 default must get the documented
    recommended production sampling (Card 6g 10000 1000 1), not the manual's
    truncation example."""
    assert fresh_app.nc_num_directions.get() == "10000"
    assert fresh_app.nc_multiphonon_num_directions.get() == "1000"
    assert (fresh_app.nc_mesh_nx.get(), fresh_app.nc_mesh_ny.get(),
            fresh_app.nc_mesh_nz.get()) == ("40", "40", "40")
    assert int(fresh_app.nc_auto_order_var.get()) == 1   # safe by construction
    assert fresh_app.lat.get().startswith("1 —")         # matches reset default


def test_fresh_app_mode_radio_labels_match_ns_panel_wording(fresh_app):
    """One physics mode, one name: the ENDF tab must not call mode 2
    'coh. approx.' while the NS panel calls it exact/coherent."""
    radios = _find_all(fresh_app.root, ttk.Radiobutton)
    labels = {str(r.cget("text")) for r in radios}
    assert "2 — coherent (exact 1-phonon)" in labels
    assert "1 — incoherent approx." in labels
    assert not any("coh. approx" in lb for lb in labels)


def test_bragg_grouping_default_on_and_help_agrees(fresh_app):
    """SPG-5: the Bragg-edge grouping checkbox is DELIBERATELY constructed
    checked with 50 bins/decade prefilled (docs/gui.md: 'on by default');
    the in-app help must describe that default, not claim 'Off by default'
    / '0 (default) = OFF' as it used to."""
    import inspect
    from irma.gui import endf_form
    assert fresh_app.coh_edge_group_enable_var.get() is True
    assert fresh_app.coh_edge_group_bpd.get() == "50"
    assert fresh_app.coh_edge_group_thr.get() == "1.0"
    src = inspect.getsource(endf_form)
    assert "0 (default) = OFF" not in src
    assert "option). Off by default" not in src
    assert "The GUI enables it by default" in src


# ---------------- material identity is blank, methodology is not ------------

def test_fresh_form_asserts_no_material_identity(fresh_app):
    """A prefilled ZA/lattice/atom row is IRMA asserting a material it cannot
    know, and a plausible-but-wrong value survives review far more easily than
    an empty field (a BeO evaluation carrying graphite's numbers ships a tape
    for the wrong material). Every identity field starts empty."""
    st = _identity_state(fresh_app)
    assert (st["za"], st["mat"], st["awr"], st["spr"]) == ("", "", "", "")
    assert st["lattice"] == ("",) * 6
    assert st["atoms"].strip() == ""


def test_fresh_form_keeps_the_methodology_defaults(fresh_app):
    """...while everything the validation campaign settled stays prefilled:
    blanking identity must not blank the settings a user should not have to
    think about."""
    assert (fresh_app.nc_mesh_nx.get(), fresh_app.nc_mesh_ny.get(),
            fresh_app.nc_mesh_nz.get()) == ("40", "40", "40")
    assert fresh_app.nc_num_directions.get() == "10000"
    assert fresh_app.nc_multiphonon_num_directions.get() == "1000"
    assert int(fresh_app.nc_auto_order_var.get()) == 1
    assert fresh_app.npr.get() == "1"
    assert float(fresh_app.temps_var.get()) == 296.0
    assert fresh_app.nphon.get() == "100"
    assert fresh_app.smin.get() == "1e-75"
    assert fresh_app.coh_edge_group_enable_var.get() is True
    assert fresh_app.coh_edge_group_bpd.get() == "50"


def test_identity_hints_name_the_ways_to_fill_the_blank_fields():
    """Blank must read as deliberately required, not merely forgotten: one
    gray hint per blanked group, naming the routes that fill it."""
    from irma.gui import endf_form
    hints = (endf_form.IDENTITY_HINT_SCATTERER, endf_form.IDENTITY_HINT_MAT,
             endf_form.IDENTITY_HINT_LATTICE, endf_form.IDENTITY_HINT_ATOMS)
    for hint in hints:
        assert "YOUR material" in hint or "YOUR evaluation" in hint
        assert "Import Input File" in hint
        assert "examples/tsl" in hint
    for hint in (endf_form.IDENTITY_HINT_LATTICE, endf_form.IDENTITY_HINT_ATOMS):
        assert endf_form.STRUCTURE_FILL_TITLE in hint
    assert "Fill AWR + sigma_free from ZA" in endf_form.IDENTITY_HINT_SCATTERER


def test_reset_lands_in_the_same_blank_state_as_a_fresh_form(new_app):
    """Reset (deck import runs it first) and a freshly built form must agree.
    Reset used to leave ZA/MAT/AWR/spr at '0' and the lattice at '0' where the
    builder had graphite's numbers -- two different 'defaults', neither blank."""
    fresh = _identity_state(new_app)
    new_app.za.set("4009")
    new_app.mat.set("29")
    new_app.awr.set("8.93478")
    new_app.spr.set("6.15")
    for w in (new_app.latt_a, new_app.latt_b, new_app.latt_c):
        w.set("2.2856")
    for w in (new_app.latt_alpha, new_app.latt_beta, new_app.latt_gamma):
        w.set("90.0")
    new_app.atoms_text.insert("1.0", "4 9 8.93478 7.79 0.0018 2 0 0 0 "
                                     "0.333333 0.666667 0.5\n")
    new_app._reset_form_to_defaults()
    assert _identity_state(new_app) == fresh
    assert fresh["za"] == "" and fresh["lattice"] == ("",) * 6


def test_importing_an_example_deck_fills_the_blank_identity(new_app):
    """The advertised way out of a blank form: a committed example deck fills
    every field the form leaves empty."""
    new_app._import_leapr_from_path(_example_deck("graphite_iel10_classic.input"))
    st = _identity_state(new_app)
    assert st["za"] == "6000"                      # natural carbon
    assert st["mat"] == "30"
    assert float(st["awr"]) == pytest.approx(11.898, rel=1e-6)
    assert float(st["spr"]) == pytest.approx(4.73918, rel=1e-6)
    assert [float(v) for v in st["lattice"]] == pytest.approx(
        [2.4612, 2.4612, 6.7079, 90.0, 90.0, 120.0])
    assert st["atoms"].split()[:2] == ["6", "0"]   # Card 6d row present


def test_blank_form_generation_names_the_missing_cards(new_app):
    """Pressing Run on a fresh form must produce IRMA's friendly card-naming
    validation error, not a traceback and not a silently short Card 4."""
    with pytest.raises(ValueError) as exc:
        new_app._generate_input_text()
    msg = str(exc.value)
    for token in ("MAT (Card 4)", "ZA (Card 4)", "AWR (Card 5)",
                  "sigma_free (Card 5)", "Import Input File"):
        assert token in msg
    assert "could not convert" not in msg       # not a bare conversion error


def test_blank_crystal_names_its_own_cards(new_app):
    """With the scatterer identity given, the iel=10 crystal cards report
    themselves the same way (6c lattice, then 6d atom types)."""
    new_app.mat.set("28")
    new_app.za.set("6000")
    new_app.awr.set("11.9078")
    new_app.spr.set("4.72633")
    with pytest.raises(ValueError, match=r"Card 6c"):
        new_app._generate_input_text()
    for w, v in ((new_app.latt_a, "2.461"), (new_app.latt_b, "2.461"),
                 (new_app.latt_c, "6.708"), (new_app.latt_alpha, "90.0"),
                 (new_app.latt_beta, "90.0"), (new_app.latt_gamma, "120.0")):
        w.set(v)
    with pytest.raises(ValueError, match=r"at least one atom type \(Card 6d\)"):
        new_app._generate_input_text()


def test_fill_from_za_fills_awr_and_sigma_free_from_the_table(new_app):
    """The successor to the old 'fresh defaults agree with the button' pin:
    with ZA blank there is nothing to agree with, so what matters is that
    setting a ZA and pressing the button fills the other two from the
    built-in table (in the deck's FREE-atom convention)."""
    from unittest import mock
    from irma.gui import endf_form
    from irma.core.nuclear_data import lookup
    new_app.za.set("6000")
    with mock.patch.object(endf_form, "messagebox") as mb:
        new_app._fill_from_za()
    assert not mb.showerror.called, mb.showerror.call_args
    nat = lookup((6, 0))
    assert float(new_app.awr.get()) == pytest.approx(nat.awr, rel=1e-5)
    spr = nat.sigma_bound_b * (nat.awr / (1.0 + nat.awr)) ** 2
    assert float(new_app.spr.get()) == pytest.approx(spr, rel=1e-5)


def test_fill_from_za_with_a_blank_za_reports_and_changes_nothing(new_app):
    """The first click on a fresh form hits an empty ZA: say so plainly
    (messagebox stubbed -- a real modal would block the suite) and touch
    nothing."""
    from unittest import mock
    from irma.gui import endf_form
    with mock.patch.object(endf_form, "messagebox") as mb:
        new_app._fill_from_za()
    assert mb.showerror.called
    message = " ".join(str(a) for a in mb.showerror.call_args[0])
    assert "ZA" in message and "Enter a ZA first" in message
    assert new_app.awr.get() == "" and new_app.spr.get() == ""
