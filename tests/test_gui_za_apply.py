"""ENDF tab: 'Apply ZA' keeps Card 5 and the Card 6d atom rows consistent.

The engine requires the principal (Z, A) on Card 4 to be one of the atom
rows. 'Fill structure from phonopy.yaml' fills the rows as natural elements
(A = 0), and the user then names an isotope on the Scattering tab; the
button relabels that element's row to the same nuclide, positions kept.
The contract pinned here:
- the button is the only thing that changes a row; setting ZA does not;
- a row whose constants are the table's own is relabelled without a
  question, a row with custom constants only after a yes, and 'No' changes
  nothing at all (Card 5 included);
- other elements' rows are untouched (BeO: O stays natural);
- several rows of one element are never chosen between;
- a nuclide with no tabulated constants or energy-dependent ones is refused
  before anything changes;
- deck generation offers the same relabel when ZA and the rows disagree,
  and aborts with the engine's message on 'No'.
Requires a display (Tk); skipped headless (CI).
"""
import pytest

tk = pytest.importorskip("tkinter")

NATURAL_C = "6  0  11.907820  6.647200  0.001000  2  0.000000 0.000000 0.250000  0.000000 0.000000 0.750000\n"
NATURAL_BE_O = ("4  0  8.934780  7.790000  0.001800  2  0.333333 0.666667 0.000000  0.666667 0.333333 0.500000\n"
                "8  0  15.857510  5.803000  0.000000  2  0.333333 0.666667 0.375000  0.666667 0.333333 0.875000\n")


@pytest.fixture(scope="module")
def app():
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("no display available for Tk")
    root.withdraw()
    from irma.gui.app import IrmaApp
    application = IrmaApp(root)
    yield application
    root.destroy()


def _set_text(widget, text):
    widget.delete("1.0", "end")
    widget.insert("1.0", text)


def _atoms(app):
    return app.atoms_text.get("1.0", "end-1c").strip()


def _rows(app):
    from irma.gui.deck_text import parse_atoms_text
    return parse_atoms_text(_atoms(app))


@pytest.fixture(autouse=True)
def _quiet_dialogs(app, monkeypatch):
    from irma.gui import endf_form
    monkeypatch.setattr(endf_form.messagebox, "showerror", lambda *a, **k: None)
    monkeypatch.setattr(app, "_ask_yes_no", lambda *a, **k: True)
    app.inelastic_mode_var.set(2)
    app.za_status_var.set("")
    yield


def test_setting_za_changes_no_row(app):
    _set_text(app.atoms_text, NATURAL_C)
    app.za.set("6012")
    assert _atoms(app) == NATURAL_C.strip()


def test_apply_relabels_the_natural_row_to_the_isotope_and_back(app):
    from irma.core.nuclear_data import lookup
    _set_text(app.atoms_text, NATURAL_C)
    app.za.set("6012")
    app._fill_from_za()
    rows = _rows(app)
    c12 = lookup((6, 12))
    assert (rows[0]["Z"], rows[0]["A"]) == (6, 12)
    assert rows[0]["awr"] == pytest.approx(c12.awr, rel=1e-6)
    assert rows[0]["b_coh"] == pytest.approx(c12.b_coh_fm, rel=1e-6)
    assert rows[0]["positions"] == [(0.0, 0.0, 0.25), (0.0, 0.0, 0.75)]
    assert app.awr.get() == f"{c12.awr:.6g}"
    assert "C -> C-12" in app.za_status_var.get()
    assert "phonopy model" in app.za_status_var.get()
    app.za.set("6000")
    app._fill_from_za()
    assert _rows(app)[0]["A"] == 0
    assert "C-12 -> C" in app.za_status_var.get()


def test_apply_again_with_the_same_za_is_a_no_op_for_the_row(app):
    _set_text(app.atoms_text, NATURAL_C)
    app.za.set("6000")
    app._fill_from_za()
    assert _atoms(app).split()[:5] == NATURAL_C.split()[:5]
    assert "already C" in app.za_status_var.get()


def test_custom_constants_need_a_yes_and_no_changes_nothing(app, monkeypatch):
    custom = "6  0  11.898000  6.646000  0.001000  1  0.000000 0.000000 0.000000\n"
    _set_text(app.atoms_text, custom)
    app.awr.set("11.898")
    app.spr.set("4.739")
    app.za.set("6012")
    monkeypatch.setattr(app, "_ask_yes_no", lambda *a, **k: False)
    app._fill_from_za()
    assert _atoms(app) == custom.strip()      # row untouched
    assert app.awr.get() == "11.898"                # Card 5 untouched too
    assert app.za_status_var.get() == "nothing changed"
    monkeypatch.setattr(app, "_ask_yes_no", lambda *a, **k: True)
    app._fill_from_za()
    assert _rows(app)[0]["A"] == 12


def test_other_elements_are_left_alone(app):
    _set_text(app.atoms_text, NATURAL_BE_O)
    app.za.set("4009")
    app._fill_from_za()
    rows = _rows(app)
    assert (rows[0]["Z"], rows[0]["A"]) == (4, 9)
    assert (rows[1]["Z"], rows[1]["A"]) == (8, 0)           # O stays natural
    assert rows[1]["b_coh"] == pytest.approx(5.803)
    app.za.set("8016")
    app._fill_from_za()
    rows = _rows(app)
    assert (rows[0]["Z"], rows[0]["A"]) == (4, 9)           # Be keeps Be-9
    assert (rows[1]["Z"], rows[1]["A"]) == (8, 16)


def test_two_rows_of_one_element_are_never_chosen_between(app):
    two = (NATURAL_C + "6  13  12.891600  6.190000  0.520000  1  0.500000 0.500000 0.500000\n")
    _set_text(app.atoms_text, two)
    app.awr.set("")
    app.za.set("6012")
    app._fill_from_za()
    assert _atoms(app) == two.strip()
    assert app.awr.get() == ""                              # nothing applied
    app.za.set("6013")                                      # an exact match
    app._fill_from_za()
    assert _atoms(app) == two.strip()
    assert app.awr.get() != ""
    assert "row 2 is already C-13" in app.za_status_var.get()


def test_missing_or_energy_dependent_nuclides_are_refused_before_any_change(app):
    _set_text(app.atoms_text, NATURAL_C)
    app.awr.set("")
    app.za.set("6014")                                      # no table entry
    app._fill_from_za()
    assert _atoms(app) == NATURAL_C.strip() and app.awr.get() == ""
    gd = "64  0  155.900000  6.500000  151.000000  1  0.000000 0.000000 0.000000\n"
    _set_text(app.atoms_text, gd)
    app.za.set("64157")                                     # energy-dependent
    app._fill_from_za()
    assert _atoms(app) == gd.strip() and app.awr.get() == ""


def test_hydrogen_to_deuterium_names_the_model_caveat(app):
    h = "1  0  0.999341  -3.740900  80.260000  1  0.000000 0.000000 0.000000\n"
    _set_text(app.atoms_text, h)
    app.za.set("1002")
    app._fill_from_za()
    rows = _rows(app)
    assert rows[0]["A"] == 2 and rows[0]["b_coh"] > 0
    assert "H masses" in app.za_status_var.get()


def test_deck_generation_names_the_mismatch_and_changes_no_row(app):
    import pathlib
    app.nc_phonopy_yaml.set(str(pathlib.Path(__file__).resolve().parent
                                / "mode2_euphonic_n1_validation" / "graphite"
                                / "phonopy.yaml"))
    app.grid_mode.set("manual")
    _set_text(app.alpha_text, "0.1 0.2 0.5")
    _set_text(app.beta_text, "0.0 0.1 0.5")
    for widget, value in zip((app.latt_a, app.latt_b, app.latt_c,
                              app.latt_alpha, app.latt_beta, app.latt_gamma),
                             ("2.46", "2.46", "6.7", "90.0", "90.0", "120.0")):
        widget.set(value)
    _set_text(app.atoms_text, NATURAL_C)
    app.za.set("6012")
    app.awr.set("11.8969")
    app.spr.set("4.7338")
    app.mat.set("28")
    with pytest.raises(ValueError, match="requires a Card 6d atom row"):
        app._generate_input_text()
    assert _rows(app)[0]["A"] == 0                          # no row changed
