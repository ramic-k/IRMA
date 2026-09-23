"""ENDF tab: 'Fill structure from phonopy.yaml'.

The button prefills the iel=10 Lattice Parameters and the whole 'Atom Types
in Unit Cell' block from the phonopy model named on Card 6f. The contract
this file pins:

- it lives under the Card 6f selector, so it is on screen only for the
  phonopy-backed inelastic modes (1/2), which are pinned to iel=10;
- it NEVER fires by itself -- not when a phonopy.yaml is selected, not on
  deck import;
- the preview is a real gate: Cancel mutates nothing;
- Apply is ATOMIC -- lattice and atom block are replaced together, never one
  without the other;
- a missing path / unreadable yaml reports and mutates nothing;
- what it writes generates a deck the core parser accepts.

Requires a display (Tk); skipped headless (CI).
"""
import pathlib

import pytest

tk = pytest.importorskip("tkinter")

GRAPHITE_YAML = str(pathlib.Path(__file__).resolve().parent
                    / "mode2_euphonic_n1_validation" / "graphite"
                    / "phonopy.yaml")

SENTINEL_ATOMS = "1 1 0.999 -3.74 80.26 1 0.1 0.2 0.3\n"
SENTINEL_LATTICE = ("1.111", "2.222", "3.333", "44.4", "55.5", "66.6")


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


def _lattice(app):
    return tuple(w.get() for w in (app.latt_a, app.latt_b, app.latt_c,
                                   app.latt_alpha, app.latt_beta,
                                   app.latt_gamma))


def _atoms(app):
    # end-1c drops the newline Tk keeps at the end of every Text widget.
    return app.atoms_text.get("1.0", "end-1c")


def _set_sentinels(app):
    """A hand-typed structure the fill must replace wholesale or not at all."""
    for widget, value in zip((app.latt_a, app.latt_b, app.latt_c,
                              app.latt_alpha, app.latt_beta, app.latt_gamma),
                             SENTINEL_LATTICE):
        widget.set(value)
    _set_text(app.atoms_text, SENTINEL_ATOMS)


def _setup_mode2(app, yaml_path=GRAPHITE_YAML):
    """A runnable iel=10 / inelastic_mode=2 form pointed at a phonopy model."""
    app._reset_form_to_defaults()
    app.iel_var.set("10 — Generalized (crystal structure)")
    app.inelastic_mode_var.set(2)
    app.mat.set("1")
    app.za.set("6000")
    app.awr.set("11.907856")
    app.spr.set("4.724629")
    app.npr.set("1")
    app.nphon.set("4")
    app.elastic_mode.set("1 — SEF (Single-channel Elastic Format)")
    app.coh_edge_group_bpd.set("0")
    app.temps_var.set("296.0")
    app.grid_mode.set("manual")
    _set_text(app.alpha_text, "0.05 1.0 8.0")
    _set_text(app.beta_text, "0.0 0.6 2.0 6.0")
    app.nc_phonopy_yaml.set(yaml_path)
    app.nc_mesh_nx.set("8")
    app.nc_mesh_ny.set("8")
    app.nc_mesh_nz.set("8")
    app.nc_ncpu.set("1")
    app.nc_num_directions.set("100")
    app.nc_multiphonon_num_directions.set("100")


def _confirm(app, monkeypatch, answer, record=None):
    """Drive the preview decision without opening a real dialog."""
    def fake(path, lattice_fields, atom_rows, warnings):
        if record is not None:
            record.append((path, lattice_fields, atom_rows, warnings))
        return answer
    monkeypatch.setattr(app, "_confirm_structure_fill", fake)


def _silence_messagebox(monkeypatch):
    """Capture the error/warning dialogs instead of showing them."""
    from irma.gui import endf_form
    seen = []

    class _Box:
        @staticmethod
        def showerror(title, message):
            seen.append(("error", title, message))

        @staticmethod
        def showwarning(title, message):
            seen.append(("warning", title, message))

        @staticmethod
        def showinfo(title, message):
            seen.append(("info", title, message))

    monkeypatch.setattr(endf_form, "messagebox", _Box)
    return seen


# ------------------------------------------------------ visibility ----------

def _visible(widget):
    """True when the widget and every ancestor are still managed.

    ``winfo_ismapped`` is 0 for everything under a withdrawn root, and the
    button's own geometry manager does not change when an ANCESTOR is
    pack_forget()ed -- so walk the chain.
    """
    w = widget
    while w is not None and not isinstance(w, tk.Tk):
        manager = w.winfo_manager()
        if not manager:
            return False
        if manager not in ("pack", "grid", "place"):
            return True     # held by a container widget (notebook tab, canvas)
        w = w.master
    return True


def test_button_shown_only_for_phonopy_modes(app):
    app._reset_form_to_defaults()
    app.iel_var.set("10 — Generalized (crystal structure)")

    app.inelastic_mode_var.set(0)
    assert not _visible(app._fill_structure_btn)

    for mode in (1, 2):
        app.inelastic_mode_var.set(mode)
        assert _visible(app._fill_structure_btn)
        # modes 1/2 are pinned to iel=10, so the button is only ever on
        # screen for the decks whose Card 6c/6d it writes
        assert app._code(app.iel_var) == 10

    app.inelastic_mode_var.set(0)
    app.iel_var.set("1 — Graphite (legacy)")
    assert not _visible(app._fill_structure_btn)
    app._reset_form_to_defaults()


# ------------------------------------------------------ never automatic -----

def test_selecting_a_phonopy_path_fills_nothing(app):
    """Card 6f is a path field, not a trigger."""
    _setup_mode2(app, yaml_path="")
    _set_sentinels(app)
    app.nc_phonopy_yaml.set(GRAPHITE_YAML)
    assert _lattice(app) == SENTINEL_LATTICE
    assert _atoms(app) == SENTINEL_ATOMS


def test_deck_import_fills_nothing(app, tmp_path):
    """Importing a mode-2 deck that NAMES a phonopy.yaml must apply the
    deck's own Card 6c/6d, never the model's."""
    _setup_mode2(app)
    app.za.set("6012")          # the row below is C-12: the deck must agree
    _set_text(app.atoms_text,
              "6 12 11.907856 6.6484 0.001 2  0 0 0.25  0 0 0.75")
    for widget, value in zip((app.latt_a, app.latt_b, app.latt_c,
                              app.latt_alpha, app.latt_beta, app.latt_gamma),
                             ("2.46", "2.46", "6.7", "90.0", "90.0", "120.0")):
        widget.set(value)
    deck = tmp_path / "mode2.input"
    deck.write_text(app._generate_input_text())

    app._reset_form_to_defaults()
    app._import_leapr_from_path(str(deck))
    assert app.nc_phonopy_yaml.get() == GRAPHITE_YAML   # model IS named
    atoms = _atoms(app)
    assert atoms.split()[:6] == ["6", "12", "11.907856", "6.6484",
                                 "0.001", "2"]
    assert "0.333333" not in atoms          # the model's 4 sites did not leak
    assert _lattice(app)[2] == "6.7"


# ------------------------------------------------------ preview gate --------

def test_cancel_mutates_nothing(app, monkeypatch):
    pytest.importorskip("phonopy")
    _setup_mode2(app)
    _set_sentinels(app)
    _confirm(app, monkeypatch, answer=False)
    app._fill_structure_from_phonopy()
    assert _lattice(app) == SENTINEL_LATTICE
    assert _atoms(app) == SENTINEL_ATOMS


def test_apply_replaces_lattice_and_atom_block(app, monkeypatch):
    pytest.importorskip("phonopy")
    _setup_mode2(app)
    _set_sentinels(app)
    seen = []
    _confirm(app, monkeypatch, answer=True, record=seen)
    app._fill_structure_from_phonopy()

    assert _lattice(app) == ("2.460600", "2.460600", "6.705000",
                             "90.000000", "90.000000", "120.000000")
    rows = [ln for ln in _atoms(app).strip().split("\n") if ln.strip()]
    assert len(rows) == 1                       # one row per distinct species
    fields = rows[0].split()
    assert fields[0] == "6"                     # Z
    assert fields[1] == "0"                     # natural element, not 12
    assert fields[5] == "4"                     # npos: the 4 primitive sites
    assert len(fields) == 6 + 3 * 4

    # The preview it asked the user to confirm is exactly what was written.
    (_path, lattice_fields, atom_rows, warnings) = seen[0]
    assert tuple(lattice_fields) == _lattice(app)
    assert atom_rows == rows
    assert any("natural element" in w for w in warnings)


def test_preview_text_states_the_assumptions(app):
    text = app._structure_fill_preview_text(
        GRAPHITE_YAML,
        ("2.460600", "2.460600", "6.705000",
         "90.000000", "90.000000", "120.000000"),
        ["6  0  11.907820  6.647200  0.001000  1  0.000000 0.000000 0.000000"],
        ["C: filled as the natural element (za=6000, A=0) with "
         "natural-abundance constants"])
    assert "NATURAL ELEMENT" in text
    assert "A = 0" in text
    assert "deuterium" in text                  # elements, not isotopes
    assert "Apply" in text and "relabels" in text   # isotopes come from ZA
    assert "2.460600" in text and "11.907820" in text


def _drive_real_dialog(app, button_text):
    """Press a button in the REAL preview dialog once it is up.

    Scheduled with ``after`` so it runs inside ``wait_window``'s event loop.
    Any failure destroys the dialog anyway, so a broken lookup fails the
    assertion instead of hanging the suite.
    """
    from tkinter import ttk
    from irma.gui import endf_form

    def act():
        top = None
        try:
            top = next(w for w in app.root.winfo_children()
                       if isinstance(w, tk.Toplevel)
                       and w.title() == endf_form.STRUCTURE_FILL_TITLE)
            buttons = []

            def _walk(w):
                for child in w.winfo_children():
                    if isinstance(child, ttk.Button):
                        buttons.append(child)
                    _walk(child)

            _walk(top)
            next(b for b in buttons
                 if str(b.cget("text")) == button_text).invoke()
        except Exception:
            if top is not None:
                top.destroy()

    app.root.after(50, act)


@pytest.mark.parametrize("button_text, expect_applied",
                         [("Cancel", False), ("Apply", True)])
def test_real_preview_dialog_apply_and_cancel(app, button_text,
                                              expect_applied):
    """The dialog itself, not the test seam: build it and press its buttons."""
    pytest.importorskip("phonopy")
    _setup_mode2(app)
    _set_sentinels(app)
    from irma.core.crystal_input import (
        format_card6d_row, format_lattice_fields, species_from_sites)
    from irma.core.phonopy_io import load_phonopy_primitive_structure

    structure = load_phonopy_primitive_structure(GRAPHITE_YAML)
    species, warnings = species_from_sites(structure.symbols,
                                           structure.scaled_positions)
    lattice_fields = format_lattice_fields(structure.cellpar)
    atom_rows = [format_card6d_row(sp) for sp in species]

    _drive_real_dialog(app, button_text)
    applied = app._confirm_structure_fill(GRAPHITE_YAML, lattice_fields,
                                          atom_rows, warnings)
    assert applied is expect_applied
    # The dialog itself is a pure question: it never touches the form.
    assert _lattice(app) == SENTINEL_LATTICE
    assert _atoms(app) == SENTINEL_ATOMS


# ------------------------------------------------------ failure paths -------

def test_no_path_reports_and_mutates_nothing(app, monkeypatch):
    _setup_mode2(app, yaml_path="")
    _set_sentinels(app)
    seen = _silence_messagebox(monkeypatch)
    _confirm(app, monkeypatch, answer=True)     # would apply if it got there
    app._fill_structure_from_phonopy()
    assert seen and seen[0][0] == "error"
    assert "No phonopy.yaml" in seen[0][2]
    assert _lattice(app) == SENTINEL_LATTICE
    assert _atoms(app) == SENTINEL_ATOMS


def test_missing_file_reports_and_mutates_nothing(app, monkeypatch):
    _setup_mode2(app, yaml_path="/nonexistent/phonopy.yaml")
    _set_sentinels(app)
    seen = _silence_messagebox(monkeypatch)
    _confirm(app, monkeypatch, answer=True)
    app._fill_structure_from_phonopy()
    assert seen and seen[0][0] == "error"
    assert "does not exist" in seen[0][2]
    assert _atoms(app) == SENTINEL_ATOMS


def test_unreadable_yaml_reports_and_mutates_nothing(app, monkeypatch,
                                                     tmp_path):
    pytest.importorskip("phonopy")
    bad = tmp_path / "phonopy.yaml"
    bad.write_text("this: is not a phonopy model\n")
    _setup_mode2(app, yaml_path=str(bad))
    _set_sentinels(app)
    seen = _silence_messagebox(monkeypatch)
    _confirm(app, monkeypatch, answer=True)
    app._fill_structure_from_phonopy()
    assert seen and seen[0][0] == "error"
    assert _lattice(app) == SENTINEL_LATTICE
    assert _atoms(app) == SENTINEL_ATOMS


def test_phonopy_missing_reports_and_mutates_nothing(app, monkeypatch):
    """No phonopy installed: a clear dialog, not a traceback."""
    from irma.gui import endf_form            # noqa: F401  (module under test)
    import irma.core.phonopy_io as pio

    def boom(_path):
        raise ImportError("phonopy is required")

    monkeypatch.setattr(pio, "load_phonopy_primitive_structure", boom)
    _setup_mode2(app)
    _set_sentinels(app)
    seen = _silence_messagebox(monkeypatch)
    _confirm(app, monkeypatch, answer=True)
    app._fill_structure_from_phonopy()
    assert seen and "phonopy" in seen[0][1].lower()
    assert _atoms(app) == SENTINEL_ATOMS


def test_energy_dependent_species_is_refused(app, monkeypatch, tmp_path):
    """Gd's tabulated length is a resonance-region value: refuse, don't fill."""
    import irma.core.phonopy_io as pio
    import numpy as np

    def fake(_path):
        return pio.PhonopyPrimitiveStructure(
            cellpar=(5.0, 5.0, 5.0, 90.0, 90.0, 90.0),
            lattice_ang=np.eye(3) * 5.0,
            symbols=["Gd"],
            scaled_positions=np.zeros((1, 3)))

    monkeypatch.setattr(pio, "load_phonopy_primitive_structure", fake)
    _setup_mode2(app)
    _set_sentinels(app)
    seen = _silence_messagebox(monkeypatch)
    _confirm(app, monkeypatch, answer=True)
    app._fill_structure_from_phonopy()
    assert seen and "ENERGY-DEPENDENT" in seen[0][2]
    assert _atoms(app) == SENTINEL_ATOMS


# ------------------------------------------------------ end to end ----------

def test_filled_form_generates_a_deck_the_core_parser_accepts(app, monkeypatch,
                                                              tmp_path):
    pytest.importorskip("phonopy")
    from irma.core.deck import TokenReader, parse_leapr_input
    from irma.gui.deck_text import parse_deck_to_staging

    _setup_mode2(app)
    _set_sentinels(app)
    _confirm(app, monkeypatch, answer=True)
    app._fill_structure_from_phonopy()

    deck = tmp_path / "filled.input"
    deck.write_text(app._generate_input_text())
    tokens, lines, _start, token_lines = parse_leapr_input(str(deck))
    reader = TokenReader(tokens, token_lines=token_lines,
                         filename=str(deck), raw_lines=lines)
    staged = parse_deck_to_staging(reader, str(deck))

    assert staged["iel"] == 10
    assert staged["inelastic_mode"] == 2
    assert staged["lattice"] == pytest.approx(
        [2.4606, 2.4606, 6.705, 90.0, 90.0, 120.0])
    assert len(staged["atoms"]) == 1
    atom = staged["atoms"][0]
    assert (atom["Z"], atom["A"]) == (6, 0)
    assert atom["npos"] == 4
    # Card 4's za must name a Card 6d group; A = 0 is why za = 6000 works.
    assert int(staged["za"]) == 1000 * atom["Z"] + atom["A"]
