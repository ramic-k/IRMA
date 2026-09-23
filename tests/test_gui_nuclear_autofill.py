"""GUI nuclear-data autofill -- ElementTable symbol autofill + ENDF ZA fill.

Pins the three ways the GUI pulls scattering constants from
irma.core.nuclear_data instead of hardcoded literals:

  1. ElementTable.autofill_row -- leaving the symbol cell fills the still-empty
     nuclear columns; user-typed values always win; energy-dependent nuclides
     (B, Cd, Gd, ...) and unknown symbols are never prefilled.
  2. Panel auto-fill from phonopy.yaml -- new rows take the table's constants.
     (The panels ship ONE BLANK row: the scatterer list is material identity.)
  3. EndfFormMixin._fill_from_za -- the explicit button fills AWR + sigma_free
     (spr) from ZA, refusing energy-dependent and unknown nuclides.

Uses a withdrawn root (no display shown); skips cleanly without tkinter.
"""
from unittest import mock

import pytest

tk = pytest.importorskip("tkinter")

from irma.core.nuclear_data import lookup                   # noqa: E402
from irma.gui.element_table import ElementTable, NUCLEAR    # noqa: E402


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
def table(root):
    return ElementTable(root)


# -- 1. symbol autofill -----------------------------------------------------

def _machine_row(table, symbol):
    """A row whose constants were filled from the table, not typed."""
    r = table.add_row({"symbol": symbol})
    assert table.autofill_row(r) is True
    return r


def test_autofill_fills_blank_nuclear_columns(table):
    r = table.add_row({"symbol": "Be"})
    assert table.autofill_row(r) is True
    be = lookup("Be")
    got = {k: float(r["_var"][k].get()) for k in NUCLEAR if k != "symbol"}
    assert got["sigma_bound_b"] == pytest.approx(be.sigma_bound_b, rel=1e-5)
    assert got["awr"] == pytest.approx(be.awr, rel=1e-5)
    assert got["b_coh_fm"] == pytest.approx(be.b_coh_fm, rel=1e-5)
    assert got["sigma_inc_b"] == pytest.approx(be.sigma_inc_b, rel=1e-5)


def test_autofill_never_overwrites_user_values(table):
    r = table.add_row({"symbol": "O", "awr": "99"})
    table.autofill_row(r)
    assert r["_var"]["awr"].get() == "99"                   # user value wins
    assert r["_var"]["b_coh_fm"].get() != ""                # blanks still fill


@pytest.mark.parametrize("symbol", ["Cd", "B", "Gd", "B-10", "Li-6"])
def test_autofill_refuses_energy_dependent_nuclides(table, symbol):
    """B, B-10, and Li-6 carry complex (energy-dependent)
    scattering lengths the upstream table under-flags; the regenerated
    table marks them and autofill must refuse the whole family."""
    r = table.add_row({"symbol": symbol})
    assert table.autofill_row(r) is False
    assert r["_var"]["sigma_bound_b"].get() == ""


def test_autofill_ignores_unknown_and_blank_symbols(table):
    assert table.autofill_row(table.add_row({"symbol": "Xx"})) is False
    assert table.autofill_row(table.add_row({})) is False


def test_isotope_labels_autofill_too(table):
    r = table.add_row({"symbol": "C-13"})
    assert table.autofill_row(r) is True
    assert float(r["_var"]["awr"].get()) == pytest.approx(
        lookup("C-13").awr, rel=1e-5)


def test_symbol_change_refreshes_machine_filled_constants(table):
    """Typing a new symbol over a machine-filled
    row must refresh the constants, not silently keep the old element's."""
    r = _machine_row(table, "C")
    r["_var"]["symbol"].set("Be")
    assert table.autofill_row(r) is True
    be = lookup("Be")
    assert float(r["_var"]["awr"].get()) == pytest.approx(be.awr, rel=1e-5)
    assert float(r["_var"]["b_coh_fm"].get()) == pytest.approx(
        be.b_coh_fm, rel=1e-5)


def test_symbol_change_keeps_user_edited_fields(table):
    r = _machine_row(table, "C")
    r["_var"]["awr"].set("99")                              # user override
    r["_var"]["symbol"].set("Be")
    table.autofill_row(r)
    assert r["_var"]["awr"].get() == "99"                   # user value survives
    assert float(r["_var"]["sigma_bound_b"].get()) == pytest.approx(
        lookup("Be").sigma_bound_b, rel=1e-5)               # machine value refreshed


def test_symbol_change_to_refused_nuclide_blanks_stale_constants(table):
    """C -> Cd: carbon's machine constants must not linger on a Cd row the
    table refuses to prefill."""
    r = _machine_row(table, "C")
    r["_var"]["symbol"].set("Cd")
    assert table.autofill_row(r) is False
    assert r["_var"]["sigma_bound_b"].get() == ""
    assert r["_var"]["awr"].get() == ""


def test_typo_then_correct_symbol_recovers(table):
    r = _machine_row(table, "C")
    r["_var"]["symbol"].set("Xx")                           # typo blanks machine values
    table.autofill_row(r)
    r["_var"]["symbol"].set("Be")                           # corrected
    assert table.autofill_row(r) is True
    assert float(r["_var"]["awr"].get()) == pytest.approx(
        lookup("Be").awr, rel=1e-5)


def test_plain_table_is_unchanged_by_the_nuclide_editor_mode(table, root):
    """The MLIP emit form's nuclear-data editor is an OPT-IN mode of this
    same widget. A table built without the flag must keep the exact key
    set, column set, and controls the NS / NCrystal panels read."""
    from irma.gui.element_table import _KEYS
    assert table._keys == _KEYS
    assert table.nuclide_editor is False
    row = table.add_row({"symbol": "C"})
    assert set(table.get_rows()[0]) == set(_KEYS)
    assert "mode" not in table.get_rows()[0]
    assert row.get("_note") is None
    # the editor's own entry points are inert on a plain table
    table.sync_nuclide_row(row)
    assert set(table.get_rows()[0]) == set(_KEYS)
    # and the "+ Add element" button (which the editor drops) is present
    assert any(str(w.cget("text")) == "+ Add element"
               for f in table.winfo_children()
               for w in f.winfo_children()
               if "text" in w.keys())

    editor = ElementTable(root, nuclide_editor=True)
    assert editor._keys[:3] == ["symbol", "mode", "nuclide"]
    assert set(_KEYS) < set(editor._keys)


# -- 2. panel default rows --------------------------------------------------

def test_phonopy_autofill_pulls_constants_for_new_rows(root, monkeypatch):
    """Auto-fill from phonopy.yaml fills the new rows' nuclear columns from
    the table (blanks only -- an existing user row is carried over intact)."""
    from irma.gui import ns_panel
    from irma.gui.runner import ComputationRunner
    panel = ns_panel.NSPanel(tk.Frame(root), runner=ComputationRunner())
    panel.phonopy_yaml.set("fake.yaml")
    monkeypatch.setattr(ns_panel, "phonopy_species", lambda p: ["Be", "C"])
    panel.element_table.set_rows([{"symbol": "C", "awr": "99"}])
    panel._autofill_from_phonopy()
    rows = {r["symbol"]: r for r in panel.element_table.get_rows()}
    assert set(rows) == {"Be", "C"}
    assert float(rows["Be"]["awr"]) == pytest.approx(lookup("Be").awr, rel=1e-5)
    assert rows["C"]["awr"] == "99"                         # user value kept
    assert float(rows["C"]["b_coh_fm"]) == pytest.approx(
        lookup("C").b_coh_fm, rel=1e-5)                     # blank filled


def test_phonopy_autofill_preserves_provenance_for_later_symbol_edit(
        root, monkeypatch):
    """The machine-filled default row must keep
    its autofill provenance THROUGH the phonopy auto-fill rebuild, so a
    later symbol edit still refreshes the constants (the old
    set_rows(get_rows()) round-trip stripped the record and re-froze
    carbon's constants as if user-typed)."""
    from irma.gui import ns_panel
    from irma.gui.runner import ComputationRunner
    panel = ns_panel.NSPanel(tk.Frame(root), runner=ComputationRunner())
    panel.phonopy_yaml.set("fake.yaml")
    monkeypatch.setattr(ns_panel, "phonopy_species", lambda p: ["C"])
    # The panel ships ONE BLANK row (material identity is the user's to
    # declare); typing the symbol is what makes it a machine-filled carbon row.
    assert panel.element_table.rows[0]["_var"]["symbol"].get() == ""
    panel.element_table.rows[0]["_var"]["symbol"].set("C")
    assert panel.element_table.autofill_row(panel.element_table.rows[0]) is True
    panel._autofill_from_phonopy()
    r = panel.element_table.rows[0]
    r["_var"]["symbol"].set("Be")
    assert panel.element_table.autofill_row(r) is True
    assert float(r["_var"]["awr"].get()) == pytest.approx(
        lookup("Be").awr, rel=1e-5)                         # refreshed, not stale C


def test_set_symbols_drops_unlisted_and_carries_user_rows(table):
    table.add_row({"symbol": "H", "awr": "0.999"})          # user row
    _machine_row(table, "C")                              # machine row
    table.set_symbols(["H", "O"])
    rows = {r["symbol"]: r for r in table.get_rows()}
    assert set(rows) == {"H", "O"}                          # C dropped
    assert rows["H"]["awr"] == "0.999"                      # user value kept
    assert float(rows["O"]["awr"]) == pytest.approx(lookup("O").awr, rel=1e-5)


# -- 3. ENDF form fill-from-ZA ----------------------------------------------

class _FakeEntry:
    """Stands in for LabeledEntry: just .get()/.set() on a string."""

    def __init__(self, value=""):
        self.value = str(value)

    def get(self):
        return self.value

    def set(self, value):
        self.value = str(value)


@pytest.fixture
def form():
    """A bare EndfFormMixin instance with only the fill-from-ZA fields."""
    from irma.gui.endf_form import EndfFormMixin
    h = object.__new__(type("_Harness", (EndfFormMixin,), {}))
    h.za = _FakeEntry("6012")
    h.awr = _FakeEntry("")
    h.spr = _FakeEntry("")
    return h


def test_fill_from_za_sets_awr_and_free_xs(form):
    from irma.gui import endf_form
    with mock.patch.object(endf_form, "messagebox") as mb:
        form._fill_from_za()
    assert not mb.showerror.called
    c12 = lookup((6, 12))
    assert float(form.awr.get()) == pytest.approx(c12.awr, rel=1e-4)
    spr = c12.sigma_bound_b * (c12.awr / (1.0 + c12.awr)) ** 2
    assert float(form.spr.get()) == pytest.approx(spr, rel=1e-4)


def test_fill_from_za_refuses_energy_dependent(form):
    from irma.gui import endf_form
    form.za.set("48000")                                    # natural Cd
    with mock.patch.object(endf_form, "messagebox") as mb:
        form._fill_from_za()
    assert mb.showerror.called
    assert form.awr.get() == "" and form.spr.get() == ""    # untouched


@pytest.mark.parametrize("bad", ["banana", "", "6999", "-6012"])
def test_fill_from_za_reports_unknown_or_malformed(form, bad):
    from irma.gui import endf_form
    form.za.set(bad)
    with mock.patch.object(endf_form, "messagebox") as mb:
        form._fill_from_za()
    assert mb.showerror.called
    assert form.awr.get() == "" and form.spr.get() == ""
