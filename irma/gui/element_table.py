"""A dynamic per-element scatterer table for the Neutron-Scattering panel.

One row per scattering element, added by the user with "+ Add element". The
DOS-file, unit and multiplicity columns show only in DOS-file mode 0, the
positions column only when a mode-0 coherent-elastic crystal is used; every
column stays in the data model, and the panel's ``build_config`` keeps the
ones that apply. ``ElementTable(parent, nuclide_editor=True)`` is the MLIP emit
form's per-species nuclear-data editor, with two extra columns (``mode`` and
``nuclide``); see ``set_species``.
"""
import tkinter as tk
from tkinter import ttk, filedialog

from irma.core.nuclear_data import isotopes, lookup

# (key, header, entry width in chars)
_COLS = [
    ("symbol", "Sym", 5),
    ("sigma_bound_b", "σ_bound[b]", 10),
    ("awr", "AWR", 8),
    ("b_coh_fm", "b_coh[fm]", 9),
    ("sigma_inc_b", "σ_inc[b]", 9),
    ("multiplicity", "mult", 5),
    ("dos_unit", "unit", 6),
    ("dos_file", "DOS file", 26),
    ("positions", "positions (x y z …)", 22),
]
_KEYS = [c[0] for c in _COLS]
_UNITS = ["meV", "eV", "cm-1", "THz"]
# the always-on nuclear columns (visible in every context)
NUCLEAR = ("symbol", "sigma_bound_b", "awr", "b_coh_fm", "sigma_inc_b")
DOS_COLS = ("multiplicity", "dos_unit", "dos_file")
POS_COL = ("positions",)

# ---------------------------------------------------------------------------
# Nuclear-data-editor mode (the MLIP emit form). These two columns are not in
# _COLS, which the NS and NCrystal tables read as a fixed key set; an editor
# instance adds them to its own column list (self._cols / self._keys).
# ---------------------------------------------------------------------------
_NUCLIDE_COLS = [("mode", "source", 8), ("nuclide", "isotope", 9)]
MODE_NATURAL = "natural"
MODE_ISOTOPE = "isotope"
MODE_CUSTOM = "custom"
MODES = (MODE_NATURAL, MODE_ISOTOPE, MODE_CUSTOM)
#: the columns the editor shows (pass to ``set_visible``; the table lays
#: them out in its own canonical order: symbol, mode, nuclide, then the
#: nuclear constants)
NUCLIDE_EDITOR_COLS = ("symbol", "mode", "nuclide",
                       "awr", "b_coh_fm", "sigma_inc_b")
#: the constants a `custom` row can supply, in emitted-argument order
CUSTOM_FIELDS = ("b_coh_fm", "sigma_inc_b", "awr")
#: relative mass disagreement (isotope vs the bundle's phonopy mass) that
#: triggers a row warning. 1% separates a genuine isotope substitution
#: (13-C is 8% heavier than natural C, 2-H is 100% heavier than 1-H) from
#: the rounding between an abundance-averaged mass and a tabulated one.
MASS_WARN_REL = 0.01
_WARN_COLOR = "#e5a145"


def _safe_lookup(key):
    """``nuclear_data.lookup`` that returns None instead of raising."""
    if not key:
        return None
    try:
        return lookup(key)
    except (KeyError, ValueError):
        return None


class ElementTable(ttk.Frame):
    """Editable rows of scattering elements with context-dependent columns."""

    def __init__(self, parent, nuclide_editor=False):
        super().__init__(parent)
        self.nuclide_editor = bool(nuclide_editor)
        # the editor's two columns sit straight after the symbol, so the row
        # reads left-to-right as identity then constants
        self._cols = (_COLS[:1] + _NUCLIDE_COLS + _COLS[1:]
                      if self.nuclide_editor else list(_COLS))
        self._keys = [c[0] for c in self._cols]
        # editor mode: masses read from the bundle's phonopy.yaml, keyed by
        # symbol, used only for the isotope-vs-model mass warning
        self._masses = {}
        # a short line shown instead of an empty grid (editor mode: "no
        # bundle selected yet"); never both
        self._hint = ttk.Label(self, foreground="gray", justify=tk.LEFT,
                               wraplength=520)
        self.body = ttk.Frame(self)
        self.body.pack(anchor="w", fill=tk.X)
        self.rows = []                          # list of row dicts
        self._visible = list(self._keys)
        # header labels live in the SAME grid as the rows so columns align
        self._hdr = {}
        for key, hdr, width in self._cols:
            self._hdr[key] = ttk.Label(self.body, text=hdr, width=width,
                                       anchor="w", foreground="gray")
        btns = ttk.Frame(self)
        # The editor's rows ARE the bundle's species: adding or deleting one
        # would describe a material the phonon model does not contain, so
        # neither control exists there.
        if not self.nuclide_editor:
            btns.pack(anchor="w", pady=(3, 0))
            ttk.Button(btns, text="+ Add element",
                       command=self.add_row).pack(side=tk.LEFT)

    # -- columns ----------------------------------------------------------
    def set_visible(self, keys):
        """Show exactly ``keys`` (in canonical order); relayout."""
        self._visible = [k for k in self._keys if k in keys]
        self._relayout()

    def set_hint(self, text):
        """Show ``text`` in place of the grid (empty string hides it)."""
        self._hint.config(text=text or "")
        if text:
            self._hint.pack(anchor="w", before=self.body)
        else:
            self._hint.pack_forget()

    def _relayout(self):
        """Re-grid all rows after an add or remove."""
        for w in self._hdr.values():
            w.grid_forget()
        for r in self.rows:
            r["_del"].grid_forget()
            for w in r["_w"].values():
                w.grid_forget()
            if r.get("_note") is not None:
                r["_note"].grid_forget()
        col = 1
        for key in self._keys:
            if key in self._visible:
                self._hdr[key].grid(row=0, column=col, padx=1, sticky="w")
                col += 1
        ncol = col - 1
        for i, r in enumerate(self.rows):
            # editor rows own two grid lines: the fields, then the row's own
            # note (warnings belong to the row that caused them, and a note
            # gridded beside the fields would widen the whole form column)
            grow = 1 + (2 * i if self.nuclide_editor else i)
            if not self.nuclide_editor:
                r["_del"].grid(row=grow, column=0, padx=1)
            col = 1
            for key in self._keys:
                if key in self._visible:
                    if not self._cell_hidden(r, key):
                        r["_w"][key].grid(row=grow, column=col, padx=1,
                                          sticky="w")
                    col += 1
            note = r.get("_note")
            if note is not None and note.cget("text"):
                note.grid(row=grow + 1, column=1, columnspan=max(1, ncol),
                          padx=1, pady=(0, 3), sticky="w")

    def _cell_hidden(self, r, key):
        """Editor mode: semantic disclosure inside a row.

        The isotope selector appears for an `isotope` row, and stays on a
        `custom` row that already carries an isotope -- the energy-dependent
        recovery below flips such a row to `custom`, and hiding the selector
        there would silently discard the identity the user picked.
        """
        if not self.nuclide_editor or key != "nuclide":
            return False
        mode = r["_var"]["mode"].get()
        if mode == MODE_ISOTOPE:
            return False
        return not (mode == MODE_CUSTOM and r["_var"]["nuclide"].get().strip())

    # -- rows -------------------------------------------------------------
    def add_row(self, data=None):
        """Append a table row (optionally prefilled) and re-layout."""
        data = data or {}
        r = {"_var": {}, "_w": {}}
        for key, _hdr, width in self._cols:
            val = data.get(key)
            var = tk.StringVar(value="" if val is None else str(val))
            r["_var"][key] = var
            if key == "dos_unit":
                if not var.get():
                    var.set("meV")
                w = ttk.Combobox(self.body, textvariable=var, values=_UNITS,
                                 width=width, state="readonly")
            elif key == "dos_file":
                cell = ttk.Frame(self.body)
                ttk.Entry(cell, textvariable=var, width=width).pack(side=tk.LEFT)
                ttk.Button(cell, text="…", width=2,
                           command=lambda v=var: self._browse(v)).pack(side=tk.LEFT)
                w = cell
            elif key == "mode":
                if not var.get():
                    var.set(MODE_NATURAL)
                w = ttk.Combobox(self.body, textvariable=var, values=list(MODES),
                                 width=width, state="readonly")
                w.bind("<<ComboboxSelected>>",
                       lambda _e, rr=r: self.sync_nuclide_row(rr))
            elif key == "nuclide":
                w = ttk.Combobox(self.body, textvariable=var, values=[],
                                 width=width, state="readonly")
                w.bind("<<ComboboxSelected>>",
                       lambda _e, rr=r: self.sync_nuclide_row(rr))
            else:
                w = ttk.Entry(self.body, textvariable=var, width=width)
                if key == "symbol":
                    if self.nuclide_editor:
                        # the symbol comes from the bundle and is not a
                        # user field there
                        w.config(state="readonly")
                    else:
                        # leaving the symbol cell autofills the still-empty
                        # nuclear columns from the built-in table
                        w.bind("<FocusOut>",
                               lambda _e, rr=r: self.autofill_row(rr))
                        w.bind("<Return>",
                               lambda _e, rr=r: self.autofill_row(rr))
            r["_w"][key] = w
        r["_del"] = ttk.Button(self.body, text="×", width=2,
                               command=lambda rr=r: self._remove(rr))
        if self.nuclide_editor:
            r["_note"] = ttk.Label(self.body, text="", justify=tk.LEFT,
                                   wraplength=520, foreground=_WARN_COLOR)
        self.rows.append(r)
        self._relayout()
        return r

    def autofill_row(self, r):
        """Fill the row's empty nuclear columns from irma.core.nuclear_data.

        Only blank fields are filled, so values the user typed win.
        Energy-dependent nuclides (B, Cd, Gd, ...) are never prefilled, and
        unknown symbols are left for the panel's validation. The row records
        the values it filled (``r["_autofill"]``); when the symbol changes,
        those still unedited are cleared so they refresh for the new symbol.
        """
        symbol = r["_var"]["symbol"].get().strip()
        prev = r.get("_autofill")
        if prev and prev["symbol"] != symbol:
            for key, machine_value in prev["values"].items():
                if r["_var"][key].get() == machine_value:
                    r["_var"][key].set("")
            r["_autofill"] = {"symbol": symbol, "values": {}}
        nuc = _safe_lookup(symbol)
        if nuc is None or nuc.energy_dependent:
            return False
        filled = {}
        for key, value in (("sigma_bound_b", nuc.sigma_bound_b),
                           ("awr", nuc.awr),
                           ("b_coh_fm", nuc.b_coh_fm),
                           ("sigma_inc_b", nuc.sigma_inc_b)):
            var = r["_var"][key]
            if not var.get().strip():
                var.set(f"{value:.6g}")
                filled[key] = var.get()
        if filled:
            record = r.setdefault("_autofill", {"symbol": symbol, "values": {}})
            record["symbol"] = symbol
            record["values"].update(filled)
        return bool(filled)

    def set_symbols(self, syms):
        """Rebuild the table with one row per symbol, carrying matching rows
        over with their values and autofill record (so a later symbol edit
        still refreshes machine-filled values), then autofill blank columns."""
        by_sym = {}
        for r in self.rows:
            by_sym.setdefault(r["_var"]["symbol"].get().strip(), r)
        carried = []
        for s in syms:
            old = by_sym.get(s)
            if old is not None:
                data = {k: old["_var"][k].get() for k in self._keys}
                rec = old.get("_autofill")
                rec = ({"symbol": rec["symbol"], "values": dict(rec["values"])}
                       if rec else None)
            else:
                data, rec = {"symbol": s}, None
            carried.append((data, rec))
        self.clear()
        for data, rec in carried:
            r = self.add_row(data)
            if rec:
                r["_autofill"] = rec
            self.autofill_row(r)

    def _browse(self, var):
        """Open a file dialog for the row's DOS-file column."""
        p = filedialog.askopenfilename(filetypes=[
            ("DOS / text", "*.txt *.dat *.dos *.csv"), ("All files", "*.*")])
        if p:
            var.set(p)

    def _remove(self, r):
        """Remove one row and re-layout."""
        r["_del"].destroy()
        for w in r["_w"].values():
            w.destroy()
        if r.get("_note") is not None:
            r["_note"].destroy()
        self.rows.remove(r)
        self._relayout()

    def clear(self):
        """Remove all rows."""
        for r in list(self.rows):
            self._remove(r)

    # -- data -------------------------------------------------------------
    def get_rows(self):
        """List of dicts (all columns, raw strings) -- the panel filters by context."""
        return [{k: r["_var"][k].get().strip() for k in self._keys}
                for r in self.rows]

    def set_rows(self, rows):
        """Replace the table contents with ``rows``."""
        self.clear()
        for d in rows:
            self.add_row(d)

    # -- nuclear-data editor ----------------------------------------------
    def set_species(self, symbols, masses=None):
        """Editor mode: one row per species, each starting as Natural (which
        emits nothing, so an untouched table gives the CLI's default emission).

        ``masses`` (amu per species) is used only for the mass warning; an
        empty ``symbols`` clears the table. A species whose record is
        energy-dependent or missing opens as Custom.
        """
        self._masses = {str(k): float(v) for k, v in (masses or {}).items()
                        if isinstance(v, (int, float))}
        self.clear()
        for sym in symbols:
            r = self.add_row({"symbol": sym, "mode": MODE_NATURAL})
            iso = self._isotope_labels(sym)
            r["_w"]["nuclide"].config(values=iso)
            self.sync_nuclide_row(r, relayout=False)
        self._relayout()

    @staticmethod
    def _isotope_labels(symbol):
        """['12-C', '13-C'] -- the labels ``--nuclide`` and lookup() take."""
        return [f"{n.A}-{n.symbol}" for n in isotopes(symbol)]

    def sync_nuclide_row(self, r, relayout=True):
        """Re-resolve one editor row after a mode or isotope change.

        The constants shown are the ones that will be emitted: read-only
        from the resolved entry for Natural and Isotope, editable for Custom.
        An entry flagged energy-dependent (``Nuclide.energy_dependent``) or
        missing switches the row to Custom with blank boxes and a note, so the
        user fixes it in this row before anything is emitted. An isotope mass
        that differs from the model's mass adds a warning (it does not block).
        """
        if not self.nuclide_editor:
            return
        var = r["_var"]
        sym = var["symbol"].get().strip()
        mode = var["mode"].get() or MODE_NATURAL
        if mode == MODE_NATURAL:
            var["nuclide"].set("")
        label = var["nuclide"].get().strip()
        base = _safe_lookup(label or sym)
        flagged = base is None or base.energy_dependent
        if flagged and mode != MODE_CUSTOM:
            mode = MODE_CUSTOM
            var["mode"].set(MODE_CUSTOM)
        was = r.get("_mode_shown")
        r["_mode_shown"] = mode
        r["_flagged"] = bool(flagged)

        if mode in (MODE_NATURAL, MODE_ISOTOPE):
            for key in CUSTOM_FIELDS:
                var[key].set(f"{getattr(base, key):.6g}")
        elif was != MODE_CUSTOM:
            # entering Custom: seed from the table when that is safe, and
            # blank when it is not (an energy-dependent record is never a
            # silent prefill)
            for key in CUSTOM_FIELDS:
                var[key].set("" if flagged else f"{getattr(base, key):.6g}")
        editable = "normal" if mode == MODE_CUSTOM else "readonly"
        for key in CUSTOM_FIELDS:
            r["_w"][key].config(state=editable)

        notes = []
        if base is None:
            notes.append(
                f"no tabulated neutron data for {label or sym!r}: enter "
                f"b_coh and sigma_inc yourself.")
        elif base.energy_dependent:
            notes.append(
                f"{label or sym}: the tabulated scattering length is "
                f"ENERGY-DEPENDENT (a resonance-region value), not a safe "
                f"static prefill. Enter b_coh_fm and sigma_inc_b for the "
                f"energy range you are evaluating.")
        if r.get("_flagged") and not self._custom_complete(r):
            notes.append("values still needed before this can be emitted.")
        notes += self._mass_note(sym, label, base)
        note = r.get("_note")
        if note is not None:
            note.config(text=" ".join(notes))
        if relayout:
            self._relayout()

    @staticmethod
    def _custom_complete(r):
        """Both scattering constants present on a Custom row."""
        return bool(r["_var"]["b_coh_fm"].get().strip()
                    and r["_var"]["sigma_inc_b"].get().strip())

    def _mass_note(self, symbol, label, base):
        """Isotope mass vs the model's mass for the species (e.g. 2-H on a
        hydrogen bundle): an isotope changes the scattering constants but not
        the masses in the bundle's phonopy.yaml, which the Debye-Waller
        factors use. A warning only; the fix is to rebuild the bundle.
        """
        if not label or base is None:
            return []
        model_mass = self._masses.get(symbol)
        if not model_mass:
            return []
        from irma.core.constants import AMASSN
        iso_mass = base.awr * AMASSN
        if abs(iso_mass - model_mass) <= MASS_WARN_REL * abs(model_mass):
            return []
        return [f"mass check: {label} is {iso_mass:.4g} amu but the bundle's "
                f"phonopy.yaml has {model_mass:.4g} amu for {symbol}; rebuild "
                f"the bundle to change the masses."]

    def nuclide_rows(self):
        """Editor rows as plain dicts, for the panel's argv assembly.

        Each dict carries the visible columns plus ``flagged`` (the
        resolved record is energy-dependent or absent) and ``complete``
        (both scattering constants filled in).
        """
        out = []
        for r in self.rows:
            d = {k: r["_var"][k].get().strip() for k in self._keys}
            d["flagged"] = bool(r.get("_flagged"))
            d["complete"] = self._custom_complete(r)
            out.append(d)
        return out
