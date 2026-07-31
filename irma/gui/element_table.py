"""A dynamic per-element scatterer table for the Neutron-Scattering panel.

One row per scattering element, built by the user via "+ Add element" (so the
GUI never needs to know the elements ahead of time). Columns appear/disappear by
context: the DOS-file / unit / multiplicity columns only in DOS-file mode 0, the
positions column only when a mode-0 coherent-elastic crystal is in play. Each row
carries its own DOS-file Browse button, so a per-element file pick is well-defined
even though the element set is user-declared.

All columns are always present in the data model (so toggling context never loses
what you typed); the panel's ``build_config`` filters out the columns that don't
apply to the active context.
"""
import tkinter as tk
from tkinter import ttk, filedialog

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


class ElementTable(ttk.Frame):
    """Editable rows of scattering elements with context-dependent columns."""

    def __init__(self, parent, on_browse_filetypes=None):
        super().__init__(parent)
        self._browse_ft = on_browse_filetypes or [
            ("DOS / text", "*.txt *.dat *.dos *.csv"), ("All files", "*.*")]
        self.body = ttk.Frame(self)
        self.body.pack(anchor="w", fill=tk.X)
        self.rows = []                          # list of row dicts
        self._visible = list(_KEYS)
        # header labels live in the SAME grid as the rows so columns align
        self._hdr = {}
        for key, hdr, width in _COLS:
            self._hdr[key] = ttk.Label(self.body, text=hdr, width=width,
                                       anchor="w", foreground="gray")
        btns = ttk.Frame(self)
        btns.pack(anchor="w", pady=(3, 0))
        ttk.Button(btns, text="+ Add element",
                   command=self.add_row).pack(side=tk.LEFT)

    # -- columns ----------------------------------------------------------
    def set_visible(self, keys):
        """Show exactly ``keys`` (in canonical order); relayout."""
        self._visible = [k for k in _KEYS if k in keys]
        self._relayout()

    def _relayout(self):
        """Re-grid all rows after an add or remove."""
        for w in self._hdr.values():
            w.grid_forget()
        for r in self.rows:
            r["_del"].grid_forget()
            for w in r["_w"].values():
                w.grid_forget()
        col = 1
        for key in _KEYS:
            if key in self._visible:
                self._hdr[key].grid(row=0, column=col, padx=1, sticky="w")
                col += 1
        for i, r in enumerate(self.rows, start=1):
            r["_del"].grid(row=i, column=0, padx=1)
            col = 1
            for key in _KEYS:
                if key in self._visible:
                    r["_w"][key].grid(row=i, column=col, padx=1, sticky="w")
                    col += 1

    # -- rows -------------------------------------------------------------
    def add_row(self, data=None):
        """Append a table row (optionally prefilled) and re-layout."""
        data = data or {}
        r = {"_var": {}, "_w": {}}
        for key, _hdr, width in _COLS:
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
            else:
                w = ttk.Entry(self.body, textvariable=var, width=width)
                if key == "symbol":
                    # leaving the symbol cell autofills the still-empty
                    # nuclear columns from the built-in table
                    w.bind("<FocusOut>",
                           lambda _e, rr=r: self.autofill_row(rr))
                    w.bind("<Return>",
                           lambda _e, rr=r: self.autofill_row(rr))
            r["_w"][key] = w
        r["_del"] = ttk.Button(self.body, text="×", width=2,
                               command=lambda rr=r: self._remove(rr))
        self.rows.append(r)
        self._relayout()
        return r

    def autofill_row(self, r):
        """Fill the row's EMPTY nuclear columns from the built-in table.

        Values come from irma.core.nuclear_data (the Rauch-Waschkowski /
        Sears compilation via periodictable; sigma_bound derived from
        b_coh + sigma_inc for in-deck consistency). Only blank fields are
        touched -- anything the user typed wins. Energy-dependent
        nuclides (B, Cd, Gd, ...) are never prefilled: their tabulated
        scattering lengths are resonance-region values unsuitable as
        static constants. Unknown symbols are ignored silently (the
        panel's own validation reports them at build time).

        The row remembers which values IT filled (``r["_autofill"]``).
        When the symbol later changes, still-untouched machine values are
        cleared first so they refresh for the new symbol -- otherwise
        editing the prefilled default row's symbol would silently keep
        the old element's constants. User-edited fields differ from the
        recorded value and are never cleared.
        """
        symbol = r["_var"]["symbol"].get().strip()
        prev = r.get("_autofill")
        if prev and prev["symbol"] != symbol:
            for key, machine_value in prev["values"].items():
                if r["_var"][key].get() == machine_value:
                    r["_var"][key].set("")
            r["_autofill"] = {"symbol": symbol, "values": {}}
        if not symbol:
            return False
        try:
            from irma.core.nuclear_data import lookup
            nuc = lookup(symbol)
        except (KeyError, ValueError, ImportError):
            return False
        if nuc.energy_dependent:
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

    def add_default_row(self, symbol):
        """Prefilled default row whose constants count as machine-filled.

        Unlike ``add_row(nuclear_defaults(symbol))``, the values are
        recorded in the autofill provenance, so typing a different symbol
        over the default refreshes them instead of keeping them.
        """
        r = self.add_row(self.nuclear_defaults(symbol))
        r["_autofill"] = {
            "symbol": symbol,
            "values": {k: r["_var"][k].get()
                       for k in NUCLEAR if k != "symbol"}}
        return r

    @staticmethod
    def nuclear_defaults(symbol):
        """Prefill dict for `add_row` from the built-in nuclear table."""
        from irma.core.nuclear_data import lookup
        nuc = lookup(symbol)
        return {"symbol": symbol,
                "sigma_bound_b": f"{nuc.sigma_bound_b:.6g}",
                "awr": f"{nuc.awr:.6g}",
                "b_coh_fm": f"{nuc.b_coh_fm:.6g}",
                "sigma_inc_b": f"{nuc.sigma_inc_b:.6g}"}

    def set_symbols(self, syms):
        """Rebuild the table with one row per symbol, carrying matching
        existing rows over -- values AND autofill provenance -- then
        autofilling the blank nuclear columns of every row.

        The provenance carry-over matters: rebuilding via
        ``set_rows(get_rows())`` would strip ``_autofill``, and a
        machine-filled row that loses its record silently keeps the old
        element's constants on a later symbol edit."""
        by_sym = {}
        for r in self.rows:
            by_sym.setdefault(r["_var"]["symbol"].get().strip(), r)
        carried = []
        for s in syms:
            old = by_sym.get(s)
            if old is not None:
                data = {k: old["_var"][k].get() for k in _KEYS}
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
        p = filedialog.askopenfilename(filetypes=self._browse_ft)
        if p:
            var.set(p)

    def _remove(self, r):
        """Remove one row and re-layout."""
        r["_del"].destroy()
        for w in r["_w"].values():
            w.destroy()
        self.rows.remove(r)
        self._relayout()

    def clear(self):
        """Remove all rows."""
        for r in list(self.rows):
            self._remove(r)

    # -- data -------------------------------------------------------------
    def get_rows(self):
        """List of dicts (all columns, raw strings) -- the panel filters by context."""
        return [{k: r["_var"][k].get().strip() for k in _KEYS} for r in self.rows]

    def set_rows(self, rows):
        """Replace the table contents with ``rows``."""
        self.clear()
        for d in rows:
            self.add_row(d)
