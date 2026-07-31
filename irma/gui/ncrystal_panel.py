"""NCrystal-plugin panel for the IRMA GUI.

A thin controller for IRMA's third capability: exporting the per-temperature
NCrystal scattering data (``.irmapack`` files + a ``@CUSTOM_IRMA`` NCMAT snippet)
that wire an IRMA mode-2 (coherent one-phonon + anisotropic Debye-Waller)
calculation into NCrystal. It collects inputs into an
:class:`~irma.ncrystal.NCrystalExportConfig` and runs ``irma.ncrystal.write_packs``
-- it reimplements no export logic.

``build_config()`` is a pure widget<->config mapping (no event loop needed), so
it is unit-testable against a withdrawn Tk root. The export runs out-of-process
through the shared :class:`~irma.gui.runner.ComputationRunner` (the same
cancellable subprocess path the Neutron-Scattering panel uses), driving the
``python -m irma.ncrystal`` CLI, which calls ``write_packs`` and streams its
``progress`` markers to this panel's log.
"""

import os
import sys
import tempfile
import tkinter as tk
from tkinter import ttk, messagebox

from irma.gui.widgets import (
    LabeledEntry, LabeledCombobox, FileSelector, ScrolledText, InfoLabel,
    form_section, init_form_styles,
    parse_float, parse_int)
from irma.gui.grid_form import SabGridForm
from irma.gui.element_table import ElementTable, NUCLEAR
from irma.ncrystal.config import NCrystalExportConfig
from irma.spectra.config import SpectraConfigError


# inelastic-mode dropdown labels (the leading digit is parsed back to the int).
_INELASTIC_LABELS = ["2 (coherent 1ph + multi)", "1 (incoherent approx)"]
_INELASTIC_BY_INT = {1: "1 (incoherent approx)", 2: "2 (coherent 1ph + multi)"}


# ---------------------------------------------------------------------------
# Per-field help text (the '?' buttons). Each explains what the field is, its
# units, how to choose it, and what the pre-filled default means.
# ---------------------------------------------------------------------------
HELP = {
    "phonopy_yaml": (
        "The phonopy.yaml from your phonon calculation -- the FULL file that "
        "carries the supercell + force-constant context (the one phonopy writes "
        "alongside FORCE_CONSTANTS/FORCE_SETS), NOT a primitive-cell mesh.yaml "
        "dump.\n\nIt defines the lattice, atom positions and force constants the "
        "mode-2 engine evaluates. Required."),
    "born": (
        "Optional BORN file (Born effective charges + dielectric tensor) for the "
        "non-analytical LO-TO splitting at the zone centre.\n\nDefault: blank = "
        "off. Leave blank for non-polar materials (graphite, metals); supply it "
        "for polar/ionic crystals where the LO-TO splitting matters."),
    "force_constants": (
        "Optional explicit FORCE_CONSTANTS file.\n\nDefault: blank = read the "
        "force constants embedded in phonopy.yaml, or auto-discover a sibling "
        "FORCE_CONSTANTS/FORCE_SETS next to it. Set this only if your constants "
        "live in a separate file the yaml does not point to."),
    "force_sets": (
        "Optional explicit FORCE_SETS file (the displacement/force dataset, the "
        "alternative to a FORCE_CONSTANTS file).\n\nDefault: blank = use the "
        "force constants from the yaml or a sibling file."),
    "mesh": (
        "Phonon q-point mesh 'nx ny nz' sampling the Brillouin zone for the "
        "Debye-Waller tensor + DOS.\n\nFiner mesh -> smoother result, more cost "
        "(cost ~ nx*ny*nz). 40 40 40 (the default) is the validation-suite "
        "production density."),
    "temperature": (
        "Sample temperature in Kelvin. ONE config -> ONE temperature -> one "
        "NCrystal data file (.irmapack) per principal scatterer. Re-run at each "
        "temperature for a multi-T deployment.\n\nDefault 296 K (room temperature)."),
    "scatterers": (
        "One row per DISTINCT element in your phonopy.yaml. Each species in the "
        "structure must have a matching row.\n\nColumns:\n"
        "  Sym            element symbol; must match the atoms in phonopy.yaml.\n"
        "  sigma_bound_b  bound scattering cross section [barn]; normalizes "
        "S(alpha,beta). REQUIRED.\n"
        "  AWR            atomic weight ratio A = M/m_n. Sets the recoil + "
        "Debye-Waller mass (blank -> derived from the phonopy mass).\n"
        "  b_coh_fm       coherent scattering length [fm] (can be < 0); drives "
        "the coherent Bragg comb. Needed for the coherent elastic line.\n"
        "  sigma_inc_b    incoherent bound cross section [barn]; drives the "
        "incoherent elastic Debye-Waller line.\n\n"
        "The default row is natural carbon from IRMA's built-in nuclear "
        "table (Rauch-Waschkowski/Sears); typing a symbol and leaving the "
        "cell autofills the blank columns from the same table."),
    "material_id": (
        "Identifier stamped into the output filenames and the NCMAT snippet "
        "(data files are written as '<material_id>__<symbol>.irmapack'). Required."),
    "inelastic_mode": (
        "Fidelity of the mode-2 inelastic export.\n\n"
        "  2 (default): exact coherent one-phonon dispersion + multiphonon "
        "background -- the recommended choice for the NCrystal plugin and the "
        "validated path.\n"
        "  1: directional incoherent approximation (n=1 incoherent + "
        "incoherent-approx multiphonons); cheaper."),
    "num_directions": (
        "Golden-spiral powder-average directions for the ONE-phonon term.\n\n"
        "Default 10000 (the validation-campaign value). Cost is roughly linear; "
        "drop to ~4000 for a faster look."),
    "multiphonon_num_directions": (
        "Powder-average directions for the MULTIPHONON background (cheaper per "
        "direction than the one-phonon term).\n\nDefault 1000; converged by "
        "~50-100, so this carries ample margin."),
    "multiphonon_max_order": (
        "Highest multiphonon order summed in the background.\n\nDefault 'auto': "
        "the order is convergence-sized to the high-Q recoil tail (recommended "
        "-- the anisotropic Debye-Waller order required to reach the free-gas "
        "limit grows with Q). Enter an integer to set it exactly (lower = "
        "faster, but a too-low value truncates the high-Q cross section)."),
    "incoherent_elastic_mode": (
        "Debye-Waller treatment of the pack's INCOHERENT elastic channel.\n\n"
        "  isotropic (default): the plugin collapses each site tensor to its "
        "trace/3 scalar (NCrystal's standard model).\n"
        "  directional: the plugin samples the powder-averaged anisotropic "
        "<exp(-Q^2 u.U.u)> per site -- larger at high Q for anisotropic "
        "crystals, and consistent with the directional multiphonon and the "
        "per-reflection coherent elastic.\n\n"
        "The ENDF tape path cannot represent this option (scalar W' only)."),
    "jobs": (
        "Number of worker processes for the parallel direction/shell sums.\n\n"
        "Default: blank = serial (1). Enter a number to parallelize."),
    "outdir": (
        "Directory the per-temperature NCrystal data (.irmapack) files and the "
        "<material_id>.ncmat (carrying its @CUSTOM_IRMA section) are written to "
        "(created if absent). Required."),
}


class NCrystalPanel(ttk.Frame):
    """IRMA -> NCrystal data export panel."""

    # exposed for the test suite (mirrors the module map)
    _INELASTIC_BY_INT = _INELASTIC_BY_INT

    def __init__(self, parent, runner, status_setter=None):
        super().__init__(parent, padding=8)
        self.runner = runner
        self._status = status_setter or (lambda msg: None)
        init_form_styles()
        self._build()

    # ------------------------------------------------------------------ UI ---
    def _build(self):
        """Build the panel widgets."""
        self.pack(fill=tk.BOTH, expand=True)
        top = ttk.Frame(self)
        top.pack(fill=tk.BOTH, expand=True)

        # The form column is taller than common windows, so it lives in a
        # vertically-scrolling canvas (mouse-wheel + scrollbar); the log column
        # fills the rest. The canvas requests the form's natural width so
        # nothing is clipped horizontally. (Mirrors NSPanel._build.)
        left_outer = ttk.Frame(top)
        left_outer.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 8))
        bg = ttk.Style().lookup("TFrame", "background")
        canvas = tk.Canvas(left_outer, highlightthickness=0, borderwidth=0,
                           background=bg or None)
        vsb = ttk.Scrollbar(left_outer, orient=tk.VERTICAL, command=canvas.yview)
        canvas.configure(yscrollcommand=vsb.set)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        left = ttk.Frame(canvas)
        canvas.create_window((0, 0), window=left, anchor="nw")

        def _sync(_e):
            """Keep the canvas scroll region matched to the inner frame."""
            canvas.configure(scrollregion=canvas.bbox("all"),
                             width=left.winfo_reqwidth())
        left.bind("<Configure>", _sync)

        def _wheel(event):
            """Scroll the canvas on mouse wheel (platform-normalized delta)."""
            delta = event.delta
            step = delta // 120 if abs(delta) >= 120 else delta
            canvas.yview_scroll(-int(step), "units")
        canvas.bind("<Enter>", lambda e: canvas.bind_all("<MouseWheel>", _wheel))
        canvas.bind("<Leave>", lambda e: canvas.unbind_all("<MouseWheel>"))

        right = ttk.Frame(top)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self._build_material(left)
        self._build_export(left)
        self._build_actions(left)
        self._build_output(right)

    @staticmethod
    def _check_with_help(parent, text, var, help_text):
        """Checkbox with an attached ⓘ help glyph."""
        row = ttk.Frame(parent)
        row.pack(anchor=tk.W, fill=tk.X, pady=2)
        ttk.Checkbutton(row, text=text, variable=var).pack(side=tk.LEFT)
        InfoLabel(row, text, help_text).pack(side=tk.LEFT, padx=(4, 0))

    def _build_material(self, parent):
        """Build the material/model section."""
        g = form_section(parent, "Material (phonon model)")

        self.phonopy_yaml = FileSelector(
            g, "phonopy.yaml:",
            filetypes=[("phonopy YAML", "*.yaml *.yml"), ("All files", "*.*")],
            help_text=HELP["phonopy_yaml"])
        self.phonopy_yaml.pack(fill=tk.X, pady=2)
        self.born = FileSelector(g, "BORN (optional):",
                                 filetypes=[("All files", "*.*")],
                                 help_text=HELP["born"])
        self.born.pack(fill=tk.X, pady=2)
        self.force_constants = FileSelector(
            g, "FORCE_CONSTANTS (opt):",
            filetypes=[("All files", "*.*")], help_text=HELP["force_constants"])
        self.force_constants.pack(fill=tk.X, pady=2)
        self.force_sets = FileSelector(
            g, "FORCE_SETS (opt):",
            filetypes=[("All files", "*.*")], help_text=HELP["force_sets"])
        self.force_sets.pack(fill=tk.X, pady=2)
        self.mesh = LabeledEntry(g, "mesh (nx ny nz):", default="40 40 40",
                                 width=12, help_text=HELP["mesh"])
        self.mesh.pack(fill=tk.X, pady=2)
        self.temperature = LabeledEntry(g, "temperature (K):", default="296",
                                        width=10, help_text=HELP["temperature"])
        self.temperature.pack(fill=tk.X, pady=2)

        el = ttk.Frame(g)
        el.pack(fill=tk.X, pady=(4, 0))
        ttk.Label(el, text="scatterers:", foreground="gray").pack(side=tk.LEFT)
        InfoLabel(el, "scatterers", HELP["scatterers"]).pack(
            side=tk.LEFT, padx=(4, 0))
        # The export only reads the nuclear columns (symbol/sigma_bound_b/awr/
        # b_coh_fm/sigma_inc_b); the DOS/positions columns are spectra-only.
        self.element_table = ElementTable(g)
        self.element_table.set_visible(NUCLEAR)
        self.element_table.pack(fill=tk.X, pady=2)
        # natural-carbon example row; typing a different symbol over it
        # refreshes the machine-filled constants (see add_default_row)
        self.element_table.add_default_row("C")

    def _build_export(self, parent):
        """Build the export-settings section."""
        g = form_section(parent, "Export")
        self.material_id = LabeledEntry(g, "material_id:", default="material",
                                        width=20, help_text=HELP["material_id"])
        self.material_id.pack(fill=tk.X, pady=2)
        self.inelastic_mode = LabeledCombobox(
            g, "inelastic mode:", _INELASTIC_LABELS,
            default=_INELASTIC_LABELS[0], help_text=HELP["inelastic_mode"])
        self.inelastic_mode.pack(fill=tk.X, pady=2)
        self.num_directions = LabeledEntry(g, "directions:", default="10000",
                                           width=8,
                                           help_text=HELP["num_directions"])
        self.num_directions.pack(fill=tk.X, pady=2)
        self.multiphonon_num_directions = LabeledEntry(
            g, "multiphonon dirs:", default="1000", width=8,
            help_text=HELP["multiphonon_num_directions"])
        self.multiphonon_num_directions.pack(fill=tk.X, pady=2)
        self.multiphonon_max_order = LabeledEntry(
            g, "multiphonon order:", default="auto", width=8,
            help_text=HELP["multiphonon_max_order"])
        self.multiphonon_max_order.pack(fill=tk.X, pady=2)
        self.incoherent_elastic_mode = LabeledCombobox(
            g, "incoh. elastic DW:", ["isotropic", "directional"],
            default="isotropic", help_text=HELP["incoherent_elastic_mode"])
        self.incoherent_elastic_mode.pack(fill=tk.X, pady=2)
        self.jobs = LabeledEntry(g, "jobs (blank=serial):", default="",
                                 width=8, help_text=HELP["jobs"])
        self.jobs.pack(fill=tk.X, pady=2)
        # Shared S(alpha,beta) grid form (automatic Q/E range OR explicit
        # alpha/beta). Self-contained help; build_config merges its fields.
        gg = form_section(g, "S(alpha,beta) grid")
        self.grid_form = SabGridForm(gg)
        self.grid_form.pack(fill=tk.X)

    def _build_actions(self, parent):
        """Build the action buttons row."""
        g = ttk.Frame(parent)
        g.pack(fill=tk.X, pady=(2, 0))
        self.outdir = FileSelector(g, "output dir:", mode="directory",
                                   help_text=HELP["outdir"])
        self.outdir.pack(fill=tk.X, pady=2)
        btns = ttk.Frame(g)
        btns.pack(fill=tk.X, pady=4)
        self.export_btn = ttk.Button(btns, text="Export NCrystal data",
                                     default="active", command=self._export)
        self.export_btn.pack(side=tk.LEFT, padx=(0, 4))
        self.cancel_btn = ttk.Button(btns, text="Cancel", command=self._cancel,
                                     state=tk.DISABLED)
        self.cancel_btn.pack(side=tk.LEFT, padx=(0, 4))

    def _build_output(self, parent):
        """Build the log/output section."""
        nb = ttk.Notebook(parent)
        nb.pack(fill=tk.BOTH, expand=True)
        logf = ttk.Frame(nb)
        nb.add(logf, text="Log")
        self.log = ScrolledText(logf, height=20)
        self.log.pack(fill=tk.BOTH, expand=True)

    # -------------------------------------------------------------- config ---
    def _row_to_scatterer(self, row):
        """One element-table row -> a scatterer dict (nuclear columns only).

        The exporter reads symbol/sigma_bound_b/awr/b_coh_fm/sigma_inc_b; blank
        optional fields are omitted so the config dataclass keeps its defaults.
        """
        sym = row["symbol"]
        d = {"symbol": sym}
        for key in ("sigma_bound_b", "awr", "b_coh_fm", "sigma_inc_b"):
            if row[key]:
                d[key] = parse_float(f"scatterer {sym} {key}", row[key])
        return d

    def build_config(self):
        """Assemble an :class:`NCrystalExportConfig` from the current widgets."""
        scat = [self._row_to_scatterer(r)
                for r in self.element_table.get_rows() if r["symbol"]]
        material = {
            "phonopy_yaml": self.phonopy_yaml.get().strip() or None,
            "mesh": [parse_int("mesh (nx ny nz)", x)
                     for x in self.mesh.get().replace(",", " ").split()],
            "temperature_K": parse_float("temperature (K)",
                                         self.temperature.get()),
            "scatterers": scat,
        }
        if self.born.get().strip():
            material["born"] = self.born.get().strip()
        if self.force_constants.get().strip():
            material["force_constants"] = self.force_constants.get().strip()
        if self.force_sets.get().strip():
            material["force_sets"] = self.force_sets.get().strip()

        mpo = self.multiphonon_max_order.get().strip()
        export = {
            "material_id": self.material_id.get().strip(),
            "inelastic_mode": int(self.inelastic_mode.get()[0]),
            "num_directions": parse_int("directions",
                                        self.num_directions.get()),
            "multiphonon_num_directions": parse_int(
                "multiphonon dirs", self.multiphonon_num_directions.get()),
            "multiphonon_max_order": ("auto" if mpo == "auto"
                                      else parse_int("multiphonon order", mpo)),
        }
        if self.jobs.get().strip():
            export["jobs"] = parse_int("jobs", self.jobs.get())
        if self.incoherent_elastic_mode.get() != "isotropic":
            export["incoherent_elastic_mode"] = self.incoherent_elastic_mode.get()
        # Merge the active-mode S(alpha,beta) grid keys (the converged automatic
        # grid knobs or explicit alpha/beta) -- only the keys for the selected mode.
        export.update(self.grid_form.export_fields())

        self._last_built_dict = {"material": material, "export": export}
        return NCrystalExportConfig.from_dict(self._last_built_dict)

    @staticmethod
    def _write_config_yaml(built, path):
        """Write the {'material', 'export'} mapping ``build_config`` assembled to
        ``path`` as YAML, so the export CLI reads it back with ``from_yaml`` exactly
        as it was validated -- no separate (de)serializer to drift out of sync.
        """
        import yaml
        with open(path, "w", encoding="utf-8") as handle:
            yaml.safe_dump(built, handle, sort_keys=False)

    # -------------------------------------------------------------- actions --
    def _export(self):
        """Collect the config and start the export subprocess."""
        if self.runner.is_running:
            messagebox.showwarning("Running",
                                   "A calculation is already in progress.")
            return
        try:
            cfg = self.build_config()
        except (SpectraConfigError, ValueError) as exc:
            messagebox.showerror("Config Error", str(exc))
            return
        outdir = self.outdir.get().strip()
        if not outdir:
            messagebox.showerror("Error", "Please specify an output directory.")
            return

        # Serialize the validated config to a temp YAML and export via the CLI
        # (python -m irma.ncrystal), which calls write_packs and streams its
        # progress to stdout -- captured by the runner and routed to the log.
        self._cfg_tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False)
        self._cfg_tmp.close()
        self._write_config_yaml(self._last_built_dict, self._cfg_tmp.name)

        self.log.clear()
        self.log.append("=== IRMA -> NCrystal data export ===\n")
        self.log.append(f"material_id={cfg.material_id}  outdir={outdir}\n\n")
        self._status("Exporting NCrystal data...")
        self.export_btn.config(state=tk.DISABLED)
        self.cancel_btn.config(state=tk.NORMAL)
        self.runner.run_command(
            [sys.executable, "-u", "-m", "irma.ncrystal",
             self._cfg_tmp.name, "-o", outdir],
            success_msg=f"NCrystal data written to {outdir}.",
            on_log=self._log_ts, on_done=self._done_ts,
            error_label="NCrystal export config error")

    def _cancel(self):
        """Cancel the running export."""
        if not self.runner.is_running:
            return
        self._status("Cancelling...")
        self.cancel_btn.config(state=tk.DISABLED)
        self.log.append("\n=== Cancelling (terminating workers) ===\n")
        self.runner.cancel()

    def _log_ts(self, text):
        """Append a log line from the worker thread (marshalled via after())."""
        self.after(0, self.log.append, text)

    def cleanup_temp_files(self):
        """Idempotently remove the panel-owned temp config (review GUI-2)."""
        tmp = getattr(self, "_cfg_tmp", None)
        if tmp is not None:
            try:
                os.unlink(tmp.name)
            except OSError:
                pass
            self._cfg_tmp = None

    def _done_ts(self, ok, msg):
        """Handle export completion from the worker thread (marshalled via after())."""
        def _update():
            """Apply the completion UI changes on the Tk thread."""
            self.export_btn.config(state=tk.NORMAL)
            self.cancel_btn.config(state=tk.DISABLED)
            self.log.append(f"\n{msg}\n")
            if ok:
                self._status("Export complete")
            elif msg.startswith("Calculation cancelled"):
                self._status("Cancelled")
            else:
                self._status("Error")
                messagebox.showerror("Export Error", msg[:500])
            self.cleanup_temp_files()
        self.after(0, _update)
