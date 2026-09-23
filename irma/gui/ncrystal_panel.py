"""NCrystal-plugin panel for the IRMA GUI.

A thin controller for IRMA's third capability: exporting the per-temperature
NCrystal scattering data (``.irmapack`` files + a ``@CUSTOM_IRMA`` NCMAT snippet)
that wire an IRMA mode-2 (coherent one-phonon + anisotropic Debye-Waller)
calculation into NCrystal. It collects inputs into an
:class:`~irma.ncrystal.NCrystalExportConfig` and runs ``irma.ncrystal.write_packs``
-- it reimplements no export logic.

``build_config()`` is a pure widget<->config mapping (no event loop needed), so
it is unit-testable against a withdrawn Tk root; ``load_config()`` is its
inverse, so an exporter config written by this panel or by ``irma mlip emit``
can be opened back into the form. The export runs out-of-process
through the shared :class:`~irma.gui.runner.ComputationRunner` (the same
cancellable subprocess path the Neutron-Scattering panel uses), driving the
``python -m irma.ncrystal`` CLI, which calls ``write_packs`` and streams its
``progress`` markers to this panel's log.
"""

import dataclasses
import os
import sys
import tempfile
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

from irma.gui.widgets import (
    LabeledEntry, LabeledCombobox, FileSelector, ScrolledText, InfoLabel,
    form_section, init_form_styles, scrolled_columns,
    parse_float, parse_int)
from irma.core.noncubic_inelastic import MIN_PHONON_ENERGY_HELP
from irma.gui.grid_form import SabGridForm, GRID_EXPORT_KEYS
from irma.gui.element_table import ElementTable, NUCLEAR
from irma.ncrystal.config import NCrystalExportConfig
from irma.spectra.config import SpectraConfigError


# inelastic-mode dropdown labels (the leading digit is parsed back to the int).
_INELASTIC_LABELS = ["2 (coherent 1ph + multi)", "1 (incoherent approx)"]
_INELASTIC_BY_INT = {1: "1 (incoherent approx)", 2: "2 (coherent 1ph + multi)"}

# The export keys this form OWNS: every key ``build_config`` can write, i.e.
# every one with a control on the panel (the grid form names its own). The
# remaining NCrystalExportConfig fields -- gain_side, elastic,
# coherent_partition_mode, lat -- have no widget here, so a
# loaded config's values for them are carried through untouched rather than
# silently reset to the dataclass defaults on the next Export. See
# ``_unrepresented_export``.
FORM_EXPORT_KEYS = frozenset(
    {"material_id", "inelastic_mode", "num_directions",
     "multiphonon_num_directions", "multiphonon_max_order", "jobs",
     "min_phonon_energy_meV", "incoherent_elastic_mode"} | set(GRID_EXPORT_KEYS))

# the nuclear columns the exporter reads off a scatterer row
_SCATTERER_KEYS = ("sigma_bound_b", "awr", "b_coh_fm", "sigma_inc_b")


def _unrepresented_export(cfg):
    """Loaded export settings this form has no control for, and cannot show.

    ``irma mlip emit`` writes ``gain_side``, ``elastic`` and
    ``coherent_partition_mode`` into every ``ncrystal.yaml``, and the panel
    exposes none of them. Rebuilding the config from the widgets alone would
    quietly rewrite them to the dataclass defaults -- turning
    ``coherent_partition_mode: auto`` into ``principal-xs-weighted`` (a
    DIFFERENT partition on a single-group material) or ``elastic: false``
    back on -- so the loader keeps them and ``build_config`` merges them back.

    Only values that DIFFER from the config default are carried: a file that
    just spells out a default needs nothing carried, and the panel reports
    exactly the settings a reader of the form would otherwise not know about.
    """
    carried = {}
    for field in dataclasses.fields(NCrystalExportConfig):
        if field.name == "material" or field.name in FORM_EXPORT_KEYS:
            continue
        value = getattr(cfg, field.name)
        if value is None or value == field.default:
            continue
        carried[field.name] = value
    return carried


# ---------------------------------------------------------------------------
# Per-field help text (the '?' buttons). Each explains what the field is, its
# units, how to choose it, and what the pre-filled default means.
# ---------------------------------------------------------------------------
HELP = {
    "phonopy_yaml": (
        "The phonopy.yaml from your phonon calculation. Use the FULL file that "
        "carries the supercell and force-constant context (the one phonopy "
        "writes alongside FORCE_CONSTANTS/FORCE_SETS), NOT a primitive-cell "
        "mesh.yaml dump.\n\nIt defines the lattice, the atom positions, and the "
        "force constants that the mode-2 engine evaluates. Required."),
    "born": (
        "Optional BORN file (Born effective charges plus dielectric tensor). It "
        "adds the non-analytical LO-TO splitting at the zone "
        "centre.\n\nDefault: blank = "
        "off. Leave blank for non-polar materials (graphite, metals); supply it "
        "for polar or ionic crystals, where the LO-TO splitting matters."),
    "force_constants": (
        "Optional explicit FORCE_CONSTANTS file.\n\nDefault: blank = read the "
        "force constants embedded in phonopy.yaml, or auto-discover a sibling "
        "FORCE_CONSTANTS or FORCE_SETS file next to it. Set this only if your "
        "force constants live in a separate file the yaml does not point to."),
    "force_sets": (
        "Optional explicit FORCE_SETS file (the displacement/force dataset, the "
        "alternative to a FORCE_CONSTANTS file).\n\nDefault: blank = use the "
        "force constants from the yaml or a sibling file."),
    "mesh": (
        "Phonon q-point mesh 'nx ny nz': how densely the Brillouin zone is "
        "sampled to build the Debye-Waller tensor and the vibrational density "
        "of states.\n\nA finer mesh "
        "gives a smoother result at more cost (cost scales as nx*ny*nz). The "
        "default, 40 40 40, is the validation-suite production density."),
    "temperature": (
        "Sample temperature in Kelvin. ONE config covers ONE temperature and "
        "writes one NCrystal data file (.irmapack) per principal scatterer "
        "(each distinct element in the material). Re-run at each temperature for a "
        "multi-temperature deployment.\n\nDefault 296 K (room temperature)."),
    "scatterers": (
        "One row per DISTINCT element in your phonopy.yaml. Each species in the "
        "structure must have a matching row.\n\nColumns:\n"
        "  Sym            element symbol; must match the atoms in phonopy.yaml.\n"
        "  sigma_bound_b  bound scattering cross section [barn]; normalizes the "
        "S(alpha,beta) table. REQUIRED.\n"
        "  AWR            atomic weight ratio A = M/m_n. Sets the recoil and "
        "Debye-Waller mass (blank = derived from the phonopy mass).\n"
        "  b_coh_fm       coherent scattering length [fm] (can be < 0); drives "
        "the coherent Bragg edges. Needed for the coherent elastic line.\n"
        "  sigma_inc_b    incoherent bound cross section [barn]; drives the "
        "incoherent elastic Debye-Waller line.\n\n"
        "The table starts with one EMPTY row on purpose -- the scatterers "
        "are your material's identity, which IRMA cannot guess; typing a "
        "symbol and leaving the cell autofills the blank columns from "
        "IRMA's built-in nuclear table (Rauch-Waschkowski/Sears)."),
    "material_id": (
        "Identifier stamped into the output filenames and the NCMAT snippet "
        "(data files are written as '<material_id>__<symbol>.irmapack'). Required."),
    "inelastic_mode": (
        "Fidelity of the mode-2 inelastic export. Inelastic scattering "
        "exchanges energy with the lattice vibrations (phonons).\n\n"
        "  2 (default): exact coherent one-phonon dispersion plus the "
        "multiphonon "
        "background. This is the recommended choice for the NCrystal plugin "
        "and the validated path.\n"
        "  1: directional incoherent approximation (one-phonon incoherent plus "
        "incoherent-approximation multiphonons); cheaper."),
    "num_directions": (
        "Number of golden-spiral directions used to powder-average the "
        "ONE-phonon term (averaged over crystal "
        "orientations).\n\nDefault 10000 (the validation-campaign value). Cost is "
        "roughly linear in this number; drop to about 4000 for a faster look."),
    "multiphonon_num_directions": (
        "Number of powder-average directions for the MULTIPHONON background. "
        "Each direction is cheaper here than in the one-phonon term.\n\nDefault "
        "1000; converged by about 50-100 directions, so the default carries "
        "ample margin."),
    "multiphonon_max_order": (
        "Highest multiphonon order summed in the background. The order is the "
        "number of lattice vibrations exchanged in one scattering event.\n\n"
        "Default 'auto': the order is convergence-sized to the recoil tail at "
        "high Q (momentum transfer), which is recommended: the anisotropic "
        "Debye-Waller order required to reach the free-gas limit grows with Q. "
        "Enter an integer to set the order exactly (lower = faster, but a "
        "too-low value truncates the high-Q cross section)."),
    "min_phonon_energy": MIN_PHONON_ENERGY_HELP,
    "incoherent_elastic_mode": (
        "Debye-Waller treatment of the pack's INCOHERENT elastic "
        "component.\n\n"
        "  isotropic (default): the plugin collapses each site's displacement "
        "tensor to its trace/3 scalar (NCrystal's standard model).\n"
        "  directional: the plugin samples the powder-averaged anisotropic "
        "<exp(-Q^2 u.U.u)> per site. This is larger at high Q (momentum "
        "transfer) for anisotropic crystals, and consistent with the "
        "directional multiphonon and the per-reflection coherent elastic "
        "components.\n\n"
        "The ENDF tape path cannot represent this option (scalar W' only)."),
    "jobs": (
        "Number of worker processes for the parallel direction/shell sums.\n\n"
        "Default: blank = every CPU core on this machine. Enter a number to "
        "cap it -- lower it when memory is tight (each worker holds its own "
        "copy of the phonon arrays) or to leave cores for other work; 1 runs "
        "serially."),
    "outdir": (
        "Directory where the per-temperature NCrystal data files (.irmapack) "
        "and the <material_id>.ncmat file (carrying its @CUSTOM_IRMA section) "
        "are written (created if absent). Required."),
}

# The scatterer list is MATERIAL IDENTITY: a prefilled carbon row is IRMA
# asserting a material it cannot know, and a plausible-but-wrong row survives
# review far more easily than an empty one. The export settings below it
# (mesh, directions, grid) are methodology and stay prefilled.
IDENTITY_HINT_SCATTERERS = (
    "Blank on purpose — the scatterers describe YOUR material, and must be "
    "the species of the phonopy.yaml above. Type a symbol and the nuclear "
    "constants autofill from the built-in table.")


class NCrystalPanel(ttk.Frame):
    """IRMA -> NCrystal data export panel."""

    # exposed for the test suite (mirrors the module map)
    _INELASTIC_BY_INT = _INELASTIC_BY_INT

    def __init__(self, parent, runner, status_setter=None):
        super().__init__(parent, padding=8)
        self.runner = runner
        self._status = status_setter or (lambda msg: None)
        # export settings a loaded config carried that this form cannot show
        # (see _unrepresented_export); empty until Open Config... loads one.
        self._carried_export = {}
        init_form_styles()
        self._build()

    # ------------------------------------------------------------------ UI ---
    def _build(self):
        """Build the panel widgets."""
        self.pack(fill=tk.BOTH, expand=True)
        left, right, _ = scrolled_columns(self)

        self._build_material(left)
        self._build_export(left)
        self._build_actions(left)
        self._build_output(right)

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
        ttk.Label(g, text=IDENTITY_HINT_SCATTERERS, foreground="gray",
                  justify=tk.LEFT, wraplength=540).pack(anchor=tk.W)
        # The export only reads the nuclear columns (symbol/sigma_bound_b/awr/
        # b_coh_fm/sigma_inc_b); the DOS/positions columns are spectra-only.
        self.element_table = ElementTable(g)
        self.element_table.set_visible(NUCLEAR)
        self.element_table.pack(fill=tk.X, pady=2)
        # One EMPTY row: the scatterer list is the user's material, so IRMA
        # asserts nothing about it. Typing a symbol and leaving the cell fills
        # the nuclear constants from the built-in table (see autofill_row).
        self.element_table.add_row()

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
        self.min_phonon_energy = LabeledEntry(
            g, "min phonon energy [meV]:", default="", width=8,
            help_text=HELP["min_phonon_energy"])
        self.min_phonon_energy.pack(fill=tk.X, pady=2)
        self.incoherent_elastic_mode = LabeledCombobox(
            g, "incoh. elastic DW:", ["isotropic", "directional"],
            default="isotropic", help_text=HELP["incoherent_elastic_mode"])
        self.incoherent_elastic_mode.pack(fill=tk.X, pady=2)
        self.jobs = LabeledEntry(g, "jobs (blank=all cores):", default="",
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
        # The export writes a config the CLI runs; this reads one back, so an
        # emitted ncrystal.yaml can be inspected and adjusted here instead of
        # retyped. (The Neutron Scattering tab's button of the same name.)
        self.open_btn = ttk.Button(btns, text="Open Config...",
                                   command=self._open_config)
        self.open_btn.pack(side=tk.LEFT, padx=(0, 4))

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
        for key in _SCATTERER_KEYS:
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
        # Settings a loaded config carried that this form has no control for
        # go in FIRST, so anything the widgets own still wins (by construction
        # the two key sets are disjoint -- see FORM_EXPORT_KEYS).
        export = dict(self._carried_export)
        export.update({
            "material_id": self.material_id.get().strip(),
            "inelastic_mode": int(self.inelastic_mode.get()[0]),
            "num_directions": parse_int("directions",
                                        self.num_directions.get()),
            "multiphonon_num_directions": parse_int(
                "multiphonon dirs", self.multiphonon_num_directions.get()),
            "multiphonon_max_order": ("auto" if mpo == "auto"
                                      else parse_int("multiphonon order", mpo)),
            "min_phonon_energy_meV": (
                parse_float("min phonon energy [meV]", self.min_phonon_energy.get())
                if self.min_phonon_energy.get().strip() else 0.0),
        })
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

    # ---------------------------------------------------------------- load ---
    @staticmethod
    def _set_constant(row, key, value):
        """Put ONE nuclear constant into a freshly autofilled scatterer row.

        The row arrives with the built-in table's values in place and recorded
        as machine-filled (``autofill_row``), which is the state typing the
        symbol into an empty row produces. A config value equal to the table's
        is therefore left exactly as the autofill wrote it, provenance intact,
        so a later symbol edit refreshes it instead of carrying the old
        element's constants; a value that differs overwrites the machine
        string and thereby reads as user data, which is what it is. Compared
        numerically, so 5.551 from the file and '5.551' from the table match
        whatever either side's formatting is. ``None`` (the config omitted the
        field, e.g. awr derived from the phonopy mass) clears the cell, so the
        next build omits it again.
        """
        var = row["_var"][key]
        if value is None:
            var.set("")
            return
        machine = var.get().strip()
        if machine and float(machine) == float(value):
            return
        var.set(str(value))

    def _load_scatterers(self, scatterers):
        """Rebuild the scatterer table, one row per config species."""
        table = self.element_table
        table.clear()
        for s in scatterers:
            row = table.add_row({"symbol": s.symbol})
            # exactly what typing the symbol and leaving the cell does
            table.autofill_row(row)
            for key in _SCATTERER_KEYS:
                self._set_constant(row, key, getattr(s, key))

    def load_config(self, cfg):
        """Populate every widget from an :class:`NCrystalExportConfig`.

        The inverse of :meth:`build_config`, field for field: anything that
        method writes, this one reads back, so an exported config -- or an
        ``ncrystal.yaml`` from ``irma mlip emit`` -- round-trips through the
        form. Settings with no control here are carried instead (see
        :func:`_unrepresented_export`).

        Total by construction: it only formats values off an already-validated
        config, so it cannot fail halfway and leave a half-loaded form. The
        caller validates FIRST (``from_yaml``) and calls this only on success.
        """
        m = cfg.material
        self.phonopy_yaml.set(m.phonopy_yaml or "")
        self.born.set(m.born or "")
        self.force_constants.set(m.force_constants or "")
        self.force_sets.set(m.force_sets or "")
        self.mesh.set(" ".join(str(x) for x in m.mesh))
        self.temperature.set(m.temperature_K)
        self._load_scatterers(m.scatterers)

        self.material_id.set(cfg.material_id)
        self.inelastic_mode.set(_INELASTIC_BY_INT[cfg.inelastic_mode])
        self.num_directions.set(cfg.num_directions)
        self.multiphonon_num_directions.set(cfg.multiphonon_num_directions)
        self.multiphonon_max_order.set(cfg.multiphonon_max_order)
        cutoff = float(getattr(cfg, "min_phonon_energy_meV", 0.0))
        self.min_phonon_energy.set("" if cutoff == 0.0 else f"{cutoff:g}")
        self.jobs.set("" if cfg.jobs is None else cfg.jobs)
        self.incoherent_elastic_mode.set(cfg.incoherent_elastic_mode)
        # the grid form owns its own keys and picks its mode from them
        self.grid_form.load_fields({k: getattr(cfg, k)
                                    for k in GRID_EXPORT_KEYS})
        self._carried_export = _unrepresented_export(cfg)

    # -------------------------------------------------------------- actions --
    def _open_config(self):
        """Load an NCrystal exporter config YAML into the form.

        Parsed by the EXPORTER's own loader, so the form can never accept a
        config the export would reject and the two cannot drift apart. A
        cancelled dialog, an unreadable file, or a config the exporter refuses
        changes NOTHING: the config is fully validated before the first widget
        is touched, so there is no partial application to undo.
        """
        path = filedialog.askopenfilename(
            filetypes=[("NCrystal export config", "*.yaml *.yml"),
                       ("All files", "*.*")])
        if not path:
            return
        try:
            cfg = NCrystalExportConfig.from_yaml(path)
        except Exception as exc:
            messagebox.showerror("Open Config", str(exc))
            return
        self.load_config(cfg)
        if self._carried_export:
            # Visible, not hidden: these have no control on the form, so say
            # so rather than let the next Export write settings nobody can see.
            self.log.append(
                "note: carried from the file (no control on this form; the "
                "next Export writes them back unchanged): "
                + ", ".join(f"{k}={v!r}"
                            for k, v in sorted(self._carried_export.items()))
                + "\n")
        self._status(f"Loaded config from {os.path.basename(path)}")

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
