"""MLIP phonon-model panel for the IRMA GUI.

A thin controller for the ``irma mlip`` front end: build a phonon-model
bundle from a structure file + a pretrained machine-learned interatomic
potential,
validate it, and emit prefilled inputs for the three IRMA consumers. It
reimplements no front-end logic -- every action is the corresponding
``python -m irma mlip ...`` CLI invocation, run out-of-process through the
shared :class:`~irma.gui.runner.ComputationRunner` (the same cancellable
subprocess path the other panels use), so the GUI and the CLI can never
disagree about behavior.

``build_command()`` / ``emit_command()`` are pure widget->argv mappings
(no event loop needed), unit-testable against a withdrawn Tk root.
"""

import os
import sys
import tkinter as tk
from tkinter import ttk, messagebox

from irma.gui.widgets import (
    LabeledEntry, LabeledCombobox, FileSelector, ScrolledText, InfoLabel,
    form_section, init_form_styles, parse_int)
from irma.mlip.calculators import POTENTIALS

_TARGETS = ("endf", "spectra", "ncrystal")


# ---------------------------------------------------------------------------
# Per-field help text (the '?' buttons).
# ---------------------------------------------------------------------------
HELP = {
    "structure": (
        "The crystal structure to build a phonon model for -- any format ASE "
        "reads (CIF, VASP POSCAR, xyz with a cell, ...).\n\nThe structure is "
        "relaxed with the SAME potential before displacements, so a "
        "reasonable experimental or DFT geometry is a fine starting point."),
    "potential": (
        "The pretrained machine-learned interatomic potential that "
        "computes the "
        "forces. All backends are CPU-only by design and wired to "
        "conservative forces.\n\nQuality/character notes live in the manual "
        "('MLIP phonon models'): nequip was the best all-around performer "
        "in the validation campaign; grace has an academic-use license; "
        "mace-off is organic molecules only; dpa3 cannot bind van-der-Waals "
        "layered crystals.\n\nA potential whose packages are not installed "
        "here can still run through a registered dedicated environment -- "
        "see the Environments section below."),
    "model": (
        "Specific checkpoint name or path (blank = the potential's default)."
        "\n\nPer-backend syntax:\n"
        "  pet-mad   NAME@VERSION (e.g. pet-mad-s@1.5.0); bare name pins "
        "the newest release at build start\n"
        "  dpa3      MODEL::HEAD (default head MP_traj_v024_alldata_mixu)\n"
        "  nequip    model-zoo id (mir-group/NequIP-OAM-L:0.1) or a "
        "compiled .nequip.pt2 path\n"
        "  mace      size names or foundation names (medium-omat-0 -- ASL "
        "license note is printed and recorded)\n"
        "  grace     foundation name (GRACE-2L-OAM, ...)\n"
        "  others    checkpoint name or file path"),
    "supercell": (
        "Supercell for the finite-displacement force constants: explicit "
        "'n1 n2 n3', or a single number = minimum lattice-parameter length "
        "in Angstrom (the default corresponds to 12).\n\nBigger supercell = "
        "longer-ranged force constants and more cost. Blank = default "
        "(12 A rule; 1 1 1 under 'disordered')."),
    "mesh": (
        "Quick-look phonon mesh 'nx ny nz' for the bundle's DOS and "
        "imaginary-mode census.\n\nBlank = an automatic density (Gamma "
        "point under 'disordered'). The validation campaign used 40 40 40 "
        "for production comparisons."),
    "delta": (
        "Finite-displacement amplitude in Angstrom.\n\nDefault 0.03 -- the "
        "value the validation campaign used everywhere."),
    "fmax": (
        "Relaxation force convergence in eV/A (FIRE optimizer).\n\n"
        "Default 0.01. Tighten for soft or nearly-unstable materials."),
    "jitter_cycles": (
        "On an unconverged relaxation, kick the highest-force atoms "
        "(~0.05 A) and re-relax up to this many extra cycles, keeping the "
        "lowest-residual frame seen.\n\nEscapes MLIP force-noise stalls -- "
        "common on glasses, where FIRE can stall an order of magnitude "
        "above fmax. 0 = plain single-pass relaxation."),
    "relax_cell": (
        "Also relax the cell (lattice vectors + volume) with the potential, "
        "not just atomic positions.\n\nNeeds a checkpoint that provides "
        "stress. Validated: from starting points 3% apart, builds converge "
        "to the same lattice within ~0.001 A."),
    "disordered": (
        "Declare a disordered/amorphous model: the box is treated as its "
        "own supercell, the mesh defaults to the Gamma point, and emission "
        "switches to the DOS-driven classic path (incoherent elastic scaled "
        "by the total bound cross section). Never auto-detected."),
    "snap_symmetry": (
        "After relaxation, snap the positions onto the exact orbits of "
        "the detected spacegroup (tolerance 1e-2 A).\n\nFloat32 "
        "relaxations often land a hair off the ideal Wyckoff sites; "
        "phonopy then sees P1, generates several times more "
        "displacements, and a symmetry-reduced BORN file stops "
        "matching. The snap repairs all three (the shift is recorded "
        "in the manifest)."),
    "force": (
        "Proceed past an unconverged relaxation (the residual is recorded "
        "in the bundle). Without this, a relaxation that fails to reach "
        "fmax stops the build (exit 3)."),
    "born": (
        "Optional phonopy BORN file (Born effective charges + dielectric "
        "tensor). Embedded into the bundle -- NAC is never read from the "
        "working directory.\n\nEssential for ionic crystals: without it "
        "even a perfect model shows spurious imaginary modes."),
    "jobs": (
        "Parallel displacement workers (spawn processes), one native "
        "thread each by default (threads/worker widens them).\n\nThis is THE performance knob for "
        "heavy models: pick a value that divides the displacement "
        "count evenly, up to your core count (ZrO2 benchmark, 18 "
        "displacements: jobs=9 was 21x faster than the unfixed "
        "serial path). Fast models (mattersim) finish in seconds "
        "regardless."),
    "worker_threads": (
        "Native threads per parallel worker (keep jobs x threads <= "
        "cores). Default 1 -- the measured-safe convention."),
    "threads": (
        "Native threads for the SERIAL force path (jobs = 1). Blank = "
        "all cores.\n\nNote: compiled-artifact backends (nequip) are "
        "single-threaded by construction, so this has little effect "
        "there -- use jobs instead."),
    "outdir": (
        "Bundle output directory (created; must be empty unless "
        "'overwrite'). Everything lands here: phonopy.yaml with embedded "
        "force constants, relaxed structure, DOS, manifest, force cache."),
    "overwrite": (
        "Replace an existing bundle in the output directory (the cached "
        "displacement forces inside it are still reused when the "
        "fingerprint matches)."),
    "bundle": (
        "A bundle directory produced by a build (this panel fills it in "
        "after a successful build). Validate re-checks it; Emit generates "
        "IRMA inputs from it."),
    "targets": (
        "Which prefilled inputs to generate from the bundle:\n"
        "  endf      one ready-to-run deck per principal scatterer\n"
        "  spectra   an `irma spectra` YAML config\n"
        "  ncrystal  an exporter YAML for `irma ncrystal`\n\n"
        "Nothing is executed -- inspect the generated files, then run the "
        "consumer yourself (the emit log prints the exact commands)."),
    "mats": (
        "ENDF MAT number per species, as SYM=INT pairs separated by "
        "spaces or commas (e.g.  C=31  or  Be=26, O=48). Required for the "
        "endf target; the species must exist in the bundle."),
    "nuclides": (
        "Optional isotope selection per species, SYM=NUCLIDE pairs (e.g. "
        "C=13-C). Switches both the ENDF ZA identity and the scattering "
        "constants to that isotope."),
    "species_over": (
        "Optional scattering-constant overrides, one SYM:field=value[,...] "
        "group per species (fields: awr, b_coh_fm, sigma_inc_b; e.g. "
        "C:b_coh_fm=6.646,sigma_inc_b=0.001). sigma_bound_b is always "
        "derived and only cross-checked."),
    "temperature": (
        "Temperature in K for the emitted inputs.\n\nDefault 296."),
    "principal": (
        "Spectra principal scatterer symbol (blank = first species)."),
    "emit_inelastic_mode": (
        "Physics level of the emitted ENDF decks (--inelastic-mode).\n\n"
        "  default: omit the flag; the CLI emits mode-2 decks.\n"
        "  0: classic isotropic path from the bundle's species-projected "
        "DOS (principal spectrum on the classic cards, Card 6e partial "
        "spectra for the other species).\n"
        "  1/2: phonopy-backed directional decks.\n\n"
        "Not applicable to disordered bundles (the CLI rejects an explicit "
        "mode there -- they always use the DOS-driven classic path)."),
    "emit_elastic_format": (
        "Elastic output convention of the emitted ENDF decks "
        "(--elastic-format).\n\n"
        "  default: omit the flag; the CLI emits MEF decks.\n"
        "  mef: both elastic components for every species.\n"
        "  sef: the complete coherent component assigned to the "
        "designated-coherent atom.\n\n"
        "An EXPLICIT value -- even 'mef', which names the default -- is a "
        "crystal-deck selector and is rejected for disordered bundles, so "
        "'default' and 'mef' are distinct choices here."),
    "material_id": (
        "NCrystal export material id (--material-id). Blank = the CLI "
        "default (mlip_model)."),
    "emit_outdir": (
        "Directory for the emitted inputs (blank = the bundle directory)."),
    "env_potential": (
        "Potential to provision a dedicated environment for. Use this when "
        "the packages conflict with this environment (e.g. mace's "
        "e3nn==0.4.4 pin) or are simply not installed.\n\n'Create' builds "
        "a standard virtual environment (uv or python -m venv -- never "
        "conda), installs the known-good packages, verifies the import, "
        "and registers it; builds with that potential then use it "
        "automatically. 'Remove' unregisters and deletes a provisioned "
        "env."),
}


class MlipPanel(ttk.Frame):
    """Structure + pretrained MLIP -> phonon-model bundle -> IRMA inputs."""

    def __init__(self, parent, runner, status_setter=None):
        super().__init__(parent, padding=8)
        self.runner = runner
        self._status = status_setter or (lambda msg: None)
        init_form_styles()
        self._build()

    # ------------------------------------------------------------------ UI ---
    def _build(self):
        """Build the panel widgets (scrolling form column + log column)."""
        self.pack(fill=tk.BOTH, expand=True)
        top = ttk.Frame(self)
        top.pack(fill=tk.BOTH, expand=True)

        left_outer = ttk.Frame(top)
        left_outer.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 8))
        bg = ttk.Style().lookup("TFrame", "background")
        canvas = tk.Canvas(left_outer, highlightthickness=0, borderwidth=0,
                           background=bg or None)
        vsb = ttk.Scrollbar(left_outer, orient=tk.VERTICAL,
                            command=canvas.yview)
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
            """Scroll the canvas on mouse wheel (platform-normalized)."""
            delta = event.delta
            step = delta // 120 if abs(delta) >= 120 else delta
            canvas.yview_scroll(-int(step), "units")
        canvas.bind("<Enter>",
                    lambda e: canvas.bind_all("<MouseWheel>", _wheel))
        canvas.bind("<Leave>", lambda e: canvas.unbind_all("<MouseWheel>"))

        right = ttk.Frame(top)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self._build_build(left)
        self._build_bundle(left)
        self._build_emit(left)
        self._build_envs(left)
        self._build_output(right)

    def _check_with_help(self, parent, text, var, help_text):
        """Checkbox with an attached help glyph."""
        row = ttk.Frame(parent)
        row.pack(anchor=tk.W, fill=tk.X, pady=2)
        ttk.Checkbutton(row, text=text, variable=var).pack(side=tk.LEFT)
        InfoLabel(row, text, help_text).pack(side=tk.LEFT, padx=(4, 0))

    def _build_build(self, parent):
        """Build the bundle-build section."""
        g = form_section(parent, "Build a phonon-model bundle")
        self.structure = FileSelector(
            g, "structure:",
            filetypes=[("Structures", "*.cif *.vasp *.xyz POSCAR*"),
                       ("All files", "*.*")],
            help_text=HELP["structure"])
        self.structure.pack(fill=tk.X, pady=2)
        self.potential = LabeledCombobox(
            g, "potential:", list(POTENTIALS), default=POTENTIALS[0],
            help_text=HELP["potential"])
        self.potential.pack(fill=tk.X, pady=2)
        self.model = LabeledEntry(g, "model (blank=default):", default="",
                                  width=34, help_text=HELP["model"])
        self.model.pack(fill=tk.X, pady=2)
        self.supercell = LabeledEntry(g, "supercell (opt):", default="",
                                      width=12, help_text=HELP["supercell"])
        self.supercell.pack(fill=tk.X, pady=2)
        self.mesh = LabeledEntry(g, "mesh (opt):", default="", width=12,
                                 help_text=HELP["mesh"])
        self.mesh.pack(fill=tk.X, pady=2)
        self.delta = LabeledEntry(g, "delta (A):", default="0.03", width=8,
                                  help_text=HELP["delta"])
        self.delta.pack(fill=tk.X, pady=2)
        self.fmax = LabeledEntry(g, "fmax (eV/A):", default="0.01", width=8,
                                 help_text=HELP["fmax"])
        self.fmax.pack(fill=tk.X, pady=2)
        self.jitter_cycles = LabeledEntry(g, "jitter cycles:", default="0",
                                          width=6,
                                          help_text=HELP["jitter_cycles"])
        self.jitter_cycles.pack(fill=tk.X, pady=2)
        self.born = FileSelector(g, "BORN (optional):",
                                 filetypes=[("All files", "*.*")],
                                 help_text=HELP["born"])
        self.born.pack(fill=tk.X, pady=2)

        self.relax_cell = tk.BooleanVar(value=False)
        self._check_with_help(g, "relax the cell too", self.relax_cell,
                              HELP["relax_cell"])
        self.disordered = tk.BooleanVar(value=False)
        self._check_with_help(g, "disordered / amorphous", self.disordered,
                              HELP["disordered"])
        self.snap_symmetry = tk.BooleanVar(value=False)
        self._check_with_help(g, "snap to symmetry after relaxation",
                              self.snap_symmetry, HELP["snap_symmetry"])
        self.force = tk.BooleanVar(value=False)
        self._check_with_help(g, "proceed past unconverged relaxation",
                              self.force, HELP["force"])
        self.overwrite = tk.BooleanVar(value=False)
        self._check_with_help(g, "overwrite existing bundle", self.overwrite,
                              HELP["overwrite"])

        self.jobs = LabeledEntry(g, "jobs:", default="1", width=6,
                                 help_text=HELP["jobs"])
        self.jobs.pack(fill=tk.X, pady=2)
        self.worker_threads = LabeledEntry(g, "threads/worker:", default="1",
                                           width=6,
                                           help_text=HELP["worker_threads"])
        self.worker_threads.pack(fill=tk.X, pady=2)
        self.threads = LabeledEntry(g, "serial threads (opt):", default="",
                                    width=6, help_text=HELP["threads"])
        self.threads.pack(fill=tk.X, pady=2)

        self.outdir = FileSelector(g, "save bundle to:",
                                   mode="directory",
                                   help_text=HELP["outdir"])
        self.outdir.pack(fill=tk.X, pady=2)

        btns = ttk.Frame(g)
        btns.pack(fill=tk.X, pady=4)
        self.build_btn = ttk.Button(btns, text="Build bundle",
                                    default="active", command=self._run_build)
        self.build_btn.pack(side=tk.LEFT, padx=(0, 4))
        self.cancel_btn = ttk.Button(btns, text="Cancel",
                                     command=self._cancel, state=tk.DISABLED)
        self.cancel_btn.pack(side=tk.LEFT)

    def _build_bundle(self, parent):
        """Build the bundle validate section."""
        g = form_section(parent, "Use a bundle")
        ttk.Label(g, text="Validate, Show DOS, and Generate IRMA inputs below\nall operate on this bundle (auto-filled after a successful build).",
                  justify=tk.LEFT, foreground="gray").pack(
            anchor=tk.W, pady=(0, 2))
        self.bundle = FileSelector(g, "existing bundle:",
                                   mode="directory",
                                   help_text=HELP["bundle"])
        self.bundle.pack(fill=tk.X, pady=2)
        row = ttk.Frame(g)
        row.pack(fill=tk.X, pady=2)
        self.validate_btn = ttk.Button(row, text="Validate",
                                       command=self._run_validate)
        self.validate_btn.pack(side=tk.LEFT)
        ttk.Button(row, text="Show DOS", command=self._show_dos).pack(
            side=tk.LEFT, padx=(4, 0))

    def _build_emit(self, parent):
        """Build the emit section."""
        g = form_section(parent,
                         "Generate IRMA inputs (from the bundle above)")
        row = ttk.Frame(g)
        row.pack(anchor=tk.W, fill=tk.X, pady=2)
        ttk.Label(row, text="targets:").pack(side=tk.LEFT)
        self.targets = {}
        for name in _TARGETS:
            var = tk.BooleanVar(value=(name == "endf"))
            ttk.Checkbutton(row, text=name, variable=var).pack(
                side=tk.LEFT, padx=(6, 0))
            self.targets[name] = var
        InfoLabel(row, "targets", HELP["targets"]).pack(side=tk.LEFT,
                                                        padx=(6, 0))

        self.mats = LabeledEntry(g, "MAT numbers:", default="", width=28,
                                 help_text=HELP["mats"])
        self.mats.pack(fill=tk.X, pady=2)
        self.nuclides = LabeledEntry(g, "nuclides (opt):", default="",
                                     width=28, help_text=HELP["nuclides"])
        self.nuclides.pack(fill=tk.X, pady=2)
        self.species_over = LabeledEntry(g, "species overrides (opt):",
                                         default="", width=28,
                                         help_text=HELP["species_over"])
        self.species_over.pack(fill=tk.X, pady=2)
        self.temperature = LabeledEntry(g, "temperature (K):", default="296",
                                        width=8,
                                        help_text=HELP["temperature"])
        self.temperature.pack(fill=tk.X, pady=2)
        self.principal = LabeledEntry(g, "principal (opt):", default="",
                                      width=8, help_text=HELP["principal"])
        self.principal.pack(fill=tk.X, pady=2)
        # ENDF deck physics/format selectors + NCrystal material id: the
        # panel is a pure argv mapping onto `irma mlip emit`, so every emit
        # flag needs a control ('default' = omit the flag; an explicit value
        # is meaningful even when it names the CLI default, because the CLI
        # rejects explicit crystal-deck selectors on disordered bundles).
        self.emit_inelastic_mode = LabeledCombobox(
            g, "inelastic mode:", ["default", "0", "1", "2"],
            default="default", help_text=HELP["emit_inelastic_mode"])
        self.emit_inelastic_mode.pack(fill=tk.X, pady=2)
        self.emit_elastic_format = LabeledCombobox(
            g, "elastic format:", ["default", "mef", "sef"],
            default="default", help_text=HELP["emit_elastic_format"])
        self.emit_elastic_format.pack(fill=tk.X, pady=2)
        self.material_id = LabeledEntry(g, "material id (opt):", default="",
                                        width=18,
                                        help_text=HELP["material_id"])
        self.material_id.pack(fill=tk.X, pady=2)
        self.emit_outdir = FileSelector(g, "output dir (opt):",
                                        mode="directory",
                                        help_text=HELP["emit_outdir"])
        self.emit_outdir.pack(fill=tk.X, pady=2)
        self.allow_unstable = tk.BooleanVar(value=False)
        self._check_with_help(
            g, "allow unstable (imaginary modes)", self.allow_unstable,
            "Proceed with DOS-driven emission although the bundle records "
            "imaginary modes (the CLI refuses by default).")
        self.emit_overwrite = tk.BooleanVar(value=False)
        self._check_with_help(
            g, "overwrite emitted files", self.emit_overwrite,
            "Replace previously emitted inputs in the output directory.")
        row = ttk.Frame(g)
        row.pack(fill=tk.X, pady=4)
        self.emit_btn = ttk.Button(row, text="Emit inputs",
                                   command=self._run_emit)
        self.emit_btn.pack(side=tk.LEFT)

    def _build_envs(self, parent):
        """Build the potential-environments section."""
        g = form_section(parent, "Potential environments")
        intro = ttk.Label(
            g, justify=tk.LEFT, foreground="gray", wraplength=430,
            text=("Only needed when a potential\'s packages conflict with this\n"
                  "Python environment or are not installed (e.g. mace pins\n"
                  "e3nn==0.4.4, which sevennet/nequip forbid; grace runs on\n"
                  "TensorFlow). Create registers a dedicated env once; builds\n"
                  "with that potential then use it automatically. Potentials\n"
                  "importable here run in-process and need nothing below."))
        intro.pack(anchor=tk.W, pady=(0, 4))
        self.env_list = ttk.Label(g, text=self._env_summary(),
                                  justify=tk.LEFT, foreground="gray")
        self.env_list.pack(anchor=tk.W, pady=(0, 4))
        row = ttk.Frame(g)
        row.pack(fill=tk.X, pady=2)
        self.env_potential = LabeledCombobox(
            row, "potential:", list(POTENTIALS), default=POTENTIALS[0],
            help_text=HELP["env_potential"])
        self.env_potential.pack(side=tk.LEFT)
        self.env_create_btn = ttk.Button(row, text="Create",
                                         command=self._env_create)
        self.env_create_btn.pack(side=tk.LEFT, padx=(6, 0))
        self.env_remove_btn = ttk.Button(row, text="Remove",
                                         command=self._env_remove)
        self.env_remove_btn.pack(side=tk.LEFT, padx=(4, 0))
        ttk.Button(row, text="Refresh", command=self._env_refresh).pack(
            side=tk.LEFT, padx=(4, 0))

    def _build_output(self, parent):
        """Build the log + DOS-plot column."""
        nb = ttk.Notebook(parent)
        nb.pack(fill=tk.BOTH, expand=True)
        logf = ttk.Frame(nb)
        nb.add(logf, text="Log")
        self.log = ScrolledText(logf, height=24)
        self.log.pack(fill=tk.BOTH, expand=True)

        # DOS tab: a live matplotlib canvas of the bundle's dos.dat, drawn
        # after a successful build (or on demand for any selected bundle) --
        # the same lazy-FigureCanvasTkAgg pattern the NS panel uses.
        dosf = ttk.Frame(nb)
        nb.add(dosf, text="DOS")
        ctl = ttk.Frame(dosf)
        ctl.pack(fill=tk.X, pady=(2, 0))
        ttk.Label(ctl, text="y-axis:").pack(side=tk.LEFT, padx=(4, 2))
        self.dos_yscale = tk.StringVar(value="linear")
        for txt in ("linear", "log"):
            ttk.Radiobutton(ctl, text=txt, value=txt,
                            variable=self.dos_yscale,
                            command=self._replot_dos).pack(side=tk.LEFT)
        self._dos_holder = ttk.Frame(dosf)
        self._dos_holder.pack(fill=tk.BOTH, expand=True)
        self._out_nb = nb
        self._dos_tab = dosf
        self._dos_canvas = None
        self._dos_data = None

    # ---------------------------------------------------------------- argv ---
    @staticmethod
    def _pairs(label, text):
        """'A=1, B=2' -> ['A=1', 'B=2'] (validated non-empty tokens)."""
        toks = [t for t in text.replace(",", " ").split() if t]
        for t in toks:
            if "=" not in t or t.startswith("=") or t.endswith("="):
                raise ValueError(f"{label}: expected SYM=VALUE pairs, "
                                 f"got {t!r}")
        return toks

    def build_command(self):
        """Assemble the `irma mlip build` argv from the current widgets."""
        structure = self.structure.get().strip()
        if not structure:
            raise ValueError("structure file is required")
        outdir = self.outdir.get().strip()
        if not outdir:
            raise ValueError("bundle output directory is required")
        cmd = [sys.executable, "-u", "-m", "irma", "mlip", "build",
               structure, "-o", outdir,
               "--potential", self.potential.get()]
        if self.model.get().strip():
            cmd += ["--model", self.model.get().strip()]
        if self.supercell.get().strip():
            cmd += ["--supercell", self.supercell.get().strip()]
        if self.mesh.get().strip():
            cmd += ["--mesh", self.mesh.get().strip()]
        if self.delta.get().strip() not in ("", "0.03"):
            cmd += ["--delta", self.delta.get().strip()]
        if self.fmax.get().strip() not in ("", "0.01"):
            cmd += ["--fmax", self.fmax.get().strip()]
        if self.born.get().strip():
            cmd += ["--born", self.born.get().strip()]
        if self.relax_cell.get():
            cmd += ["--relax-cell"]
        if self.disordered.get():
            cmd += ["--disordered"]
        if self.snap_symmetry.get():
            cmd += ["--snap-symmetry"]
        jc = self.jitter_cycles.get().strip()
        if jc and jc != "0":
            cmd += ["--jitter-cycles", jc]
        if self.force.get():
            cmd += ["--force"]
        if self.overwrite.get():
            cmd += ["--overwrite"]
        jobs = parse_int("jobs", self.jobs.get() or "1")
        if jobs != 1:
            cmd += ["--jobs", str(jobs)]
        wt = parse_int("threads/worker", self.worker_threads.get() or "1")
        if wt != 1:
            cmd += ["--worker-threads", str(wt)]
        if self.threads.get().strip():
            cmd += ["--threads",
                    str(parse_int("serial threads", self.threads.get()))]
        return cmd

    def emit_command(self):
        """Assemble the `irma mlip emit` argv from the current widgets."""
        bundle = self.bundle.get().strip()
        if not bundle:
            raise ValueError("bundle directory is required")
        targets = [n for n in _TARGETS if self.targets[n].get()]
        if not targets:
            raise ValueError("select at least one emit target")
        cmd = [sys.executable, "-u", "-m", "irma", "mlip", "emit",
               bundle, "--to", ",".join(targets)]
        for pair in self._pairs("MAT numbers", self.mats.get()):
            cmd += ["--mat", pair]
        for pair in self._pairs("nuclides", self.nuclides.get()):
            cmd += ["--nuclide", pair]
        for group in [t for t in self.species_over.get().split() if t]:
            cmd += ["--species", group]
        if self.temperature.get().strip() not in ("", "296"):
            cmd += ["--temperature", self.temperature.get().strip()]
        if self.principal.get().strip():
            cmd += ["--principal", self.principal.get().strip()]
        # 'default' omits the flag (CLI default); explicit values are
        # forwarded verbatim -- including an explicit 'mef', which the CLI
        # distinguishes from the absent flag (rejected on disordered
        # bundles, exactly like --inelastic-mode).
        if self.emit_inelastic_mode.get() != "default":
            cmd += ["--inelastic-mode", self.emit_inelastic_mode.get()]
        if self.emit_elastic_format.get() != "default":
            cmd += ["--elastic-format", self.emit_elastic_format.get()]
        if self.material_id.get().strip():
            cmd += ["--material-id", self.material_id.get().strip()]
        if self.emit_outdir.get().strip():
            cmd += ["--out-dir", self.emit_outdir.get().strip()]
        if self.allow_unstable.get():
            cmd += ["--allow-unstable"]
        if self.emit_overwrite.get():
            cmd += ["--overwrite"]
        return cmd

    # -------------------------------------------------------------- actions --
    def _start(self, cmd, banner, success_msg, error_label, on_ok=None):
        """Run one mlip CLI command through the shared runner."""
        if self.runner.is_running or getattr(self, "_busy", False):
            messagebox.showwarning("Running",
                                   "A calculation is already in progress.")
            return
        self.log.clear()
        self.log.append(f"=== {banner} ===\n")
        self.log.append("$ " + " ".join(cmd[3:]) + "\n\n")
        self._status(banner + "...")
        self._set_busy(True)
        # token-scoped completion: the runner clears its running flag
        # before the Tk-thread update executes, so a stale completion
        # must never touch a newer action's state (review finding)
        token = object()
        self._active_token = token

        def done(ok, msg):
            self.after(0, self._finish, token, on_ok, ok, msg)
        self.runner.run_command(
            cmd, success_msg=success_msg,
            on_log=self._log_ts, on_done=done,
            error_label=error_label)

    def _run_build(self):
        """Start a bundle build."""
        try:
            cmd = self.build_command()
        except ValueError as exc:
            messagebox.showerror("Build", str(exc))
            return
        outdir = self.outdir.get().strip()

        def _ok():
            """Point the bundle section at the fresh build; show its DOS."""
            self.bundle.set(outdir)
            self._plot_dos(outdir)
        self._start(cmd, "MLIP bundle build",
                    f"Bundle written to {outdir}.", "MLIP build error",
                    on_ok=_ok)

    def _run_validate(self):
        """Validate the selected bundle."""
        bundle = self.bundle.get().strip()
        if not bundle:
            messagebox.showerror("Validate", "Select a bundle directory.")
            return
        self._start([sys.executable, "-u", "-m", "irma", "mlip",
                     "validate", bundle],
                    "MLIP bundle validation", "Bundle is valid.",
                    "MLIP validation error")

    def _run_emit(self):
        """Emit IRMA inputs from the selected bundle."""
        try:
            cmd = self.emit_command()
        except ValueError as exc:
            messagebox.showerror("Emit", str(exc))
            return
        self._start(cmd, "MLIP input generation",
                    "Inputs written (see the log for next steps).",
                    "MLIP emit error")

    # ---------------------------------------------------------------- envs ---
    @staticmethod
    def _env_summary():
        """One-line-per-potential summary of registered environments."""
        try:
            from irma.mlip import envs
            table = envs.list_envs()
        except Exception:
            return "registered: (unavailable)"
        if not table:
            return "registered: none (potentials run in this environment)"
        return "registered:\n" + "\n".join(
            f"  {p} -> {table[p]}" for p in sorted(table))

    def _env_refresh(self):
        """Refresh the registered-environments summary."""
        self.env_list.config(text=self._env_summary())

    def _env_create(self):
        """Provision + register a dedicated env for the chosen potential."""
        pot = self.env_potential.get()
        self._start([sys.executable, "-u", "-m", "irma", "mlip", "env",
                     "create", pot],
                    f"Provision environment for {pot}",
                    f"Environment for {pot} ready.",
                    "environment provisioning error",
                    on_ok=self._env_refresh)

    def _env_remove(self):
        """Unregister/delete the chosen potential's environment."""
        pot = self.env_potential.get()
        if not messagebox.askyesno(
                "Remove environment",
                f"Unregister {pot} (and delete its auto-provisioned "
                f"environment, if any)?"):
            return
        self._start([sys.executable, "-u", "-m", "irma", "mlip", "env",
                     "remove", pot],
                    f"Remove environment for {pot}",
                    f"Environment registration for {pot} removed.",
                    "environment removal error",
                    on_ok=self._env_refresh)

    # ------------------------------------------------------------ DOS plot --
    def _show_dos(self):
        """Plot the selected bundle's DOS on demand."""
        bundle = self.bundle.get().strip()
        if not bundle:
            messagebox.showerror("DOS", "Select a bundle directory.")
            return
        self._plot_dos(bundle)

    def _plot_dos(self, bundle_dir):
        """Load dos.dat (+ manifest phonon summary) and draw the DOS tab."""
        import json
        dos_path = os.path.join(bundle_dir, "dos.dat")
        if not os.path.isfile(dos_path):
            messagebox.showerror("DOS", f"no dos.dat in {bundle_dir}")
            return
        if os.path.getsize(dos_path) > 32 * 1024 * 1024:
            messagebox.showerror("DOS", f"{dos_path} is implausibly large "
                                        f"for a DOS table (>32 MB)")
            return
        try:
            import numpy as np
            data = np.loadtxt(dos_path, ndmin=2)
            if data.ndim != 2 or data.shape[1] < 2 or data.shape[0] < 2 \
                    or not np.isfinite(data[:, :2]).all():
                raise ValueError("expected a finite Nx2 table")
            title = os.path.basename(os.path.normpath(bundle_dir))
            try:
                m = json.load(open(os.path.join(bundle_dir,
                                                "manifest.json")))
                ph, calc = m["phonons"], m["calculator"]
                title = (f"{calc.get('potential', '?')} — freq_max "
                         f"{ph['freq_max_meV']:.2f} meV, "
                         f"{ph['n_imaginary']} imaginary, mesh "
                         f"{'x'.join(str(x) for x in ph['mesh'])}")
            except Exception:
                pass
            self._dos_data = (data, title)
        except Exception as exc:
            messagebox.showerror("DOS", f"could not read {dos_path}: {exc}")
            return
        self._replot_dos()
        self._out_nb.select(self._dos_tab)

    def _replot_dos(self):
        """(Re)draw the cached DOS data at the selected y-scale."""
        if self._dos_data is None:
            return
        try:
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
            from matplotlib.figure import Figure
        except ImportError:
            self.log.append("\n(matplotlib is not installed -- the DOS "
                            "plot tab is unavailable)\n")
            return
        data, title = self._dos_data
        fig = Figure(figsize=(6.4, 4.6), dpi=100)
        ax = fig.add_subplot(111)
        ax.plot(data[:, 0], data[:, 1], lw=1.4)
        ax.set_xlabel("energy (meV)")
        ax.set_ylabel("DOS (states/meV)")
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.3)
        if self.dos_yscale.get() == "log":
            positive = data[data[:, 1] > 0, 1]
            if positive.size:
                ax.set_yscale("log")
                ax.set_ylim(bottom=max(positive.min(), positive.max() * 1e-6))
        fig.tight_layout()
        if self._dos_canvas is not None:
            self._dos_canvas.get_tk_widget().destroy()
        self._dos_canvas = FigureCanvasTkAgg(fig, master=self._dos_holder)
        self._dos_canvas.draw()
        self._dos_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    # ------------------------------------------------------------- plumbing --
    def _cancel(self):
        """Cancel the running command."""
        if not self.runner.is_running:
            return
        self._status("Cancelling...")
        self.cancel_btn.config(state=tk.DISABLED)
        self.log.append("\n=== Cancelling (terminating workers) ===\n")
        self.runner.cancel()

    def _log_ts(self, text):
        """Append a log line from the worker thread (via after())."""
        self.after(0, self.log.append, text)

    def _set_busy(self, busy):
        """Enable/disable every action button as one unit."""
        state = tk.DISABLED if busy else tk.NORMAL
        for btn in (self.build_btn, self.emit_btn, self.validate_btn,
                    self.env_create_btn, self.env_remove_btn):
            btn.config(state=state)
        self.cancel_btn.config(state=tk.NORMAL if busy
                               else tk.DISABLED)
        self._busy = busy

    def _finish(self, token, on_ok, ok, msg):
        """Completion on the Tk thread; ignores superseded runs."""
        if token is not getattr(self, "_active_token", None):
            return                        # a newer action owns the panel
        self._active_token = None
        self._set_busy(False)
        self.log.append(f"\n{msg}\n")
        if ok:
            self._status("Done")
            if on_ok is not None:
                on_ok()
        elif msg.startswith("Calculation cancelled"):
            self._status("Cancelled")
        else:
            self._status("Error")
            messagebox.showerror("MLIP", msg[:500])

    def cleanup_temp_files(self):
        """No panel-owned temp files (the CLI owns its outputs)."""
