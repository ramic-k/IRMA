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

from irma.gui.element_table import (
    ElementTable, CUSTOM_FIELDS, MODE_CUSTOM, MODE_ISOTOPE,
    NUCLIDE_EDITOR_COLS)
from irma.gui.widgets import (
    LabeledEntry, LabeledCombobox, FileSelector, ScrolledText, InfoLabel,
    form_section, init_form_styles, parse_float, parse_int)
from irma.mlip.calculators import POTENTIALS

_TARGETS = ("endf", "spectra", "ncrystal")

_NO_BUNDLE_HINT = ("select a bundle above and its species appear here, one "
                   "row each, with the nuclear constants that will be "
                   "emitted.")
_NO_SPECIES_HINT = ("could not read a species list from this bundle's "
                    "phonopy.yaml; emission still works and uses the "
                    "natural element for every species.")


# ---------------------------------------------------------------------------
# Species discovery: the bundle's OWN phonopy.yaml, read as plain YAML.
#
# The GUI process is not guaranteed to have ase or phonopy (the potentials
# often live in their own environments), so neither the structure file nor
# phonopy itself may be touched here. The symbol list comes from the same
# light parser the NS tab's auto-fill button uses, so the two tabs can never
# disagree about what a model contains.
# ---------------------------------------------------------------------------
def _bundle_phonopy_yaml(bundle_dir):
    """The bundle's phonopy.yaml path, or None."""
    if not bundle_dir or not os.path.isdir(bundle_dir):
        return None
    try:
        from irma.mlip.bundle import load_bundle
        path = load_bundle(bundle_dir).phonopy_yaml
    except Exception:
        # not a bundle (yet), or an unreadable manifest: the conventional
        # name is still worth a try, and a miss just yields no species
        path = os.path.join(bundle_dir, "phonopy.yaml")
    return path if os.path.isfile(path) else None


def _phonopy_masses(path):
    """{symbol: mass in amu} from a phonopy.yaml, best effort.

    Walks the same cells in the same order as
    ``irma.spectra.config.phonopy_species`` so the masses line up with the
    symbols that function returns. First occurrence of a symbol wins; any
    parse failure yields {} and only costs the mass warning.

    Only the text BEFORE the embedded ``force_constants`` block is parsed.
    A bundle's phonopy.yaml carries its force constants inline and they are
    the overwhelming bulk of the file (94% of a 4-atom aluminium bundle,
    and the fraction grows with the supercell), while every cell is written
    ahead of them. Parsing the whole document here would double the cost of
    a bundle selection on the Tk thread for nothing.
    """
    try:
        import yaml
        head = []
        with open(path) as fh:
            for line in fh:
                if line.startswith("force_constants:"):
                    break
                head.append(line)
        doc = yaml.safe_load("".join(head))
    except Exception:
        return {}
    if not isinstance(doc, dict):
        return {}
    for key in ("primitive_cell", "unit_cell", "supercell"):
        cell = doc.get(key)
        if not (isinstance(cell, dict) and isinstance(cell.get("points"), list)):
            continue
        out = {}
        for pt in cell["points"]:
            if not isinstance(pt, dict):
                continue
            sym, mass = pt.get("symbol"), pt.get("mass")
            if sym and isinstance(mass, (int, float)) \
                    and not isinstance(mass, bool) and str(sym) not in out:
                out[str(sym)] = float(mass)
        if out:
            return out
    return {}


# The species read costs one plain-YAML parse of a file that embeds force
# constants, and it runs on the Tk thread from a variable trace. Memoize on
# (path, mtime, size) so re-selecting or retyping the same bundle is free
# and only a genuinely new (or rebuilt) bundle pays.
_SPECIES_CACHE = {}
_SPECIES_CACHE_MAX = 16


def bundle_species(bundle_dir):
    """``(symbols, {symbol: mass_amu})`` for a bundle directory.

    Returns ``([], {})`` for every degenerate case -- no directory, not a
    bundle, missing or unparseable phonopy.yaml, a cell the light parser
    does not recognise -- so the caller shows a hint instead of an empty
    grid and never has to catch anything.
    """
    path = _bundle_phonopy_yaml(bundle_dir)
    if not path:
        return [], {}
    try:
        st = os.stat(path)
    except OSError:
        return [], {}
    key = (path, st.st_mtime_ns, st.st_size)
    if key not in _SPECIES_CACHE:
        try:
            from irma.spectra.config import phonopy_species
            symbols = list(phonopy_species(path))
        except Exception:
            symbols = []
        masses = _phonopy_masses(path) if symbols else {}
        if len(_SPECIES_CACHE) >= _SPECIES_CACHE_MAX:
            _SPECIES_CACHE.clear()
        _SPECIES_CACHE[key] = (symbols, masses)
    symbols, masses = _SPECIES_CACHE[key]
    return list(symbols), dict(masses)


# ---------------------------------------------------------------------------
# Per-field help text (the '?' buttons).
# ---------------------------------------------------------------------------
HELP = {
    "structure": (
        "The crystal structure to build a phonon calculation (the material's "
        "atomic vibrations) for, in any format the ASE library "
        "reads: CIF, VASP POSCAR, xyz with a cell, and others.\n\nThe "
        "structure is relaxed (its atomic positions settled into the "
        "potential's energy minimum) with the same potential before the "
        "displacements, so a reasonable experimental or DFT geometry is a "
        "fine starting point."),
    "potential": (
        "The pretrained machine-learned interatomic potential that computes "
        "the forces. All backends are CPU-only by design and wired to "
        "conservative forces (forces derived from an energy function).\n\n"
        "Quality and character notes are in the manual section 'MLIP "
        "phonon calculations': nequip was the best all-around performer in the "
        "validation campaign; grace has an academic-use license; mace-off "
        "is for organic molecules only; dpa3 cannot bind van-der-Waals "
        "layered crystals.\n\nA potential whose packages are not installed "
        "here can still run through a registered dedicated environment; "
        "see the Environments section below."),
    "model": (
        "Specific checkpoint name or path (blank = the potential's default)."
        "\n\nPer-backend syntax:\n"
        "  pet-mad   NAME@VERSION (e.g. pet-mad-s@1.5.0); a bare name pins "
        "the newest release at build start\n"
        "  dpa3      MODEL::HEAD (default head MP_traj_v024_alldata_mixu)\n"
        "  nequip    model-zoo id (mir-group/NequIP-OAM-L:0.1) or a "
        "compiled .nequip.pt2 path\n"
        "  mace      size names or foundation names (medium-omat-0; an ASL "
        "license note is printed and recorded)\n"
        "  grace     foundation name (GRACE-2L-OAM, ...)\n"
        "  others    checkpoint name or file path"),
    "supercell": (
        "Supercell used for the "
        "finite-displacement force constants: an explicit 'n1 n2 n3', or a "
        "single number meaning the minimum lattice-parameter length in "
        "Angstrom (the default corresponds to 12).\n\nA bigger supercell "
        "captures longer-ranged force constants and costs more. Blank = "
        "the default (1 1 1 when 'disordered' is set)."),
    "mesh": (
        "Quick-look phonon mesh 'nx ny nz': the sampling grid for the "
        "bundle's DOS (density of states) and its census of imaginary "
        "modes.\n\n"
        "Blank = an automatic density (the Gamma point when 'disordered' "
        "is set). The validation campaign used 40 40 40 for production "
        "comparisons."),
    "delta": (
        "Finite-displacement amplitude in Angstrom: how far each atom is "
        "displaced when the forces are sampled.\n\nDefault 0.03, the value "
        "the validation campaign used everywhere."),
    "fmax": (
        "Force-convergence target for the relaxation, in eV/A: the FIRE "
        "optimizer stops once the largest remaining force is below this."
        "\n\nDefault 0.01. Tighten it for soft or nearly-unstable "
        "materials."),
    "jitter_cycles": (
        "When a relaxation fails to converge, kick the highest-force atoms "
        "(~0.05 A) and re-relax, up to this many extra cycles, keeping the "
        "lowest-residual frame seen.\n\nThis escapes stalls caused by MLIP "
        "force noise, common on glasses, where the FIRE optimizer can "
        "stall an order of magnitude above fmax. 0 = plain single-pass "
        "relaxation."),
    "relax_cell": (
        "Also relax the cell (lattice vectors + volume) with the potential, "
        "not just atomic positions.\n\nNeeds a checkpoint that provides "
        "stress. Validated: from starting points 3% apart, builds converge "
        "to the same lattice within ~0.001 A."),
    "disordered": (
        "Declare a disordered or amorphous model: the box is treated as "
        "its own supercell, the mesh defaults to the Gamma point, and "
        "emission switches to DOS-driven classic input files (incoherent "
        "elastic scattering scaled by the total bound cross section). "
        "Never auto-detected: you must check this box yourself."),
    "snap_symmetry": (
        "After relaxation, snap the atomic positions onto the exact orbits "
        "of the detected spacegroup (tolerance 1e-2 A).\n\nFloat32 "
        "relaxations often land a hair off the ideal Wyckoff sites (the "
        "exact symmetry positions); phonopy then sees P1 (no symmetry), "
        "generates several times more displacements, and a "
        "symmetry-reduced BORN file stops matching. The snap repairs all "
        "three (the shift is recorded in the manifest)."),
    "force": (
        "Proceed past an unconverged relaxation (the residual force is "
        "recorded in the bundle). Without this, a relaxation that fails to "
        "reach fmax stops the build (exit code 3)."),
    "born": (
        "Optional phonopy BORN file (Born effective charges plus "
        "dielectric tensor). It is embedded into the bundle; the "
        "non-analytical correction (NAC) is never read from the working "
        "directory.\n\nEssential for ionic crystals: without it even a "
        "perfect model shows spurious imaginary modes."),
    "jobs": (
        "How many displaced supercells are computed at once, as "
        "separate worker processes. This is the main speed knob for "
        "heavy potentials: raise it toward your core count, ideally a "
        "divisor of the displacement count (ZrO2, 18 displacements: "
        "jobs=9 was 21x faster than serial). Fast potentials "
        "(mattersim) finish in seconds regardless.\n\nWith jobs = 1 "
        "everything runs in one process and 'serial threads' applies; "
        "with jobs > 1 each worker gets 'threads/worker' CPU threads. "
        "The form shows whichever of the two the current jobs value "
        "makes live."),
    "worker_threads": (
        "CPU threads each parallel worker's math libraries may use; "
        "read only when jobs > 1. Keep jobs x threads at or below your "
        "core count. The measured optimum is 1: several workers with "
        "threaded math libraries oversubscribe the cores and run "
        "slower, not faster."),
    "threads": (
        "CPU threads for the single-process path; read only when "
        "jobs = 1. Blank = all cores. Compiled backends (nequip) are "
        "single-threaded by construction, so this has little effect "
        "there; raise jobs instead."),
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
        "  endf      one ready-to-run input file per principal scatterer (the "
        "species an ENDF evaluation is written for)\n"
        "  spectra   an `irma spectra` YAML config\n"
        "  ncrystal  an exporter YAML for `irma ncrystal`\n\n"
        "Nothing is executed. Inspect the generated files, then run each "
        "program yourself; the emit log prints the exact commands."),
    "mats": (
        "ENDF target only. The ENDF MAT number (the integer that "
        "identifies a material in an ENDF library) for each species, as "
        "SYM=INT pairs separated by spaces or commas (e.g.  C=31  or  "
        "Be=26, O=48). Required for the endf target; the species must "
        "exist in the bundle."),
    "nuclear_data": (
        "Applies to every target: this is where the emitted scattering "
        "constants come from. One row per species in the bundle, listed "
        "from the bundle's own phonopy.yaml once you select it above.\n\n"
        "Each row picks where its constants come from:\n"
        "  natural   the natural element (the default, and what the CLI "
        "emits with no flags): ENDF codes the identity A = 0 and the "
        "constants are that element's natural-abundance values, from the "
        "same table entry.\n"
        "  isotope   choose one of the element's isotopes; the identity "
        "AND the constants both switch to it.\n"
        "  custom    type b_coh_fm and sigma_inc_b (and awr) yourself. "
        "sigma_bound_b is always derived from b_coh_fm and sigma_inc_b, "
        "never entered.\n\n"
        "The constants are shown for every row, read-only unless the row "
        "is custom, so you can see what will be written before you emit.\n\n"
        "Some tabulated scattering lengths (natural B, Cd, In, Sm, Eu, Gd; "
        "and per isotope, e.g. 6-Li but not natural Li, 10-B but not "
        "11-B) are resonance-region, energy-dependent values that are not "
        "safe as static constants. Those rows open as custom by "
        "themselves and ask you for the numbers.\n\n"
        "Picking an isotope changes the scattering constants but not the "
        "masses in the phonon calculation: the row warns when the two disagree "
        "materially, because the emitted inputs keep pointing at this "
        "bundle's phonopy.yaml."),
    "temperature": (
        "Temperature in K for the emitted inputs.\n\nDefault 296."),
    "emit_inelastic_mode": (
        "ENDF target only. "
        "Physics level of the emitted ENDF input files (--inelastic-mode).\n\n"
        "  default: omit the flag; the CLI emits mode-2 input files.\n"
        "  0: the classic isotropic option built from the bundle's "
        "species-projected DOS (the principal spectrum on the classic "
        "cards, Card 6e partial spectra for the other species).\n"
        "  1/2: directional input files computed from the phonopy calculation.\n\n"
        "Not applicable to disordered bundles: the CLI rejects an explicit "
        "mode there, since they always use the DOS-driven classic option."),
    "emit_elastic_format": (
        "ENDF target only. Elastic output convention of the emitted ENDF "
        "input files (--elastic-format).\n\n"
        "  default: omit the flag; the CLI emits MEF input files.\n"
        "  mef (mixed elastic format): both elastic components for every "
        "species.\n"
        "  sef (single-channel elastic format): the complete coherent "
        "component assigned to the designated-coherent (DC) atom.\n\n"
        "An EXPLICIT value, even 'mef', which names the default, is a "
        "crystal-input selector and is rejected for disordered bundles, so "
        "'default' and 'mef' are distinct choices here."),
    "material_id": (
        "NCrystal target only. NCrystal export material id (--material-id). Blank = the CLI "
        "default (mlip_model)."),
    "emit_outdir": (
        "Directory for the emitted inputs (blank = the bundle directory)."),
    "env_potential": (
        "Potential to provision a dedicated environment for. Use this when "
        "the packages conflict with this environment (e.g. mace's "
        "e3nn==0.4.4 pin) or are simply not installed.\n\n'Create' builds "
        "a standard virtual environment (uv or python -m venv, never "
        "conda; with uv the environment is pinned to Python 3.12, since "
        "the potential packages lag new interpreters), installs the "
        "potential's packages at whatever versions the package index "
        "currently serves, verifies the import plus a torch/NumPy "
        "compatibility probe, and registers the environment; builds with "
        "that potential then use it automatically. 'Remove' unregisters "
        "and deletes a provisioned environment."),
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
        """Checkbox with an attached help glyph; returns the row frame so
        callers can show/hide it."""
        row = ttk.Frame(parent)
        row.pack(anchor=tk.W, fill=tk.X, pady=2)
        ttk.Checkbutton(row, text=text, variable=var).pack(side=tk.LEFT)
        InfoLabel(row, text, help_text).pack(side=tk.LEFT, padx=(4, 0))
        return row

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
        self.potential_status = ttk.Label(g, justify=tk.LEFT,
                                          foreground="gray", wraplength=430)
        self.potential_status.pack(anchor=tk.W, pady=(0, 2))
        self.potential.combo.bind(
            "<<ComboboxSelected>>",
            lambda e: self._update_potential_status())
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
        # Progressive disclosure: only the thread field the current jobs
        # value makes live is shown (serial threads for jobs = 1,
        # threads/worker for jobs > 1).
        self.jobs.var.trace_add("write",
                                lambda *a: self._sync_thread_rows())
        self._sync_thread_rows()
        self._update_potential_status()

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
            var.trace_add("write", lambda *a: self._sync_emit_rows())
            self.targets[name] = var
        InfoLabel(row, "targets", HELP["targets"]).pack(side=tk.LEFT,
                                                        padx=(6, 0))

        self.mats = LabeledEntry(g, "MAT numbers:", default="", width=28,
                                 help_text=HELP["mats"])
        self.mats.pack(fill=tk.X, pady=2)

        # Per-species nuclear-data editor (the --nuclide / --species
        # mini-languages, made visible). It replaces two free-text fields
        # that required knowing both syntaxes and showed nothing of what
        # would actually be emitted.
        self.species_frame = ttk.Frame(g)
        self.species_frame.pack(anchor=tk.W, fill=tk.X, pady=(6, 0))
        hdr = ttk.Frame(self.species_frame)
        hdr.pack(anchor=tk.W, fill=tk.X)
        ttk.Label(hdr, text="nuclear data (per species):",
                  foreground="gray").pack(side=tk.LEFT)
        InfoLabel(hdr, "nuclear data", HELP["nuclear_data"]).pack(
            side=tk.LEFT, padx=(4, 0))
        self.species_table = ElementTable(self.species_frame,
                                          nuclide_editor=True)
        self.species_table.set_visible(NUCLIDE_EDITOR_COLS)
        self.species_table.pack(fill=tk.X, pady=2)
        self.species_table.set_hint(_NO_BUNDLE_HINT)
        # the table follows the bundle selection, including the auto-fill
        # after a successful build
        self.bundle.var.trace_add("write",
                                  lambda *a: self._sync_species_table())
        self._sync_species_table()

        self.temperature = LabeledEntry(g, "temperature (K):", default="296",
                                        width=8,
                                        help_text=HELP["temperature"])
        self.temperature.pack(fill=tk.X, pady=2)
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
        # --allow-unstable is read by the endf and spectra emitters only
        # (never by ncrystal), so _sync_emit_rows shows/hides this row.
        self._allow_unstable_row = self._check_with_help(
            g, "allow unstable (imaginary modes)", self.allow_unstable,
            "ENDF and spectra targets only. Proceed with DOS-driven "
            "emission even though the bundle records imaginary modes; the "
            "CLI refuses by default.")
        self.emit_overwrite = tk.BooleanVar(value=False)
        self._overwrite_row = self._check_with_help(
            g, "overwrite emitted files", self.emit_overwrite,
            "Replace previously emitted inputs in the output directory.")
        row = ttk.Frame(g)
        row.pack(fill=tk.X, pady=4)
        self._sync_emit_rows()
        self.emit_btn = ttk.Button(row, text="Emit inputs",
                                   command=self._run_emit)
        self.emit_btn.pack(side=tk.LEFT)

    def _build_envs(self, parent):
        """Build the potential-environments section."""
        g = form_section(parent, "Potential environments")
        intro = ttk.Label(
            g, justify=tk.LEFT, foreground="gray", wraplength=430,
            text=("Each potential is a stack of heavy Python packages, and\n"
                  "most stacks cannot share one environment (mace pins a\n"
                  "library version nequip and sevennet forbid; grace runs\n"
                  "on TensorFlow). A potential runs one of two ways:\n"
                  "installed in THIS environment, in-process, with nothing\n"
                  "needed here; or in a dedicated environment IRMA builds\n"
                  "for it. For the second, pick the potential and press\n"
                  "Create, once per potential (a few minutes of downloads).\n"
                  "Every later build with that potential then uses its\n"
                  "environment automatically. Remove deletes a dedicated\n"
                  "environment; Refresh re-reads the list."))
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
        # Only the thread field the jobs value makes LIVE is parsed and
        # forwarded, the same rule _sync_thread_rows uses to show it:
        # threads/worker for jobs > 1, serial threads for jobs = 1. The CLI
        # reads --threads for the parent relaxation whatever jobs says, so a
        # stale value left in the hidden field used to cap the relaxation
        # silently (and a nonnumeric one blocked Build from a field the user
        # could not reach).
        if jobs > 1:
            wt = parse_int("threads/worker", self.worker_threads.get() or "1")
            if wt != 1:
                cmd += ["--worker-threads", str(wt)]
        elif self.threads.get().strip():
            cmd += ["--threads",
                    str(parse_int("serial threads", self.threads.get()))]
        return cmd

    def _sync_species_table(self):
        """Fill the nuclear-data table from the selected bundle.

        Runs on the Tk thread from a variable trace, so it must never
        raise: every failure mode collapses into "no rows plus a hint".
        """
        try:
            path = self.bundle.get().strip()
            symbols, masses = (bundle_species(path) if path else ([], {}))
            self.species_table.set_species(symbols, masses)
            if not path:
                self.species_table.set_hint(_NO_BUNDLE_HINT)
            elif not symbols:
                self.species_table.set_hint(_NO_SPECIES_HINT)
            else:
                self.species_table.set_hint("")
        except Exception as exc:                       # never on the UI thread
            try:
                self.species_table.set_species([])
                self.species_table.set_hint(
                    f"{_NO_SPECIES_HINT} ({exc.__class__.__name__})")
            except Exception:
                pass

    def _nuclear_data_args(self):
        """``[(flag, value), ...]`` from the per-species nuclear-data table.

        The CLI is untouched, so this is a pure assembly of the two flags
        the free-text fields used to carry:

        * ``natural`` (the default of every row) contributes NOTHING. A
          table nobody edited therefore produces an argv with neither
          ``--nuclide`` nor ``--species``, which is byte-for-byte the
          emission the CLI already performed.
        * ``isotope`` contributes ``--nuclide SYM=<A>-SYM``; the CLI takes
          the identity and the constants from that one table entry.
        * ``custom`` contributes ``--species SYM:field=value,...`` over the
          non-empty boxes, in the emitter's own field order. A custom row
          that also carries an isotope (the energy-dependent recovery keeps
          the identity the user picked) contributes both flags, which is
          exactly how the CLI composes them: identity from the isotope,
          constants from the override.

        The two refusals mirror ``irma.mlip.emit.resolve_species`` so the
        form reports them before the subprocess starts, in the language of
        the row rather than of a flag.
        """
        args = []
        for row in self.species_table.nuclide_rows():
            sym = row["symbol"]
            if not sym:
                continue
            mode, label = row["mode"], row["nuclide"]
            if mode in (MODE_ISOTOPE, MODE_CUSTOM) and label:
                args.append(("--nuclide", f"{sym}={label}"))
            if mode != MODE_CUSTOM:
                continue
            fields = [(k, row[k]) for k in CUSTOM_FIELDS if row[k]]
            for key, value in fields:
                # the group syntax is comma- and colon-delimited, so a
                # non-numeric entry has to be caught by NAME here rather
                # than reaching the CLI as a malformed group
                parse_float(f"species {sym} {key}", value)
            if row["flagged"] and not row["complete"]:
                raise ValueError(
                    f"species {sym}: the tabulated scattering length for "
                    f"{label or sym} is energy-dependent (a resonance-region "
                    f"value) and cannot be used as a static constant; enter "
                    f"both b_coh_fm and sigma_inc_b in its row")
            if not fields:
                raise ValueError(
                    f"species {sym}: 'custom' is selected but no scattering "
                    f"constants were entered; fill the row, or set it back "
                    f"to 'natural'")
            args.append(("--species", f"{sym}:" + ",".join(
                f"{k}={v}" for k, v in fields)))
        return args

    def emit_command(self):
        """Assemble the `irma mlip emit` argv from the current widgets.

        Only the fields the SELECTED targets read are parsed and forwarded,
        mirroring the disclosure rules of _sync_emit_rows exactly (MAT
        numbers, inelastic mode and elastic format: endf; material id:
        ncrystal). A hidden field is neither validated nor emitted, so a
        leftover MAT from an earlier ENDF emit cannot block an
        NCrystal-only emit through a check whose control is off screen.
        """
        bundle = self.bundle.get().strip()
        if not bundle:
            raise ValueError("bundle directory is required")
        targets = [n for n in _TARGETS if self.targets[n].get()]
        if not targets:
            raise ValueError("select at least one emit target")
        cmd = [sys.executable, "-u", "-m", "irma", "mlip", "emit",
               bundle, "--to", ",".join(targets)]
        if "endf" in targets:
            for pair in self._pairs("MAT numbers", self.mats.get()):
                cmd += ["--mat", pair]
        # the per-species nuclear-data table applies to every target (it
        # sets the emitted constants), so it is never target-scoped
        for flag, value in self._nuclear_data_args():
            cmd += [flag, value]
        if self.temperature.get().strip() not in ("", "296"):
            cmd += ["--temperature", self.temperature.get().strip()]
        # 'default' omits the flag (CLI default); explicit values are
        # forwarded verbatim -- including an explicit 'mef', which the CLI
        # distinguishes from the absent flag (rejected on disordered
        # bundles, exactly like --inelastic-mode).
        if "endf" in targets:
            if self.emit_inelastic_mode.get() != "default":
                cmd += ["--inelastic-mode", self.emit_inelastic_mode.get()]
            if self.emit_elastic_format.get() != "default":
                cmd += ["--elastic-format", self.emit_elastic_format.get()]
        if "ncrystal" in targets and self.material_id.get().strip():
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

    def _sync_emit_rows(self):
        """Progressive disclosure for the emit form: show only the
        fields the selected targets read (MAT numbers, inelastic mode
        and elastic format are ENDF-only, material id is NCrystal-only,
        allow-unstable is read by the endf and spectra emitters;
        everything else applies to every target)."""
        if not hasattr(self, "_overwrite_row"):
            return                      # section not fully built yet
        endf = self.targets["endf"].get()
        spectra = self.targets["spectra"].get()
        ncrystal = self.targets["ncrystal"].get()
        for w in (self.mats, self.emit_inelastic_mode,
                  self.emit_elastic_format, self.material_id,
                  self._allow_unstable_row):
            w.pack_forget()
        if endf:
            self.mats.pack(fill=tk.X, pady=2, before=self.species_frame)
        # the conditional middle rows anchor on the always-visible
        # temperature field, keeping the construction order
        # (inelastic mode, elastic format) stable
        if endf:
            self.emit_inelastic_mode.pack(fill=tk.X, pady=2,
                                          after=self.temperature)
            self.emit_elastic_format.pack(fill=tk.X, pady=2,
                                          after=self.emit_inelastic_mode)
        if ncrystal:
            self.material_id.pack(fill=tk.X, pady=2,
                                  before=self.emit_outdir)
        if endf or spectra:
            self._allow_unstable_row.pack(anchor=tk.W, fill=tk.X, pady=2,
                                          before=self._overwrite_row)

    def _sync_thread_rows(self):
        """Show the thread field the current jobs value makes live:
        'threads/worker' is read only when jobs > 1, 'serial threads'
        only when jobs = 1."""
        try:
            parallel = int(float(self.jobs.get().strip() or "1")) > 1
        except ValueError:
            parallel = False
        if parallel:
            self.threads.pack_forget()
            self.worker_threads.pack(fill=tk.X, pady=2, before=self.outdir)
        else:
            self.worker_threads.pack_forget()
            self.threads.pack(fill=tk.X, pady=2, before=self.outdir)

    def _update_potential_status(self):
        """One line under the build selector: can the chosen potential
        run right now, and if not, what to do about it."""
        name = self.potential.get()
        try:
            from irma.mlip.envs import registered_interpreter
            interp = registered_interpreter(name)
        except Exception:
            interp = None
        if interp:
            text, color = "ready: runs in its dedicated environment", "gray"
        else:
            from importlib.util import find_spec
            from irma.mlip.calculators import _PACKAGES
            module = _PACKAGES.get(name, ("", ""))[1]
            try:
                found = bool(module) and find_spec(module) is not None
            except (ImportError, ValueError):
                found = False
            if found:
                text, color = "ready: installed in this environment", "gray"
            else:
                text, color = ("not installed: create its environment in "
                               "'Potential environments' below, or pip "
                               "install it into this one", "#e5a145")
        self.potential_status.config(text=text, foreground=color)

    def _env_refresh(self):
        """Refresh the registered-environments summary."""
        self.env_list.config(text=self._env_summary())
        self._update_potential_status()

    def _env_create(self):
        """Build + register a dedicated env for the chosen potential."""
        pot = self.env_potential.get()
        self._start([sys.executable, "-u", "-m", "irma", "mlip", "env",
                     "create", pot],
                    f"Build environment for {pot}",
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
