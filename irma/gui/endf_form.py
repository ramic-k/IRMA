"""ENDF-evaluation form for the IRMA GUI (the LEAPR-parity deck panel).

Extracted verbatim from :mod:`irma.gui.app` so the ENDF form is reviewable as
one module, mirroring the NS/NCrystal panel extraction. It is a MIXIN rather
than a self-contained panel class deliberately: the form's ~700 widget
attributes live directly on the application object (``self.awr``,
``self.iel_var``, ...), and the deck round-trip tests and
:mod:`irma.gui.deck_text` address them there — a separate namespace would
force hundreds of delegation shims for zero behavioral gain. ``IrmaApp``
inherits this mixin; every attribute name is unchanged.

The card-by-card deck format reference lives in ``docs/input-reference.md``.
"""

import os
import tempfile
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

from irma.core.noncubic_inelastic import MIN_PHONON_ENERGY_HELP
from irma.gui.widgets import (
    LabeledEntry, LabeledCombobox, FileSelector, ScrolledText, InfoLabel,
    ToolTip, fixed_font, form_section, init_form_styles,
    parse_float, parse_int
)
from irma.gui.deck_text import (
    _quote, fmt_array, emit_comment_lines, parse_atoms_text,
    parse_deck_to_staging,
)
from irma.core.extinction import EXTINCTION_MODELS

# Distribution options per extinction-model family (Becker-Coppens vs Sabine).
# The dropdown is restricted to the active model's family (see
# _on_ext_model_change); the deck parser independently validates the per-model
# allowed set. _EXT_DISTS is the full union, used only to size the widget.
_EXT_DIST_BC = ["Gauss", "Lorentz", "Fresnel"]
_EXT_DIST_SABINE = ["rect", "tri"]
_EXT_DISTS = _EXT_DIST_BC + _EXT_DIST_SABINE

# iel dropdown entries. The menu is narrowed to _IEL_GENERALIZED alone while a
# phonopy-backed inelastic mode is selected (see _sync_modes).
_IEL_GENERALIZED = "10 — Generalized (crystal structure)"
_IEL_CHOICES = ["0 — None", "1 — Graphite (legacy)",
                "2 — Be (legacy)", "3 — BeO (legacy)",
                "4 — Al (legacy)", "5 — Pb (legacy)",
                "6 — Fe (legacy)", _IEL_GENERALIZED]


def _set_values(box, values):
    """Replace a text box's contents with the values in %.6e."""
    box.delete("1.0", tk.END)
    box.insert("1.0", " ".join(f"{v:.6e}" for v in values))


def _detail_lines(delta, rho, twt_c_tbeta, e_box, w_box, what):
    """Deck lines of one temperature's phonon model: the spectrum
    (Cards 11-12), twt c tbeta (Card 13) and the oscillators (Cards 14-16).
    what names the oscillator boxes in errors ("Discrete", "Secondary")."""
    e = [parse_float(f"{what} oscillator energies", x)
         for x in e_box.get("1.0", tk.END).split()]
    w = [parse_float(f"{what} oscillator weights", x)
         for x in w_box.get("1.0", tk.END).split()]
    if len(e) != len(w):
        raise ValueError(
            f"{what} oscillators: {len(e)} energies but {len(w)} weights; "
            "give one weight per energy, or leave both empty.")
    out = [f"{delta:.6e} {len(rho)} /", fmt_array(rho) + " /",
           twt_c_tbeta + " /"]
    if e:
        out += [f"{len(e)} /", " ".join(f"{v:.6e}" for v in e) + " /",
                " ".join(f"{v:.6e}" for v in w) + " /"]
    else:
        out.append("0 /")
    return out


def _show(widget, on, **pack):
    """Pack the widget with the given options, or unpack it."""
    if on:
        widget.pack(**pack)
    else:
        widget.pack_forget()


STRUCTURE_FILL_TITLE = "Fill structure from phonopy.yaml"
STRUCTURE_FILL_HELP = (
    "Prefills the iel=10 crystal structure -- the Lattice Parameters fields "
    "and the whole 'Atom Types in Unit Cell' block -- from the phonopy calculation "
    "named above (Card 6f).\n\n"
    "The cell read is the phonopy PRIMITIVE cell, loaded through phonopy "
    "itself. That is the cell the input file must describe: for inelastic_mode 1/2 "
    "IRMA matches every Card 6d position against the phonopy primitive cell's "
    "atom positions and refuses the input file when they do not correspond.\n\n"
    "Nothing happens until you press Apply in the preview: the button never "
    "fires on its own, not when you pick a phonopy.yaml, not on input-file import, "
    "not on reset. Apply is all-or-nothing -- it replaces the lattice AND the "
    "whole atom block together, so phonopy positions can never end up beside "
    "a hand-typed lattice.\n\n"
    "Each species is filled as the NATURAL ELEMENT (A = 0) with "
    "natural-abundance constants, because a phonopy calculation names elements, not "
    "isotopes. Isotopic materials must be edited afterwards.")

# Material-IDENTITY fields (ZA, MAT, AWR, sigma_free, the lattice, the Card 6d
# atom types) ship BLANK. A prefilled value there is IRMA asserting something
# about the user's material that it cannot know, and a plausible-but-wrong
# number survives review far more easily than an empty field. Methodology
# defaults (mesh, ndir/mpdir, auto-order, the grid knobs, npr, temperature)
# stay prefilled: those encode what the validation campaign established.
# One gray hint line per blanked group names the ways to fill it.
_FILL_ROUTES = ("Import Input File (top of the Material part) or a committed "
                "input file under examples/tsl/")
IDENTITY_HINT_SCATTERER = (
    "Blank on purpose — ZA, AWR and sigma_free describe YOUR material. Type "
    "them in, press 'Fill AWR + sigma_free from ZA', or load an input file with "
    + _FILL_ROUTES + ".")
IDENTITY_HINT_MAT = (
    "Blank on purpose — the MAT number labels YOUR evaluation. Assign one, or "
    "load an input file with " + _FILL_ROUTES + ".")
IDENTITY_HINT_LATTICE = (
    "Blank on purpose — the unit cell describes YOUR material. Type it in, "
    "press '" + STRUCTURE_FILL_TITLE + "' (inelastic_mode 1/2), or load an "
    "input file with " + _FILL_ROUTES + ".")
IDENTITY_HINT_ATOMS = (
    "Blank on purpose — the atom types describe YOUR material. Type them in, "
    "press '" + STRUCTURE_FILL_TITLE + "' (inelastic_mode 1/2), or load an "
    "input file with " + _FILL_ROUTES + ".")

EXT_HELP = {
    "about": (
        "Crystalline EXTINCTION is a reduction of the Bragg-peak intensity in "
        "a real crystallite: once a beam is strongly diffracted it cannot "
        "diffract again, so the measured coherent-elastic cross section is "
        "LOWER than the ideal (kinematic) theory predicts.\n\n"
        "Extinction is a property of the SAMPLE (crystallite size, mosaic, "
        "grain size), NOT of the material itself. Fit l/g/L to a measured "
        "transmission (as in Xu 2025) or take them from the known "
        "microstructure. OFF by default: the tape is then the ideal-crystal "
        "Bragg edges, byte-identical to today's output.\n\n"
        "Extinction is strongest at long wavelength (low energy) and vanishes "
        "above ~0.1 eV. IRMA applies a per-plane factor y in (0,1] to the "
        "coherent-elastic Bragg edges and writes a standard histogram MF7/MT2 "
        "table; the high-energy edges are unchanged.\n\n"
        "The models and recipes are ported from the NCrystal CrysXT plugin. "
        "See the '?' on the attribution line for references."),
    "model": (
        "Extinction model (5 available):\n"
        "  - Sabine_uncorr / Sabine_corr: Sabine's analytic block model "
        "(Int. Tables Vol C, 6.4); a single knob.\n"
        "  - BC_pure: Becker-Coppens, primary OR secondary extinction only "
        "(not both).\n"
        "  - BC_mix: Becker-Coppens, coupled primary + secondary (the general "
        "case).\n"
        "  - BC_mod: Becker-Coppens 'modified', secondary only, with no "
        "primary extinction factor (the block size l still enters the "
        "secondary term).\n\n"
        "BC_mix and BC_mod couple the two extinction mechanisms and REQUIRE "
        "l>0, g>0 and L>0; l is needed even for the no-primary BC_mod because "
        "it enters the secondary term. For primary-only (one knob) use BC_pure "
        "with only l set; for secondary-only use BC_pure with g>0 and L>0 "
        "(l=0)."),
    "l": (
        "l is the crystallite (mosaic-block) size in Å. It drives PRIMARY "
        "extinction (multiple scattering within one perfect block). For "
        "BC_pure, leave l=0 to select the pure-secondary mechanism instead; "
        "BC_mix and BC_mod require l>0. Fitted values are typically microns "
        "(1 µm = 1e4 Å)."),
    "g": (
        "g is the mosaic spread in rad⁻¹ (the width of the distribution of "
        "block orientations). Together with L it drives SECONDARY extinction "
        "(block-to-block beam depletion). Must be > 0 for BC_mix/BC_mod and "
        "for any secondary mechanism."),
    "L": (
        "L is the grain (particle) size in Å. Together with g it drives "
        "SECONDARY extinction. Set 0 (with g=0) to disable the secondary "
        "mechanism. Must be > 0 for BC_mix/BC_mod."),
    "dist": (
        "Shape of the tilt (mosaic) distribution, the assumed spread of block "
        "orientations.\n"
        "  - Becker-Coppens models: Gauss / Lorentz / Fresnel.\n"
        "  - Sabine models: rect / tri.\n"
        "Choose a value valid for the selected model; the default follows the "
        "model (Gauss for BC, rect for Sabine)."),
    "recipe": (
        "Numerical recipe for evaluating the Becker-Coppens y(x,θ) "
        "functions:\n"
        "  - std (default): the BC2025 recipe (Kittelmann 2026); accurate and "
        "robust.\n"
        "  - cls: the original BC1974 closed forms (can be numerically fragile "
        "at strong extinction or backscatter).\n"
        "The analytic Sabine models ignore this setting."),
    "rmse_tol": (
        "Tolerance for the adaptive MF7/MT2 tabulation of the extinction-"
        "corrected cross section below the cutoff energy. Smaller values give "
        "more table points and higher fidelity. The default 1e-3 gives a "
        "faithful tape (~0.04% RMSE on THERMR-step reconstruction) at a modest "
        "point count."),
    "attribution": (
        "The extinction models and recipes are PORTED (not imported) from the "
        "NCrystal CrysXT plugin:\n\n"
        "  - ncplugin-CrysXT: https://github.com/dddijulio/ncplugin-CrysXT\n\n"
        "References (please read these to understand the physics):\n\n"
        "  - T. Kittelmann, D. D. DiJulio, S. Xu & J. I. Marquez Damian, "
        "'Revisiting Becker-Coppens (1974): updated recipes for estimating "
        "extinction factors in spherical crystallites', Acta Cryst. (2026) "
        "A82, 163-178. DOI 10.1107/S2053273326001245 (the BC2025 'std' "
        "recipes).\n\n"
        "  - S. Xu et al., 'Impact of extinction effects on neutron "
        "transmission in solid beryllium metal', J. Appl. Cryst. (2025) 58, "
        "1957-1966. DOI 10.1107/S1600576725007939 (concept, motivation, and "
        "how to fit l/g/L to a measured transmission).\n\n"
        "  - P. J. Becker & P. Coppens, Acta Cryst. (1974) A30, 129.\n"
        "  - T. M. Sabine, International Tables for Crystallography (2006), "
        "Vol. C, ch. 6.4."),
}


def _phase_from_log_line(line):
    """Extract a short phase label from a streamed child log line, or None.

    The out-of-process compute child prints human phase markers ending in
    ``...`` ("Accumulating coherent one-phonon contribution...", "Writing ENDF
    output..."). Surfacing those on the status line turns a long run's static
    "Running..." into live progress. Non-marker output (data rows, ``===``
    banners, warnings) returns None so the current phase is left unchanged --
    the readout is purely additive and never wrong-by-omission.
    """
    s = line.strip()
    if not s.endswith("...") or s.startswith("==="):
        return None
    # Upper bound guards against a stray long line being pinned as a phase; 120
    # clears the longest real marker ("Running phonopy mesh (...) again with
    # symmetry reduction for incoherent one-phonon mode sums..." is ~105 chars).
    if not (3 < len(s) <= 120):
        return None
    # Every real engine marker is a multi-word sentence; requiring a space
    # rejects bare single-word progress dots a library might print
    # ("Loading...", "Retrying...") that would otherwise pin a misleading phase.
    if " " not in s[:-3].strip():
        return None
    return s


class EndfFormMixin:
    """The ENDF Evaluation panel: tab builders, deck generation, run/log
    wiring, and deck import/export. Mixed into :class:`irma.gui.app.IrmaApp`;
    reads and writes its widget attributes on ``self``."""

    def _build_endf_form(self, page):
        """Build the single-page ENDF form: a jump bar over one scrolling
        column holding the five parts (Material, Scattering, Grids, Phonon,
        Run) in deck-writing order.

        The parts were separate notebook tabs through v0.19. One page keeps
        the whole deck visible in the order it is written — fill top to
        bottom, press Run — and a deck import can no longer change a field
        on a tab the user is not looking at. The jump bar preserves the
        tabs' direct navigation: each button scrolls its part to the top.
        """
        init_form_styles()
        import tkinter.font as tkfont
        if "IrmaPartFont" not in tkfont.names():
            base = tkfont.nametofont("TkDefaultFont")
            tkfont.Font(name="IrmaPartFont", family=base.cget("family"),
                        size=base.cget("size") + 6, weight="bold")
        ttk.Style().configure("Part.TLabel", font="IrmaPartFont")

        self._endf_parts = {}
        bar = ttk.Frame(page)
        bar.pack(fill=tk.X, padx=10, pady=(6, 2))
        ttk.Label(bar, text="Jump to:", style="Hint.TLabel").pack(
            side=tk.LEFT, padx=(0, 8))
        for title in ("Material", "Scattering", "Grids", "Phonon", "Run"):
            ttk.Button(bar, text=title,
                       command=lambda t=title: self.scroll_to_part(t)).pack(
                side=tk.LEFT, padx=(0, 6))

        # The form is far taller than any screen, so it lives in a canvas
        # with a scrollbar and mouse-wheel support. The content width is
        # capped and centered: a wide window widens the margins instead of
        # stretching the fields.
        max_width = 1000
        background = ttk.Style().lookup("TFrame", "background")
        canvas = tk.Canvas(page, highlightthickness=0, borderwidth=0,
                           background=background or None)
        vsb = ttk.Scrollbar(page, orient=tk.VERTICAL, command=canvas.yview)
        canvas.configure(yscrollcommand=vsb.set)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        inner = ttk.Frame(canvas, padding=10)
        window = canvas.create_window((0, 0), window=inner, anchor="nw")
        self._endf_canvas = canvas
        self._endf_column = inner

        def _sync_inner(_event):
            """Keep the canvas scroll region matched to the inner frame.

            The x range is pinned at 0: bbox("all") starts at the centered
            item's x offset, and a scroll region starting there would clamp
            the view to it, undoing the centering."""
            bbox = canvas.bbox("all")
            if bbox:
                canvas.configure(
                    scrollregion=(0, bbox[1], bbox[2], bbox[3]),
                    width=inner.winfo_reqwidth())

        def _fit_width(e):
            """Track the canvas width, clamped and centered at max_width."""
            w = min(e.width, max_width)
            canvas.itemconfigure(window, width=w)
            canvas.coords(window, max((e.width - w) // 2, 0), 0)

        inner.bind("<Configure>", _sync_inner)
        canvas.bind("<Configure>", _fit_width)

        def _wheel(event):
            # macOS reports small per-line deltas; Windows multiples of 120
            """Scroll the canvas on mouse wheel (platform-normalized delta)."""
            delta = event.delta
            step = delta // 120 if abs(delta) >= 120 else delta
            canvas.yview_scroll(-int(step), "units")

        canvas.bind("<Enter>",
                    lambda e: canvas.bind_all("<MouseWheel>", _wheel))
        canvas.bind("<Leave>",
                    lambda e: canvas.unbind_all("<MouseWheel>"))

        self._build_material_tab()
        self._build_scattering_tab()
        self._build_grid_tab()
        self._build_phonon_tab()
        self._build_output_tab()

        # initial conditional-visibility state (traces only fire on writes)
        self._sync_modes()

    def _form_part(self, title):
        """One deck part (a former tab) inside the single-page form: a
        prominent title + separator, registered for the jump bar. Returns
        the part's body frame."""
        outer = ttk.Frame(self._endf_column)
        outer.pack(fill=tk.X, pady=(0, 20))
        head = ttk.Frame(outer)
        head.pack(fill=tk.X, pady=(6, 0))
        ttk.Label(head, text=title, style="Part.TLabel").pack(side=tk.LEFT)
        ttk.Separator(outer, orient=tk.HORIZONTAL).pack(fill=tk.X,
                                                        pady=(4, 10))
        body = ttk.Frame(outer)
        body.pack(fill=tk.X)
        self._endf_parts[title] = outer
        return body

    def scroll_to_part(self, title):
        """Scroll the ENDF form so the named part starts at the top."""
        canvas = self._endf_canvas
        canvas.update_idletasks()
        bbox = canvas.bbox("all")
        total = max(1, bbox[3] - bbox[1])
        y = self._endf_parts[title].winfo_y()
        canvas.yview_moveto(min(1.0, max(0.0, y / total)))

    # ------------------------------------------------------------------
    # Tab 1: Material Setup
    # ------------------------------------------------------------------
    def _build_material_tab(self):
        """Build the Material part widgets."""
        frame = self._form_part("Material")

        LBL = 20        # shared label-column width (chars) for this part

        # Import input file
        qs_body = form_section(frame, "Quick Start")
        qs_row = ttk.Frame(qs_body)
        qs_row.pack(fill=tk.X)
        ttk.Button(qs_row, text="Import Input File...", default="active",
                   command=self._import_leapr).pack(side=tk.LEFT, padx=(0, 12))
        ttk.Label(qs_row, style="Hint.TLabel",
                  text="Load an existing .input or .leapr file to "
                       "populate all tabs, or fill in the fields below "
                       "manually.").pack(side=tk.LEFT)

        # Inelastic mode. Parented to the tab frame, not _iel10_group: the
        # selector is a top-level choice about MF7/MT4 and stays visible for
        # every iel. Built before the elastic section so it packs above it.
        nc_body = form_section(
            frame, "Inelastic Mode",
            help_title="Inelastic Mode",
            help_text="Selects how the inelastic scattering (MF7/MT4, the "
                      "S(alpha,beta) table) is computed.\n\n"
                      "inelastic_mode values:\n"
                      "  0. Legacy cubic: isotropic Debye-Waller factor plus "
                      "the cubic/incoherent inelastic model.\n\n"
                      "  1. Directional incoherent approximation: keeps IRMA's "
                      "directional Debye-Waller / MT2 handling, but calls the "
                      "noncubic S(alpha,beta) driver for the MT4 inelastic "
                      "section, injecting incoherent-approximation one-phonon "
                      "(n=1) plus incoherent-approximation multiphonon "
                      "terms.\n\n"
                      "  2. Coherent approximation: the same noncubic path, "
                      "but injects the exact one-phonon term (coherent + "
                      "incoherent) plus incoherent-approximation multiphonon "
                      "terms.\n\n"
                      "Modes 1 and 2 are the cleaned-up noncubic hybrid paths.\n\n"
                      "Requires phonopy installed and a phonopy.yaml with "
                      "force constants (Card 6f), plus explicit noncubic "
                      "inelastic controls (Card 6g). These modes work with "
                      "either elastic_mode=1 (SEF) or elastic_mode=2 (MEF). "
                      "Card 6e partial spectra are omitted because MT4 and "
                      "the directional elastic Debye-Waller factors come from "
                      "Phonopy.\n\n"
                      "For mixed materials, keep the full crystal in Card 6d, "
                      "but still generate one principal-scatterer MT4 section "
                      "per input file. Card 5 chooses the principal scatterer; if "
                      "you need more than one principal (for example Be and O "
                      "in BeO), run separate input files.")
        self._inelastic_section = nc_body.master

        # inelastic_mode selector
        self.inelastic_mode_var = tk.IntVar(value=0)
        self.inelastic_mode_var.trace_add(
            "write", self._sync_modes)
        coh_elas_dw_row = ttk.Frame(nc_body)
        coh_elas_dw_row.pack(fill=tk.X, pady=2)
        ttk.Label(coh_elas_dw_row, text="inelastic_mode:", width=LBL,
                  anchor="e").pack(side=tk.LEFT, padx=(0, 5))
        ttk.Radiobutton(
            coh_elas_dw_row, text="0 — legacy cubic",
            variable=self.inelastic_mode_var, value=0,
            command=self._on_inelastic_mode_click).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Radiobutton(
            coh_elas_dw_row, text="1 — incoherent approx.",
            variable=self.inelastic_mode_var, value=1,
            command=self._on_inelastic_mode_click).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Radiobutton(
            coh_elas_dw_row, text="2 — coherent (exact 1-phonon)",
            variable=self.inelastic_mode_var, value=2,
            command=self._on_inelastic_mode_click).pack(side=tk.LEFT)

        # Sub-frame that is shown/hidden by either checkbox
        self._nc_subframe = ttk.Frame(nc_body)
        self._nc_subframe.pack(fill=tk.X, pady=5)

        # phonopy.yaml path
        self.nc_phonopy_yaml = FileSelector(
            self._nc_subframe, "phonopy.yaml:",
            mode="open",
            filetypes=[("YAML files", "*.yaml"), ("All files", "*.*")],
            label_width=LBL,
            help_title="phonopy.yaml",
            help_text="Path to the phonopy.yaml file describing the phonon "
                      "model.\n\n"
                      "Force constants are read from the yaml itself if "
                      "embedded there, otherwise discovered in the same "
                      "directory as one of: force_constants.hdf5, "
                      "FORCE_CONSTANTS, or FORCE_SETS.")
        self.nc_phonopy_yaml.pack(fill=tk.X, pady=2)

        # Structure prefill. Lives here, directly under the model selector,
        # so it inherits the sub-frame's progressive disclosure: it is only
        # on screen for the phonopy-backed modes (1/2), which _sync_modes
        # in turn pins to iel=10 -- exactly the decks whose Card 6c/6d it
        # writes. Click-only; see _fill_structure_from_phonopy.
        fill_row = ttk.Frame(self._nc_subframe)
        fill_row.pack(fill=tk.X, pady=2)
        ttk.Label(fill_row, text="", width=LBL).pack(side=tk.LEFT, padx=(0, 5))
        self._fill_structure_btn = ttk.Button(
            fill_row, text=STRUCTURE_FILL_TITLE,
            command=self._fill_structure_from_phonopy)
        self._fill_structure_btn.pack(side=tk.LEFT)
        InfoLabel(fill_row, STRUCTURE_FILL_TITLE, STRUCTURE_FILL_HELP).pack(
            side=tk.LEFT, padx=(4, 0))

        # Mesh dimensions row
        mesh_row = ttk.Frame(self._nc_subframe)
        mesh_row.pack(fill=tk.X, pady=2)
        ttk.Label(mesh_row, text="Mesh (nx ny nz):", width=LBL,
                  anchor="e").pack(side=tk.LEFT, padx=(0, 5))
        self.nc_mesh_nx = LabeledEntry(mesh_row, "nx:", "40", width=5,
                                       label_width=0)
        self.nc_mesh_nx.pack(side=tk.LEFT, padx=(0, 8))
        self.nc_mesh_ny = LabeledEntry(mesh_row, "ny:", "40", width=5,
                                       label_width=0)
        self.nc_mesh_ny.pack(side=tk.LEFT, padx=(0, 8))
        self.nc_mesh_nz = LabeledEntry(mesh_row, "nz:", "40", width=5,
                                       label_width=0)
        self.nc_mesh_nz.pack(side=tk.LEFT, padx=(0, 8))
        InfoLabel(mesh_row, "Mesh Dimensions",
                  "Monkhorst-Pack mesh dimensions: how finely the Brillouin "
                  "zone is sampled to compute the phonon DOS tensor.\n\n"
                  "Larger meshes give smoother DOS tensors but require more "
                  "memory and time to load. Typical values:\n"
                  "  Light test:  10×10×10\n"
                  "  Production:  40×40×40 (the default; used throughout "
                  "the validation suite)\n\n"
                  "Note: for anisotropic crystals (e.g. graphite), "
                  "an asymmetric mesh (e.g. 40×40×20) can be used to "
                  "match the crystal symmetry."
                  ).pack(side=tk.LEFT, padx=(4, 0))

        self.nc_min_phonon_energy = LabeledEntry(
            self._nc_subframe, "Minimum phonon energy [meV]:", "",
            width=10, label_width=LBL,
            help_title="Minimum Phonon Energy",
            help_text=MIN_PHONON_ENERGY_HELP)
        self.nc_min_phonon_energy.pack(fill=tk.X, pady=2)

        # ncpu row
        dir_row = ttk.Frame(self._nc_subframe)
        dir_row.pack(fill=tk.X, pady=2)
        self.nc_ncpu = LabeledEntry(
            dir_row, "ncpu:", str(os.cpu_count() or 1), width=4, label_width=LBL,
            help_title="ncpu (Parallel CPUs)",
            help_text="Number of parallel worker processes for the phonopy-backed "
                      "workflow.\n\n"
                      "This value is passed to the noncubic S(alpha,beta) "
                      "driver and related mesh processing steps.\n\n"
                      "ncpu=1: serial.\n"
                      "ncpu>1: use multiple worker processes.")
        self.nc_ncpu.pack(side=tk.LEFT, padx=(0, 10))

        # Noncubic inelastic controls (Card 6g)
        ctrl_row = ttk.Frame(self._nc_subframe)
        ctrl_row.pack(fill=tk.X, pady=2)
        self.nc_num_directions = LabeledEntry(
            ctrl_row, "ndir:", "10000", width=6, label_width=LBL,
            help_title="ndir (One-Phonon Directions)",
            help_text="Number of directions used to powder-average the "
                      "one-phonon terms: the scattering is evaluated along "
                      "this many golden-spiral directions on the sphere and "
                      "averaged over crystal orientations.\n\n"
                      "Default 10000, the validation-campaign sampling, "
                      "well converged for production evaluations; drop to "
                      "~4000 for a faster look or ~1000 for a quick one. "
                      "Cost is roughly linear in the count.")
        self.nc_num_directions.pack(side=tk.LEFT, padx=(0, 10))
        self.nc_multiphonon_num_directions = LabeledEntry(
            ctrl_row, "mpdir:", "1000", width=6, label_width=0,
            help_title="mpdir (Multiphonon Directions)",
            help_text="Number of directions used to powder-average "
                      "(orientationally average) the multiphonon background "
                      "over the unit sphere.\n\n"
                      "What it controls: with the anisotropic Debye-Waller "
                      "tensor the multiphonon S(α,β) depends on the "
                      "scattering DIRECTION: both the Debye-Waller "
                      "exponent 2W = Q²·(û·U·û) and the direction-projected "
                      "phonon DOS vary with the unit vector û. IRMA samples "
                      "this many golden-spiral (Fibonacci-sphere) directions, "
                      "equally weighted, and averages the multiphonon over "
                      "them. (The one-phonon terms use 'ndir'; this knob only "
                      "sizes the multiphonon angular average.)\n\n"
                      "Convergence: the average converges quickly; for "
                      "graphite-like crystals it is already converged by "
                      "~50–100 directions. The default of 1000 (the "
                      "validation-campaign sampling) carries ample margin; "
                      "raise it only to verify convergence on a new "
                      "material. Cost is linear in the count (the "
                      "multiphonon phase scales ∝ mpdir), so larger values "
                      "cost proportionally more for no accuracy gain once "
                      "converged.")
        self.nc_multiphonon_num_directions.pack(side=tk.LEFT, padx=(0, 10))
        # ON by default: the GUI's fixed nphon=100 with the default grids
        # (Beta max = 5 eV reaches Q ~ 98 1/A) is a documented truncation
        # case — graphite needs an order near 223 there. The deck-side
        # default stays 0 (reproducibility for hand-written decks); the GUI
        # writes the deck itself, so it defaults to the safe setting.
        self.nc_auto_order_var = tk.IntVar(value=1)
        # auto-size ON makes Card 3's nphon a dead field for modes 1/2;
        # the trace keeps the nphon entry's grey-out in step (it also
        # fires on deck import and reset).
        self.nc_auto_order_var.trace_add(
            "write", self._sync_modes)
        ttk.Checkbutton(
            ctrl_row, text="Auto-size multiphonon order",
            variable=self.nc_auto_order_var,
        ).pack(side=tk.LEFT, padx=(0, 4))
        InfoLabel(ctrl_row, "Card 6g Controls",
                  "Card 6g controls the noncubic MT4 inelastic model:\n\n"
                  "  ndir: one-phonon direction count; default 10000 (the\n"
                  "    validation-campaign sampling)\n"
                  "  mpdir: multiphonon direction count; powder-averages the\n"
                  "    anisotropic multiphonon over the sphere; default 1000,\n"
                  "    converged by ~50–100, raise only to check convergence\n"
                  "  nphon (Card 3): maximum multiphonon order\n"
                  "  Auto-size multiphonon order (Card 6g 3rd field): IRMA\n"
                  "    sizes nphon from the anisotropic Debye-Waller physics.\n"
                  "    ON by default; the GUI defaults produce Card 6g\n"
                  "    '10000 1000 1'. Turn it OFF only for\n"
                  "    deliberate low-order studies: your Card 3 nphon is then\n"
                  "    honored exactly, and a too-low value truncates the\n"
                  "    high-Q (high momentum transfer) rows of the cross\n"
                  "    section.\n\n"
                  "These controls are required for inelastic_mode=1/2."
                  ).pack(side=tk.LEFT)

        # Note: with the anisotropic Debye-Waller factor the multiphonon
        # order required to reach the free-gas limit grows with Q, so a hand-set
        # nphon that is fine at low Q will silently truncate the high-Q rows of
        # the cross section. The GUI therefore defaults "Auto-size" ON; this
        # warning is for users who deliberately turn it off.
        warn_row = ttk.Frame(self._nc_subframe)
        warn_row.pack(fill=tk.X, pady=(0, 2))
        ttk.Label(
            warn_row,
            text=("⚠ Anisotropic Debye-Waller: high Q needs a HIGH multiphonon order "
                  "(nphon). With 'Auto-size' OFF, your nphon is honored exactly and a "
                  "too-low value truncates the high-Q cross section (IRMA warns in the "
                  "terminal). Leave 'Auto-size' ON unless you are doing a deliberate "
                  "low-order study."),
            foreground="#e5484d",
            wraplength=640,
            justify=tk.LEFT,
        ).pack(side=tk.LEFT, anchor="w")

        # Coherent one-phonon powder averaging always uses the golden-spiral
        # direction quadrature (equivalent to Euphonic's 'golden' method); it is
        # not user-selectable.

        # BORN corrections row
        born_row = ttk.Frame(self._nc_subframe)
        born_row.pack(fill=tk.X, pady=2)
        self.nc_use_born_var = tk.IntVar(value=0)
        self.nc_use_born_var.trace_add("write", self._sync_modes)
        ttk.Checkbutton(
            born_row, text="Use BORN corrections (NAC)",
            variable=self.nc_use_born_var,
        ).pack(side=tk.LEFT)
        InfoLabel(born_row, "BORN Corrections (NAC)",
                  "Non-analytical correction (NAC) for polar materials, "
                  "where long-range electric fields shift the phonon "
                  "frequencies.\n\n"
                  "When enabled, provide a BORN file containing the Born "
                  "effective charges and dielectric tensor. This corrects "
                  "the LO-TO splitting near the zone centre Γ.\n\n"
                  "For non-polar materials (e.g. graphite), leave unchecked."
                  ).pack(side=tk.LEFT, padx=(4, 0))

        self.nc_born_path = FileSelector(
            self._nc_subframe, "BORN file:",
            mode="open",
            filetypes=[("BORN files", "BORN"), ("All files", "*.*")],
            label_width=LBL,
            help_title="BORN File",
            help_text="Path to the phonopy BORN file with Born effective charges "
                      "and the macroscopic dielectric tensor.\n\n"
                      "Generated by phonopy with: phonopy --born")
        self.nc_born_path.pack(fill=tk.X, pady=2)

        # Elastic mode
        mode_body = form_section(frame, "Elastic Scattering Mode")

        self.iel_var = tk.StringVar(value="10")
        row = ttk.Frame(mode_body)
        row.pack(fill=tk.X, pady=2)
        ttk.Label(row, text="iel:", width=LBL, anchor="e").pack(
            side=tk.LEFT, padx=(0, 5))
        self._iel_combo = ttk.Combobox(
            row, textvariable=self.iel_var, values=list(_IEL_CHOICES),
            width=35, state="readonly")
        self._iel_combo.pack(side=tk.LEFT)
        self._iel_combo.set(_IEL_GENERALIZED)
        InfoLabel(row, "iel (Elastic Scattering Type)",
                  "Controls the coherent elastic treatment: the Bragg-edge "
                  "scattering.\n\n"
                  "iel=0: No coherent elastic scattering.\n\n"
                  "iel=1-6: Legacy built-in materials (graphite, Be, BeO, "
                  "Al, Pb, Fe), using hardcoded crystal structures.\n\n"
                  "iel=10: Generalized. Uses the crystal structure you "
                  "define in this part to compute Bragg edges for ANY "
                  "material. This is the recommended option for new "
                  "evaluations."
                  ).pack(side=tk.LEFT, padx=(6, 0))

        self.elastic_mode = LabeledCombobox(
            mode_body, "Elastic format:",
            ["1 — SEF (Single-channel Elastic Format)",
             "2 — MEF (Mixed Elastic Format)"],
            default="1 — SEF (Single-channel Elastic Format)",
            label_width=LBL,
            help_title="Elastic Format",
            help_text="Determines how the coherent and incoherent elastic "
                      "scattering components are stored in the output ENDF "
                      "file.\n\n"
                      "SEF (elastic_mode=1), the single-channel elastic "
                      "format:\n"
                      "  The designated-coherent (DC) atom, the one with the "
                      "smallest incoherent contribution, gets coherent "
                      "elastic (LTHR=1). All other atoms are treated as "
                      "incoherent elastic (LTHR=2). This is the standard "
                      "ENDF format supported by all transport codes. SEF was "
                      "called the current ENDF format (CEF) when the mixed "
                      "elastic format was introduced (Ramić et al., NIM-A "
                      "1027 (2022) 166227).\n\n"
                      "MEF (elastic_mode=2), the mixed elastic format:\n"
                      "  All atoms get both coherent and incoherent elastic "
                      "(LTHR=3). More physically accurate for polyatomic "
                      "materials, but the transport code must support the "
                      "mixed elastic format.")
        self.elastic_mode.pack(fill=tk.X, pady=2)

        # Everything below only applies to the generalized treatment: the
        # crystal structure and the elastic-output options are iel=10
        # features. Hidden for iel=0-6 (built-in materials). The inelastic
        # mode is NOT in here; it is built above, outside this group.
        self._iel10_group = ttk.Frame(frame)
        self._iel10_group.pack(fill=tk.X)
        self.iel_var.trace_add("write", self._sync_modes)

        # Lattice parameters
        latt_help = ("Unit cell lattice parameters defining the crystal "
                     "structure.\n\n"
                     "a, b, c: Lattice constants in Angstrom. These are the "
                     "lengths of the three unit cell edge vectors.\n\n"
                     "alpha: Angle between b and c axes (degrees)\n"
                     "beta: Angle between a and c axes (degrees)\n"
                     "gamma: Angle between a and b axes (degrees)\n\n"
                     "Examples:\n"
                     "  Cubic: a=b=c, alpha=beta=gamma=90\n"
                     "  Hexagonal: a=b, alpha=beta=90, gamma=120\n"
                     "  FCC NaCl: a=b=c=5.69, all angles 90")

        latt_body = form_section(self._iel10_group, "Lattice Parameters",
                                 help_title="Lattice Constants",
                                 help_text=latt_help)
        ttk.Label(latt_body, style="Hint.TLabel", justify=tk.LEFT,
                  wraplength=640, text=IDENTITY_HINT_LATTICE).pack(anchor=tk.W)
        latt_grid = ttk.Frame(latt_body)
        latt_grid.pack(anchor=tk.W)
        # Blank by construction: the cell describes the USER's material, and a
        # plausible-but-wrong prefill (someone else's graphite) is worse than an
        # empty field. See IDENTITY_HINT_LATTICE for the ways to fill it.
        self.latt_a = LabeledEntry(latt_grid, "a [Ang]:", "",
                                   label_width=12)
        self.latt_b = LabeledEntry(latt_grid, "b [Ang]:", "",
                                   label_width=12)
        self.latt_c = LabeledEntry(latt_grid, "c [Ang]:", "",
                                   label_width=12)
        self.latt_alpha = LabeledEntry(latt_grid, "alpha [deg]:", "",
                                       label_width=12)
        self.latt_beta = LabeledEntry(latt_grid, "beta [deg]:", "",
                                      label_width=12)
        self.latt_gamma = LabeledEntry(latt_grid, "gamma [deg]:", "",
                                       label_width=12)
        for i, cell in enumerate((self.latt_a, self.latt_b, self.latt_c,
                                  self.latt_alpha, self.latt_beta,
                                  self.latt_gamma)):
            cell.grid(row=i // 3, column=i % 3, sticky=tk.W,
                      padx=(0, 24), pady=3)

        # Atom types
        atom_body = form_section(
            self._iel10_group, "Atom Types in Unit Cell",
            help_title="Atom Types",
            help_text="Define each distinct atom species in the unit cell, "
                      "one line per species.\n\n"
                      "Fields (space-separated):\n"
                      "  Z         Atomic number (e.g. 6 for carbon)\n"
                      "  A         Mass number (e.g. 12 for C-12), or 0 for\n"
                      "            the natural element\n"
                      "  AWR       Atomic weight ratio to neutron mass\n"
                      "  b_coh     Coherent scattering length in fm\n"
                      "  sigma_inc Incoherent cross section in barns\n"
                      "  npos      Number of positions in the unit cell\n"
                      "  x y z ... Fractional coordinates of each position\n\n"
                      "Example (graphite, 4 C atoms per cell; A = 0 because\n"
                      "the constants are natural-carbon values):\n"
                      "6  0  11.9078  6.6472  0.001  4  "
                      "0.0 0.0 0.0  0.0 0.0 0.5  "
                      "0.333 0.667 0.0  0.667 0.333 0.5\n\n"
                      "For polyatomic materials (e.g. NaCl), use one line "
                      "per species:\n"
                      "11 23 22.99 3.63 1.62 4 0 0 0 0 .5 .5 .5 0 .5 .5 .5 0\n"
                      "17 35 34.97 11.65 4.7 4 .5 .5 .5 .5 0 0 0 .5 0 0 0 .5\n\n"
                      "Scattering lengths and cross sections can be found in "
                      "the NIST neutron scattering length tables.")
        ttk.Label(atom_body, style="Hint.TLabel",
                  text="Define each distinct atom type. Format per row:  "
                       "Z  A  AWR  b_coh  sigma_inc  npos  x1 y1 z1 ...",
                  justify=tk.LEFT).pack(anchor=tk.W)
        ttk.Label(atom_body, style="Hint.TLabel", justify=tk.LEFT,
                  wraplength=640, text=IDENTITY_HINT_ATOMS).pack(anchor=tk.W)
        # Deliberately empty: the atom block is the crystal's identity (see
        # IDENTITY_HINT_ATOMS). A leftover example row reads as "IRMA already
        # knows my material" and ships a tape for someone else's crystal.
        self.atoms_text = tk.Text(atom_body, height=6, width=80,
                                  font=fixed_font(), padx=8, pady=6)
        self.atoms_text.pack(fill=tk.BOTH, expand=True, pady=(6, 0))

        # Coherent-elastic output options (iel=10) — the Bragg-edge grouping
        # (ENDF-102 7.2.2) is an MF7/MT2 ELASTIC option that applies to every
        # iel=10 inelastic_mode (including the classic mode 0), so it lives in
        # its own section rather than under the phonopy-inelastic heading.
        grp_overview = (
            "The coherent elastic table S(E,T) is a staircase with one step "
            "per Bragg edge; the cross section is sigma = S(E,T)/E. Above "
            "~1 eV the edges crowd together and the steps are tiny, so "
            "ENDF-102 sec 7.2.2 lets you GROUP them into fewer steps while "
            "preserving the average cross section.\n\n"
            "Set 'bins/decade' > 0 to enable. The edges above 'above "
            "(eV)' are merged into log-uniform bins (this many per "
            "decade of energy); everything at or below the threshold is "
            "kept unchanged. Cumulative S and the total cross section are "
            "preserved exactly; only the placement of the merged "
            "high-energy steps is approximated (IRMA prints the resulting "
            "integral cross-section error per temperature).\n\n"
            "Applies to ANY iel=10 inelastic_mode (it is an elastic "
            "MF7/MT2 option). The GUI enables it by default, which keeps "
            "tapes compact; uncheck the box to write every edge. Written as "
            "optional fields 5 and 6 on Card 6b: 'elastic_mode nat nspec "
            "inelastic_mode [bins_per_decade] [threshold_eV]'; an input file "
            "without them (bins_per_decade 0) has grouping off.")
        grp_bpd_help = (
            "Reduces the number of coherent-elastic Bragg edges written to "
            "MF7/MT2 above the threshold energy.\n\n"
            "WHY: above ~1 eV the Bragg edges of many crystals (especially "
            "low-symmetry space groups) become extremely dense, and each "
            "'stair step' of S(E,T) is tiny. ENDF-102 sec 7.2.2 permits "
            "grouping these into fewer steps 'while still preserving the "
            "average value of the cross section'. Some materials otherwise "
            "emit thousands of edges between 1 eV and emax.\n\n"
            "WHAT 'bins/decade' MEANS: above the threshold the energy axis "
            "is split into log-uniform bins, this many per factor of 10 in "
            "energy. All edges in a bin are merged into ONE step, placed at "
            "the structure-factor-weighted log-mean energy (the placement "
            "that exactly preserves that bin's average cross section). The "
            "number of grouped steps above the threshold is about "
            "bins_per_decade × log10(emax/threshold) regardless of how many "
            "raw edges existed; for example 20/decade over 1→5 eV keeps ~14 "
            "steps. Larger values are finer (more steps, closer to "
            "ungrouped); smaller values are coarser.\n\n"
            "0 = OFF: keep every edge (the input-file default; the GUI "
            "prefills 50 with the grouping checkbox ON). Edges at or below "
            "the threshold are always kept individually. The cumulative S, "
            "the total bound cross section, and the high-energy 1/E tail "
            "stay exact in either case.")
        grp_thr_help = (
            "Energy (eV) above which Bragg edges may be grouped. ENDF-102 "
            "sec 7.2.2 recommends 1 eV (the default), where the stair steps "
            "are small enough that grouping preserves the average cross "
            "section. Edges at or below this energy are always kept "
            "individually. Only used when bins/decade > 0.")
        grp_body = form_section(
            self._iel10_group, "Coherent-Elastic Output (iel=10)",
            help_title="Bragg-Edge Grouping (ENDF-102 7.2.2)",
            help_text=(grp_overview
                       + "\n\nbins/decade\n" + grp_bpd_help
                       + "\n\nabove (eV)\n" + grp_thr_help))
        grp_row = ttk.Frame(grp_body)
        grp_row.pack(fill=tk.X, pady=2)
        # Grouping is ON by default (compact tapes; uncheck to keep every edge).
        self.coh_edge_group_enable_var = tk.BooleanVar(value=True)
        grp_chk = ttk.Checkbutton(grp_row, text="Bragg-edge grouping",
                                  variable=self.coh_edge_group_enable_var)
        grp_chk.pack(side=tk.LEFT, padx=(0, 14))
        ToolTip(grp_chk,
                "Merge the dense high-energy Bragg edges into log-uniform "
                "bins (ENDF-102 7.2.2). Uncheck to write every edge.")
        # Progressive disclosure: the bins/decade and threshold fields are
        # read only while grouping is enabled, so they live in a sub-frame
        # of the same row (the row/section identity is pinned by tests) and
        # a var trace shows/hides them -- covering the checkbox and every
        # programmatic set (deck import, reset) alike, like the extinction
        # toggle below.
        grp_fields = ttk.Frame(grp_row)
        self._grp_fields_frame = grp_fields
        grp_fields.pack(side=tk.LEFT)
        self.coh_edge_group_bpd = LabeledEntry(
            grp_fields, "bins/decade:", "50", width=6, label_width=0,
            tooltip="Log-uniform bins per decade of energy for merging "
                    "edges above the threshold. Larger = finer (closer to "
                    "ungrouped); 0 keeps every edge.")
        self.coh_edge_group_bpd.pack(side=tk.LEFT, padx=(0, 14))
        self.coh_edge_group_thr = LabeledEntry(
            grp_fields, "above (eV):", "1.0", width=6, label_width=0,
            tooltip="Edges at or below this energy are always kept "
                    "individually; ENDF-102 recommends 1 eV. Only used "
                    "when bins/decade > 0.")
        self.coh_edge_group_thr.pack(side=tk.LEFT)

        def _toggle_grp_fields(*_):
            if self.coh_edge_group_enable_var.get():
                grp_fields.pack(side=tk.LEFT)
            else:
                grp_fields.pack_forget()
        self.coh_edge_group_enable_var.trace_add("write", _toggle_grp_fields)

        # Crystalline extinction (optional iel=10 coherent-elastic correction).
        # Off by default; reduces the Bragg edges for dynamical-diffraction
        # extinction (a SAMPLE property: crystallite l, mosaic g, grain L).
        # Models ported from the NCrystal CrysXT plugin.
        ext_body = form_section(
            self._iel10_group, "Crystalline Extinction (Optional)",
            help_title="Crystalline Extinction", help_text=EXT_HELP["about"])
        self.ext_enable_var = tk.BooleanVar(value=False)
        ext_chk = ttk.Checkbutton(ext_body, text="Enable extinction correction",
                                  variable=self.ext_enable_var)
        ext_chk.pack(anchor=tk.W, pady=(0, 2))
        ToolTip(ext_chk,
                "Apply the dynamical-diffraction extinction reduction to "
                "the coherent-elastic Bragg edges. Off: the ideal-crystal "
                "Bragg edges (byte-identical tape).")
        # Progressive disclosure: the parameter fields exist only while
        # the correction is enabled; a var trace covers the checkbox and
        # every programmatic set (deck import, reset) alike.
        ext_fields = ttk.Frame(ext_body)
        self._ext_fields_frame = ext_fields

        def _toggle_ext_fields(*_):
            if self.ext_enable_var.get():
                ext_fields.pack(fill=tk.X, after=ext_chk)
            else:
                ext_fields.pack_forget()
        self.ext_enable_var.trace_add("write", _toggle_ext_fields)
        self.ext_model = LabeledCombobox(
            ext_fields, "model:", list(EXTINCTION_MODELS), default="BC_mix",
            label_width=LBL,
            help_title="Extinction model", help_text=EXT_HELP["model"])
        self.ext_model.pack(fill=tk.X, pady=2)
        self.ext_model.combo.bind("<<ComboboxSelected>>",
                                  self._on_ext_model_change)
        self.ext_l = LabeledEntry(ext_fields, "l — crystallite (Å):", "8550",
                                  width=12, label_width=LBL,
                                  help_text=EXT_HELP["l"])
        self.ext_l.pack(fill=tk.X, pady=2)
        self.ext_g = LabeledEntry(ext_fields, "g — mosaic (rad⁻¹):", "170",
                                  width=12, label_width=LBL,
                                  help_text=EXT_HELP["g"])
        self.ext_g.pack(fill=tk.X, pady=2)
        self.ext_L = LabeledEntry(ext_fields, "L — grain (Å):", "75750",
                                  width=12, label_width=LBL,
                                  help_text=EXT_HELP["L"])
        self.ext_L.pack(fill=tk.X, pady=2)
        self.ext_dist = LabeledCombobox(
            ext_fields, "distribution:", _EXT_DISTS, default="Gauss",
            label_width=LBL, help_text=EXT_HELP["dist"])
        self.ext_dist.pack(fill=tk.X, pady=2)
        self.ext_recipe = LabeledCombobox(
            ext_fields, "recipe:", ["std", "cls"], default="std",
            label_width=LBL, help_text=EXT_HELP["recipe"])
        self.ext_recipe.pack(fill=tk.X, pady=2)
        self.ext_rmse_tol = LabeledEntry(
            ext_fields, "rmse_tol:", "1e-3", width=12, label_width=LBL,
            help_text=EXT_HELP["rmse_tol"])
        self.ext_rmse_tol.pack(fill=tk.X, pady=2)
        ext_attr = ttk.Frame(ext_body)
        ext_attr.pack(fill=tk.X, pady=(4, 0))
        ttk.Label(ext_attr, style="Hint.TLabel",
                  text="Models ported from the NCrystal CrysXT plugin "
                       "(Kittelmann 2026, Xu 2025).").pack(side=tk.LEFT)
        InfoLabel(ext_attr, "Extinction (attribution & references)",
                  EXT_HELP["attribution"]).pack(side=tk.LEFT, padx=(6, 0))

    def _sync_modes(self, *_):
        """Show the sections that the selected iel and inelastic_mode read.

        Modes 1/2 exist only for iel=10, so they narrow the iel menu to the
        generalized entry. Deck generation forces inelastic_mode=0 when
        iel != 10, so the phonopy layout (phonon cards and special modes
        hidden behind notes, nphon greyed out under auto-size) follows the
        effective mode. Hiding the special modes also clears them: a value
        left behind a hidden section would fail the run with the control
        off screen.
        """
        mode = self.inelastic_mode_var.get()
        if mode in (1, 2):
            self._iel_combo.configure(values=[_IEL_GENERALIZED])
            if self._code(self.iel_var) != 10:
                self.iel_var.set(_IEL_GENERALIZED)  # the trace re-enters here
                return
        else:
            self._iel_combo.configure(values=list(_IEL_CHOICES))
        gen = self._code(self.iel_var) == 10
        phonopy = gen and mode > 0
        _show(self.elastic_mode, gen, fill=tk.X, pady=2)
        _show(self._iel10_group, gen, fill=tk.X)
        _show(self._nc_subframe, mode > 0, fill=tk.X, pady=5)
        _show(self.nc_born_path, mode > 0 and self.nc_use_born_var.get(),
              fill=tk.X, pady=2)
        if phonopy:
            self._clear_special_scatter_fields()
            self._sec2_frame.pack_forget()
        for w, info in self._phonon_card_sections + self._special_scatter_sections:
            _show(w, not phonopy, **info)
        for note in (self._phonon_ignored_note, self._special_modes_note):
            _show(note, phonopy, fill=tk.X, pady=(0, 8))
        if not phonopy:
            self._update_scattering_groups()
        self.nphon.entry.config(
            state=tk.DISABLED if phonopy and self.nc_auto_order_var.get()
            else tk.NORMAL)

    IINT_LOGLIN = "0 — log-lin (INT=4)"
    IINT_LINLIN = "1 — lin-lin (INT=2)"

    def _on_inelastic_mode_click(self):
        """A user's click on an inelastic_mode radio button.

        Besides the layout update the variable trace already does, the
        click sets the Card 4 interpolation default that suits the mode:
        the coherent one-phonon law (mode 2) has structural near-zeros
        that log interpolation floors, so it gets lin-lin; modes 0 and 1
        get the classic log-lin. Only a click does this, so a deck import
        (which sets the mode variable and then the deck's own iint) and the
        user's later choice of iint are left alone.
        """
        self.iint.set(self.IINT_LINLIN if self.inelastic_mode_var.get() == 2
                      else self.IINT_LOGLIN)

    # ------------------------------------------------------------------
    # Structure prefill from the Card 6f phonopy model
    # ------------------------------------------------------------------
    _STRUCTURE_FILL_ASSUMPTIONS = (
        "Assumptions -- read before applying:\n"
        "  - Each species is filled as the NATURAL ELEMENT. ENDF codes that\n"
        "    as A = 0, and AWR, b_coh and sigma_inc are that element's\n"
        "    natural-abundance values, so the identity and the constants\n"
        "    always come from one table entry.\n"
        "  - A phonopy model names ELEMENTS, not isotopes. It typically\n"
        "    labels deuterium as H, and it says nothing about enrichment.\n"
        "  - For an isotope, set ZA on the Scattering tab and press 'Apply\n"
        "    ZA': it relabels that element's row with the isotope's identity\n"
        "    and constants (positions kept). Enrichment or a nuclide without\n"
        "    tabulated constants is edited on the row by hand.\n"
        "  - Positions are the phonopy PRIMITIVE cell's fractional\n"
        "    coordinates, which is the cell inelastic_mode 1/2 requires.\n\n"
        "Apply replaces the lattice fields AND the whole atom block; Cancel\n"
        "changes nothing.")

    def _fill_structure_from_phonopy(self):
        """Prefill Card 6c/6d from the phonopy model named on Card 6f.

        Explicit click only. It never fires on its own -- not when a
        phonopy.yaml is selected, not on deck import, not on reset -- and
        nothing is written until the user confirms the preview. The
        replacement is ATOMIC: Apply writes the lattice and the whole atom
        block together, Cancel writes neither, so phonopy positions can
        never end up beside a hand-typed lattice.
        """
        from irma.core.crystal_input import (
            format_card6d_row, format_lattice_fields, species_from_sites)
        # Lazy, inside the handler: irma.core.phonopy_io imports phonopy only
        # when a loader is actually called, so the GUI still runs (and this
        # module still imports) with no phonopy installed.
        from irma.core.phonopy_io import load_phonopy_primitive_structure

        path = self.nc_phonopy_yaml.get().strip()
        if not path:
            messagebox.showerror(
                STRUCTURE_FILL_TITLE,
                "No phonopy.yaml is selected.\n\nChoose the phonopy.yaml "
                "above (Card 6f) first; the structure is read from it.")
            return
        if not os.path.isfile(path):
            messagebox.showerror(
                STRUCTURE_FILL_TITLE,
                f"The selected phonopy.yaml does not exist:\n{path}")
            return
        if self.runner.is_running:
            # The loader pins the process cwd (isolated_phonopy_cwd), which
            # would race a calculation doing the same.
            messagebox.showwarning(
                "Busy", "Finish or cancel the running calculation before "
                "filling the structure (both use the process working "
                "directory).")
            return

        try:
            structure = load_phonopy_primitive_structure(path)
        except ImportError:
            messagebox.showerror(
                "phonopy not available",
                "phonopy is required to read the crystal structure from a "
                "phonopy.yaml.\nInstall with: pip install phonopy\n\n"
                "Until then, enter the lattice and atom types by hand.")
            return
        except Exception as exc:
            messagebox.showerror(
                STRUCTURE_FILL_TITLE,
                f"Could not read the crystal structure from\n{path}\n\n{exc}")
            return

        try:
            species, warnings = species_from_sites(
                structure.symbols, structure.scaled_positions)
        except (KeyError, ValueError) as exc:
            messagebox.showerror(
                STRUCTURE_FILL_TITLE,
                f"Could not resolve the scattering constants for this "
                f"structure:\n\n{exc}")
            return

        lattice_fields = format_lattice_fields(structure.cellpar)
        atom_rows = [format_card6d_row(sp) for sp in species]
        if not self._confirm_structure_fill(path, lattice_fields, atom_rows,
                                            warnings):
            return
        self._apply_structure_fill(lattice_fields, atom_rows)

    def _structure_fill_preview_text(self, path, lattice_fields, atom_rows,
                                     warnings):
        """The exact text the confirmation dialog shows."""
        a, b, c, alpha, beta, gamma = lattice_fields
        lines = [
            f"Source: {path}",
            "",
            "Lattice Parameters (Angstrom / degrees)",
            f"  a = {a}    b = {b}    c = {c}",
            f"  alpha = {alpha}    beta = {beta}    gamma = {gamma}",
            "",
            "Atom Types in Unit Cell",
            "  (Z  A  AWR  b_coh  sigma_inc  npos  x1 y1 z1 ...)",
        ]
        lines += [f"  {row}" for row in atom_rows]
        lines += ["", self._STRUCTURE_FILL_ASSUMPTIONS]
        if warnings:
            lines += ["", "Notes:"] + [f"  - {w}" for w in warnings]
        return "\n".join(lines)

    def _confirm_structure_fill(self, path, lattice_fields, atom_rows,
                                warnings):
        """Modal preview of the exact fill; True only when Apply is pressed.

        A separate method so the tests can drive the decision without a
        real dialog: the click handler mutates nothing before this returns
        True.
        """
        text = self._structure_fill_preview_text(path, lattice_fields,
                                                 atom_rows, warnings)
        top = tk.Toplevel(self.root)
        top.title(STRUCTURE_FILL_TITLE)
        top.transient(self.root)
        frame = ttk.Frame(top, padding=15)
        frame.pack(fill=tk.BOTH, expand=True)
        ttk.Label(frame, text="This will replace the lattice fields and the "
                              "whole atom block:",
                  justify=tk.LEFT).pack(anchor=tk.W, pady=(0, 8))
        body = ttk.Frame(frame)
        body.pack(fill=tk.BOTH, expand=True)
        view = tk.Text(body, wrap=tk.NONE, width=88, height=24,
                       font=fixed_font(), padx=8, pady=6)
        vsb = ttk.Scrollbar(body, orient=tk.VERTICAL, command=view.yview)
        hsb = ttk.Scrollbar(frame, orient=tk.HORIZONTAL, command=view.xview)
        view.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        view.insert("1.0", text)
        view.config(state=tk.DISABLED)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        view.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        hsb.pack(fill=tk.X)

        decision = {"apply": False}

        def _apply():
            decision["apply"] = True
            top.destroy()

        btn_row = ttk.Frame(frame)
        btn_row.pack(fill=tk.X, pady=(10, 0))
        ttk.Button(btn_row, text="Cancel", command=top.destroy).pack(
            side=tk.RIGHT)
        ttk.Button(btn_row, text="Apply", default="active",
                   command=_apply).pack(side=tk.RIGHT, padx=(0, 8))
        top.bind("<Escape>", lambda _e: top.destroy())

        top.update_idletasks()
        try:
            top.grab_set()
        except tk.TclError:
            pass            # not viewable yet (headless): modality only
        top.focus_set()
        top.wait_window()
        return decision["apply"]

    def _apply_structure_fill(self, lattice_fields, atom_rows):
        """Write the confirmed structure: lattice + atom block, together."""
        for entry, value in zip(
                (self.latt_a, self.latt_b, self.latt_c,
                 self.latt_alpha, self.latt_beta, self.latt_gamma),
                lattice_fields):
            entry.set(value)
        self._write_atom_rows(atom_rows)

    def _write_atom_rows(self, rows):
        """Replace the atom block with the given Card 6d row texts."""
        self.atoms_text.delete("1.0", tk.END)
        self.atoms_text.insert("1.0", "\n".join(rows) + "\n")

    # ------------------------------------------------------------------
    # Tab 2: Scattering Parameters
    # ------------------------------------------------------------------
    APPLY_ZA_TITLE = "Apply ZA"

    def _ask_yes_no(self, title, text):
        """One place for the yes/no dialogs, so tests can answer them."""
        return bool(messagebox.askyesno(title, text))

    def _set_za_status(self, text):
        if hasattr(self, "za_status_var"):
            self.za_status_var.set(text)

    def _fill_from_za(self):
        """Apply the principal ZA: fill AWR + sigma_free (spr) from the
        built-in nuclear table, and relabel the atom row of that element
        to the same nuclide.

        Explicit click only, never automatic: the fields carry values the
        user may have edited, and an overwrite must be a deliberate click.
        The whole action is one transaction: a refusal or a cancelled
        dialog changes nothing, on Card 5 or in the atom block.

        The atom row rule: with exactly one row of the ZA's element, the
        row is relabelled to the nuclide (A, AWR, b_coh, sigma_inc from
        the table entry; positions kept) when its constants are the
        table's own numbers for its current nuclide, and only after a
        confirmation showing old and new values when they are not (they
        are then someone's custom constants). Several rows of one element
        are never chosen between. Nuclides with no tabulated constants,
        or with energy-dependent ones, are refused outright.
        """
        from irma.core.crystal_input import (
            format_atom_row, nuclide_label, principal_row_match, relabel_row,
            row_constants_match_table, split_za)
        from irma.core.nuclear_data import lookup
        from irma.gui.deck_text import parse_atoms_text
        za_text = str(self.za.get()).strip()
        if not za_text:
            # The fresh form ships ZA blank (it names the user's material), so
            # an empty field is the ordinary first click here, not a typo.
            messagebox.showerror(
                self.APPLY_ZA_TITLE,
                "Enter a ZA first — this button reads the scatterer identity "
                "and fills AWR and sigma_free from the built-in nuclear "
                "table.\n\nZA is Z*1000+A (e.g. 6000 for natural carbon, "
                "6012 for C-12).")
            return
        try:
            za = int(za_text)
            z, a = split_za(za)
            nuc = lookup((z, a))
        except (KeyError, ValueError) as exc:
            messagebox.showerror(
                self.APPLY_ZA_TITLE,
                f"No tabulated scattering constants for ZA="
                f"{self.za.get()!r} ({exc}).\n\nZA must be Z*1000+A "
                f"(e.g. 6012 for C-12). The natural element (A=000) "
                f"usually has constants; isotopes without measured "
                f"values must be entered manually, on Card 5 and in the "
                f"atom row alike.")
            return
        if nuc.energy_dependent:
            messagebox.showerror(
                self.APPLY_ZA_TITLE,
                f"{nuc.symbol} has energy-dependent scattering lengths "
                f"(resonance region); the tabulated value is not a static "
                f"constant. Enter constants appropriate for your energy "
                f"range manually.")
            return
        label = nuclide_label(z, a)
        spr = nuc.sigma_bound_b * (nuc.awr / (1.0 + nuc.awr)) ** 2

        # The atom rows: decide the row change before touching anything.
        atoms_text = self.atoms_text.get("1.0", tk.END) if hasattr(self, "atoms_text") else ""
        row_note, new_rows = "", None
        if atoms_text.strip():
            try:
                rows = parse_atoms_text(atoms_text)
            except ValueError as exc:
                messagebox.showerror(
                    self.APPLY_ZA_TITLE,
                    f"The atom types block does not parse, so the row for "
                    f"{label} cannot be checked:\n\n{exc}\n\nNothing was "
                    f"changed.")
                return
            match = principal_row_match(za, rows)
            if match["exact"]:
                i = match["exact"][0]
                if row_constants_match_table(rows[i]):
                    row_note = f"atom row {i + 1} is already {label}"
                else:
                    row_note = (f"atom row {i + 1} is already {label} and keeps "
                                f"its own constants (they differ from the table)")
            elif len(match["same_z"]) == 1:
                i = match["same_z"][0]
                old_label = nuclide_label(rows[i]["Z"], rows[i]["A"])
                try:
                    new_row, changes = relabel_row(rows[i], za)
                except (KeyError, ValueError) as exc:
                    messagebox.showerror(
                        self.APPLY_ZA_TITLE,
                        f"Atom row {i + 1} ({old_label}) cannot be relabelled "
                        f"to {label}: {exc}\n\nNothing was changed.")
                    return
                change_text = ", ".join(
                    f"{name} {old:g} -> {new:g}" if name != "A" else f"A {old} -> {new}"
                    for name, old, new in changes)
                if not row_constants_match_table(rows[i]):
                    if not self._ask_yes_no(
                            self.APPLY_ZA_TITLE,
                            f"Atom row {i + 1} ({old_label}) carries constants "
                            f"that differ from the table entry for {old_label}, "
                            f"so they look like your own values.\n\n"
                            f"Replace the row with the table's {label} values?\n"
                            f"  {change_text}\n\n"
                            f"Positions are kept. 'No' changes nothing, on Card 5 "
                            f"either."):
                        self._set_za_status("nothing changed")
                        return
                new_rows = list(rows)
                new_rows[i] = new_row
                relabelled = (rows[i]["Z"], rows[i]["A"]), new_row["A"]
                row_note = (f"atom row {i + 1}: {old_label} -> {label} "
                            f"({change_text}); positions kept")
            elif match["same_z"]:
                listing = "; ".join(
                    f"row {i + 1}: {nuclide_label(rows[i]['Z'], rows[i]['A'])}"
                    for i in match["same_z"])
                messagebox.showerror(
                    self.APPLY_ZA_TITLE,
                    f"Several atom rows are Z={z} ({listing}) and none is "
                    f"{label}. Set the intended row's A and constants by "
                    f"hand, or set ZA to one of those rows.\n\nNothing was "
                    f"changed.")
                return
            else:
                row_note = (f"no atom row is Z={z}; the structure's rows must "
                            f"include {label} before the deck can run")
        else:
            row_note = (f"no atom rows yet; fill the structure, then press this "
                        f"button again to relabel the {nuc.symbol} row")

        # Commit: Card 5, then the atom block, together.
        self.awr.set(f"{nuc.awr:.6g}")
        self.spr.set(f"{spr:.6g}")
        if new_rows is not None:
            self._write_atom_rows([format_atom_row(r) for r in new_rows])
            # imported Card 6e spectra follow their row to the new nuclide
            (old_z, old_a), new_a = relabelled
            for sp in self._imported_partial_spectra:
                if (sp["Z"], sp["A"]) == (old_z, old_a):
                    sp["A"] = new_a
        model_note = ""
        if new_rows is not None and self.inelastic_mode_var.get() in (1, 2):
            model_note = ("; the phonopy model's masses and phonons are not "
                          "changed by this")
            if (z, a) in ((1, 2), (1, 3)):
                model_note += (" (a model that labels these sites H still "
                               "carries H masses)")
        self._set_za_status(
            f"ZA {za} ({label}): AWR {nuc.awr:.6g}, sigma_free {spr:.6g} b; "
            f"{row_note}{model_note}")

    def _build_scattering_tab(self):
        """Build the Scattering part widgets."""
        frame = self._form_part("Scattering")

        # Principal scatterer
        pf = form_section(frame, "Principal Scatterer")
        ttk.Label(pf, style="Hint.TLabel", justify=tk.LEFT, wraplength=640,
                  text=IDENTITY_HINT_SCATTERER).pack(anchor=tk.W)

        # ZA/AWR/sigma_free start blank: they name the user's scatterer, and
        # the Fill-from-ZA button below turns a ZA into the other two.
        self.za = LabeledEntry(
            pf, "ZA:", "",
            help_title="ZA (Scatterer Identity)",
            help_text="Identifies the principal scatterer atom (the atom this "
                      "evaluation is written for) as Z*1000 + A, where Z is "
                      "the atomic number and A is the mass number.\n\n"
                      "This must match one of the atom types defined in the "
                      "Material part.\n\n"
                      "Examples:\n"
                      "  1001 = H-1 (hydrogen)\n"
                      "  4009 = Be-9 (beryllium)\n"
                      "  6012 = C-12 (carbon)\n"
                      "  8016 = O-16 (oxygen)\n"
                      "  11023 = Na-23 (sodium)\n"
                      "  6000 = natural carbon (A = 0 names the natural\n"
                      "         element; use it when the constants beside\n"
                      "         it are natural-abundance values)")
        self.za.pack(fill=tk.X, pady=2)
        fill_row = ttk.Frame(pf)
        fill_row.pack(fill=tk.X, pady=(0, 2))
        ttk.Button(fill_row, text="Apply ZA: fill AWR + sigma_free, "
                   "relabel the atom row",
                   command=self._fill_from_za).pack(side=tk.LEFT)
        ttk.Label(fill_row,
                  text="(built-in Rauch-Waschkowski/Sears table)",
                  foreground="gray").pack(side=tk.LEFT, padx=(6, 0))
        # What the last click did, in one line: the row it relabelled and
        # the constants it wrote, so a change in the Material part is never
        # silent.
        self.za_status_var = tk.StringVar(value="")
        ttk.Label(pf, textvariable=self.za_status_var, style="Hint.TLabel",
                  justify=tk.LEFT, wraplength=640).pack(anchor=tk.W)
        self.awr = LabeledEntry(
            pf, "AWR:", "",
            help_title="AWR (Atomic Weight Ratio)",
            help_text="Ratio of the atomic mass of the principal scatterer "
                      "to the neutron mass.\n\n"
                      "Examples:\n"
                      "  H-1:   0.99917\n"
                      "  Be-9:  8.93478\n"
                      "  carbon (natural): 11.898\n"
                      "  O-16:  15.858\n"
                      "  Na-23: 22.990\n\n"
                      "This value affects the recoil kinematics and the "
                      "scaling of the alpha (momentum transfer) grid.")
        self.awr.pack(fill=tk.X, pady=2)
        self.spr = LabeledEntry(
            pf, "sigma_free [barn]:", "",
            help_title="sigma_free (spr): Free-Atom Scattering Cross Section",
            help_text="The FREE-atom scattering cross section of the "
                      "principal scatterer in barns, exactly as on LEAPR "
                      "Card 5 (spr).\n\n"
                      "IRMA derives the bound cross section that normalizes "
                      "the S(alpha, beta) table internally as\n"
                      "  sigma_b = sigma_free * ((1 + AWR) / AWR)^2\n"
                      "and writes B(1) = npr * sigma_free to the tape.\n\n"
                      "Examples (all FREE values):\n"
                      "  C-12:  4.739 barn\n"
                      "  Be-9:  6.154 barn\n"
                      "  H-1:   20.45 barn  (bound ~82 barn; do NOT enter "
                      "the bound value: every MT4 cross section would come "
                      "out ~4x too large for hydrogen)\n\n"
                      "Values can be found in nuclear data tables; if a "
                      "table lists the bound value, divide by "
                      "((1 + AWR) / AWR)^2 first.")
        self.spr.pack(fill=tk.X, pady=2)
        self.npr = LabeledEntry(
            pf, "npr:", "1",
            help_title="npr (Principal Atom Count)",
            help_text="Number of principal scattering atoms in the "
                      "compound, not in the unit cell (must be >= 1).\n\n"
                      "Examples:\n"
                      "  Graphite: 1 (one C per unit)\n"
                      "  H in H2O: 2 (two H per molecule)\n"
                      "  Be in BeO: 1\n\n"
                      "For the legacy built-in coherent-elastic materials "
                      "(iel=1-6) it also scales the Bragg-edge cross "
                      "sections.\n\n"
                      "For inelastic_mode=1/2, IRMA still writes one "
                      "principal-scatterer MT4 section per input file, even when "
                      "the crystal contains multiple atom types.")
        self.npr.pack(fill=tk.X, pady=2)

        # ENDF output
        ef = form_section(frame, "ENDF Output Control")
        ttk.Label(ef, style="Hint.TLabel", justify=tk.LEFT, wraplength=640,
                  text=IDENTITY_HINT_MAT).pack(anchor=tk.W)

        # Blank by construction: MAT identifies the user's evaluation in their
        # library, not a graphite tape IRMA guessed for them.
        self.mat = LabeledEntry(
            ef, "MAT number:", "",
            help_title="MAT (ENDF Material Number)",
            help_text="Material number used in the ENDF output file.\n\n"
                      "This is an identifier for the output library. "
                      "Standard thermal-scattering (TSL) MAT numbers are "
                      "assigned by ENDF conventions, but any value works for "
                      "testing.\n\n"
                      "Common assignments:\n"
                      "  26 = H in H2O\n"
                      "  27 = H in ZrH\n"
                      "  28 = graphite\n"
                      "  29 = Be\n"
                      "  30 = BeO\n"
                      "  37 = H in polyethylene")
        self.mat.pack(fill=tk.X, pady=2)
        self.nphon = LabeledEntry(
            ef, "nphon:", "100",
            help_title="nphon (Maximum Phonon Order)",
            help_text="Maximum number of phonon expansion terms.\n\n"
                      "The S(alpha,beta) table is computed as a sum over "
                      "phonon orders n=1, 2, ..., nphon (one-phonon, "
                      "two-phonon, and so on). Higher orders contribute "
                      "at larger momentum transfers (high alpha).\n\n"
                      "For inelastic_mode=1/2, this is also the multiphonon "
                      "maximum order used by the noncubic MT4 model.\n\n"
                      "Default: 100. This is sufficient for most materials.\n\n"
                      "Light scatterers (H, D) reach much larger alpha for "
                      "the same beta grid (alpha = E_r/(A*kT) grows as the "
                      "mass A shrinks) and generally need MORE terms before "
                      "the short-collision-time (SCT) substitution takes "
                      "over; heavy scatterers converge with fewer. The classic "
                      "path follows LEAPR and substitutes the SCT form "
                      "where the expansion underflows; for "
                      "inelastic_mode=1/2 make sure nphon (or Auto-size, "
                      "on by default) covers the requested beta grid.")
        self.nphon.pack(fill=tk.X, pady=2)
        self.isabt = LabeledCombobox(
            ef, "isabt:", ["0 — S(α,β) (standard)", "1 — S(α,β) (asymmetric)"],
            default="0 — S(α,β) (standard)",
            help_title="isabt (Output Law Form, Card 4)",
            help_text="Selects the stored form of the S(alpha,beta) "
                      "table.\n\n"
                      "0: Standard symmetric S(alpha, beta), required for "
                      "ENDF TSL libraries processed by THERMR (default).\n\n"
                      "1: The asymmetric form (writes the full "
                      "beta-asymmetric table; isym is raised by 2 in "
                      "MF7/MT4). Mainly for diagnostics, NOT for standard "
                      "library production.")
        self.isabt.pack(fill=tk.X, pady=2)
        self.ilog = LabeledCombobox(
            ef, "ilog:", ["0 — S values", "1 — ln(S) values"],
            default="0 — S values",
            help_title="ilog (Output Storage Form, Card 4)",
            help_text="0: Store the S(alpha, beta) values directly, linear "
                      "(default, NJOY-faithful).\n\n"
                      "1: Store ln(S) instead (ENDF LLN=1 log storage).\n\n"
                      "IMPORTANT for LOW TEMPERATURE: with ilog=0 the symmetric "
                      "table is stored as S*exp(-beta/2). Because beta scales "
                      "as 1/T, at cryogenic temperatures the high-energy (e.g. "
                      "optic) phonon values become ~1e-100 and are written as "
                      "ZERO, silently losing that structure on read-back. "
                      "For T below ~50-100 K, set ilog=1 to preserve it (the "
                      "run log will WARN if this happens). The reader must "
                      "support LLN=1; THERMR and IRMA's own readers do. "
                      "Harmless at room temperature.")
        self.ilog.pack(fill=tk.X, pady=2)
        self.iint = LabeledCombobox(
            ef, "iint:", ["0 — log-lin (INT=4)", "1 — lin-lin (INT=2)"],
            default="0 — log-lin (INT=4)",
            help_title="iint (S(α,β) Interpolation Form, Card 4)",
            help_text="Sibling of ilog: ilog sets how S is STORED, iint sets "
                      "how S is INTERPOLATED between stored points (the ENDF "
                      "INT flag on the MF7/MT4 section, for both the alpha "
                      "and beta tables).\n\n"
                      "0: log-lin (ENDF INT=4), classic and NJOY-faithful "
                      "(the default for inelastic_mode 0 and 1).\n\n"
                      "1: lin-lin (ENDF INT=2), selected when inelastic_mode 2 "
                      "is chosen. Coherent one-phonon "
                      "S(alpha,beta) tables have structural near-zeros (e.g. "
                      "the graphite (002) dip) that logarithmic interpolation "
                      "floors, biasing the cross section LOW in the thermal "
                      "minimum. lin-lin preserves them. Honored by INT-aware "
                      "THERMR.\n\n"
                      "IMPORTANT: use iint=1 only with a Q/alpha grid converged "
                      "near sharp coherent peaks; lin-lin still chords straight "
                      "across a peak that the grid does not resolve.")
        self.iint.pack(fill=tk.X, pady=2)
        self.smin = LabeledEntry(
            ef, "smin:", "1e-75",
            help_title="smin (Minimum S Threshold, Card 4)",
            help_text="S(alpha,beta) values below this threshold are "
                      "treated as zero in the output.\n\n"
                      "Default: 1e-75 (LEAPR convention).")
        self.smin.pack(fill=tk.X, pady=2)

        # Cold/Skold
        cf = form_section(frame, "Special Modes")

        self.ncold = LabeledCombobox(
            cf, "ncold:", ["0 — None", "1 — Ortho-H", "2 — Para-H",
                           "3 — Ortho-D", "4 — Para-D"],
            default="0 — None",
            help_title="ncold (Cold Hydrogen/Deuterium)",
            help_text="Special treatment for molecular hydrogen or deuterium "
                      "at low temperatures, where the quantum rotational "
                      "states of the molecule (the ortho and para spin forms) "
                      "matter.\n\n"
                      "0: None (default for most materials)\n"
                      "1: Ortho-hydrogen\n"
                      "2: Para-hydrogen\n"
                      "3: Ortho-deuterium\n"
                      "4: Para-deuterium\n\n"
                      "Only relevant for liquid/solid hydrogen or deuterium "
                      "moderators. For all other materials, leave as 0.")
        self.ncold.pack(fill=tk.X, pady=2)
        self.nsk = LabeledCombobox(
            cf, "nsk:", ["0 — None", "1 — Vineyard", "2 — Skold"],
            default="0 — None",
            help_title="nsk (Pair-Correlation Treatment)",
            help_text="Treatment of intermolecular coherent scattering using "
                      "a static structure factor S(kappa), which describes "
                      "how atom positions are correlated (LEAPR Card 5 "
                      "convention).\n\n"
                      "0: None (default)\n"
                      "1: Vineyard: the S(kappa) data is read and ignored "
                      "(not implemented in NJOY2016 or IRMA); use nsk=2 "
                      "(Skold) for an applied pair-correlation correction.\n"
                      "2: Skold: applies the Skold approximation to the "
                      "coherent part of S(alpha, beta)\n\n"
                      "When nsk > 0 (or ncold > 0), provide the S(kappa) "
                      "table below; for nsk > 0 also provide the coherent "
                      "fraction cfrac. Not available with "
                      "inelastic_mode=1/2.")
        self.nsk.pack(fill=tk.X, pady=2)
        # S(kappa) inputs only apply when nsk or ncold is active
        self._ska_group = ttk.Frame(cf)
        self._ska_group.pack(fill=tk.X)
        self.ska_dka = LabeledEntry(
            self._ska_group, "dka [1/Å]:", "0",
            help_title="dka (S(kappa) Grid Spacing, Card 17)",
            help_text="Spacing of the uniform kappa grid for the S(kappa) "
                      "table, in inverse Angstroms.\n\n"
                      "Required when nsk > 0 or ncold > 0; ignored "
                      "otherwise.")
        self.ska_dka.pack(fill=tk.X, pady=2)
        ska_label_row = ttk.Frame(self._ska_group)
        ska_label_row.pack(fill=tk.X)
        ttk.Label(ska_label_row,
                  text="S(kappa) values (space-separated):").pack(
            side=tk.LEFT, anchor=tk.W)
        InfoLabel(ska_label_row, "S(kappa) Values (Card 18)",
                   "Static structure factor values on the uniform kappa "
                   "grid (spacing dka), in increasing kappa order.\n\n"
                   "Required when nsk > 0 (Vineyard/Skold) or ncold > 0 "
                   "(cold hydrogen/deuterium); leave empty otherwise."
                   ).pack(side=tk.LEFT, padx=(5, 0))
        self.ska_text = tk.Text(self._ska_group, height=2, width=60,
                                font=fixed_font())
        self.ska_text.pack(fill=tk.X, pady=2)
        # cfrac (Card 19) is read only for nsk > 0 -- an ncold-only deck
        # carries the S(kappa) table but no Card 19 -- so it gets its own
        # show/hide group instead of riding along with _ska_group.
        self._cfrac_group = ttk.Frame(cf)
        self._cfrac_group.pack(fill=tk.X)
        self.cfrac = LabeledEntry(
            self._cfrac_group, "cfrac:", "0",
            help_title="cfrac (Coherent Fraction, Card 19)",
            help_text="Fraction of the scattering that is coherent, used by "
                      "the Vineyard/Skold options.\n\n"
                      "Required when nsk > 0; ignored otherwise.")
        self.cfrac.pack(fill=tk.X, pady=2)

        # Secondary scatterer
        sf = form_section(frame, "Secondary Scatterer (optional)")

        self.nss = LabeledCombobox(
            sf, "nss:", ["0 — None", "1 — One secondary scatterer"],
            default="0 — None",
            help_title="nss (Secondary Scatterer Count, Card 6)",
            help_text="Number of secondary scatterer species in a compound "
                      "moderator (0 or 1). A secondary scatterer is the "
                      "other atom species of the compound (for example O "
                      "when evaluating H in H2O).\n\n"
                      "0: No secondary scatterer (single-species, or when "
                      "generating separate tables per species)\n\n"
                      "1: One secondary scatterer, combined into the "
                      "principal S(alpha,beta) table. The MODEL used for it "
                      "is selected by b7 below.\n\n"
                      "For polyatomic materials, you typically generate "
                      "separate tables for each species (nss=0) rather than "
                      "using the mixed-moderator approach.")
        self.nss.pack(fill=tk.X, pady=2)
        # model fields only apply when a secondary scatterer is present
        self._sec_group = ttk.Frame(sf)
        self._sec_group.pack(fill=tk.X)
        self.b7 = LabeledCombobox(
            self._sec_group, "b7:", ["1 — Free gas", "2 — Diffusion",
                        "0 — Bound (two-pass)"],
            default="1 — Free gas",
            help_title="b7 (Secondary Scatterer Model, Card 6)",
            help_text="Model for the secondary scatterer (written to the "
                      "ENDF analytic-functions flag B(7)).\n\n"
                      "1: Free gas. The secondary atom is treated as "
                      "unbound (most common; e.g. O in H2O). Only the "
                      "ENDF flags are written; THERMR adds the free-gas "
                      "term downstream.\n\n"
                      "2: Diffusion. Free-gas treatment in LEAPR; the "
                      "ENDF flag tells downstream codes to apply the "
                      "diffusion model.\n\n"
                      "0: Bound (two-pass). The secondary scatterer gets "
                      "its own complete phonon-spectrum pass (e.g. O in "
                      "BeO) and the two S(alpha,beta) tables are merged "
                      "with bound-cross-section weighting. Fill in the "
                      "'Secondary phonon model' fields below. Not available "
                      "together with ncold/nsk in the GUI.\n\n"
                      "Only used when nss = 1.")
        self.b7.pack(fill=tk.X, pady=2)
        self.aws = LabeledEntry(
            self._sec_group, "AWS:", "0",
            help_title="AWS (Secondary Scatterer AWR)",
            help_text="Atomic weight ratio of the secondary scatterer "
                      "to the neutron mass.\n\n"
                      "Only used when nss = 1 (must be > 0).")
        self.aws.pack(fill=tk.X, pady=2)
        self.sps = LabeledEntry(
            self._sec_group, "sigma_s [barn]:", "0",
            help_title="sigma_s (Secondary Free Cross Section)",
            help_text="Free-atom scattering cross section of the secondary "
                      "scatterer in barns.\n\n"
                      "Only used when nss = 1 (must be > 0).")
        self.sps.pack(fill=tk.X, pady=2)
        self.mss = LabeledEntry(
            self._sec_group, "mss:", "1",
            help_title="mss (Secondary Atom Count, Card 6)",
            help_text="Number of secondary-scatterer atoms in the material "
                      "unit (must be >= 1 when nss = 1).\n\n"
                      "Examples: O in H2O -> 1; O in BeO -> 1.\n\n"
                      "Written to ENDF as B(13) = mss and "
                      "B(9) = mss * sigma_s.")
        self.mss.pack(fill=tk.X, pady=2)

        # Secondary phonon model — used only for b7 = 0 (two-pass): the
        # secondary scatterer's own detail block, emitted as a complete
        # second temperature pass (shared-spectrum convention: first
        # temperature positive, the rest negative).
        s2 = form_section(frame,
                          "Secondary phonon model (b7 = 0 two-pass only)")
        # show/hide toggles the whole section (title + separator + body)
        self._sec2_frame = s2.master

        self.sec_dos_delta = LabeledEntry(
            s2, "delta [eV]:", "0",
            help_title="Secondary Spectrum Spacing (second-pass Card 11)",
            help_text="Energy grid spacing of the secondary scatterer's "
                      "phonon spectrum, in eV.\n\n"
                      "Only used when nss = 1 and b7 = 0 (two-pass).")
        self.sec_dos_delta.pack(fill=tk.X, pady=2)
        sec_rho_row = ttk.Frame(s2)
        sec_rho_row.pack(fill=tk.X)
        ttk.Label(sec_rho_row,
                  text="Secondary rho values (space-separated):").pack(
            side=tk.LEFT, anchor=tk.W)
        InfoLabel(sec_rho_row, "Secondary Phonon Spectrum (Card 12)",
                   "The secondary scatterer's phonon density of states on "
                   "the equidistant grid (spacing delta), emitted as the "
                   "second temperature pass.\n\n"
                   "Required when b7 = 0; leave empty otherwise."
                   ).pack(side=tk.LEFT, padx=(5, 0))
        self.sec_dos_rho_text = tk.Text(s2, height=3, width=70,
                                        font=fixed_font())
        self.sec_dos_rho_text.pack(fill=tk.X, pady=2)
        self.sec_twt = LabeledEntry(
            s2, "twt:", "0.0",
            help_title="Secondary Translational Weight (Card 13)",
            help_text="Weight of the translational (diffusive or free-gas) "
                      "part of the secondary scatterer's spectrum (two-pass "
                      "only).")
        self.sec_twt.pack(fill=tk.X, pady=2)
        self.sec_c_diff = LabeledEntry(
            s2, "c (diffusion):", "0.0",
            help_title="Secondary Diffusion Constant (Card 13)",
            help_text="Diffusion constant for the secondary scatterer's "
                      "translational component (0 = free gas shape).")
        self.sec_c_diff.pack(fill=tk.X, pady=2)
        self.sec_tbeta = LabeledEntry(
            s2, "tbeta:", "1.0",
            help_title="Secondary Continuous Weight (Card 13)",
            help_text="Weight of the secondary scatterer's continuous "
                      "spectrum (must be > 0 when b7 = 0).")
        self.sec_tbeta.pack(fill=tk.X, pady=2)
        ttk.Label(s2, text="Secondary oscillator energies [eV] "
                           "(space-separated):").pack(anchor=tk.W)
        self.sec_osc_energies = tk.Text(s2, height=2, width=60,
                                        font=fixed_font())
        self.sec_osc_energies.pack(fill=tk.X, pady=2)
        ttk.Label(s2, text="Secondary oscillator weights "
                           "(space-separated):").pack(anchor=tk.W)
        self.sec_osc_weights = tk.Text(s2, height=2, width=60,
                                       font=fixed_font())
        self.sec_osc_weights.pack(fill=tk.X, pady=2)

        # Conditional visibility: variable traces fire on user changes
        # AND on deck imports (LabeledCombobox.set writes the same var).
        for widget in (self.ncold, self.nsk, self.nss, self.b7):
            widget.var.trace_add(
                "write", lambda *_: self._update_scattering_groups())
        self._update_scattering_groups()

        # sec_c_diff (second-pass Card 13) is read only when the secondary
        # twt is > 0: grey it out otherwise (a trace covers typing, deck
        # import and reset alike; an unparsable twt counts as 0).
        def _sync_sec_c_diff(*_):
            try:
                twt = float(self.sec_twt.get())
            except ValueError:
                twt = 0.0
            self.sec_c_diff.entry.config(
                state=tk.NORMAL if twt > 0 else tk.DISABLED)
        self.sec_twt.var.trace_add("write", _sync_sec_c_diff)
        _sync_sec_c_diff()

        # iel=10 + inelastic_mode 1/2 rejects any nonzero special mode or
        # secondary scatterer at deck generation, so both sections hide
        # behind a one-line note in that state (see _sync_modes; the pack
        # options are captured once, right after construction).
        self._special_scatter_sections = [
            (w, {k: v for k, v in w.pack_info().items() if k != "in"})
            for w in (cf.master, sf.master)]
        self._special_modes_note = ttk.Label(
            frame,
            text=("Special modes and the secondary scatterer apply to "
                  "inelastic_mode = 0 decks only. Hiding them CLEARED "
                  "ncold, nsk, the S(kappa) table and the secondary "
                  "scatterer: the phonopy-backed modes compute MF7/MT4 "
                  "directly from the force constants, so a pair-correlation "
                  "or secondary-scatterer correction has nothing to apply "
                  "to. Going back to inelastic_mode = 0 does not restore "
                  "the values — re-enter them there."),
            foreground="gray", wraplength=860, justify=tk.LEFT)

    def _clear_special_scatter_fields(self):
        """Reset the Special Modes and Secondary Scatterer fields to off.

        Called by _sync_modes when it hides those sections (iel=10 with
        inelastic_mode 1/2, where deck generation rejects them) and by
        _reset_form_to_defaults.
        """
        self.ncold.set("0 — None")
        self.nsk.set("0 — None")
        self.ska_dka.set("0")
        self.ska_text.delete("1.0", tk.END)
        self.cfrac.set("0")
        self.nss.set("0 — None")
        self.b7.set("1 — Free gas")
        self.aws.set("0")
        self.sps.set("0")
        self.mss.set("1")
        self.sec_dos_delta.set("0")
        self.sec_dos_rho_text.delete("1.0", tk.END)
        self.sec_twt.set("0.0")
        self.sec_c_diff.set("0.0")
        self.sec_tbeta.set("1.0")
        self.sec_osc_energies.delete("1.0", tk.END)
        self.sec_osc_weights.delete("1.0", tk.END)

    def _update_scattering_groups(self):
        """Show only the special-mode fields the selected modes use."""
        def lead(widget):
            """Leading integer of a combo widget's selection."""
            try:
                return int(widget.get().split()[0])
            except (ValueError, IndexError):
                return 0

        ncold, nsk = lead(self.ncold), lead(self.nsk)
        nss, b7 = lead(self.nss), lead(self.b7)
        if nsk > 0 or ncold > 0:
            self._ska_group.pack(fill=tk.X, after=self.nsk)
        else:
            self._ska_group.pack_forget()
        # cfrac is a Card 19 field: emitted (and read) for nsk > 0 only --
        # ncold alone carries the S(kappa) table but no coherent fraction.
        if nsk > 0:
            self._cfrac_group.pack(fill=tk.X, after=self._ska_group)
        else:
            self._cfrac_group.pack_forget()
        if nss > 0:
            self._sec_group.pack(fill=tk.X)
        else:
            self._sec_group.pack_forget()
        if nss > 0 and b7 == 0:
            self._sec2_frame.pack(fill=tk.X, pady=(2, 12))
        else:
            self._sec2_frame.pack_forget()

    # ------------------------------------------------------------------
    # Tab 3: Grid Generation
    # ------------------------------------------------------------------
    def _build_grid_tab(self):
        """Build the Grids part widgets."""
        frame = self._form_part("Grids")

        # Temperatures
        tf = form_section(frame, "Temperatures")

        temp_row = ttk.Frame(tf)
        temp_row.pack(fill=tk.X)
        ttk.Label(temp_row, text="Enter temperatures in K, separated by spaces:"
                  ).pack(side=tk.LEFT, anchor=tk.W)
        InfoLabel(temp_row, "Temperatures",
                   "List of temperatures (in Kelvin) at which to evaluate "
                   "the S(alpha, beta) table.\n\n"
                   "The phonon spectrum is read only for the first "
                   "temperature. All subsequent temperatures reuse the same "
                   "phonon spectrum but recompute S(a,b) at the new "
                   "temperature (the Boltzmann population changes).\n\n"
                   "For ENDF TSL libraries, typical temperature sets are:\n"
                   "  Graphite: 296 400 500 600 700 800 1000 1200 1600 2000\n"
                   "  Be: 77 100 200 293.6 296 400 500 600 700 800 1000 1200\n"
                   "  H2O: 283.6 293.6 300 323.6 350 373.6 400 423.6 450 "
                   "473.6 500 523.6 550 573.6 600 623.6 650 800"
                   ).pack(side=tk.LEFT, padx=(5, 0))
        self.temps_var = tk.StringVar(value="296.0")
        ttk.Entry(tf, textvariable=self.temps_var, width=60).pack(
            fill=tk.X, pady=2)

        # LAT
        lat_row = ttk.Frame(tf)
        lat_row.pack(fill=tk.X, pady=2)
        self.lat = LabeledCombobox(
            lat_row, "LAT:",
            ["0 — alpha/beta in kT units",
             "1 — alpha/beta in kT_thermal (0.0253 eV)"],
            default="1 — alpha/beta in kT_thermal (0.0253 eV)",
            width=35,
            help_title="LAT (Alpha/Beta Scaling)",
            help_text="Controls the energy unit used to scale the alpha "
                      "(momentum transfer) and beta (energy transfer) "
                      "values.\n\n"
                      "LAT=0: alpha and beta are in units of the "
                      "temperature-dependent kT = k_B * T. The grid "
                      "changes meaning at each temperature.\n\n"
                      "LAT=1: alpha and beta are in units of a fixed "
                      "thermal energy kT_thermal = 0.0253 eV (corresponding "
                      "to T = 293.6 K). This is the standard convention "
                      "for ENDF TSL libraries and is recommended.\n\n"
                      "With LAT=1, the same alpha/beta grid is used at all "
                      "temperatures, which simplifies interpolation in "
                      "transport codes.")
        self.lat.pack(fill=tk.X)

        # Alpha/Beta grids
        gf = form_section(frame, "Alpha / Beta Grids")

        grid_desc = ttk.Frame(gf)
        grid_desc.pack(fill=tk.X, pady=(0, 5))
        ttk.Label(grid_desc,
                  text="S(alpha, beta) is evaluated on alpha and beta grids. "
                       "Choose how to define them:",
                  foreground="gray").pack(side=tk.LEFT)
        InfoLabel(grid_desc, "Alpha and Beta Grids",
                   "The S(alpha, beta) table is a 2D function of "
                   "dimensionless momentum transfer (alpha) and "
                   "dimensionless energy transfer (beta).\n\n"
                   "alpha = E_recoil / (A * kT)\n"
                   "  Proportional to Q^2 (momentum transfer squared).\n"
                   "  Larger alpha = higher momentum transfer.\n\n"
                   "beta = E_transfer / kT\n"
                   "  Dimensionless energy transfer.\n"
                   "  beta > 0 = neutron loses energy (down-scattering).\n"
                   "  beta = 0 = elastic/quasi-elastic.\n\n"
                   "The grids define where S(a,b) is tabulated. They must "
                   "cover the physically relevant kinematic range.\n\n"
                   "Automatic mode:\n"
                   "  BETA grid: three regions tied to the phonon "
                   "spectrum. A logarithmic low-beta tail (N lower), a "
                   "linear region spanning [0, freq_max] (N phonon), and "
                   "a logarithmic high-beta tail up to Beta max (N "
                   "upper).\n"
                   "  ALPHA grid: linear in momentum transfer Q. Points "
                   "every dQ up to Q cut (covering the thermal scattering "
                   "window), then a logarithmic tail out to the beta "
                   "grid's kinematic reach (alpha_max = 4*beta_max/A). "
                   "The thermal-energy cross section is controlled by "
                   "S(alpha, beta) at small alpha, so alpha must be dense "
                   "where Q is small; an alpha grid that simply mirrors "
                   "the beta layout under-integrates the thermal "
                   "inelastic cross section by 10-20%.\n\n"
                   "Manual mode: enter your own alpha and beta values "
                   "directly. Useful when matching a reference evaluation "
                   "or when you need precise control over grid placement."
                   ).pack(side=tk.LEFT, padx=(5, 0))

        self.grid_mode = tk.StringVar(value="auto")
        radio_frame = ttk.Frame(gf)
        radio_frame.pack(fill=tk.X, pady=(0, 5))
        ttk.Radiobutton(radio_frame, text="Automatic grid generation",
                        variable=self.grid_mode, value="auto",
                        command=self._toggle_grid_mode).pack(
            side=tk.LEFT, padx=(0, 20))
        ttk.Radiobutton(radio_frame, text="Manual alpha/beta entry",
                        variable=self.grid_mode, value="manual",
                        command=self._toggle_grid_mode).pack(side=tk.LEFT)

        self.auto_frame = ttk.Frame(gf)
        self.auto_frame.pack(fill=tk.X, pady=5)

        freq_row = ttk.Frame(self.auto_frame)
        freq_row.pack(fill=tk.X, pady=2)
        self.freq_max = LabeledEntry(
            freq_row, "Max phonon freq [eV]:", "",
            help_title="Maximum Phonon Frequency",
            help_text="The maximum phonon frequency of the material in eV "
                      "(the highest vibrational energy in its phonon "
                      "spectrum). This sets the upper bound of the linearly "
                      "spaced grid region.\n\n"
                      "Can be auto-detected with the buttons to the right, "
                      "or entered manually.\n\n"
                      "Typical values:\n"
                      "  Graphite: ~0.20 eV\n"
                      "  Be: ~0.10 eV\n"
                      "  H2O: ~0.48 eV\n"
                      "  UO2: ~0.09 eV")
        self.freq_max.pack(side=tk.LEFT)
        ttk.Button(freq_row, text="Detect from DOS",
                   command=self._detect_freq_from_dos).pack(
            side=tk.LEFT, padx=5)
        self._detect_fc_btn = ttk.Button(
            freq_row, text="Detect from phonopy.yaml",
            command=self._detect_freq_from_fc)
        self._detect_fc_btn.pack(side=tk.LEFT, padx=5)
        # the seven converged-grid knobs, shared with the NCrystal panel
        from irma.gui.grid_form import build_auto_grid_entries
        for name, w in build_auto_grid_entries(self.auto_frame).items():
            setattr(self, name, w)

        # Manual grid entry
        self.manual_frame = ttk.Frame(gf)

        alpha_header = ttk.Frame(self.manual_frame)
        alpha_header.pack(fill=tk.X)
        ttk.Label(alpha_header,
                  text="Alpha values (space-separated, ascending):"
                  ).pack(side=tk.LEFT, anchor=tk.W)
        InfoLabel(alpha_header, "Alpha Grid",
                   "Enter alpha values in ascending order, "
                   "space-separated.\n\n"
                   "Alpha is the dimensionless momentum transfer "
                   "(alpha = E_r / (A * kT), proportional to Q^2).\n\n"
                   "The grid should cover from small alpha (~1e-6) to "
                   "large alpha (where S(a,b) becomes negligible).\n\n"
                   "Typical range for graphite (LAT=1):\n"
                   "  ~0.003 to ~90 with ~150 points\n\n"
                   "You can extract grids from reference ENDF evaluations "
                   "or use the automatic mode as a starting point."
                   ).pack(side=tk.LEFT, padx=(5, 0))
        self.alpha_text = tk.Text(self.manual_frame, height=4, width=70,
                                  font=fixed_font())
        self.alpha_text.pack(fill=tk.X, pady=2)

        beta_header = ttk.Frame(self.manual_frame)
        beta_header.pack(fill=tk.X)
        ttk.Label(beta_header,
                  text="Beta values (space-separated, ascending, start with 0):"
                  ).pack(side=tk.LEFT, anchor=tk.W)
        InfoLabel(beta_header, "Beta Grid",
                   "Enter beta values in ascending order, "
                   "space-separated. The first value should be 0.\n\n"
                   "Beta is the dimensionless energy transfer "
                   "(beta = E / kT).\n\n"
                   "The grid should cover from 0 to the maximum energy "
                   "transfer of interest (typically several eV / kT).\n\n"
                   "Typical range for graphite at 296K (LAT=1):\n"
                   "  0 to ~200 with ~400 points\n"
                   "  (beta=200 at kT_thermal corresponds to ~5 eV)\n\n"
                   "You can extract grids from reference ENDF evaluations "
                   "or use the automatic mode as a starting point."
                   ).pack(side=tk.LEFT, padx=(5, 0))
        self.beta_text = tk.Text(self.manual_frame, height=4, width=70,
                                 font=fixed_font())
        self.beta_text.pack(fill=tk.X, pady=2)

        # Preview button
        self._grid_btn_frame = ttk.Frame(gf)
        self._grid_btn_frame.pack(fill=tk.X, pady=5)
        ttk.Button(self._grid_btn_frame, text="Preview Grid Sizes",
                   command=self._preview_grids).pack(side=tk.LEFT)
        self.grid_info_var = tk.StringVar(value="")
        ttk.Label(self._grid_btn_frame, textvariable=self.grid_info_var,
                  foreground="#3b9eff").pack(side=tk.LEFT, padx=10)

    def _detect_freq_from_dos(self):
        """Detect max phonon frequency from a phonopy total_dos.dat file."""
        dos_path = self.dos_file.get().strip()
        if not dos_path:
            # Ask user to select a file
            dos_path = filedialog.askopenfilename(
                title="Select phonopy total_dos.dat",
                filetypes=[("DAT files", "*.dat"), ("All files", "*.*")])
            if not dos_path:
                return
            self.dos_file.set(dos_path)
        try:
            import numpy as np
            from irma.core.constants import THZ_TO_EV
            freq, dos = np.loadtxt(dos_path, usecols=(0, 1), unpack=True)
            # the last point where the DOS is above 1% of its peak
            above = np.nonzero(dos > 0.01 * dos.max())[0]
            fmax = (freq[above[-1]] if above.size else freq[-1]) * THZ_TO_EV
            self.freq_max.set(f"{fmax:.4f}")
            self.grid_info_var.set(f"Detected freq_max = {fmax*1000:.1f} meV")
        except Exception as e:
            messagebox.showerror("Error", f"Could not read DOS file:\n{e}")

    def _marshal(self, func, *args):
        """Queue ``func(*args)`` onto the Tk thread, no-op if it is torn down.

        Worker threads marshal results back via root.after; once the window is
        destroyed the interpreter is gone and after() raises TclError/
        RuntimeError. Swallowing that turns a late result from a quit-during-work
        thread into a clean no-op instead of a spurious traceback.
        """
        try:
            self.root.after(0, func, *args)
        except (RuntimeError, tk.TclError):
            pass

    def _detect_freq_from_fc(self):
        """Detect max phonon frequency from phonopy.yaml via phonopy API.

        phonopy.load + run_mesh can take seconds-to-minutes on a large
        supercell, so it runs on a background thread; the GUI stays responsive
        and the result is marshalled back to the Tk thread via root.after.

        Single-flight: the worker enters isolated_phonopy_cwd(), a *process*
        os.chdir, so two concurrent detections (or a calculation launched while
        one runs) would race the global cwd. The detect button is disabled for
        the duration, and detection is refused while a calculation owns the cwd.
        """
        if self.runner.is_running:
            messagebox.showwarning(
                "Busy", "Finish or cancel the running calculation before "
                "detecting freq_max (both use the process working directory).")
            return
        fc_path = filedialog.askopenfilename(
            title="Select phonopy.yaml",
            filetypes=[("YAML files", "*.yaml"), ("All files", "*.*")])
        if not fc_path:
            return
        self._detect_fc_btn.config(state=tk.DISABLED)
        self.grid_info_var.set("Loading force constants (phonopy)...")
        threading.Thread(
            target=self._detect_freq_from_fc_worker,
            args=(os.path.abspath(fc_path),), daemon=True).start()

    def _detect_freq_from_fc_worker(self, fc_path):
        """Heavy phonopy mesh evaluation; runs OFF the Tk thread."""
        try:
            import numpy as np
            from irma.core.phonopy_io import load_phonopy
            from irma.core.constants import THZ_TO_EV
            # The except-handler below turns an unsafe or unreadable yaml
            # into an error dialog.
            ph = load_phonopy(fc_path)
            ph.run_mesh([8, 8, 8], with_eigenvectors=False)
            freqs_ev = ph.mesh.frequencies * THZ_TO_EV
            positive = freqs_ev[freqs_ev > 1e-4 * THZ_TO_EV]
            if len(positive) == 0:
                raise ValueError("No positive frequencies found in mesh.")
            fmax = float(np.max(positive))
        except ImportError:
            self._marshal(self._detect_freq_from_fc_failed, "import", None)
            return
        except Exception as e:
            self._marshal(self._detect_freq_from_fc_failed, "error", str(e))
            return
        self._marshal(self._detect_freq_from_fc_done, fmax)

    def _detect_freq_from_fc_done(self, fmax):
        """Apply a successful freq_max detection (back on the Tk thread)."""
        self._detect_fc_btn.config(state=tk.NORMAL)
        self.freq_max.set(f"{fmax:.4f}")
        self.grid_info_var.set(f"Detected freq_max = {fmax*1000:.1f} meV")

    def _detect_freq_from_fc_failed(self, kind, detail):
        """Report a failed freq_max detection (back on the Tk thread)."""
        self._detect_fc_btn.config(state=tk.NORMAL)
        self.grid_info_var.set("")
        if kind == "import":
            messagebox.showerror(
                "phonopy not available",
                "phonopy is required to detect freq_max from force constants.\n"
                "Install with: pip install phonopy\n\n"
                "Alternatively, use 'Detect from DOS' with a phonopy "
                "total_dos.dat file, or enter freq_max manually.")
        else:
            messagebox.showerror("Error",
                                 f"Could not read force constants:\n{detail}")

    def _toggle_grid_mode(self):
        """Enable the widgets of the selected grid mode (automatic/manual)."""
        if self.grid_mode.get() == "auto":
            self.manual_frame.pack_forget()
            self.auto_frame.pack(fill=tk.X, pady=5, before=self._grid_btn_frame)
        else:
            self.auto_frame.pack_forget()
            self.manual_frame.pack(fill=tk.X, pady=5, before=self._grid_btn_frame)

    def _preview_grids(self):
        """Compute and display the alpha/beta grids for the current settings."""
        try:
            if self.grid_mode.get() == "auto":
                from irma.core.grids import describe_beta_grid
                alpha, beta, t_ref = self._auto_grids(self._parse_temperatures())
                self.grid_info_var.set(
                    f"nalpha={len(alpha)}, nbeta={len(beta)}, "
                    f"alpha=[{alpha[0]:.4e}..{alpha[-1]:.4e}], "
                    f"beta=[{beta[0]:.1f}..{beta[-1]:.4e}]\n"
                    + describe_beta_grid(beta, t_ref, self._code(self.iint)))
            else:
                alpha = self._parse_manual_array(self.alpha_text, "alpha grid")
                beta = self._parse_manual_array(self.beta_text, "beta grid")
                self.grid_info_var.set(
                    f"nalpha={len(alpha)}, nbeta={len(beta)}")
        except Exception as e:
            self.grid_info_var.set(f"Error: {e}")

    # ------------------------------------------------------------------
    # Tab 4: Phonon Parameters
    def _auto_grids(self, temps):
        """The automatic (alpha, beta) grids and their reference temperature.

        Under lat=1 the deck stores alpha/beta in fixed 0.0253 eV units, so
        the grids are anchored at that temperature, not at temps[0]; iint=1
        gets the step-capped lin-lin beta tail, iint=0 the log tail.
        """
        from irma.core.grids import (generate_beta_grid, generate_alpha_grid,
                                     grid_reference_temperature_K)
        freq_max = parse_float("Max phonon freq [eV]", self.freq_max.get())
        t_ref = grid_reference_temperature_K(self._code(self.lat), temps[0])
        awr = parse_float("AWR", self.awr.get())
        beta = generate_beta_grid(
            freq_max, t_ref, iint=self._code(self.iint), awr=awr,
            n_lower=parse_int("N lower (log)", self.n_lower.get()),
            n_phonon=parse_int("N phonon (linear)", self.n_phonon.get()),
            n_upper=parse_int("N upper (log)", self.n_upper.get()),
            beta_max_eV=parse_float("Beta max [eV]", self.beta_max.get()),
            evaluation_temperatures_K=temps)
        alpha = generate_alpha_grid(
            beta, awr, t_ref,
            dq_ang_inv=parse_float("Alpha dQ [1/A]", self.alpha_dq.get()),
            q_cut_ang_inv=parse_float("Alpha Q cut [1/A]",
                                      self.alpha_qcut.get()),
            n_log=parse_int("Alpha N log", self.alpha_nlog.get()))
        return alpha, beta, t_ref

    def _toggle_dos_source(self):
        """Show only the selected DOS source's fields, under its radio."""
        if self.dos_source.get() == "phonopy":
            self.dos_manual_frame.pack_forget()
            self.dos_phonopy_frame.pack(fill=tk.X, padx=(24, 0),
                                        before=self._dos_manual_radio)
        else:
            self.dos_phonopy_frame.pack_forget()
            self.dos_manual_frame.pack(fill=tk.X, padx=(24, 0))

    # ------------------------------------------------------------------
    def _build_phonon_tab(self):
        """Build the Phonon part widgets."""
        frame = self._form_part("Phonon")

        # Continuous distribution
        self._phonon_ignored_note = ttk.Label(
            frame,
            text=("inelastic_mode 1/2 is selected in the Material part: "
                  "the phonopy-backed modes compute MT4 directly from the "
                  "force constants, so the phonon distribution, "
                  "translational, and oscillator cards do not apply and "
                  "are hidden. They return with inelastic_mode = 0."),
            foreground="#e5484d", wraplength=860, justify=tk.LEFT)
        cf = form_section(frame, "Continuous Phonon Distribution")

        # Each source's fields live directly under its radio button and
        # only the selected source's fields are shown.
        self.dos_source = tk.StringVar(value="phonopy")
        ttk.Radiobutton(cf, text="From phonopy total_dos.dat",
                        variable=self.dos_source, value="phonopy",
                        command=self._toggle_dos_source).pack(anchor=tk.W)

        self.dos_phonopy_frame = ttk.Frame(cf)
        self.dos_file = FileSelector(
            self.dos_phonopy_frame, "total_dos.dat:", mode="open",
            filetypes=[("DAT files", "*.dat"), ("All files", "*.*")],
            help_title="Phonon DOS File",
            help_text="Path to a phonopy total_dos.dat file containing the "
                      "phonon density of states.\n\n"
                      "To generate this file with phonopy:\n"
                      "  phonopy -c POSCAR --dim=\"N1 N2 N3\" --readfc \\\n"
                      "    --mp=\"M1 M2 M3\" --dos --fmin=0 --sigma=0.2\n\n"
                      "The file contains two columns: frequency (THz) and "
                      "DOS. IRMA automatically converts from THz to eV "
                      "and normalizes the spectrum.")
        self.dos_file.pack(fill=tk.X, pady=5)

        self._dos_manual_radio = ttk.Radiobutton(
            cf, text="Manual entry (delta_e and rho values)",
            variable=self.dos_source, value="manual",
            command=self._toggle_dos_source)
        self._dos_manual_radio.pack(anchor=tk.W)

        self.dos_manual_frame = ttk.Frame(cf)
        self.dos_delta = LabeledEntry(
            self.dos_manual_frame, "delta_e [eV]:", "",
            help_title="delta_e (DOS Energy Spacing)",
            help_text="Uniform energy grid spacing for the phonon density "
                      "of states, in eV.\n\n"
                      "The DOS values (rho) below are assumed to be on an "
                      "equally spaced grid starting at E=0 with this spacing.")
        self.dos_delta.pack(fill=tk.X, pady=2)
        ttk.Label(self.dos_manual_frame,
                  text="rho values (space-separated):").pack(anchor=tk.W)
        self.dos_rho_text = tk.Text(self.dos_manual_frame, height=3, width=70,
                                    font=fixed_font())
        self.dos_rho_text.pack(fill=tk.X)

        self._toggle_dos_source()

        # Translational mode
        tf = form_section(frame, "Translational Mode")

        self.twt = LabeledEntry(
            tf, "twt:", "0.0",
            help_title="twt (Translational Weight)",
            help_text="Weight of the translational (diffusive or free-gas) "
                      "scattering component.\n\n"
                      "twt=0: No translational motion (crystalline solid).\n\n"
                      "twt>0: Part of the scattering comes from center-of-mass "
                      "translation (liquids, gases). The value is the "
                      "fraction of scattering from translation.\n\n"
                      "The sum tbeta + twt + sum(oscillator_weights) = 1.")
        self.twt.pack(fill=tk.X, pady=2)
        self.c_diff = LabeledEntry(
            tf, "c (diffusion):", "0.0",
            help_title="c (Diffusion Constant)",
            help_text="Diffusion constant for the translational mode.\n\n"
                      "c=0: Free-gas translation (no diffusion).\n\n"
                      "c>0: Diffusion model. The translational scattering "
                      "follows Egelstaff-Schofield diffusion with this "
                      "constant.\n\n"
                      "Only used when twt > 0.")
        self.c_diff.pack(fill=tk.X, pady=2)

        # c is read only when twt > 0: grey it out otherwise (the trace
        # covers typing, deck import and reset alike; an unparsable twt
        # counts as 0).
        def _sync_c_diff(*_):
            try:
                twt = float(self.twt.get())
            except ValueError:
                twt = 0.0
            self.c_diff.entry.config(
                state=tk.NORMAL if twt > 0 else tk.DISABLED)
        self.twt.var.trace_add("write", _sync_c_diff)
        _sync_c_diff()
        self.tbeta = LabeledEntry(
            tf, "tbeta:", "1.0",
            help_title="tbeta (Continuous Distribution Weight)",
            help_text="Normalization weight for the continuous phonon "
                      "distribution.\n\n"
                      "The weights must satisfy:\n"
                      "  tbeta + twt + sum(oscillator weights) = 1\n\n"
                      "For a simple crystalline solid with no oscillators "
                      "and no translational mode, tbeta = 1.0.\n\n"
                      "For H in H2O: tbeta ~0.444, plus two discrete "
                      "oscillators for the internal H2O modes.")
        self.tbeta.pack(fill=tk.X, pady=2)

        # Discrete oscillators
        of = form_section(frame, "Discrete Oscillators")
        # (widget, original pack options) for the mode-1/2 hide/show in
        # _sync_modes; captured once, right after construction.
        self._phonon_card_sections = [
            (w, {k: v for k, v in w.pack_info().items() if k != "in"})
            for w in (cf.master, tf.master, of.master)]

        osc_header = ttk.Frame(of)
        osc_header.pack(fill=tk.X)
        ttk.Label(osc_header,
                  text="Oscillator energies [eV] (space-separated):"
                  ).pack(side=tk.LEFT, anchor=tk.W)
        InfoLabel(osc_header, "Discrete Oscillators",
                   "Discrete oscillators represent isolated vibrational "
                   "modes that are treated analytically rather than through "
                   "the continuous phonon expansion.\n\n"
                   "Commonly used for intramolecular modes, such as the "
                   "bending and stretching modes of H2O.\n\n"
                   "Energies: The energy of each oscillator in eV.\n"
                   "Weights: The fraction of scattering from each "
                   "oscillator.\n\n"
                   "Example (H in H2O):\n"
                   "  Energies: 0.205 0.436\n"
                   "  Weights:  0.166 0.389\n"
                   "  (bending at 205 meV, stretching at 436 meV)\n\n"
                   "Leave empty if no discrete oscillators are needed "
                   "(most crystalline solids)."
                   ).pack(side=tk.LEFT, padx=(5, 0))
        self.osc_energies = tk.Text(of, height=2, width=60,
                                    font=fixed_font())
        self.osc_energies.pack(fill=tk.X, pady=2)
        ttk.Label(of, text="Oscillator weights (space-separated):").pack(
            anchor=tk.W)
        self.osc_weights = tk.Text(of, height=2, width=60,
                                   font=fixed_font())
        self.osc_weights.pack(fill=tk.X, pady=2)

    # ------------------------------------------------------------------
    # Tab 5: Output & Run
    # ------------------------------------------------------------------
    def _build_output_tab(self):
        """Build the Run part widgets."""
        frame = self._form_part("Run")

        # Output file
        of = form_section(frame, "Output")

        self.output_file = FileSelector(
            of, "Output ENDF file:", mode="save",
            filetypes=[("ENDF files", "*.endf"), ("All files", "*.*")],
            help_title="Output ENDF File",
            help_text="Path where the output ENDF-6 file will be written.\n\n"
                      "The output contains the S(alpha, beta) table in "
                      "MF7/MT4 format, plus coherent elastic Bragg edges in "
                      "MF7/MT2 (if iel > 0).\n\n"
                      "This file can be processed by THERMR/ACER in NJOY "
                      "for use in Monte Carlo transport codes (MCNP, "
                      "OpenMC, Serpent, etc.).")
        self.output_file.pack(fill=tk.X, pady=2)

        # ENDF comments (MF1/MT451)
        cf = form_section(frame, "ENDF Comment Cards (MF1/MT451)")

        cf_top = ttk.Frame(cf)
        cf_top.pack(fill=tk.X)
        ttk.Label(cf_top,
                  text="These lines are written into the ENDF output file "
                       "header. Each line becomes one comment record.",
                  foreground="gray").pack(side=tk.LEFT)
        InfoLabel(cf_top, "ENDF Comment Cards",
                   "Comment cards are stored in MF1/MT451 of the ENDF file.\n\n"
                   "The standard ENDF layout is:\n"
                   "  Line 1: ZSYMAM + Lab + Date + Author\n"
                   "  Line 2: Reference + Dist/Rev/End dates\n"
                   "  Lines 3-5: Sub-library identifiers\n"
                   "  Lines 6+: Free-form description\n\n"
                   "Each line is truncated to 66 characters.\n\n"
                   "Example first line:\n"
                   "  6-C-12   LANL     EVAL-JAN24 A.I. Hawari\n\n"
                   "You can include as many description lines as needed "
                   "to document the evaluation (method, references, etc.)."
                   ).pack(side=tk.RIGHT)

        self.comments_text = ScrolledText(cf, height=8)
        self.comments_text.pack(fill=tk.BOTH, expand=True, pady=(5, 0))

        # Run controls
        rf = ttk.Frame(frame)
        rf.pack(fill=tk.X, pady=(0, 10))

        self.run_btn = ttk.Button(
            rf, text="Run Calculation", default="active", command=self._run)
        self.run_btn.pack(side=tk.LEFT, padx=(0, 10))

        self.cancel_btn = ttk.Button(
            rf, text="Cancel", command=self._cancel, state=tk.DISABLED)
        self.cancel_btn.pack(side=tk.LEFT, padx=(0, 10))

        self.export_btn = ttk.Button(
            rf, text="Export Input File...", command=self._export_input)
        self.export_btn.pack(side=tk.LEFT, padx=(0, 10))

        self.progress = ttk.Progressbar(rf, mode="indeterminate", length=200)
        self.progress.pack(side=tk.LEFT, padx=(0, 10))

        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(rf, textvariable=self.status_var).pack(side=tk.LEFT)

        # Live phase readout, fed from the child's streamed log markers (#12).
        self.phase_var = tk.StringVar(value="")
        ttk.Label(rf, textvariable=self.phase_var,
                  foreground="#3b9eff").pack(side=tk.LEFT, padx=(10, 0))

        # Log output
        lf = form_section(frame, "Log", expand=True)

        self.log = ScrolledText(lf, height=20)
        self.log.pack(fill=tk.BOTH, expand=True)

        log_btns = ttk.Frame(lf)
        log_btns.pack(fill=tk.X, pady=(5, 0))
        ttk.Button(log_btns, text="Clear Log",
                   command=self.log.clear).pack(side=tk.LEFT)

    # ------------------------------------------------------------------
    # Bottom bar
    # ------------------------------------------------------------------
    def _build_bottom_bar(self):
        """Build the Run/Cancel/progress bottom bar."""
        bar = ttk.Frame(self.root)
        bar.pack(fill=tk.X, padx=5, pady=(0, 5))
        from irma import __version__
        ttk.Label(bar, text=f"IRMA v{__version__}",
                  foreground="gray").pack(side=tk.LEFT)

    # ------------------------------------------------------------------
    # Input generation
    # ------------------------------------------------------------------
    def _parse_temperatures(self):
        """Parse the temperature-list field into floats."""
        text = self.temps_var.get().strip()
        return [parse_float("temperatures [K]", t) for t in text.split()]

    def _parse_manual_array(self, text_widget, label):
        """Parse a manual grid text field into floats."""
        import numpy as np
        text = text_widget.get("1.0", tk.END).strip()
        return np.array([parse_float(label, x) for x in text.split()])

    @staticmethod
    def _code(widget):
        """Leading integer of an 'N — label' combobox or variable."""
        return int(widget.get().split("—")[0])

    @staticmethod
    def _set_code(widget, code, values=None):
        """Select the 'N — label' entry whose leading integer is code."""
        values = values or widget.combo["values"]
        widget.set(next((v for v in values if v.split(" —")[0] == str(code)),
                        str(code)))

    def _read_phonopy_dos(self, filename):
        """Read a phonopy total_dos.dat as (delta_e [eV], rho with unit area)."""
        import numpy as np
        from irma.core.constants import THZ_TO_EV
        freq, dos = np.loadtxt(filename, usecols=(0, 1), unpack=True)
        keep = freq >= 0
        freq_ev = freq[keep] * THZ_TO_EV
        dos_ev = dos[keep] / THZ_TO_EV
        dos_ev[dos_ev < 0] = 0
        integral = np.trapezoid(dos_ev, freq_ev)
        if integral > 0:
            dos_ev /= integral
        return freq_ev[1] - freq_ev[0], dos_ev

    def _on_ext_model_change(self, event=None):
        """When the extinction model changes, restrict the distribution dropdown to
        that model's family and snap the value to the family default (Gauss for
        Becker-Coppens, rect for Sabine) so a cross-family value can't be picked.
        The recipe dropdown only steers the numeric Becker-Coppens y(x)
        evaluation; the analytic Sabine models ignore it, so it is hidden for
        them (visibility only -- the stored value survives a model switch)."""
        if self.ext_model.get().startswith("Sabine"):
            family, default = _EXT_DIST_SABINE, "rect"
            self.ext_recipe.pack_forget()
        else:
            family, default = _EXT_DIST_BC, "Gauss"
            if not self.ext_recipe.winfo_manager():
                self.ext_recipe.pack(fill=tk.X, pady=2,
                                     before=self.ext_rmse_tol)
        self.ext_dist.combo["values"] = family
        self.ext_dist.set(default)

    def _generate_input_text(self):
        """Generate IRMA input text (LEAPR-based format) from GUI parameters."""
        import numpy as np

        temps = self._parse_temperatures()
        ntempr = len(temps)
        iel = self._code(self.iel_var)
        ncold = self._code(self.ncold)
        nsk = self._code(self.nsk)
        nss = self._code(self.nss)
        lat = self._code(self.lat)
        isabt = self._code(self.isabt)
        ilog = self._code(self.ilog)
        iint = self._code(self.iint)

        npr = parse_int("npr (principal atom count)",
                        self.npr.get().strip() or "1")

        # inelastic_mode is an iel=10 concept; 0 (classic) for everything else.
        inelastic_mode_val = int(self.inelastic_mode_var.get()) if iel == 10 else 0

        # S(kappa) table (Cards 17-19) backing ncold/nsk
        ska_vals = []
        ska_raw = self.ska_text.get("1.0", tk.END).strip()
        if ska_raw:
            ska_vals = [parse_float("S(kappa) values", x) for x in ska_raw.split()]
        if (nsk > 0 or ncold > 0) and not ska_vals:
            raise ValueError(
                "nsk > 0 or ncold > 0 requires the S(kappa) table "
                "(dka and the S(kappa) values).")

        # Card 4/5 material identity. The fresh form ships these blank (they
        # name the USER's material), so the first Run on an unfilled form must
        # say which field is missing. Without the check the blanks are simply
        # interpolated away and the card is written short, which re-reads as a
        # different, silently wrong deck (smin landing in isabt, and so on).
        missing = [label for label, widget in (
            ("MAT (Card 4)", self.mat), ("ZA (Card 4)", self.za),
            ("AWR (Card 5)", self.awr), ("sigma_free (Card 5)", self.spr))
            if not str(widget.get()).strip()]
        if missing:
            raise ValueError(
                "Material identity is empty: " + ", ".join(missing)
                + ". IRMA leaves these blank because they describe YOUR "
                "material — fill them in (the 'Fill AWR + sigma_free from ZA' "
                "button fills AWR and sigma_free once ZA is set), or load a "
                "input file with Import Input File, e.g. one of the examples/tsl "
                "input files.")

        lines = []
        # Card 1
        lines.append("20 /")
        # Card 2 (title preserved through import -> export)
        lines.append(f"{_quote(self._imported_title)} /")
        # Card 3 (iprint preserved through import -> export)
        lines.append(f"{ntempr} {self._imported_iprint} {self.nphon.get()} /")
        # Card 4 (iint appended only when non-default, so classic decks
        # round-trip byte-identically)
        smin_s = self.smin.get().strip() or '1e-75'
        card4 = f"{self.mat.get()} {self.za.get()} {isabt} {ilog} {smin_s}"
        if iint:
            card4 += f" {iint}"
        lines.append(card4 + " /")
        # Card 5
        lines.append(f"{self.awr.get()} {self.spr.get()} "
                      f"{npr} {iel} {ncold} {nsk} /")
        # Card 6
        b7 = 1
        sec_rho = []
        if nss > 0:
            b7 = self._code(self.b7)
            mss = int(parse_float("mss (secondary atom count)",
                                  self.mss.get().strip() or "1"))
            if b7 <= 0:
                # Two-pass: the secondary scatterer needs its own phonon
                # model, emitted below as a complete second temperature pass.
                sec_rho_raw = self.sec_dos_rho_text.get("1.0", tk.END).strip()
                if sec_rho_raw:
                    sec_rho = [parse_float("secondary phonon spectrum (rho)", x)
                               for x in sec_rho_raw.split()]
                if not sec_rho:
                    raise ValueError(
                        "b7 = 0 (two-pass) requires the secondary phonon "
                        "spectrum (delta and the rho values).")
                if ncold > 0 or nsk > 0:
                    raise ValueError(
                        "ncold/nsk together with a two-pass secondary "
                        "scatterer is not supported in the GUI — author "
                        "that input file directly.")
            lines.append(f"{nss} {b7} {self.aws.get()} {self.sps.get()} "
                         f"{mss} /")
        else:
            lines.append("0 0 0 0 0 /")

        # Generalized elastic cards (iel=10)
        if iel == 10:
            # Card 6c/6d are the crystal's identity and ship blank with the
            # form, so name the card and the ways to fill it (same reason as
            # the Card 4/5 check above).
            blank_latt = [label for label, widget in (
                ("a", self.latt_a), ("b", self.latt_b), ("c", self.latt_c),
                ("alpha", self.latt_alpha), ("beta", self.latt_beta),
                ("gamma", self.latt_gamma))
                if not str(widget.get()).strip()]
            if blank_latt:
                raise ValueError(
                    "iel=10 (generalized) requires the lattice parameters "
                    "(Card 6c) in the Material part; empty: "
                    + ", ".join(blank_latt)
                    + ". Type them in, press '" + STRUCTURE_FILL_TITLE
                    + "' (inelastic_mode 1/2), or load an input file with Import "
                    "Input File.")
            atoms = parse_atoms_text(self.atoms_text.get("1.0", tk.END))
            nat = len(atoms)
            if nat < 1:
                raise ValueError(
                    "iel=10 (generalized) requires at least one atom type "
                    "(Card 6d) in the Material part. Type the rows in, press "
                    "'" + STRUCTURE_FILL_TITLE + "' (inelastic_mode 1/2), or "
                    "load an input file with Import Input File.")
            # The principal ZA must be one of the rows (the engine's rule;
            # Apply ZA relabels a row).
            from irma.core.crystal_input import principal_mismatch_message
            mismatch = principal_mismatch_message(
                int(parse_float("ZA", self.za.get())), atoms)
            if mismatch:
                raise ValueError(mismatch)
            elastic_mode = self._code(self.elastic_mode)

            # Card 6e partial spectra ride through import -> export verbatim;
            # they are a classic-path (inelastic_mode=0) feature only.
            partial_spectra = (
                self._imported_partial_spectra
                if inelastic_mode_val == 0 else [])
            nspec = len(partial_spectra)

            # Card 6b: elastic_mode nat nspec inelastic_mode
            #          [edge_group_bins_per_decade] [edge_group_threshold_eV]
            bpd_i = (parse_int("Bragg-edge bins per decade",
                               self.coh_edge_group_bpd.get().strip() or "0")
                     if self.coh_edge_group_enable_var.get() else 0)
            if bpd_i > 0:
                thr = self.coh_edge_group_thr.get().strip() or "1.0"
                lines.append(
                    f"{elastic_mode} {nat} {nspec} {inelastic_mode_val} {bpd_i} {thr} /")
            else:
                lines.append(f"{elastic_mode} {nat} {nspec} {inelastic_mode_val} /")
            # Card 6c
            lines.append(f"{self.latt_a.get()} {self.latt_b.get()} "
                          f"{self.latt_c.get()} {self.latt_alpha.get()} "
                          f"{self.latt_beta.get()} {self.latt_gamma.get()} /")
            # Card 6d
            for at in atoms:
                lines.append(
                    f"{at['Z']} {at['A']} {at['awr']} "
                    f"{at['b_coh']} {at['sigma_inc']} {at['npos']} /")
                coords = '  '.join(
                    f"{x} {y} {z}" for x, y, z in at['positions'])
                lines.append(f"{coords} /")
            # Card 6e: preserved partial spectra (per-species DW, mode 0)
            for sp in partial_spectra:
                lines.append(f"{sp['Z']} {sp['A']} {sp['delta']:.6e} "
                             f"{sp['ni']} /")
                lines.append(fmt_array(sp['rho']) + ' /')

            # Card 6f: phonopy mesh parameters (inelastic_mode=1/2; Card 6e omitted)
            if inelastic_mode_val in (1, 2):
                yaml_path = self.nc_phonopy_yaml.get().strip()
                if not yaml_path:
                    raise ValueError(
                        "inelastic_mode=1 or 2 requires a phonopy.yaml path.")
                lines.append(f"{_quote(yaml_path)} /")
                nx = self.nc_mesh_nx.get().strip() or "40"
                ny = self.nc_mesh_ny.get().strip() or "40"
                nz = self.nc_mesh_nz.get().strip() or "40"
                ncpu_v = self.nc_ncpu.get().strip() or "1"
                use_born_v = int(self.nc_use_born_var.get())
                lines.append(f"{nx} {ny} {nz} {ncpu_v} {use_born_v} /")
                if use_born_v == 1:
                    born_p = self.nc_born_path.get().strip()
                    if not born_p:
                        raise ValueError(
                            "BORN corrections requested but BORN path is empty.")
                    lines.append(f"{_quote(born_p)} /")

                cutoff_raw = self.nc_min_phonon_energy.get().strip()
                if cutoff_raw:
                    cutoff = parse_float(
                        "Minimum phonon energy [meV]", cutoff_raw)
                    if not np.isfinite(cutoff) or cutoff < 0.0:
                        raise ValueError(
                            "Minimum phonon energy [meV] must be finite and "
                            "nonnegative.")
                    lines.append(f"{cutoff:g} /")

                ndir_v = self.nc_num_directions.get().strip() or "10000"
                mpdir_v = self.nc_multiphonon_num_directions.get().strip() or "1000"
                auto_order_v = int(self.nc_auto_order_var.get())
                if auto_order_v == 1:
                    lines.append(f"{ndir_v} {mpdir_v} {auto_order_v} /")
                else:
                    lines.append(f"{ndir_v} {mpdir_v} /")

            # Optional extinction card (last card of the iel=10 block, before
            # Card 7). Emitted only when the toggle is on; off => ideal edges.
            if self.ext_enable_var.get():
                model = self.ext_model.get().strip()
                ext_l = self.ext_l.get().strip() or "0"
                ext_g = self.ext_g.get().strip() or "0"
                ext_L = self.ext_L.get().strip() or "0"
                card = f"extinction {model} l={ext_l} g={ext_g} L={ext_L}"
                dist = self.ext_dist.get().strip()
                rec = self.ext_recipe.get().strip()
                tol = self.ext_rmse_tol.get().strip()
                if dist:
                    card += f" dist={dist}"
                if rec:
                    card += f" rec={rec}"
                if tol:
                    card += f" rmse_tol={tol}"
                lines.append(card + " /")

        # Card 7: alpha, beta
        if self.grid_mode.get() == "auto":
            alpha, beta, _ = self._auto_grids(temps)
        else:
            alpha = self._parse_manual_array(self.alpha_text, "alpha grid")
            beta = self._parse_manual_array(self.beta_text, "beta grid")

        lines.append(f"{len(alpha)} {len(beta)} {lat} /")

        # Card 8: alpha (fmt_array is the Tk-free deck-text helper)

        lines.append(fmt_array(alpha) + ' /')
        # Card 9: beta
        lines.append(fmt_array(beta) + ' /')

        # Temperature loop
        for itemp, temp in enumerate(temps):
            if inelastic_mode_val in (1, 2):
                lines.append(f"{temp:.4f} /")
            elif itemp == 0:
                lines.append(f"{temp:.4f} /")

                # Phonon DOS
                if self.dos_source.get() == "phonopy":
                    dos_path = self.dos_file.get()
                    if not (dos_path and os.path.exists(dos_path)):
                        raise ValueError(
                            "Phonopy DOS file not found. "
                            "Please select a valid total_dos.dat file.")
                    delta_e, rho = self._read_phonopy_dos(dos_path)
                else:
                    delta_e = parse_float("delta_e [eV] (DOS spacing)",
                                          self.dos_delta.get())
                    rho_text = self.dos_rho_text.get(
                        "1.0", tk.END).strip()
                    rho = np.array([parse_float("phonon spectrum (rho)", x)
                                    for x in rho_text.split()])
                lines += _detail_lines(
                    delta_e, rho,
                    f"{self.twt.get()} {self.c_diff.get()} {self.tbeta.get()}",
                    self.osc_energies, self.osc_weights, "Discrete")

                # Cards 17/18: S(kappa) table (nsk > 0 or ncold > 0);
                # Card 19: coherent fraction (nsk > 0)
                if nsk > 0 or ncold > 0:
                    dka_v = parse_float("dka [1/Å]",
                                        self.ska_dka.get().strip() or "0")
                    lines.append(f"{len(ska_vals)} {dka_v:.6e} /")
                    lines.append(fmt_array(np.asarray(ska_vals)) + ' /')
                if nsk > 0:
                    cfrac_raw = self.cfrac.get().strip()
                    if not cfrac_raw:
                        raise ValueError(
                            "nsk > 0 requires cfrac (coherent fraction, "
                            "Card 19).")
                    lines.append(f"{parse_float('cfrac', cfrac_raw):g} /")
            else:
                # Subsequent temperatures: negative to reuse DOS
                lines.append(f"-{temp:.4f} /")

        # Second pass: with a bound (b7 <= 0) secondary scatterer the engine
        # re-reads the whole temperature block sequence for the secondary's
        # own phonon model and merges the two laws (LEAPR convention).
        if nss > 0 and b7 <= 0:
            for itemp, temp in enumerate(temps):
                if itemp > 0:
                    lines.append(f"-{temp:.4f} /")
                    continue
                lines.append(f"{temp:.4f} /")
                sec_delta = parse_float("secondary delta [eV]",
                                        self.sec_dos_delta.get().strip())
                lines += _detail_lines(
                    sec_delta, sec_rho,
                    f"{self.sec_twt.get().strip() or '0.0'} "
                    f"{self.sec_c_diff.get().strip() or '0.0'} "
                    f"{self.sec_tbeta.get().strip() or '1.0'}",
                    self.sec_osc_energies, self.sec_osc_weights, "Secondary")

        # Comment cards (MF1/MT451). Emit exactly what was stored: interior
        # and leading/trailing whitespace round-trips inside the quotes.
        comments_raw = self.comments_text.get_text()
        lines.extend(emit_comment_lines(comments_raw))
        lines.append("/")

        return '\n'.join(lines) + '\n'

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------
    def _run(self):
        """Collect the form into a deck and start the calculation subprocess."""
        if self.runner.is_running:
            messagebox.showwarning("Running",
                                   "A calculation is already in progress.")
            return

        output_path = self.output_file.get().strip()
        if not output_path:
            messagebox.showerror("Error", "Please specify an output file path.")
            return

        try:
            input_text = self._generate_input_text()
        except Exception as e:
            messagebox.showerror("Input Error", str(e))
            return

        # Write to temporary input file
        self._tmp_input = tempfile.NamedTemporaryFile(
            mode='w', suffix='.input', delete=False)
        self._tmp_input.write(input_text)
        self._tmp_input.close()

        self.log.clear()
        self.log.append("=== IRMA Calculation ===\n\n")
        self.log.append(f"Output: {output_path}\n")
        self.log.append(f"Input file: {self._tmp_input.name}\n\n")

        self.status_var.set("Running...")
        self.phase_var.set("")
        self.run_btn.config(state=tk.DISABLED)
        self.cancel_btn.config(state=tk.NORMAL)
        self.progress.start(10)

        self.runner.run(self._tmp_input.name, output_path)

    def _cancel(self):
        """Cancel the running calculation."""
        if not self.runner.is_running:
            return
        self.status_var.set("Cancelling...")
        self.cancel_btn.config(state=tk.DISABLED)
        self.log.append("\n=== Cancelling (terminating workers) ===\n")
        self.runner.cancel()

    def _log_threadsafe(self, text):
        """Append a log line from the worker thread (marshalled via after())."""
        self.root.after(0, self._append_log_line, text)

    def _append_log_line(self, text):
        """Append a streamed log line and reflect any phase marker (#12)."""
        self.log.append(text)
        phase = _phase_from_log_line(text)
        if phase:
            self.phase_var.set(phase)

    def cleanup_temp_files(self):
        """Idempotently remove the form-owned temp deck (review GUI-2):
        the completion callback that used to own this dies with Tk on a
        close-during-run, so app close also calls it synchronously."""
        tmp = getattr(self, "_tmp_input", None)
        if tmp is not None:
            try:
                os.unlink(tmp.name)
            except OSError:
                pass
            self._tmp_input = None

    def _done_threadsafe(self, success, message):
        """Handle run completion from the worker thread (marshalled via after())."""
        def _update():
            """Apply the completion UI changes on the Tk thread."""
            self.progress.stop()
            self.phase_var.set("")
            self.run_btn.config(state=tk.NORMAL)
            self.cancel_btn.config(state=tk.DISABLED)
            if success:
                self.status_var.set("Completed")
                self.log.append(f"\n{message}\n")
            elif message.startswith("Calculation cancelled"):
                self.status_var.set("Cancelled")
                self.log.append(f"\n{message}\n")
            else:
                self.status_var.set("Error")
                self.log.append(f"\n{message}\n")
                messagebox.showerror("Calculation Error", message)
            self.cleanup_temp_files()
        self.root.after(0, _update)

    def _export_input(self):
        """Write the current form as a deck .input file."""
        try:
            input_text = self._generate_input_text()
        except Exception as e:
            messagebox.showerror("Input Error", str(e))
            return

        path = filedialog.asksaveasfilename(
            filetypes=[("IRMA input", "*.input"),
                       ("LEAPR input", "*.leapr"),
                       ("All files", "*.*")],
            defaultextension=".input")
        if path:
            with open(path, 'w') as f:
                f.write(input_text)
            messagebox.showinfo("Saved", f"Input file saved to:\n{path}")

    def _import_leapr(self):
        """Import a IRMA/LEAPR input file and populate all GUI fields."""
        path = filedialog.askopenfilename(
            title="Import IRMA Input File",
            filetypes=[("IRMA input", "*.input"),
                       ("LEAPR input", "*.leapr"),
                       ("All files", "*.*")])
        if not path:
            return
        try:
            summary = self._import_leapr_from_path(path)
        except Exception as e:
            messagebox.showerror(
                "Import Error",
                f"Failed to import input file:\n\n{e}")
            return
        messagebox.showinfo("Import Complete", summary)

    def _import_leapr_from_path(self, path):
        """Populate all GUI fields from a deck file. Returns a summary
        string; raises on any unsupported or malformed deck."""
        from irma.core.engine import parse_leapr_input, TokenReader
        tokens, raw_lines, start_line, token_lines = parse_leapr_input(path)
        reader = TokenReader(tokens, token_lines=token_lines, filename=path)

        # Transactional import: parse and validate the ENTIRE deck into a
        # plain staging dict (Tk-free) before mutating any widget. A
        # malformed deck raises here, leaving the form untouched.
        staging = parse_deck_to_staging(reader, path)

        # Reset every import-derived field to defaults first, so cards absent
        # from this deck cannot retain a previous import's value (stale
        # noncubic / DOS / inelastic_mode / partial-spectra leak), then apply.
        self._reset_form_to_defaults()
        self._apply_imported_state(staging)

        summary = (f"Successfully imported input from:\n{path}\n\n"
                   f"nalpha={staging['nalpha']}, nbeta={staging['nbeta']}, "
                   f"ntempr={staging['ntempr']}, iel={staging['iel']}")
        if self._imported_partial_spectra:
            summary += (f"\nPreserved {len(self._imported_partial_spectra)} "
                        f"Card 6e partial spectrum block(s); they will be "
                        f"re-emitted on export.")
        return summary

    def _reset_form_to_defaults(self):
        """Restore every import-derived widget/var to its classic-deck
        default, so an import is a full replace and absent cards fall back
        deterministically (no leak across back-to-back imports)."""
        # mode first: under mode 1/2 _sync_modes pins iel to 10
        self.inelastic_mode_var.set(0)
        self.iel_var.set("0 — None")
        self._imported_partial_spectra = []
        self._imported_title = "IRMA calculation"
        self._imported_iprint = 0

        self.nphon.set("100")
        # Material identity clears to BLANK, exactly as a freshly built form
        # ships it: a reset that left MAT/ZA/AWR/spr at 0 would hand the deck
        # writer a nuclide that does not exist instead of an empty field the
        # generator names. (Fresh == reset is pinned by the GUI tests.)
        self.mat.set("")
        self.za.set("")
        self.isabt.set("0 — S(α,β) (standard)")
        self.ilog.set("0 — S values")
        self.iint.set("0 — log-lin (INT=4)")
        self.smin.set("1e-75")
        self.awr.set("")
        self.spr.set("")
        self.npr.set("1")
        # special modes, S(kappa), secondary scatterer
        self._clear_special_scatter_fields()

        # Generalized elastic / noncubic (iel=10)
        self.elastic_mode.set("1 — SEF (Single-channel Elastic Format)")
        self.coh_edge_group_enable_var.set(True)        # grouping on by default
        self.coh_edge_group_bpd.set("50")
        self.coh_edge_group_thr.set("1.0")
        # Crystalline extinction (iel=10) -> defaults / off
        self.ext_enable_var.set(False)
        self.ext_model.set("BC_mix")
        self._on_ext_model_change()     # restore the BC recipe visibility
        self.ext_l.set("8550")
        self.ext_g.set("170")
        self.ext_L.set("75750")
        self.ext_dist.set("Gauss")
        self.ext_recipe.set("std")
        self.ext_rmse_tol.set("1e-3")
        for var in (self.latt_a, self.latt_b, self.latt_c,
                    self.latt_alpha, self.latt_beta, self.latt_gamma):
            var.set("")             # blank, like a freshly built form
        self.atoms_text.delete("1.0", tk.END)
        self.nc_phonopy_yaml.set("")
        self.nc_mesh_nx.set("40")
        self.nc_mesh_ny.set("40")
        self.nc_mesh_nz.set("40")
        self.nc_ncpu.set("1")
        self.nc_use_born_var.set(0)
        self.nc_born_path.set("")
        self.nc_min_phonon_energy.set("")
        self.nc_num_directions.set("10000")
        self.nc_multiphonon_num_directions.set("1000")
        # Matches the fresh-form default (safe by construction). Irrelevant
        # for import faithfulness: any mode-1/2 deck carries Card 6g, whose
        # parsed auto_order (0 when the 3rd field is omitted) overwrites this.
        self.nc_auto_order_var.set(1)

        # Grid. LAT matches the builder default (Card 7 always carries lat on
        # import, so this only governs a reset form — keeping the two defaults
        # identical avoids a latent 0-vs-1 grid-units trap).
        self.lat.set("1 — alpha/beta in kT_thermal (0.0253 eV)")
        self.alpha_text.delete("1.0", tk.END)
        self.beta_text.delete("1.0", tk.END)

        # Phonon model (continuous DOS + oscillators)
        self.dos_source.set("manual")
        self.dos_delta.set("0.005")
        self.dos_rho_text.delete("1.0", tk.END)
        self.twt.set("0.0")
        self.c_diff.set("0.0")
        self.tbeta.set("1.0")
        self.osc_energies.delete("1.0", tk.END)
        self.osc_weights.delete("1.0", tk.END)

        # Comments
        self.comments_text.clear()

    def _apply_imported_state(self, st):
        """Apply a parsed staging dict (from parse_deck_to_staging) to the
        widgets. Called only after a successful parse and a reset to
        defaults, so only the cards present in the deck are set here."""
        self._imported_title = st['title']
        self._imported_iprint = st['iprint']
        self.nphon.set(str(st['nphon']))

        self.mat.set(str(st['mat']))
        za = st['za']
        self.za.set(str(int(za)) if za == int(za) else str(za))
        self._set_code(self.isabt, st['isabt'])
        self._set_code(self.ilog, st['ilog'])
        self._set_code(self.iint, st['iint'])
        self.smin.set(f"{st['smin']:g}")

        self.awr.set(str(st['awr']))
        self.spr.set(str(st['spr']))
        self.npr.set(str(st['npr']))

        iel = st['iel']
        self._set_code(self.iel_var, iel, _IEL_CHOICES)
        self._set_code(self.ncold, st['ncold'])
        self._set_code(self.nsk, st['nsk'])

        # Secondary scatterer
        self._set_code(self.nss, st['nss'])
        if st['nss'] > 0:
            self._set_code(self.b7, st['b7'])
            self.mss.set(str(st['mss']))
        self.aws.set(str(st['aws']))
        self.sps.set(str(st['sps']))

        # Generalized elastic cards (iel=10)
        self._imported_partial_spectra = st['partial_spectra']
        if iel == 10:
            self._set_code(self.elastic_mode, st['elastic_mode'])
            # Grouping toggle follows the deck; keep a sensible bins/decade in the
            # field when the deck has none so re-checking the box is meaningful.
            _has_group = st['edge_group_bpd'] > 0
            self.coh_edge_group_enable_var.set(_has_group)
            self.coh_edge_group_bpd.set(str(st['edge_group_bpd']) if _has_group else "50")
            self.coh_edge_group_thr.set(
                f"{st['edge_group_thr']:g}" if st['edge_group_thr'] > 0
                else "1.0")
            self.inelastic_mode_var.set(st['inelastic_mode'])
            ext = st['coherent_extinction']
            if ext is not None:
                self.ext_enable_var.set(True)
                self.ext_model.set(ext['model'])
                # populate the distribution dropdown with the model's family before
                # restoring the imported value (programmatic .set won't fire the
                # combobox event); the deck's dist is valid for that family.
                self._on_ext_model_change()
                self.ext_l.set(f"{ext['l']:g}")
                self.ext_g.set(f"{ext['g']:g}")
                self.ext_L.set(f"{ext['L']:g}")
                self.ext_dist.set(ext['dist'])
                self.ext_recipe.set(ext['recipe'])
                self.ext_rmse_tol.set(f"{ext['rmse_tol']:g}")

            lat6 = st['lattice']
            self.latt_a.set(str(lat6[0]))
            self.latt_b.set(str(lat6[1]))
            self.latt_c.set(str(lat6[2]))
            self.latt_alpha.set(str(lat6[3]))
            self.latt_beta.set(str(lat6[4]))
            self.latt_gamma.set(str(lat6[5]))

            atom_lines = []
            for at in st['atoms']:
                coord_strs = [f"{x} {y} {z}" for x, y, z in at['coords']]
                atom_lines.append(
                    f"{at['Z']}  {at['A']}  {at['awr']}  {at['b_coh']}  "
                    f"{at['sigma_inc']}  {at['npos']}  "
                    + "  ".join(coord_strs))
            if atom_lines:
                self.atoms_text.insert("1.0", "\n".join(atom_lines) + "\n")

            nc = st['noncubic']
            if nc is not None:
                self.nc_phonopy_yaml.set(nc['yaml'])
                self.nc_mesh_nx.set(str(nc['mesh_nx']))
                self.nc_mesh_ny.set(str(nc['mesh_ny']))
                self.nc_mesh_nz.set(str(nc['mesh_nz']))
                self.nc_ncpu.set(str(nc['ncpu']))
                self.nc_use_born_var.set(nc['use_born'])
                self.nc_born_path.set(nc['born_path'])
                cutoff = nc['min_phonon_energy_mev']
                self.nc_min_phonon_energy.set(
                    "" if cutoff == 0.0 else f"{cutoff:g}")
                self.nc_num_directions.set(str(nc['ndir']))
                self.nc_multiphonon_num_directions.set(str(nc['mpdir']))
                self.nc_auto_order_var.set(nc['auto_order'])

        # Grid (manual; populated from the deck)
        self._set_code(self.lat, st['lat'])
        self.grid_mode.set("manual")
        self._toggle_grid_mode()
        _set_values(self.alpha_text, st['alpha'])
        _set_values(self.beta_text, st['beta'])

        # Temperatures
        self.temps_var.set(
            " ".join(f"{t:.4f}" for t in st['temperatures']))

        # Phonon model (continuous DOS + oscillators)
        self.twt.set(st['twt'])
        self.c_diff.set(st['c_diff'])
        self.tbeta.set(st['tbeta'])
        if st['delta1'] is not None and st['rho'] is not None:
            self.dos_source.set("manual")
            self._toggle_dos_source()
            self.dos_delta.set(f"{st['delta1']:.6e}")
            _set_values(self.dos_rho_text, st['rho'])

        # S(kappa) table and coherent fraction (Cards 17-19)
        if st['ska'] is not None:
            self.ska_dka.set(f"{st['dka']:.6e}")
            _set_values(self.ska_text, st['ska'])
        if st['cfrac'] is not None:
            self.cfrac.set(f"{st['cfrac']:g}")

        # Oscillators
        if st['osc_e']:
            _set_values(self.osc_energies, st['osc_e'])
            _set_values(self.osc_weights, st['osc_w'])

        # Secondary phonon model (two-pass)
        if st['two_pass'] and st['sec_rho'] is not None:
            self.sec_dos_delta.set(f"{st['sec_delta']:.6e}")
            _set_values(self.sec_dos_rho_text, st['sec_rho'])
            self.sec_twt.set(st['sec_twt'])
            self.sec_c_diff.set(st['sec_c'])
            self.sec_tbeta.set(st['sec_tbeta'])
            if st['sec_osc_e']:
                _set_values(self.sec_osc_energies, st['sec_osc_e'])
                _set_values(self.sec_osc_weights, st['sec_osc_w'])

        # Comment cards (MF1/MT451) — stored verbatim (whitespace preserved)
        if st['comments']:
            self.comments_text.append("\n".join(st['comments']))
