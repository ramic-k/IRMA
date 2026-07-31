"""Reusable S(alpha,beta) grid input widgets for the IRMA GUI.

Two shared surfaces live here:

* :func:`build_auto_grid_entries` -- the ONE builder for the seven converged
  auto-grid knob fields (n_lower ... alpha_nlog), used by BOTH the ENDF
  Evaluation grid tab and the NCrystal panel. Field labels, help text, and
  defaults are defined once (defaults from
  :data:`irma.core.grids.AUTO_GRID_DEFAULTS`), so a default or wording change
  propagates to both panels automatically.
* :class:`SabGridForm` -- the NCrystal panel's grid-mode selector (automatic
  vs explicit alpha/beta), wrapping the shared knob fields.

The AUTOMATIC mode is the converged ENDF-style grid (irma.core.grids:
generate_beta_grid + generate_alpha_grid); freq_max is auto-estimated from the
phonopy mesh when left blank. This is NOT a uniform Q/E grid -- that
under-integrates the thermal cross section. EXPLICIT mode takes BOTH
alpha_grid and beta_grid as float lists (dimensionless, ENDF lat convention).

No config type is imported; ``export_fields()`` returns ONLY the active mode's
keys as a plain dict that maps straight onto NCrystalExportConfig's export keys.
"""

import tkinter as tk
from tkinter import ttk

from irma.core.grids import AUTO_GRID_DEFAULTS
from irma.gui.widgets import LabeledEntry, LabeledCombobox, parse_float, parse_int


# grid-mode dropdown labels <-> the mode token export_fields() switches on.
_AUTO_LABEL = "automatic (converged, from phonopy)"
_EXPLICIT_LABEL = "explicit (alpha/beta)"
_MODE_LABELS = [_AUTO_LABEL, _EXPLICIT_LABEL]
_MODE_BY_LABEL = {_AUTO_LABEL: "auto", _EXPLICIT_LABEL: "explicit"}


# ---------------------------------------------------------------------------
# The ONE field specification for the seven converged auto-grid knobs, shared
# by the ENDF Evaluation grid tab and the NCrystal panel: (attribute name,
# label, help title, help text). Defaults are injected from
# irma.core.grids.AUTO_GRID_DEFAULTS so the numbers can never drift between
# the two panels or the export config.
# ---------------------------------------------------------------------------
_D = AUTO_GRID_DEFAULTS
AUTO_GRID_FIELDS = (
    ("n_lower", "N lower (log):", "N Lower — Low-Beta Logarithmic Points",
     "Number of logarithmically spaced points in the low-energy transfer "
     "(small beta) region, below the phonon spectrum.\n\n"
     "This region captures the thermal quasi-elastic scattering at very "
     "small energy transfers. Logarithmic spacing is used because the "
     "physics varies rapidly near beta=0.\n\n"
     f"Default: {_D['n_lower']}. Increase for better resolution of the "
     "thermal peak."),
    ("n_phonon", "N phonon (linear):", "N Phonon — Linear Phonon Region Points",
     "Number of linearly spaced points covering the phonon spectrum energy "
     "range [0, freq_max].\n\n"
     "This is the main region where phonon features (peaks, van Hove "
     "singularities) appear. Linear spacing ensures all spectral features "
     "are resolved uniformly.\n\n"
     f"Default: {_D['n_phonon']}. Use more points for materials with sharp "
     "phonon features. For smooth spectra, fewer points may suffice."),
    ("n_upper", "N upper (log):", "N Upper — High-Beta Logarithmic Points",
     "Number of logarithmically spaced points in the high-energy tail above "
     "the phonon spectrum, extending up to Beta max.\n\n"
     "This region captures multiphonon contributions and the exponential "
     "tail. Logarithmic spacing is efficient since S(a,b) decays smoothly "
     "at high energy transfers.\n\n"
     f"Default {_D['n_upper']} is conservative: it keeps the high-beta step "
     "fine enough that a lin-lin (INT=2) law does not overshoot the "
     "free-atom cross section at high incident energy. Fewer points (~20) "
     "suffice for log-lin (INT=4) laws or thermal-only evaluations; "
     "demanding lin-lin cases may need more. Check grid convergence for "
     "your material and incident-energy range and adjust up or down."),
    ("beta_max", "Beta max [eV]:", "Beta Max — Maximum Energy Transfer",
     "Maximum energy transfer in eV for the upper tail of the beta grid.\n\n"
     "Should be large enough to capture all significant scattering, but not "
     "so large that unnecessary grid points are added.\n\n"
     f"Default: {_D['beta_max_eV']} eV. For most solid moderators this is "
     "sufficient. For hydrogen, 5 eV may be too small due to the large "
     "recoil; consider 10 eV or more."),
    ("alpha_dq", "Alpha dQ [1/A]:", "Alpha Grid — Linear Q Spacing",
     "Q spacing of the alpha grid's linear segment, in 1/Angstrom. Alpha "
     "points are placed at Q = dQ, 2*dQ, ... up to Q cut, with "
     "alpha = Q^2*(hbar^2/2m)/(A*kT).\n\n"
     "The thermal-energy inelastic cross section integrates S(alpha, beta) "
     "over upscatter windows at Q ~ 1-6 1/A; dQ sets how well those windows "
     "are resolved.\n\n"
     f"Default: {_D['alpha_dq_invA']}. On graphite this matches the thermal "
     "accuracy of a 2000-column reference grid at ~400 columns; a "
     "converged dQ = 0.01 grid sits only 3-5% above it."),
    ("alpha_qcut", "Alpha Q cut [1/A]:", "Alpha Grid — End of the Linear Segment",
     "Q where the alpha grid switches from linear spacing to a logarithmic "
     "tail, in 1/Angstrom.\n\n"
     "Set by neutron kinematics, not by the material: the linear segment "
     "must cover the thermal upscatter windows (Q up to ~6-10 1/A for "
     "incident energies through the thermal region).\n\n"
     f"Default: {_D['alpha_qcut_invA']}. Rarely needs changing."),
    ("alpha_nlog", "Alpha N log:", "Alpha Grid — High-Q Logarithmic Points",
     "Number of logarithmically spaced alpha points from Q cut out to the "
     "grid maximum. The maximum preserves the beta grid's kinematic reach "
     "(alpha_max = 4*beta_max/A, i.e. Q ~ 98 1/A for Beta max = 5 eV).\n\n"
     "This region feeds the epithermal cross section and the approach to "
     "the free-gas limit; S(a,b) is smooth there, so logarithmic spacing "
     "is efficient.\n\n"
     f"Default: {_D['alpha_nlog']}. Must be >= 2 (the log tail carries the "
     "grid from Q cut to the kinematic maximum)."),
)

# attribute name -> default (strings, ready for LabeledEntry)
AUTO_GRID_ENTRY_DEFAULTS = {
    "n_lower": str(_D["n_lower"]),
    "n_phonon": str(_D["n_phonon"]),
    "n_upper": str(_D["n_upper"]),
    "beta_max": str(_D["beta_max_eV"]),
    "alpha_dq": str(_D["alpha_dq_invA"]),
    "alpha_qcut": str(_D["alpha_qcut_invA"]),
    "alpha_nlog": str(_D["alpha_nlog"]),
}


def build_auto_grid_entries(parent, width=12):
    """Create the seven shared auto-grid knob fields inside ``parent``.

    The single builder both panels use: each field is a
    :class:`~irma.gui.widgets.LabeledEntry` (packed ``fill=X, pady=2``) with
    the shared label, help text, and default from :data:`AUTO_GRID_FIELDS` /
    :data:`AUTO_GRID_ENTRY_DEFAULTS`. Returns ``{name: LabeledEntry}`` in
    field order; callers assign the entries to their own attribute names.
    """
    entries = {}
    for name, label, help_title, help_text in AUTO_GRID_FIELDS:
        w = LabeledEntry(parent, label, default=AUTO_GRID_ENTRY_DEFAULTS[name],
                         width=width, help_title=help_title,
                         help_text=help_text)
        w.pack(fill=tk.X, pady=2)
        entries[name] = w
    return entries


# Per-field help for the NCrystal-panel-specific fields (mode selector,
# freq_max, explicit grids); the seven shared knobs' help lives in
# AUTO_GRID_FIELDS above.
HELP = {
    "grid_mode": (
        "How the S(alpha,beta) grid is set.\n\n"
        "  automatic (converged, from phonopy) (default): the SAME converged "
        "grid the ENDF evaluator builds -- generate_beta_grid (log/linear/log "
        "in beta) + generate_alpha_grid (linear in Q to the cut, then a log "
        "tail). This is the recommended choice; a uniform Q/E grid would "
        "under-integrate the thermal cross section.\n"
        "  explicit (alpha/beta): you supply the dimensionless alpha and beta "
        "grids yourself (ENDF lat convention). Both lists are required."),
    "freq_max_eV": (
        "Max phonon frequency [eV] used to size the beta grid's linear phonon "
        "region.\n\nLEAVE BLANK to auto-estimate it from the phonopy mesh "
        "(recommended). Set a value to pin it (e.g. 0.20). Used only in "
        "automatic mode."),
    "alpha_grid": (
        "Explicit alpha grid: dimensionless alpha values (ENDF lat convention) "
        "as a space- or comma-separated list, e.g. '0.1 0.5 1.0, 2.0'.\n\n"
        "Used only in explicit mode; required (together with beta_grid) there."),
    "beta_grid": (
        "Explicit beta grid: dimensionless beta values (ENDF lat convention) "
        "as a space- or comma-separated list, e.g. '0.0 0.5 1.0, 2.0'.\n\n"
        "Used only in explicit mode; required (together with alpha_grid) there."),
}


def _parse_float_list(label, text):
    """Parse a space/comma-separated float list, naming the field on failure.

    Splits ``text`` on commas and whitespace and ``float()``s each token, so the
    error dialog says WHICH grid field is bad (parse_float-style) instead of a
    bare conversion message. Returns ``[]`` for blank/whitespace-only input.
    """
    tokens = text.replace(",", " ").split()
    return [parse_float(label, tok) for tok in tokens]


class SabGridForm(ttk.Frame):
    """Grid-mode selector + the converged auto-grid knobs + explicit alpha/beta.

    All fields are always present (simplest, most robust headless); the inactive
    mode's fields are merely disabled, and ``export_fields()`` returns ONLY the
    active mode's keys. No config import -- pure widget values -> dict.
    """

    def __init__(self, parent, padding=0):
        super().__init__(parent, padding=padding)

        self.grid_mode = LabeledCombobox(
            self, "grid mode:", _MODE_LABELS, default=_AUTO_LABEL,
            help_text=HELP["grid_mode"])
        self.grid_mode.pack(fill=tk.X, pady=2)
        self.grid_mode.combo.bind("<<ComboboxSelected>>",
                                  lambda _e: self._sync_enabled())

        # AUTOMATIC: freq_max (blank -> auto-estimate from phonopy) + the seven
        # shared converged-grid knobs, built by the ONE builder both panels use
        # (labels/help/defaults from AUTO_GRID_FIELDS; defaults trace back to
        # irma.core.grids.AUTO_GRID_DEFAULTS).
        self.freq_max = LabeledEntry(self, "freq_max (eV, blank=auto):",
                                     default="", width=12,
                                     help_text=HELP["freq_max_eV"])
        self.freq_max.pack(fill=tk.X, pady=2)
        _knobs = build_auto_grid_entries(self)
        self.n_lower = _knobs["n_lower"]
        self.n_phonon = _knobs["n_phonon"]
        self.n_upper = _knobs["n_upper"]
        self.beta_max = _knobs["beta_max"]
        self.alpha_dq = _knobs["alpha_dq"]
        self.alpha_qcut = _knobs["alpha_qcut"]
        self.alpha_nlog = _knobs["alpha_nlog"]

        # EXPLICIT: dimensionless alpha/beta float lists (blank by default).
        self.alpha_grid = LabeledEntry(self, "alpha_grid:", default="",
                                       width=28, help_text=HELP["alpha_grid"])
        self.alpha_grid.pack(fill=tk.X, pady=2)
        self.beta_grid = LabeledEntry(self, "beta_grid:", default="",
                                      width=28, help_text=HELP["beta_grid"])
        self.beta_grid.pack(fill=tk.X, pady=2)

        # ordered for the disable sweep
        self._auto_entries = (self.freq_max, self.n_lower, self.n_phonon,
                              self.n_upper, self.beta_max, self.alpha_dq,
                              self.alpha_qcut, self.alpha_nlog)
        self._explicit_entries = (self.alpha_grid, self.beta_grid)
        self._sync_enabled()

    # ------------------------------------------------------------------ mode --
    def mode(self):
        """Active grid mode token: ``"auto"`` or ``"explicit"``."""
        return _MODE_BY_LABEL[self.grid_mode.get()]

    def _sync_enabled(self):
        """Grey out the inactive mode's entry fields (cosmetic; export_fields
        is the source of truth for which keys are returned)."""
        explicit = self.mode() == "explicit"
        for w in self._auto_entries:
            w.entry.config(state=tk.DISABLED if explicit else tk.NORMAL)
        for w in self._explicit_entries:
            w.entry.config(state=tk.NORMAL if explicit else tk.DISABLED)

    # ---------------------------------------------------------------- export --
    def export_fields(self):
        """Return ONLY the active mode's grid keys, ready to merge into the
        export dict.

        AUTOMATIC -> {n_lower, n_phonon, n_upper, beta_max_eV, alpha_dq_invA,
                      alpha_qcut_invA, alpha_nlog} (+ freq_max_eV ONLY if set;
                      blank means auto-estimate from the phonopy mesh).
        EXPLICIT  -> {alpha_grid, beta_grid} (lists of floats).

        Raises a clear :class:`ValueError` (parse-style, naming the field) on a
        bad number, and in explicit mode requires BOTH lists non-empty.
        """
        if self.mode() == "explicit":
            alpha = _parse_float_list("alpha_grid", self.alpha_grid.get())
            beta = _parse_float_list("beta_grid", self.beta_grid.get())
            if not alpha or not beta:
                raise ValueError(
                    "explicit grid: provide BOTH alpha_grid and beta_grid "
                    "(non-empty float lists), or switch to automatic mode")
            return {"alpha_grid": alpha, "beta_grid": beta}
        fields = {
            "n_lower": parse_int("n_lower", self.n_lower.get()),
            "n_phonon": parse_int("n_phonon", self.n_phonon.get()),
            "n_upper": parse_int("n_upper", self.n_upper.get()),
            "beta_max_eV": parse_float("beta_max (eV)", self.beta_max.get()),
            "alpha_dq_invA": parse_float("alpha dQ (1/A)", self.alpha_dq.get()),
            "alpha_qcut_invA": parse_float("alpha Q cut (1/A)",
                                           self.alpha_qcut.get()),
            "alpha_nlog": parse_int("alpha N log", self.alpha_nlog.get()),
        }
        fm = self.freq_max.get().strip()
        if fm:  # blank -> auto-estimate from phonopy (omit the key)
            fields["freq_max_eV"] = parse_float("freq_max (eV)", fm)
        return fields
