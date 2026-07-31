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
keys as a plain dict that maps straight onto NCrystalExportConfig's export keys,
and ``load_fields()`` reads that same dict back (the panel builds it off a
loaded config), so the two directions share one key list.
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
    ("n_lower", "Beta N lower (log):", "N Lower (Low-Beta Logarithmic Points)",
     "Number of logarithmically spaced points in the low-energy-transfer "
     "(small beta) region, below the phonon spectrum. Beta is the "
     "dimensionless energy transfer, E / kT.\n\n"
     "This region captures the thermal quasi-elastic scattering at very "
     "small energy transfers. Logarithmic spacing is used because the "
     "physics varies rapidly near beta=0.\n\n"
     f"Default: {_D['n_lower']}. Increase for better resolution of the "
     "thermal peak."),
    ("n_phonon", "Beta N phonon (linear):", "N Phonon (Linear Phonon Region Points)",
     "Number of linearly spaced points covering the phonon-spectrum energy "
     "range [0, freq_max]; phonons are the lattice vibrations of the "
     "material.\n\n"
     "This is the main region where phonon features appear (peaks, and the "
     "sharp steps and spikes called van Hove singularities). Linear spacing "
     "resolves all spectral features uniformly.\n\n"
     f"Default: {_D['n_phonon']}. Use more points for materials with sharp "
     "phonon features. For smooth spectra, fewer points may suffice."),
    ("n_upper", "Beta N upper (log):", "N Upper (High-Beta Logarithmic Points)",
     "Number of logarithmically spaced points in the high-energy tail above "
     "the phonon spectrum, extending up to Beta max.\n\n"
     "This region captures multiphonon contributions and the exponential "
     "tail. Logarithmic spacing is efficient since the S(alpha,beta) table "
     "decays smoothly at high energy transfers.\n\n"
     f"Default {_D['n_upper']} is conservative: it keeps the high-beta step "
     "fine enough that lin-lin (INT=2) interpolation does not overshoot the "
     "free-atom cross section at high incident energy. Fewer points (about "
     "20) suffice for log-lin (INT=4) tapes or thermal-only evaluations; "
     "demanding lin-lin cases may need more. Check grid convergence for "
     "your material and incident-energy range and adjust up or down."),
    ("beta_max", "Beta max [eV]:", "Beta Max (Maximum Energy Transfer)",
     "Maximum energy transfer in eV for the upper tail of the beta "
     "grid.\n\n"
     "Should be large enough to capture all significant scattering, but not "
     "so large that unnecessary grid points are added.\n\n"
     f"Default: {_D['beta_max_eV']} eV. For most solid moderators this is "
     "sufficient. For hydrogen, 5 eV may be too small because of the large "
     "recoil; consider 10 eV or more."),
    ("alpha_dq", "Alpha dQ [1/A]:", "Alpha Grid (Linear Q Spacing)",
     "Q spacing of the alpha grid's linear segment, in 1/Angstrom. Q is the "
     "momentum transfer, and alpha is its dimensionless form. Alpha points "
     "are placed at "
     "Q = dQ, 2*dQ, ... up to Q cut, with "
     "alpha = Q^2*(hbar^2/2m)/(A*kT).\n\n"
     "The thermal-energy inelastic cross section integrates S(alpha,beta) "
     "over upscatter (energy-gain) windows at Q of roughly 1-6 1/A; dQ sets "
     "how well those windows are resolved.\n\n"
     f"Default: {_D['alpha_dq_invA']}. On graphite the default matches a "
     "dQ = 0.01 reference to -0.33% on the thermal integral (largest local "
     "difference 4.3%), at a fifth of the columns."),
    ("alpha_qcut", "Alpha Q cut [1/A]:", "Alpha Grid (End of the Linear Segment)",
     "Q where the alpha grid switches from linear spacing to a logarithmic "
     "tail, in 1/Angstrom (Q is the momentum transfer).\n\n"
     "Set by neutron kinematics, not by the material: the linear segment "
     "must cover the thermal upscatter (energy-gain) windows, Q up to about "
     "6-10 1/A for incident energies through the thermal region.\n\n"
     f"Default: {_D['alpha_qcut_invA']}. Rarely needs changing."),
    ("alpha_nlog", "Alpha N log:", "Alpha Grid (High-Q Logarithmic Points)",
     "Number of logarithmically spaced alpha points from Q cut out to the "
     "grid maximum; Q is the momentum transfer and alpha is its "
     "dimensionless form. The maximum preserves the beta grid's kinematic "
     "reach (alpha_max = 4*beta_max/A, i.e. Q of about 98 1/A for "
     "Beta max = 5 eV).\n\n"
     "This region feeds the epithermal cross section and the approach to "
     "the free-gas limit; the S(alpha,beta) table is smooth there, so "
     "logarithmic spacing is efficient.\n\n"
     f"Default: {_D['alpha_nlog']}. Must be >= 2 (the log tail carries the "
     "grid from Q cut to the kinematic maximum)."),
)

# The ONE mapping between the seven knob WIDGETS and the export/config keys
# they carry: (widget attribute, export key, parser, parse-error label).
# export_fields() and load_fields() both walk this tuple, so a key can never
# be written in one direction and missed in the other.
_AUTO_EXPORT_FIELDS = (
    ("n_lower", "n_lower", parse_int, "n_lower"),
    ("n_phonon", "n_phonon", parse_int, "n_phonon"),
    ("n_upper", "n_upper", parse_int, "n_upper"),
    ("beta_max", "beta_max_eV", parse_float, "beta_max (eV)"),
    ("alpha_dq", "alpha_dq_invA", parse_float, "alpha dQ (1/A)"),
    ("alpha_qcut", "alpha_qcut_invA", parse_float, "alpha Q cut (1/A)"),
    ("alpha_nlog", "alpha_nlog", parse_int, "alpha N log"),
)

#: Every export key this form can produce, in BOTH modes. A panel loading a
#: config reads these attributes off it to build ``load_fields``' argument, so
#: the form stays the single owner of its own key set.
GRID_EXPORT_KEYS = (tuple(key for _a, key, _p, _l in _AUTO_EXPORT_FIELDS)
                    + ("freq_max_eV", "alpha_grid", "beta_grid"))

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
        "How the grid for the S(alpha,beta) table (scattering probability "
        "versus momentum exchange, alpha, and energy exchange, beta) is "
        "set.\n\n"
        "  automatic (converged, from phonopy) (default): the SAME converged "
        "grid the ENDF evaluator builds: generate_beta_grid (log/linear/log "
        "in beta) plus generate_alpha_grid (linear in Q to the cut, then a log "
        "tail). This is the recommended choice; a uniform Q/E grid would "
        "under-integrate the thermal cross section.\n"
        "  explicit (alpha/beta): you supply the dimensionless alpha and beta "
        "grids yourself (ENDF lat convention). Both lists are required."),
    "freq_max_eV": (
        "Maximum phonon frequency [eV], used to size the linear phonon region "
        "of the beta grid (phonons are the lattice vibrations of the "
        "material).\n\nLEAVE BLANK to auto-estimate it from the phonopy mesh "
        "(recommended). Set a value to pin it (e.g. 0.20). Used only in "
        "automatic mode."),
    "alpha_grid": (
        "Explicit alpha grid: dimensionless alpha (momentum transfer) values "
        "in the ENDF lat convention, as a space- or comma-separated list, "
        "e.g. '0.1 0.5 1.0, 2.0'.\n\n"
        "Used only in explicit mode; required there, together with beta_grid."),
    "beta_grid": (
        "Explicit beta grid: dimensionless beta (energy transfer) values in "
        "the ENDF lat convention, as a space- or comma-separated list, "
        "e.g. '0.0 0.5 1.0, 2.0'.\n\n"
        "Used only in explicit mode; required there, together with "
        "alpha_grid."),
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

    Progressive disclosure: only the active mode's fields are shown
    (automatic -> the converged-grid knobs, explicit -> the two grid
    lists), and ``export_fields()`` returns ONLY the active mode's
    keys. ``load_fields()`` is its inverse and picks the mode from the
    data. No config import -- pure widget values <-> dict.
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
        self._auto_frame = ttk.Frame(self)
        self._auto_frame.pack(fill=tk.X)
        self.freq_max = LabeledEntry(self._auto_frame,
                                     "freq_max (eV, blank=auto):",
                                     default="", width=12,
                                     help_text=HELP["freq_max_eV"])
        self.freq_max.pack(fill=tk.X, pady=2)
        _knobs = build_auto_grid_entries(self._auto_frame)
        self.n_lower = _knobs["n_lower"]
        self.n_phonon = _knobs["n_phonon"]
        self.n_upper = _knobs["n_upper"]
        self.beta_max = _knobs["beta_max"]
        self.alpha_dq = _knobs["alpha_dq"]
        self.alpha_qcut = _knobs["alpha_qcut"]
        self.alpha_nlog = _knobs["alpha_nlog"]

        # EXPLICIT: dimensionless alpha/beta float lists (blank by default).
        self._explicit_frame = ttk.Frame(self)
        self.alpha_grid = LabeledEntry(self._explicit_frame, "alpha_grid:",
                                       default="",
                                       width=28, help_text=HELP["alpha_grid"])
        self.alpha_grid.pack(fill=tk.X, pady=2)
        self.beta_grid = LabeledEntry(self._explicit_frame, "beta_grid:",
                                      default="",
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
        """Show only the active mode's fields (progressive disclosure;
        export_fields stays the source of truth for the returned keys)."""
        if self.mode() == "explicit":
            self._auto_frame.pack_forget()
            self._explicit_frame.pack(fill=tk.X)
        else:
            self._explicit_frame.pack_forget()
            self._auto_frame.pack(fill=tk.X)

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
        fields = {key: parser(label, getattr(self, attr).get())
                  for attr, key, parser, label in _AUTO_EXPORT_FIELDS}
        fm = self.freq_max.get().strip()
        if fm:  # blank -> auto-estimate from phonopy (omit the key)
            fields["freq_max_eV"] = parse_float("freq_max (eV)", fm)
        return fields

    # ---------------------------------------------------------------- load ---
    def reset(self):
        """Put BOTH mode groups back to the values a fresh form ships and
        select automatic mode."""
        self.grid_mode.set(_AUTO_LABEL)
        self.freq_max.set("")
        for attr, _key, _parser, _label in _AUTO_EXPORT_FIELDS:
            getattr(self, attr).set(AUTO_GRID_ENTRY_DEFAULTS[attr])
        self.alpha_grid.set("")
        self.beta_grid.set("")
        self._sync_enabled()

    def load_fields(self, fields):
        """Inverse of :meth:`export_fields`: populate the form from a mapping
        of export keys (the panel reads them off a loaded config).

        The MODE comes from the data, exactly as the config decides it:
        ``alpha_grid`` AND ``beta_grid`` both present -> explicit, anything
        else -> automatic. Both groups are reset first, so the inactive mode's
        widgets hold their shipped defaults rather than a half-filled mixture
        of this config and whatever was on screen before; a value of None (the
        config's "omitted") leaves the default in place too. Values are set,
        not parsed -- a mapping built off a validated config carries numbers,
        so this cannot fail partway and leave the form inconsistent.
        """
        self.reset()
        alpha, beta = fields.get("alpha_grid"), fields.get("beta_grid")
        if alpha is not None and beta is not None:
            self.grid_mode.set(_EXPLICIT_LABEL)
            self.alpha_grid.set(" ".join(str(v) for v in alpha))
            self.beta_grid.set(" ".join(str(v) for v in beta))
        else:
            for attr, key, _parser, _label in _AUTO_EXPORT_FIELDS:
                value = fields.get(key)
                if value is not None:
                    getattr(self, attr).set(value)
            freq_max = fields.get("freq_max_eV")
            # blank = auto-estimate from the phonopy mesh, which is what an
            # omitted freq_max_eV means in the config
            self.freq_max.set("" if freq_max is None else freq_max)
        self._sync_enabled()
