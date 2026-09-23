"""Neutron-scattering panel for the IRMA GUI.

A self-contained controller for IRMA's second capability: it binds field-for-
field to a :class:`~irma.spectra.config.SpectraConfig`, so Save/Open are exactly
``config.dump`` / ``config.load`` and the GUI run is the same ``run_spectra`` the
CLI drives (via the shared :class:`~irma.gui.runner.ComputationRunner`, routed
to this panel's log). Three geometry sub-tabs (VISION / generic indirect /
generic direct) share one material + physics + grid form.

``build_config()`` / ``load_config()`` are pure widget<->config mappings (no
event loop needed), so they are unit-testable against a withdrawn Tk root.
"""

import argparse
import dataclasses
import os
import sys
import tempfile
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

from irma.gui.widgets import (
    LabeledEntry, LabeledCombobox, FileSelector, ScrolledText, InfoLabel,
    RunPanel, build_phonopy_fields, check_with_help, form_section,
    init_form_styles, load_phonopy_fields, phonopy_material,
    scrolled_columns, parse_float, parse_int)
from irma.core.noncubic_inelastic import MIN_PHONON_ENERGY_HELP
from irma.gui.element_table import ElementTable
from irma.spectra.config import (
    GridConfig, InstrumentConfig, PhysicsConfig, SpectraConfig,
    SpectraConfigError, dump, load, phonopy_species)
from irma.spectra.cli import parse_angles, parse_coeffs
from irma.spectra.chopper_resolution import (
    available_instruments, available_packages, default_frequency, default_coverage)

# Config fields the panel has no control for. A loaded config's non-default
# values are carried through build_config instead of being reset to the
# defaults on the next Run or Save (the NCrystal panel keeps its export
# settings the same way).
_NO_WIDGET = {"physics": (PhysicsConfig, ("elastic_from_tape",)),
              "grid": (GridConfig, ("q_pad_invA",)),
              "instrument": (InstrumentConfig, ("bank_halfwidth_deg", "combine"))}


def _unrepresented_fields(cfg):
    """{(section, field): value} for the loaded fields in _NO_WIDGET whose
    value differs from the config default."""
    carried = {}
    for section, (cls, names) in _NO_WIDGET.items():
        defaults = {f.name: f.default for f in dataclasses.fields(cls)}
        for name in names:
            value = getattr(getattr(cfg, section), name)
            if value != defaults[name]:
                carried[(section, name)] = value
    return carried


_GEOMETRIES = ("indirect", "direct")

# Direct-tab resolution-model dropdown labels <-> config resolution_model values
_RES_MODEL_LABELS = {
    "width polynomial": "poly",
    "chopper (auto, real instrument)": "chopper",
}
_RES_MODEL_REV = {v: k for k, v in _RES_MODEL_LABELS.items()}

# Physics-tab gain-side dropdown labels <-> config gain_side values
_GAIN_SIDE_LABELS = {
    "direct (explicit Bose factors)": "direct",
    "detailed balance (mirror)": "detailed_balance",
}
_GAIN_SIDE_REV = {v: k for k, v in _GAIN_SIDE_LABELS.items()}

# Direct-tab Output selector + fixed-cuts sub-mode dropdown labels <-> config values
_OUTPUT_LABELS = {"fixed cuts": "cuts", "2-D map": "map"}
_OUTPUT_REV = {v: k for k, v in _OUTPUT_LABELS.items()}
_CUT_BY_LABELS = {"detector angles": "angles", "constant-Q": "q"}
_CUT_BY_REV = {v: k for k, v in _CUT_BY_LABELS.items()}

# Direct-tab detector-angle pre-fill; also the fallback when the (hidden)
# angles field is blank in constant-Q mode, where validate still requires
# angles_deg but the constant-Q path never reads them.
_DIR_ANGLES_DEFAULT = "30,60,90,120"


class _CoeffFields:
    """The three resolution-width-polynomial coefficient fields.

    The energy resolution width is a quadratic in energy transfer ``E``::

        sigma(E) = c0 + c1*|E| + c2*E**2      [meV]

    (evaluated at |E|, so the energy-gain side mirrors the loss-side fit),
    interpreted as the Gaussian sigma (shape=gaussian) or the Lorentzian HWHM
    (shape=lorentzian). Each coefficient gets its own labelled entry + '?'
    button so the user can set and understand them independently instead of
    typing one opaque comma-separated string.

    Config mapping (``instrument.sigma_coeffs``):

    * all three blank  -> ``None`` (use the VISION preset polynomial);
    * any non-blank    -> blanks read as ``0.0``; trailing blanks are dropped so
      a short coefficient list (e.g. ``[c0, c1]``) round-trips to itself exactly.
    """

    def __init__(self, parent):
        self.c0 = LabeledEntry(parent, "c0  constant (meV):", default="",
                               width=14, help_text=HELP["sigma_c0"])
        self.c1 = LabeledEntry(parent, "c1 × E (dimensionless):", default="",
                               width=14, help_text=HELP["sigma_c1"])
        self.c2 = LabeledEntry(parent, "c2  x E^2 (1/meV):", default="",
                               width=14, help_text=HELP["sigma_c2"])
        for w in self._entries:
            w.pack(fill=tk.X, pady=1)

    @property
    def _entries(self):
        """The three resolution-coefficient entry widgets."""
        return (self.c0, self.c1, self.c2)

    def get_coeffs(self):
        """Return the coefficient list, ``None`` if every field is blank."""
        strs = [w.get().strip() for w in self._entries]
        present = [i for i, s in enumerate(strs) if s]
        if not present:
            return None
        return [float(s) if s else 0.0 for s in strs[:present[-1] + 1]]

    def set_coeffs(self, seq):
        """Populate the fields from a coefficient list (or ``None`` -> blank)."""
        seq = list(seq) if seq is not None else []
        for i, w in enumerate(self._entries):
            w.set(str(seq[i]) if i < len(seq) else "")


# ---------------------------------------------------------------------------
# Per-field help text (the '?' buttons). Each explains what the field is, its
# units, how to choose it, and what the pre-filled default means.
# ---------------------------------------------------------------------------
HELP = {
    "phonopy_yaml": (
        "The phonopy.yaml from your phonon calculation (phonons are the "
        "lattice vibrations that produce inelastic scattering). Use the FULL "
        "file that carries the supercell and force-constant context, the one "
        "phonopy writes with FORCE_CONSTANTS/FORCE_SETS. Do NOT use a "
        "primitive-cell mesh.yaml dump: it fails with a 'FORCE_CONSTANTS "
        "inconsistent / p2s_map' error.\n\nThis file defines the lattice, "
        "atom positions and force constants from which S(Q,E) and the "
        "Debye-Waller factors are computed.\n\nNo default. This field is "
        "required."),
    "born": (
        "Optional BORN file (Born effective charges plus the dielectric "
        "tensor) for the non-analytical LO-TO splitting at the zone "
        "centre.\n\nDefault: blank = off. Leave blank for "
        "non-polar materials (graphite, metals); supply it for polar/ionic "
        "crystals where the LO-TO splitting matters."),
    "force_constants": (
        "Optional explicit FORCE_CONSTANTS file.\n\nDefault: blank = read the "
        "force constants embedded in phonopy.yaml, or auto-discover a sibling "
        "FORCE_CONSTANTS/FORCE_SETS next to it. Only set this if your "
        "constants live in a separate file the yaml does not point to."),
    "force_sets": (
        "Optional explicit FORCE_SETS file (the displacement/force dataset, the "
        "alternative to a FORCE_CONSTANTS file).\n\nDefault: blank = use the "
        "force constants from the yaml or a sibling file. Set this if your phonon "
        "data is a FORCE_SETS the yaml does not point to."),
    "mesh": (
        "Phonon q-point mesh 'nx ny nz' sampling the Brillouin zone.\n\nA "
        "finer mesh gives a smoother S(Q,E) and smoother Debye-Waller "
        "factors, at more cost "
        "(cost ~ nx*ny*nz). 20x20x20 is fine for a survey; 40x40x40 (the "
        "default) for production. Use the same density on each axis unless "
        "the cell is very anisotropic."),
    "temperature": (
        "Sample temperature in Kelvin.\n\nSets the Bose mode populations (how "
        "strongly each lattice vibration is thermally excited, which controls "
        "up-scattering and the multiphonon background) and the Debye-Waller "
        "attenuation (seen as a fall-off of intensity at high momentum "
        "transfer Q). Default 296 K "
        "(room temperature). Use the actual experiment temperature, for "
        "example 5 K for a cryogenic VISION run."),
    "inelastic_mode": (
        "Fidelity of the inelastic (phonon) calculation, for the Phonopy-model "
        "input.\n\n"
        "  1 = incoherent approximation (default): each atom is treated as "
        "scattering independently. Fast, phonon-DOS-like; the standard choice "
        "for most materials and surveys.\n"
        "  2 = coherent one-phonon + incoherent-approximation multiphonon: the "
        "exact one-phonon dispersion with an incoherent-approximation "
        "multiphonon tail. Needed for strong coherent scatterers (graphite, "
        "Be, ...) where "
        "the dispersion structure matters; several times slower.\n"
        "  0 = DOS + isotropic Debye-Waller: the lightweight incoherent-"
        "approximation phonon expansion, with the partial phonon density of "
        "states (DOS) derived from this phonopy calc (no eigenvector engine). "
        "Cheapest; per-atom normalized like modes 1/2. (To feed your OWN DOS "
        "files instead, switch 'phonon input' to 'DOS files'.)"),
    "lattice": (
        "Unit cell for the inelastic-mode-0 COHERENT-elastic Bragg peaks (the "
        "iel=10 analogue): "
        "a,b,c,alpha,beta,gamma (Angstrom, degrees), comma- or space-"
        "separated. Used only by mode 0 (modes 1/2 read the cell from "
        "phonopy.yaml). Pair it with each element's b_coh_fm plus the "
        "per-element fractional 'positions' column."),
    "input_source": (
        "Where the phonon (lattice-vibration) information comes from.\n\n"
        "  Phonopy model (default): a finished phonopy calculation "
        "(phonopy.yaml + mesh). The 'inelastic mode' selects the fidelity: "
        "1 incoherent / 2 coherent eigenvector engines, or 0 (a fast DOS + "
        "isotropic Debye-Waller derived from the same phonopy calc).\n"
        "  DOS files (mode 0): you provide a 2-column phonon density-of-states "
        "(DOS) file per element; no phonopy needed. This is the lightweight "
        "DOS path for H-rich / incoherent / disordered materials, or when you "
        "only have a DOS."),
    "autofill": (
        "Read the distinct element symbols from the loaded phonopy.yaml, add a "
        "table row for each, and prefill the blank scattering columns from "
        "IRMA's built-in nuclear table. Rows whose symbol matches the yaml are "
        "carried over unchanged (your typed values win); rows for symbols not "
        "in the yaml are dropped. Energy-dependent nuclides (B, Cd, Gd, ...) "
        "stay blank; enter their constants by hand."),
    "elements": (
        "One row per DISTINCT scattering element. Build the list with "
        "'+ Add element' (each row is an element; the per-row DOS Browse picks "
        "that element's DOS file in DOS-files mode).\n\n"
        "Columns:\n"
        "  Sym            element symbol; must match the atoms in your model.\n"
        "  sigma_bound_b  bound cross section [barn] (~ coh + inc).  REQUIRED.\n"
        "  AWR            atomic weight ratio A = M/m_n.             REQUIRED.\n"
        "  b_coh_fm       coherent scattering length [fm] (can be < 0); drives\n"
        "                 the coherent-elastic Bragg peaks. Optional.\n"
        "  sigma_inc_b    incoherent bound cross section [barn]; drives the\n"
        "                 incoherent elastic Debye-Waller line. Optional.\n"
        "  mult           (DOS-files mode 0) atoms of this element in the cell.\n"
        "  unit           (DOS-files mode 0) frequency unit of the DOS file.\n"
        "  DOS file       (DOS-files mode 0) the 2-column phonon density of\n"
        "                 states (DOS) for this element.\n"
        "  positions      (mode-0 coherent elastic) fractional sites, flat\n"
        "                 'x1 y1 z1 x2 y2 z2 ...' (same format as the ENDF deck);\n"
        "                 the number of triplets must equal 'mult'.\n\n"
        "The FIRST row is the principal scatterer (sets the overall normalization "
        "for the engine modes 1/2)."),
    "dos_format": (
        "DOS file (phonon density of states): 2 columns, frequency and DOS "
        "intensity. Whitespace OR "
        "comma separated. Lines starting with '#' (and non-numeric header "
        "lines) are skipped. Intensity units are arbitrary (renormalized "
        "internally). The frequency unit of column 1 is chosen per element "
        "(the 'unit' column)."),
    "max_phonon_order": (
        "Highest multiphonon order summed in the smooth background under the "
        "spectrum.\n\nDefault 'auto': "
        "in modes 1/2 the order is convergence-sized to the high-energy / "
        "high-Q recoil tail; in mode 0 it is convergence-sized from the DOS "
        "Debye-Waller lambda (Poisson(f0*alpha) at the largest Q). Or "
        "enter an integer to set it exactly (lower = faster, but truncates "
        "the high-E wing). Keep 'auto' unless you are deliberately limiting "
        "cost."),
    "min_phonon_energy": MIN_PHONON_ENERGY_HELP,
    "n_directions": (
        "Number of powder-average sampling directions for the ONE-phonon term "
        "(the spectrum is averaged over crystal orientations).\n\nMore "
        "directions give a smoother spectrum and less "
        "orientational sampling noise, at more cost (roughly linear). Default "
        "10000 is well converged for production runs (the value the validation "
        "campaign used, and the NCrystal exporter default); drop to ~1000 for "
        "a quick look."),
    "mp_directions": (
        "Number of powder-average directions for the MULTIPHONON background. "
        "Each direction is cheaper here than in the one-phonon term.\n\n"
        "Default "
        "1000 (converged by ~50-100). The multiphonon part is smooth, so it "
        "needs far fewer directions than the one-phonon term."),
    "jobs": (
        "Number of worker processes for the parallel direction/shell sums.\n\n"
        "Default: blank = use ALL CPU cores. Enter a number to limit it (e.g. to "
        "leave cores free for other work)."),
    "elastic": (
        "Whether to add an elastic line at zero energy transfer (scattering in "
        "which the neutron loses no energy; the peak at E = 0).\n\n"
        "  on (default): include the elastic line (see 'elastic kind').\n"
        "  off: inelastic spectrum only (no E=0 peak)."),
    "elastic_kind": (
        "Which elastic scattering components to include (when elastic line = "
        "on).\n\n"
        "  both (default): the coherent Bragg peaks plus the incoherent "
        "Debye-Waller line. This is the "
        "physically complete elastic for most crystals; both components are "
        "present in real materials.\n"
        "  coherent: Bragg peaks only (needs b_coh_fm in the scatterers).\n"
        "  incoherent: Debye-Waller line only (needs sigma_inc_b), for example "
        "for amorphous or purely-incoherent scatterers with no Bragg peaks."),
    "incoherent_elastic_mode": (
        "Debye-Waller treatment of the INCOHERENT elastic line.\n\n"
        "  isotropic (default): scalar exponent from the trace/3 of each "
        "species' displacement tensor (the ENDF-convention form).\n"
        "  directional: powder average of the full anisotropic exponential "
        "<exp(-Q^2 u.U.u)> per atom. Larger at high Q for anisotropic "
        "crystals (e.g. graphite), and consistent with the directional "
        "multiphonon. Needs inelastic mode 1 or 2 (displacement tensors).\n\n"
        "The coherent Bragg peaks are unaffected (already directional per "
        "reflection). The ENDF tape path cannot represent this option."),
    "include_gain": (
        "Add the energy-GAIN (anti-Stokes, E < 0) side, where the neutron "
        "takes energy from the material instead of depositing it.\n\n"
        "Default: on, giving a physical double-differential spectrum with both "
        "energy-loss and energy-gain. Turn off for energy-loss only.\n\n"
        "How the gain side is computed is set by 'gain-side method' below."),
    "gain_side": (
        "How the energy-GAIN (E < 0) side, where the neutron takes energy from "
        "the material, is computed when 'include energy-gain' is on.\n\n"
        "'direct (explicit Bose factors)': evaluate the gain side from first "
        "principles with explicit Bose phonon-annihilation factors. This is "
        "the more accurate default; gain and loss are computed "
        "independently.\n\n"
        "'detailed balance (mirror)': obtain the gain side by mirroring the "
        "energy-loss side, S(-E) = exp(-E/kT) S(E) (the classic LEAPR-style "
        "approximation; equals the direct result only as dE/kT -> 0)."),
    "kinematic": (
        "Multiply by the kinematic factor kf/ki so the result is the DOUBLE-"
        "DIFFERENTIAL scattering cross section d2sigma/dOmega/dE', what a "
        "detector actually records (a count rate), rather than the bare "
        "S(Q,omega).\n\n"
        "The measured double-differential cross section is\n"
        "  d2sigma/dOmega/dE' = (kf/ki) * (sigma_b/4pi) * S(Q,omega),\n"
        "and IRMA stores the (sigma_b/4pi) S(Q,omega) part WITHOUT the kf/ki "
        "flux factor. Turning this on applies that kf/ki (kf = final, ki = "
        "incident wavevector), giving the full d2sigma/dOmega/dE'.\n\n"
        "Default: off, matching the OCLIMAX S(Q,omega) convention. Turn on to "
        "compare against raw detector counts / an experimental d2sigma/dOmega/dE'."),
    "e_min": (
        "Lowest energy transfer in the output spectrum [meV] (negative "
        "values mean the neutron gains energy).\n\nDefault 0.0 (energy-loss "
        "side only). Use a NEGATIVE value (e.g. -50) to include the "
        "energy-gain side in the output window (requires 'include "
        "energy-gain')."),
    "e_max": (
        "Highest energy transfer in the output spectrum [meV] (the largest "
        "energy transfer the output records).\n\nMust cover the phonon "
        "(lattice-vibration) band plus its multiphonon tail. Default 250 meV "
        "suits most molecular/lattice modes; raise it for hard modes (e.g. "
        "C-H stretches ~400)."),
    "de": (
        "Output energy bin width [meV], and the engine's energy work-grid step.\n\n"
        "Smaller -> finer spectrum but more cost. Default 0.5 meV (also the "
        "config-file default). Match it to your instrument resolution; "
        "0.1-0.5 meV is typical for VISION."),
    "dq": (
        "Spacing [1/A] of the momentum-transfer support for the "
        "underlying S(Q,E).\n\nSmaller means finer Q sampling along the "
        "instrument locus, at more cost. Default 0.05 1/A. This is the ONE "
        "meaning of dQ (the S(Q,E) shell spacing); it is not the per-shell "
        "sub-sampling."),
    "q_max": (
        "Optional hard cap on the Q-support [1/A] (Q is the momentum "
        "transfer).\n\nDefault: blank "
        "= derive the Q range from the instrument locus over the chosen energy "
        "range. Set a value only to truncate it."),
    "ind_ef": (
        "Fixed final energy Ef [meV] for an indirect-geometry instrument (a "
        "crystal analyzer selects Ef, the energy neutrons leave with; the "
        "incident energy is scanned).\n\nDefault 3.5 meV, which together with "
        "the 45/135 deg banks below reproduces the VISION spectrometer. Change "
        "Ef and the angles for a different indirect instrument."),
    "ind_angles": (
        "Detector bank centre angles 2-theta [deg] (2-theta is the scattering "
        "angle, the angle by which the neutron is deflected).\n\nEither a "
        "comma list '30,60,90,120,150' or a range 'start:stop:step' (e.g. "
        "'30:150:30'). One spectrum is computed per bank (each bank keeps its "
        "own curve).\n\nDefault '45,135' = the VISION forward/backward banks."),
    "dir_ei": (
        "Fixed INCIDENT energy Ei [meV] for a direct-geometry instrument (Ei "
        "is set by a chopper; the final energy is measured).\n\nNote: E max "
        "must be "
        "below Ei; the defaults (Ei = 300 meV, E max = 250 meV) already "
        "satisfy this. The neutron cannot lose more than its incident "
        "energy."),
    "dir_angles": (
        "Detector bank 2-theta [deg] (the scattering angle): a comma list "
        "'10,60,120' or a range 'start:stop:step' (e.g. '5:135:5')."),
    "q_cuts": (
        "Optional constant-|Q| cuts (Q is the momentum transfer): a comma "
        "list of Q values [1/A] "
        "(e.g. '2.0,4.0,6.0'). For each Q you get I(E) at that FIXED |Q|, a "
        "vertical slice through S(Q,E), which reads off the vibrational modes "
        "living at that Q.\n\n"
        "This is a DIRECT-geometry product (hence it lives on the Direct tab): "
        "a direct instrument spans a broad Q-E region, so a fixed-|Q| slice is "
        "meaningful. An indirect instrument instead samples a fixed Q(E) locus "
        "per detector bank, where a constant-Q cut has no instrument meaning, "
        "so it is not offered for indirect geometry.\n\nDefault: blank = no "
        "Q-cuts. The S(Q,E) support is auto-extended to cover the ones you list."),
    "output_mode": (
        "What the Run button computes for this DIRECT-geometry instrument.\n\n"
        "  fixed cuts: 1-D spectra, either along detector-angle loci or as "
        "constant-|Q| slices (pick which below). Writes to the 'output' file.\n"
        "  2-D map: the dense 2-D S(Q,E) heat-map, shown in the Plot "
        "tab (and exportable there). Direct-geometry only.\n\n"
        "Indirect geometry always produces its bank spectra, so it has no 2-D "
        "map option."),
    "cut_by": (
        "How the 'fixed cuts' 1-D spectra are taken.\n\n"
        "  detector angles: one spectrum per scattering angle 2theta; the "
        "neutron is sampled along that detector's curved Q(E) locus (a real "
        "detector group).\n"
        "  constant-Q: vertical slices of S(Q,E) at fixed momentum-transfer "
        "values |Q| you list, each integrated over a band of width 'cut "
        "dQ'.\n\n"
        "For an arbitrary angle- or Q-integration not covered here, compute the "
        "2-D map and post-process it yourself."),
    "cut_dq": (
        "Half-width [1/A] over which each constant-Q cut is AVERAGED. A real "
        "measurement integrates a finite band of momentum transfer |Q|, not an "
        "infinitely-thin line. The cut at Q0 averages S(Q,E) over |Q| in "
        "[Q0 - dQ, Q0 + dQ].\n\n"
        "Default: blank = an infinitely-thin interpolated slice at exactly Q0."),
    "map_coverage": (
        "Detector angular coverage 2theta (min,max in degrees) the 2-D map "
        "masks to: the instrument's accessible-(q,E) 'arch', the region its "
        "detectors can actually reach. Enter as 'min,max' (e.g. '3,135').\n\n"
        "Pre-filled from the selected real instrument's detector span (ARCS "
        "~2.4-136, MARI ~3.4-134, SEQUOIA ~2-62, ...; the PyChop tthlims). Edit "
        "it for a custom range. Blank = the full map (no kinematic mask), which "
        "is the natural choice for the generic 'width polynomial' model."),
    "map_mask": (
        "Blank the 2-D map (and its export) OUTSIDE the detector coverage band "
        "above: show only the kinematically accessible region (the "
        "Euphonic-style arch, the region the instrument's detectors can reach) "
        "rather than the full computed S(Q,E) surface with the envelope drawn "
        "over it.\n\n"
        "On by default for a real instrument; off (full map) for the generic "
        "model. The Plot tab also has a live toggle to flip the view after a run."),
    "resolution_shape": (
        "Instrument energy-resolution line shape (the blurring the instrument "
        "applies to every peak), applied by convolving the spectrum with a "
        "kernel of energy-dependent width.\n\n"
        "  gaussian (default): a normal peak; the usual choice and the "
        "OCLIMAX-equivalent (OCLIMAX applies only a Gaussian). The width "
        "polynomial below is the Gaussian sigma(E).\n"
        "  lorentzian: a Cauchy peak with heavier tails (for spectrometers whose "
        "resolution is Lorentzian-tailed); the width polynomial is then the "
        "Lorentzian HWHM(E).\n\n"
        "Both use the SAME width polynomial c0,c1,c2 below; only the peak shape "
        "(and tail weight) differ."),
    "sigma_c0": (
        "c0, the CONSTANT term of the resolution-width polynomial sigma(E) = "
        "c0 + c1*E + c2*E^2, in meV.\n\n"
        "It is the width at zero energy transfer (E=0), the elastic-line "
        "resolution: the Gaussian sigma (shape=gaussian) or Lorentzian HWHM "
        "(shape=lorentzian). This sets how sharp the elastic peak is and is the "
        "dominant term for an indirect spectrometer like VISION.\n\n"
        "Units: meV.  VISION value: 0.31.\n"
        "Leave ALL THREE coefficients blank to use the VISION preset "
        "(0.31, 0.005, 8.1e-7). If you set any one, blank fields read as 0."),
    "sigma_c1": (
        "c1, the LINEAR coefficient of sigma(E) = c0 + c1*E + c2*E^2.\n\n"
        "The width grows by c1 per meV of energy transfer E, so a larger c1 "
        "means the resolution broadens faster as you move away from the elastic "
        "line. For a direct-geometry instrument the width typically grows "
        "roughly linearly with energy transfer, so c1 carries most of that "
        "energy dependence.\n\n"
        "Units: dimensionless (meV of width per meV of E).  VISION value: 0.005."),
    "sigma_c2": (
        "c2, the QUADRATIC coefficient of sigma(E) = c0 + c1*E + c2*E^2.\n\n"
        "A small curvature term: it adds c2*E^2 to the width, fine-tuning the "
        "broadening at large energy transfer. Usually very small (often "
        "negligible); raise it only if the resolution clearly curves upward at "
        "high E.\n\n"
        "Units: 1/meV (meV of width per meV^2 of E).  VISION value: 8.1e-7 "
        "(0.00000081)."),
    "resolution_model": (
        "How the DIRECT-geometry energy resolution (the instrument's "
        "energy-dependent blurring) is determined.\n\n"
        "  width polynomial: use the shared 'resolution width c0,c1,c2' below "
        "(a Gaussian sigma polynomial); the simple generic option.\n"
        "  chopper (auto, real instrument): pick a real direct-geometry "
        "spectrometer + chopper package + frequency, and IRMA computes the "
        "physical, energy-dependent resolution from the moderator + chopper + "
        "aperture + detector geometry (reproduces Mantid PyChop to <1%). All 8 "
        "PyChop instruments are built in: the Fermi-chopper machines ARCS, "
        "SEQUOIA, MAPS, MARI, MERLIN, HYSPEC and the disk-chopper machines CNCS "
        "and LET. The width is derived from the Ei field above; no manual "
        "numbers needed. RECOMMENDED for a real direct-geometry instrument.\n\n"
        "Both feed a Gaussian kernel; only the WIDTH source differs. This "
        "selector is direct-geometry only."),
    "chop_instrument": (
        "Direct-geometry chopper spectrometer to model. The flight paths, "
        "chopper packages, moderator pulse model and detector geometry are "
        "built in; the resolution is computed for your incident energy Ei and "
        "the chopper package + frequency below. ARCS, SEQUOIA, MAPS, MARI, "
        "MERLIN and HYSPEC are Fermi-chopper instruments; CNCS and LET are "
        "disk-chopper instruments."),
    "chop_package": (
        "Chopper package / mode. For a Fermi instrument this is the slit/"
        "curvature package: different packages trade flux for resolution and "
        "have different usable Ei ranges (a '100 meV' package will not transmit "
        "a 300 meV beam). Disk-chopper instruments (CNCS, LET) expose a single "
        "resolution mode here. If the choice does not pass your Ei at the chosen "
        "frequency, the run reports no transmission; pick a higher-energy "
        "package or change the frequency."),
    "chop_frequency": (
        "Chopper rotation frequency (Hz): the Fermi-chopper spin for ARCS/"
        "SEQUOIA/MAPS/MARI/MERLIN/HYSPEC, or the resolution-disk frequency for "
        "CNCS/LET. A higher frequency gives sharper resolution but lower flux. "
        "Fermi machines run ~120-600 Hz; the disk machines ~60-300 Hz. The "
        "resolution narrows and the burst time shortens as frequency rises."),
    "output": (
        "Output file for the computed spectrum.\n\nThe extension picks the "
        "format: .csv (default; E + one column per cut), .npz (NumPy arrays), or "
        ".json. By default each cut writes only its TOTAL; tick 'inelastic / "
        "elastic breakdown' below to also keep the component columns."),
    "export_components": (
        "Whether to keep the INELASTIC and ELASTIC breakdown of each cut, or just "
        "the TOTAL.\n\n"
        "  OFF (default): the saved file and the 1-D plot show only each cut's "
        "TOTAL; the lean, uncluttered default for both a single cut and many.\n"
        "  ON: each cut additionally carries its inelastic and elastic scattering "
        "components, written to the file (extra columns/arrays) and overlaid on "
        "the plot.\n\n"
        "Read when you press Run, so it governs the saved file; it also toggles "
        "the components on the current 1-D plot live (when that run kept them). "
        "The 2-D map is a single S(Q,E) surface and is unaffected by this."),
}

# The scatterer list is MATERIAL IDENTITY: a prefilled carbon row is IRMA
# asserting a material it cannot know, and a plausible-but-wrong row survives
# review far more easily than an empty one. Everything else on this panel
# (mesh, directions, Ef, angles, grids) is methodology and stays prefilled.
IDENTITY_HINT_ELEMENTS = (
    "Blank on purpose — the scatterers describe YOUR material. Type a symbol "
    "(the nuclear constants autofill), press 'Auto-fill elements from "
    "phonopy.yaml', or load a config with Open Config... (see the committed "
    "examples/spectra configs).")


class NSPanel(RunPanel):
    """Neutron-scattering forward-spectrum panel."""

    error_title = "Spectrum Error"

    def __init__(self, parent, runner, status_setter=None):
        super().__init__(parent, padding=8)
        self.runner = runner
        self._status = status_setter or (lambda msg: None)
        self._last_output = None
        self._carried = {}          # see _unrepresented_fields
        init_form_styles()
        self._build()
        # temp 2-D map files (.npz) live for the panel's lifetime only
        self.bind("<Destroy>", self._on_destroy, add=True)

    # ------------------------------------------------------------------ UI ---
    def _build(self):
        """Build the panel widgets."""
        self.pack(fill=tk.BOTH, expand=True)
        # the canvas is exposed for tooling (screenshot scroll)
        left, right, self._form_canvas = scrolled_columns(self)

        self._build_material(left)
        self._build_physics(left)
        self._build_grid(left)
        self._build_geometry(left)
        self._build_actions(left)
        self._build_output(right)
        self._sync_ns_context()          # set initial field/column visibility

    # mode-dropdown labels (the digit-0 of each is parsed back to the int mode)
    _MODE_LABELS = ["1 (incoherent approx)", "2 (coherent 1ph+multi)",
                    "0 (DOS + isotropic DW)"]
    _MODE_BY_INT = {int(label[0]): label for label in _MODE_LABELS}

    def _mode(self):
        """Active inelastic mode (DOS-files input always means mode 0)."""
        if self.input_source.get() == "dos_files":
            return 0
        return int(self.inelastic_mode.get()[0])

    def _build_material(self, parent):
        """Build the material section."""
        g = form_section(parent, "Material (phonon model)")

        # --- primary gate: where the phonon information comes from -----------
        self.input_source = tk.StringVar(value="phonopy")
        src = ttk.Frame(g)
        src.pack(fill=tk.X, pady=(0, 4))
        ttk.Label(src, text="phonon input:", width=20, anchor="e").pack(
            side=tk.LEFT, padx=(0, 5))
        ttk.Radiobutton(src, text="Phonopy model", value="phonopy",
                        variable=self.input_source,
                        command=self._sync_ns_context).pack(side=tk.LEFT)
        ttk.Radiobutton(src, text="DOS files (mode 0)", value="dos_files",
                        variable=self.input_source,
                        command=self._sync_ns_context).pack(side=tk.LEFT, padx=(10, 0))
        InfoLabel(src, "phonon input", HELP["input_source"]).pack(
            side=tk.LEFT, padx=(4, 0))

        # --- source slot: exactly one of phonopy_frame / dos_frame visible ---
        self._source_slot = ttk.Frame(g)
        self._source_slot.pack(fill=tk.X)

        # phonopy branch
        self.phonopy_frame = ttk.Frame(self._source_slot)
        build_phonopy_fields(self, self.phonopy_frame, HELP)
        self.inelastic_mode = LabeledCombobox(
            self.phonopy_frame, "inelastic mode:", self._MODE_LABELS,
            default=self._MODE_LABELS[1], help_text=HELP["inelastic_mode"])
        self.inelastic_mode.pack(fill=tk.X, pady=2)
        self.inelastic_mode.combo.bind(
            "<<ComboboxSelected>>", lambda e: self._sync_ns_context())
        af = ttk.Frame(self.phonopy_frame)
        af.pack(fill=tk.X, pady=2)
        ttk.Label(af, text="", width=20).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(af, text="Auto-fill elements from phonopy.yaml",
                   command=self._autofill_from_phonopy).pack(side=tk.LEFT)
        InfoLabel(af, "auto-fill", HELP["autofill"]).pack(side=tk.LEFT, padx=(4, 0))

        # DOS-files branch (mode 0, manual)
        self.dos_frame = ttk.Frame(self._source_slot)
        mrow = ttk.Frame(self.dos_frame)
        mrow.pack(fill=tk.X, pady=2)
        ttk.Label(mrow, text="inelastic mode:", width=20, anchor="e").pack(
            side=tk.LEFT, padx=(0, 5))
        ttk.Label(mrow, text="0 — DOS + isotropic Debye-Waller").pack(side=tk.LEFT)
        InfoLabel(mrow, "inelastic mode", HELP["inelastic_mode"]).pack(
            side=tk.LEFT, padx=(4, 0))
        fmt = ttk.Frame(self.dos_frame)
        fmt.pack(fill=tk.X, pady=(2, 0))
        ttk.Label(fmt, text=HELP["dos_format"], foreground="gray",
                  justify=tk.LEFT, wraplength=540).pack(side=tk.LEFT)

        # --- temperature (every context) ------------------------------------
        self.temperature = LabeledEntry(g, "temperature (K):", default="296",
                                        width=10, help_text=HELP["temperature"])
        self.temperature.pack(fill=tk.X, pady=2)

        # --- lattice (mode-0 coherent elastic; toggled inside its box) -------
        self.lattice_box = ttk.Frame(g)
        self.lattice_box.pack(fill=tk.X)
        self.lattice = LabeledEntry(self.lattice_box, "lattice a,b,c,al,be,ga:",
                                    default="", width=22, help_text=HELP["lattice"])

        # --- element table ---------------------------------------------------
        el = ttk.Frame(g)
        el.pack(fill=tk.X, pady=(4, 0))
        ttk.Label(el, text="elements (scatterers):", foreground="gray").pack(side=tk.LEFT)
        InfoLabel(el, "elements", HELP["elements"]).pack(side=tk.LEFT, padx=(4, 0))
        ttk.Label(g, text=IDENTITY_HINT_ELEMENTS, foreground="gray",
                  justify=tk.LEFT, wraplength=540).pack(anchor=tk.W)
        self.element_table = ElementTable(g)
        self.element_table.pack(fill=tk.X, pady=2)
        # One EMPTY row: the scatterer list is the user's material, so IRMA
        # asserts nothing about it. Typing a symbol and leaving the cell fills
        # the nuclear constants from the built-in table (see autofill_row).
        self.element_table.add_row()

    def _build_physics(self, parent):
        """Build the physics section."""
        g = form_section(parent, "Physics")
        self.max_phonon_order = LabeledEntry(g, "max phonon order:", default="auto",
                                             width=8, help_text=HELP["max_phonon_order"])
        self.max_phonon_order.pack(fill=tk.X, pady=2)
        self.min_phonon_energy = LabeledEntry(
            g, "min phonon energy [meV]:", default="", width=8,
            help_text=HELP["min_phonon_energy"])
        self.min_phonon_energy.pack(fill=tk.X, pady=2)
        # directions / multiphonon dirs / jobs apply to the eigenvector engine
        # (modes 1/2) only -- hidden for mode 0.
        self.eng_only_box = ttk.Frame(g)
        self.eng_only_box.pack(fill=tk.X)
        self.n_directions = LabeledEntry(self.eng_only_box, "directions:",
                                         default="10000", width=8,
                                         help_text=HELP["n_directions"])
        self.n_directions.pack(fill=tk.X, pady=2)
        self.mp_directions = LabeledEntry(self.eng_only_box, "multiphonon dirs:",
                                          default="1000", width=8,
                                          help_text=HELP["mp_directions"])
        self.mp_directions.pack(fill=tk.X, pady=2)
        self.jobs = LabeledEntry(self.eng_only_box, "jobs (blank=all cores):",
                                 default="", width=8, help_text=HELP["jobs"])
        self.jobs.pack(fill=tk.X, pady=2)
        self.elastic = LabeledCombobox(g, "elastic line:", ["on", "off"],
                                       default="on", help_text=HELP["elastic"])
        self.elastic.pack(fill=tk.X, pady=2)
        self.elastic.combo.bind("<<ComboboxSelected>>", lambda e: self._sync_ns_context())
        self.elastic_kind = LabeledCombobox(
            g, "elastic kind:", ["both", "coherent", "incoherent"], default="both",
            help_text=HELP["elastic_kind"])
        self.elastic_kind.pack(fill=tk.X, pady=2)
        self.elastic_kind.combo.bind("<<ComboboxSelected>>", lambda e: self._sync_ns_context())
        self.incoherent_elastic_dw = LabeledCombobox(
            g, "incoh. elastic DW:", ["isotropic", "directional"],
            default="isotropic", help_text=HELP["incoherent_elastic_mode"])
        self.incoherent_elastic_dw.pack(fill=tk.X, pady=2)
        self.include_gain = tk.BooleanVar(value=True)
        self._gain_row = check_with_help(
            g, "include energy-gain side", self.include_gain,
            HELP["include_gain"])
        self.gain_side = LabeledCombobox(
            g, "gain-side method:", list(_GAIN_SIDE_LABELS.keys()),
            default=_GAIN_SIDE_REV["direct"], help_text=HELP["gain_side"])
        self.gain_side.pack(fill=tk.X, pady=2)
        # the gain-side method is read only when the gain side is included;
        # a var trace covers the checkbox and load_config alike
        self.include_gain.trace_add(
            "write", lambda *a: self._sync_gain_side())
        self.kinematic = tk.BooleanVar(value=False)
        check_with_help(g, "kinematic kf/ki (count rate)", self.kinematic,
                              HELP["kinematic"])

    # ------------------------------------------------------------ context ---
    def _sync_ns_context(self):
        """Show/hide field groups + table columns for the active (input, mode)."""
        from irma.gui.element_table import NUCLEAR, DOS_COLS, POS_COL
        src = self.input_source.get()
        mode = self._mode()
        if src == "phonopy":
            self.dos_frame.pack_forget()
            self.phonopy_frame.pack(fill=tk.X)
        else:
            self.phonopy_frame.pack_forget()
            self.dos_frame.pack(fill=tk.X)
        # The mode-0 COHERENT Bragg peaks need the crystal (lattice + per-element
        # positions for the structure factor); the incoherent line runs
        # lattice-free from multiplicities. Show the crystal fields whenever a
        # mode-0 elastic line is requested so the peaks are one field away.
        want_crystal = (mode == 0 and self.elastic.get() == "on")
        if want_crystal:
            self.lattice.pack(fill=tk.X, pady=2)
        else:
            self.lattice.pack_forget()
        # engine-only knobs hidden for mode 0
        if mode == 0:
            self.eng_only_box.pack_forget()
        else:
            self.eng_only_box.pack(fill=tk.X)
        # elastic 'off' leaves nothing for elastic_kind or the incoherent-
        # elastic DW selector to steer; kind 'coherent' additionally drops
        # the DW selector (it shapes the incoherent line only). Visibility
        # only: the stored values still round-trip through build_config /
        # load_config, and a mode-0 Save keeps the crystal (see
        # build_config's want_crystal note).
        elastic_on = self.elastic.get() == "on"
        anchor = self.elastic
        if elastic_on:
            self.elastic_kind.pack(fill=tk.X, pady=2, after=anchor)
            anchor = self.elastic_kind
        else:
            self.elastic_kind.pack_forget()
        if elastic_on and self.elastic_kind.get() != "coherent":
            self.incoherent_elastic_dw.pack(fill=tk.X, pady=2, after=anchor)
        else:
            self.incoherent_elastic_dw.pack_forget()
        # table columns by context
        cols = list(NUCLEAR)
        if src == "dos_files":
            cols += list(DOS_COLS)
        if want_crystal:
            cols += list(POS_COL)
        self.element_table.set_visible(cols)

    def _sync_gain_side(self):
        """Show the gain-side method only while the gain side is included
        (nothing reads it when include_gain is off). Visibility only: the
        stored value still round-trips through build/load_config."""
        if self.include_gain.get():
            self.gain_side.pack(fill=tk.X, pady=2, after=self._gain_row)
        else:
            self.gain_side.pack_forget()

    def _autofill_from_phonopy(self):
        """Populate element rows with the distinct symbols in the phonopy.yaml."""
        from tkinter import messagebox
        path = self.phonopy_yaml.get().strip()
        if not path:
            messagebox.showinfo("Auto-fill", "Set a phonopy.yaml first.")
            return
        try:
            syms = phonopy_species(path)
        except Exception as exc:                       # parsing is best-effort
            messagebox.showerror("Auto-fill",
                                 f"Could not read symbols from {path}:\n{exc}")
            return
        if not syms:
            messagebox.showinfo("Auto-fill", "No atom symbols found in the yaml.")
            return
        # set_symbols carries values AND autofill provenance by symbol --
        # a set_rows(get_rows()) round-trip would strip the provenance and
        # re-freeze machine-filled constants as if the user had typed them
        self.element_table.set_symbols(syms)
        self._sync_ns_context()

    def _build_grid(self, parent):
        """Build the energy/Q grid section."""
        g = form_section(parent, "Grid")
        self.e_min = LabeledEntry(g, "E min (meV):", default="0.0", width=8,
                                  help_text=HELP["e_min"])
        self.e_min.pack(fill=tk.X, pady=2)
        self.e_max = LabeledEntry(g, "E max (meV):", default="250.0", width=8,
                                  help_text=HELP["e_max"])
        self.e_max.pack(fill=tk.X, pady=2)
        self.de = LabeledEntry(g, "dE (meV):", default="0.5", width=8,
                               help_text=HELP["de"])
        self.de.pack(fill=tk.X, pady=2)
        self.dq = LabeledEntry(g, "dQ (1/A):", default="0.05", width=8,
                               help_text=HELP["dq"])
        self.dq.pack(fill=tk.X, pady=2)
        self.q_max = LabeledEntry(g, "Q max (1/A, opt):", default="", width=8,
                                  help_text=HELP["q_max"])
        self.q_max.pack(fill=tk.X, pady=2)

    def _build_geometry(self, parent):
        """Build the geometry tabs (VISION / indirect / direct)."""
        g = form_section(parent, "Instrument geometry")
        self.geom_nb = ttk.Notebook(g)
        self.geom_nb.pack(fill=tk.X)
        self.geom_nb.bind("<<NotebookTabChanged>>",
                          lambda _e: self._sync_actions_rows())

        # --- Indirect tab (defaults reproduce the VISION spectrometer) -------
        ind = ttk.Frame(self.geom_nb, padding=4)
        self.geom_nb.add(ind, text="Indirect (VISION defaults)")
        ttk.Label(ind, text="Fixed final energy Ef; incident energy is scanned. "
                            "The defaults below reproduce VISION.",
                  foreground="gray", wraplength=320, justify=tk.LEFT).pack(
                      anchor=tk.W, pady=(0, 4))
        self.ind_ef = LabeledEntry(ind, "Ef (meV):", default="3.5", width=8,
                                   help_text=HELP["ind_ef"])
        self.ind_ef.pack(fill=tk.X, pady=2)
        self.ind_angles = LabeledEntry(ind, "angles (deg):", default="45,135",
                                       width=24, help_text=HELP["ind_angles"])
        self.ind_angles.pack(fill=tk.X, pady=2)
        # Indirect geometry always uses the sigma-polynomial resolution path,
        # so the shape + width controls live directly on this tab.
        self.ind_resolution_shape, self.ind_sigma_coeffs = \
            self._build_res_shape_width(ind)

        # --- Direct tab ------------------------------------------------------
        dirf = ttk.Frame(self.geom_nb, padding=4)
        self.geom_nb.add(dirf, text="Direct")
        ttk.Label(dirf, text="Fixed incident energy Ei; final energy is measured. "
                             "Choose the Output below: 1-D fixed cuts or the 2-D "
                             "S(Q,E) map. Run computes the selected output.",
                  foreground="gray", wraplength=320, justify=tk.LEFT).pack(
                      anchor=tk.W, pady=(0, 4))
        # default Ei sits above the Grid 'E max' default (250) so the fresh
        # Direct tab validates out of the box (validate requires e_max < Ei).
        self.dir_ei = LabeledEntry(dirf, "Ei (meV):", default="300.0", width=8,
                                   help_text=HELP["dir_ei"])
        self.dir_ei.pack(fill=tk.X, pady=2)
        self._build_direct_resolution(dirf)
        self._build_direct_output(dirf)

    def _build_res_shape_width(self, parent):
        """Build the sigma-polynomial resolution controls (shape dropdown + the
        three width-polynomial coefficient fields c0/c1/c2 + hint) into
        ``parent``; return ``(shape_combo, coeff_fields)`` where ``coeff_fields``
        is a :class:`_CoeffFields`.

        Shared by the Indirect tab and the Direct tab's 'width polynomial'
        branch -- the Direct chopper branch derives its own Gaussian width and
        so does not use these.
        """
        shape = LabeledCombobox(
            parent, "resolution shape:", ["gaussian", "lorentzian"],
            default="gaussian", help_text=HELP["resolution_shape"])
        shape.pack(fill=tk.X, pady=(8, 2))
        ttk.Label(parent, text="resolution width  sigma(E) = c0 + c1*E + c2*E^2:",
                  justify=tk.LEFT).pack(anchor=tk.W, pady=(4, 0))
        coeffs = _CoeffFields(parent)
        ttk.Label(
            parent, text="sigma(E) = c0 + c1*E + c2*E^2 (meV).  Leave all three "
                         "blank = VISION (0.31, 0.005, 8.1e-7).  See each ? for "
                         "what the term does.",
            foreground="gray", wraplength=320, justify=tk.LEFT).pack(anchor=tk.W)
        return shape, coeffs

    def _build_direct_resolution(self, parent):
        """Direct-only resolution-model selector with two mutually-exclusive
        parameter frames: the sigma-polynomial (shape + width) frame for the
        'width polynomial' model, and the chopper (instrument + package +
        frequency) frame for the 'chopper' model."""
        self.dir_res_model = LabeledCombobox(
            parent, "resolution model:", list(_RES_MODEL_LABELS),
            default="width polynomial", help_text=HELP["resolution_model"])
        self.dir_res_model.pack(fill=tk.X, pady=(6, 2))
        self.dir_res_model.combo.bind(
            "<<ComboboxSelected>>", lambda e: self._on_res_model_change())

        # 'width polynomial' frame: the shape + width-poly controls (same as the
        # Indirect tab), shown when the poly model is selected.
        self._dir_poly_frame = ttk.Frame(parent)
        self.dir_resolution_shape, self.dir_sigma_coeffs = \
            self._build_res_shape_width(self._dir_poly_frame)

        # 'chopper' frame (instrument + package + frequency), shown for chopper.
        self._chop_frame = ttk.Frame(parent)
        insts = available_instruments()
        self.chop_instrument = LabeledCombobox(
            self._chop_frame, "instrument:", insts,
            default=("ARCS" if "ARCS" in insts else insts[0]),
            help_text=HELP["chop_instrument"])
        self.chop_instrument.pack(fill=tk.X, pady=2)
        self.chop_instrument.combo.bind(
            "<<ComboboxSelected>>", lambda e: self._sync_chop_packages())
        self.chop_package = LabeledCombobox(
            self._chop_frame, "chopper package:", available_packages(insts[0]),
            default="ARCS-700-1.5-AST", help_text=HELP["chop_package"])
        self.chop_package.pack(fill=tk.X, pady=2)
        self.chop_frequency = LabeledEntry(
            self._chop_frame, "frequency (Hz):", default="600", width=8,
            help_text=HELP["chop_frequency"])
        self.chop_frequency.pack(fill=tk.X, pady=2)
        self._sync_res_model()

    def _sync_chop_packages(self):
        """Refresh the chopper-package list for the selected instrument."""
        inst = self.chop_instrument.get()
        try:
            pkgs = available_packages(inst)
        except ValueError:
            return
        self.chop_package.combo["values"] = pkgs
        if self.chop_package.get() not in pkgs:
            self.chop_package.set(pkgs[0])
        # seed a sensible default frequency for the new instrument so a disk
        # machine never inherits a Fermi default outside its range (load_config
        # overwrites this with the stored frequency immediately afterward).
        self.chop_frequency.set(default_frequency(inst))
        # pre-fill the 2-D-map detector-coverage band from this instrument's span
        # (editable; load_config overwrites it with any stored coverage).
        cov = default_coverage(inst)
        if cov:
            self.map_coverage.set(f"{cov[0]:g},{cov[1]:g}")

    def _on_res_model_change(self):
        """User switched the resolution model: swap the frame, and set the 2-D-map
        mask default (a real chopper instrument masks to its coverage; the generic
        width-polynomial model shows the full map). Fires only on user action --
        load_config calls _sync_res_model directly and sets the stored mask."""
        self._sync_res_model()
        model = _RES_MODEL_LABELS.get(self.dir_res_model.get(), "poly")
        self.dir_map_mask.set(model == "chopper")
        if model == "chopper":
            self._sync_chop_packages()      # pre-fill coverage from the instrument
        else:
            self.map_coverage.set("")        # generic -> full map (no coverage)

    def _sync_res_model(self):
        """Swap the Direct-tab resolution parameter frame to match the model:
        the shape + width-polynomial frame for 'width polynomial', the chopper
        frame for 'chopper' (which derives its own Gaussian width, so the
        shape/width controls are not shown for it)."""
        model = _RES_MODEL_LABELS.get(self.dir_res_model.get(), "poly")
        self._dir_poly_frame.pack_forget()
        self._chop_frame.pack_forget()
        if model == "chopper":
            self._chop_frame.pack(fill=tk.X, pady=2)
        else:
            self._dir_poly_frame.pack(fill=tk.X, pady=2)

    def _build_direct_output(self, parent):
        """Direct-only Output selector: 'fixed cuts' (1-D spectra, by detector
        angles or constant-Q) vs '2-D map' (dense S(Q,E) + coverage mask). The
        bottom Run button computes whichever is selected."""
        self.dir_output = LabeledCombobox(
            parent, "output:", list(_OUTPUT_LABELS),
            default="2-D map", help_text=HELP["output_mode"])
        self.dir_output.pack(fill=tk.X, pady=(8, 2))
        self.dir_output.combo.bind("<<ComboboxSelected>>", lambda e: self._sync_output())

        # ---- 'fixed cuts' frame -------------------------------------------
        self._cuts_frame = ttk.Frame(parent)
        self.dir_cut_by = LabeledCombobox(
            self._cuts_frame, "cut by:", list(_CUT_BY_LABELS),
            default="detector angles", help_text=HELP["cut_by"])
        self.dir_cut_by.pack(fill=tk.X, pady=2)
        self.dir_cut_by.combo.bind("<<ComboboxSelected>>", lambda e: self._sync_cut_by())
        self._cut_angles_frame = ttk.Frame(self._cuts_frame)
        self.dir_angles = LabeledEntry(self._cut_angles_frame, "angles (deg):",
                                       default=_DIR_ANGLES_DEFAULT, width=24,
                                       help_text=HELP["dir_angles"])
        self.dir_angles.pack(fill=tk.X, pady=2)
        self._cut_q_frame = ttk.Frame(self._cuts_frame)
        self.q_cuts = LabeledEntry(self._cut_q_frame, "constant-Q (1/A):", default="",
                                   width=24, help_text=HELP["q_cuts"])
        self.q_cuts.pack(fill=tk.X, pady=2)
        self.cut_dq = LabeledEntry(self._cut_q_frame, "cut dQ (1/A):", default="",
                                   width=8, help_text=HELP["cut_dq"])
        self.cut_dq.pack(fill=tk.X, pady=2)

        # ---- '2-D map' frame ----------------------------------------------
        self._mapcfg_frame = ttk.Frame(parent)
        self.map_coverage = LabeledEntry(
            self._mapcfg_frame, "detector coverage 2θ (deg):", default="",
            width=18, help_text=HELP["map_coverage"])
        self.map_coverage.pack(fill=tk.X, pady=2)
        self.dir_map_mask = tk.BooleanVar(value=True)
        check_with_help(self._mapcfg_frame, "mask map to detector coverage",
                              self.dir_map_mask, HELP["map_mask"])
        ttk.Label(self._mapcfg_frame,
                  text="Tip: for an arbitrary angle/Q integration, export the full "
                       "map (Plot tab) and post-process it yourself.",
                  foreground="gray", wraplength=320, justify=tk.LEFT).pack(anchor=tk.W)

        self._sync_output()
        self._sync_cut_by()

    def _sync_output(self):
        """Show the fixed-cuts frame or the 2-D-map frame per the Output selector."""
        mode = _OUTPUT_LABELS.get(self.dir_output.get(), "cuts")
        self._cuts_frame.pack_forget()
        self._mapcfg_frame.pack_forget()
        if mode == "map":
            self._mapcfg_frame.pack(fill=tk.X, pady=2)
        else:
            self._cuts_frame.pack(fill=tk.X, pady=2)
        self._sync_actions_rows()

    def _sync_actions_rows(self):
        """Hide the actions-row fields a Direct '2-D map' run never reads:
        the map goes to a panel-owned temp file (exported via 'Save map...'
        on the Plot tab), so the 1-D output file selector and the
        inelastic/elastic breakdown checkbox are dead in that state. Every
        other (geometry, output) combination shows them."""
        if not hasattr(self, "_actions_btns"):
            return                      # actions row not built yet
        hide = (self._geometry() == "direct"
                and _OUTPUT_LABELS.get(self.dir_output.get(), "cuts") == "map")
        if hide:
            self.output.pack_forget()
            self._export_row.pack_forget()
        else:
            if not self.output.winfo_manager():
                self.output.pack(fill=tk.X, pady=2,
                                 before=self._actions_btns)
            if not self._export_row.winfo_manager():
                self._export_row.pack(anchor=tk.W, fill=tk.X, pady=2,
                                      before=self._actions_btns)

    def _sync_cut_by(self):
        """Show the angles field or the constant-Q (+ dQ) fields per 'cut by'.

        A non-empty constant-Q entry stays visible even in angles mode:
        build_config emits q_cuts in both modes (they are computed in
        addition to the angle spectra), so hiding the field would let a
        leftover entry from a previous mode switch feed the run unseen."""
        by = _CUT_BY_LABELS.get(self.dir_cut_by.get(), "angles")
        self._cut_angles_frame.pack_forget()
        self._cut_q_frame.pack_forget()
        if by == "q":
            self._cut_q_frame.pack(fill=tk.X, pady=2)
        else:
            self._cut_angles_frame.pack(fill=tk.X, pady=2)
            if self.q_cuts.get().strip():
                self._cut_q_frame.pack(fill=tk.X, pady=2)

    def _active_res_widgets(self):
        """The ``(shape_combo, coeff_fields)`` feeding the config for the current
        geometry + model, or ``(None, None)`` when the chopper model supplies the
        width itself (a derived Gaussian sigma)."""
        if self._geometry() != "direct":
            return self.ind_resolution_shape, self.ind_sigma_coeffs
        model = _RES_MODEL_LABELS.get(self.dir_res_model.get(), "poly")
        if model == "chopper":
            return None, None
        return self.dir_resolution_shape, self.dir_sigma_coeffs

    def _build_actions(self, parent):
        """Build the action buttons row."""
        g = ttk.Frame(parent)
        g.pack(fill=tk.X, pady=(2, 0))
        self.output = FileSelector(g, "output:", mode="save",
                                   filetypes=[("CSV", "*.csv"), ("NumPy", "*.npz"),
                                              ("JSON", "*.json")],
                                   help_text=HELP["output"])
        self.output.pack(fill=tk.X, pady=2)
        # Lean default: export/plot only the TOTAL of each cut. Tick to keep the
        # inelastic + elastic breakdown in the saved file and the 1-D plot.
        self.export_components = tk.BooleanVar(value=False)
        self._export_row = check_with_help(
            g, "inelastic / elastic breakdown (plot + saved file)",
            self.export_components, HELP["export_components"])
        self.export_components.trace_add("write", lambda *a: self._replot())
        btns = ttk.Frame(g)
        btns.pack(fill=tk.X, pady=4)
        self._actions_btns = btns
        self.run_btn = ttk.Button(btns, text="Run", default="active",
                                  command=self._run)
        self.run_btn.pack(side=tk.LEFT, padx=(0, 4))
        self.cancel_btn = ttk.Button(btns, text="Cancel", command=self._cancel,
                                     state=tk.DISABLED)
        self.cancel_btn.pack(side=tk.LEFT, padx=(0, 4))
        ttk.Button(btns, text="Save Config...", command=self._save_config).pack(
            side=tk.LEFT, padx=(0, 4))
        ttk.Button(btns, text="Open Config...", command=self._open_config).pack(
            side=tk.LEFT, padx=(0, 4))
        self.plot_btn = ttk.Button(btns, text="Plot", command=self._plot,
                                   state=tk.DISABLED)
        self.plot_btn.pack(side=tk.LEFT, padx=(0, 4))
        # The 2-D map has no button of its own: it is the Direct tab's
        # 'output: 2-D map' mode, and Run computes it (see _run).
        self._sync_actions_rows()       # honor a pre-selected Direct+map state

    def _build_output(self, parent):
        """Build the log/output section."""
        nb = ttk.Notebook(parent)
        nb.pack(fill=tk.BOTH, expand=True)
        logf = ttk.Frame(nb)
        nb.add(logf, text="Log")
        self.log = ScrolledText(logf, height=20)
        self.log.pack(fill=tk.BOTH, expand=True)
        self.plotf = ttk.Frame(nb)
        nb.add(self.plotf, text="Plot")
        ctl = ttk.Frame(self.plotf)
        ctl.pack(fill=tk.X, pady=(2, 0))
        ttk.Label(ctl, text="y-axis:").pack(side=tk.LEFT, padx=(4, 2))
        self.yscale = tk.StringVar(value="linear")
        for txt in ("linear", "log"):
            ttk.Radiobutton(ctl, text=txt, value=txt, variable=self.yscale,
                            command=self._replot).pack(side=tk.LEFT)
        # 2-D map only: blank S(Q,E) outside the instrument's accessible (q,E)
        # band (Euphonic-style kinematic mask) instead of showing the full
        # surface with the envelope overlaid. Re-draws from the cached map.
        self.map_mask = tk.BooleanVar(value=False)
        # disabled until a 2-D map exists, exactly like the Save map button
        # below (the toggle only redraws a cached map).
        self.map_mask_chk = ttk.Checkbutton(
            ctl, text="mask to accessible (q,E)  [2-D map]",
            variable=self.map_mask, command=self._replot, state=tk.DISABLED)
        self.map_mask_chk.pack(side=tk.LEFT, padx=(12, 0))
        # Export the displayed 2-D map (full, or masked to the coverage arch to
        # match the toggle) as long-form CSV / npz.
        self.savemap_btn = ttk.Button(ctl, text="Save map...", command=self._save_map,
                                      state=tk.DISABLED)
        self.savemap_btn.pack(side=tk.LEFT, padx=(12, 0))
        self._plot_holder = ttk.Frame(self.plotf)
        self._plot_holder.pack(fill=tk.BOTH, expand=True)
        self._out_nb = nb
        self._canvas = None
        self._plot_data = None
        self._map_path = None
        self._pending_map = None

    # -------------------------------------------------------------- config ---
    def _geometry(self):
        """Name of the selected geometry tab."""
        return _GEOMETRIES[self.geom_nb.index(self.geom_nb.select())]

    @staticmethod
    def _row_to_scatterer(row, is_dos_file, want_positions):
        """One element-table row -> a scatterer dict, filtered to the active
        context (DOS columns only in DOS-files mode; positions only for the
        mode-0 coherent-elastic line)."""
        sym = row["symbol"]
        d = {"symbol": sym}
        for key in ("sigma_bound_b", "awr", "b_coh_fm", "sigma_inc_b"):
            if row[key]:
                d[key] = parse_float(f"scatterer {sym} {key}", row[key])
        if is_dos_file:
            if row["dos_file"]:
                d["dos_file"] = row["dos_file"]
            if row["dos_unit"]:
                d["dos_unit"] = row["dos_unit"]
            if row["multiplicity"]:
                d["multiplicity"] = parse_int(f"scatterer {sym} mult",
                                              row["multiplicity"])
        if want_positions and row["positions"].strip():
            coords = [parse_float(f"scatterer {sym} positions", x)
                      for x in row["positions"].replace(",", " ").split()]
            if len(coords) % 3 != 0:
                raise ValueError(
                    f"scatterer {sym!r}: positions need a multiple of 3 "
                    f"coordinates (x y z per site), got {len(coords)}")
            d["positions"] = [coords[i:i + 3] for i in range(0, len(coords), 3)]
            # The mult column is hidden outside DOS-files mode, so derive the
            # multiplicity from the sites -- validate requires mult ==
            # len(positions) and the user has no visible field to satisfy it.
            d.setdefault("multiplicity", len(d["positions"]))
        return d

    def build_config(self):
        """Assemble a SpectraConfig from the current widget values."""
        src = self.input_source.get()
        mode = self._mode()
        is_dos_file = (src == "dos_files")
        # Mode 0 emits the crystal (lattice + per-row positions) whenever it is
        # non-blank: gating on the elastic toggle made Save destructive (an
        # elastic-'off' save silently erased the crystal from the file).
        want_crystal = (mode == 0)

        scat = [self._row_to_scatterer(r, is_dos_file, want_crystal)
                for r in self.element_table.get_rows() if r["symbol"]]
        material = {"temperature_K": parse_float("temperature (K)",
                                                 self.temperature.get()),
                    "scatterers": scat}
        if is_dos_file:
            # DOS-files mode 0: no phonopy artifacts; mesh stays at its default
            material["phonopy_yaml"] = None
        else:
            material.update(phonopy_material(self))
        if want_crystal and self.lattice.get().strip():
            # comma- OR space-separated, matching the field help and the mesh
            # field; parse_float raises ValueError (caught by the handlers),
            # unlike the cli parsers' argparse.ArgumentTypeError.
            material["lattice"] = [parse_float("lattice", x)
                                   for x in self.lattice.get().replace(",", " ").split()]

        mpo = self.max_phonon_order.get().strip()
        physics = {
            "inelastic_mode": mode,
            # derived from the input-source gate, not an independent control
            "dos_source": "phonopy" if (mode == 0 and not is_dos_file) else "file",
            "max_phonon_order": ("auto" if mpo == "auto"
                                 else parse_int("max phonon order", mpo)),
            "min_phonon_energy_meV": (
                parse_float("min phonon energy [meV]", self.min_phonon_energy.get())
                if self.min_phonon_energy.get().strip() else 0.0),
            "elastic": self.elastic.get() == "on",
            "elastic_kind": self.elastic_kind.get(),
            "incoherent_elastic_mode": self.incoherent_elastic_dw.get(),
            "include_energy_gain": bool(self.include_gain.get()),
            "gain_side": _GAIN_SIDE_LABELS[self.gain_side.get()],
            "kinematic_kf_ki": bool(self.kinematic.get()),
        }
        # directions / multiphonon-dirs / jobs drive the eigenvector engine
        # (modes 1/2) only -- their widgets are hidden for mode 0, so don't parse
        # them there (mode 0 keeps the config defaults).
        if mode != 0:
            physics["n_directions"] = parse_int("directions",
                                                self.n_directions.get())
            physics["multiphonon_directions"] = parse_int(
                "multiphonon dirs", self.mp_directions.get())
            physics["jobs"] = (parse_int("jobs", self.jobs.get())
                               if self.jobs.get().strip() else None)

        grid = {"e_min_meV": parse_float("E min (meV)", self.e_min.get()),
                "e_max_meV": parse_float("E max (meV)", self.e_max.get()),
                "de_meV": parse_float("dE (meV)", self.de.get()),
                "dq_max_invA": parse_float("dQ (1/A)", self.dq.get())}
        if self.q_max.get().strip():
            grid["q_max_invA"] = parse_float("Q max (1/A)", self.q_max.get())

        geometry = self._geometry()
        instrument = {"geometry": geometry,
                      "export_components": bool(self.export_components.get())}
        if geometry == "indirect":
            instrument["e_fixed_meV"] = parse_float("Ef (meV)", self.ind_ef.get())
            instrument["angles_deg"] = parse_angles(self.ind_angles.get())
        else:
            instrument["e_fixed_meV"] = parse_float("Ei (meV)", self.dir_ei.get())
            # direct output selection (the Run product) + its fields
            out_mode = _OUTPUT_LABELS.get(self.dir_output.get(), "cuts")
            cut_by = _CUT_BY_LABELS.get(self.dir_cut_by.get(), "angles")
            angles = parse_angles(self.dir_angles.get())
            if not angles and cut_by == "q":
                # the angles field is hidden in constant-Q mode; a blank there
                # must still satisfy validate's angles_deg requirement (the
                # constant-Q cuts never read the angles).
                angles = parse_angles(_DIR_ANGLES_DEFAULT)
            instrument["angles_deg"] = angles
            instrument["output_mode"] = out_mode
            instrument["cut_by"] = cut_by
            instrument["map_mask"] = bool(self.dir_map_mask.get())
            # coverage is a map-mode concern; only emit it there (it's pre-filled
            # from the instrument even in cuts mode, which must not round-trip out).
            if out_mode == "map" and self.map_coverage.get().strip():
                instrument["map_coverage_deg"] = parse_coeffs(self.map_coverage.get())
            # direct-geometry resolution model (poly | chopper)
            model = _RES_MODEL_LABELS.get(self.dir_res_model.get(), "poly")
            instrument["resolution_model"] = model
            if model == "chopper":
                instrument["chopper_spec"] = {
                    "instrument": self.chop_instrument.get(),
                    "package": self.chop_package.get(),
                    "frequency": parse_float("frequency (Hz)",
                                             self.chop_frequency.get())}
        # Constant-|Q| cuts are emitted for EVERY geometry: the CLI attaches
        # --q-cuts geometry-independently and run_spectra honours q_cuts the
        # same way, so a CLI-authored vision/indirect config carrying
        # instrument.q_cuts must survive Run/Save, not be silently dropped.
        # (The editing widgets live on the Direct tab; emitted in BOTH cut
        # modes -- _sync_cut_by keeps a non-empty entry visible so it never
        # feeds the run unseen.)
        if self.q_cuts.get().strip():
            instrument["q_cuts"] = parse_coeffs(self.q_cuts.get())
        if self.cut_dq.get().strip():
            instrument["cut_dq_invA"] = parse_float("cut dQ (1/A)",
                                                    self.cut_dq.get())
        # Resolution shape + width-poly come from the active geometry/model's
        # controls; under the chopper model the width is derived (Gaussian), so
        # those controls are absent and the shape is fixed to gaussian.
        shape_w, coeff_w = self._active_res_widgets()
        if shape_w is not None:
            instrument["resolution_shape"] = shape_w.get()
            coeffs = coeff_w.get_coeffs()
            if coeffs is not None:
                instrument["sigma_coeffs"] = coeffs
        else:
            instrument["resolution_shape"] = "gaussian"

        sections = {"material": material, "physics": physics, "grid": grid,
                    "instrument": instrument}
        for (section, name), value in self._carried.items():
            sections[section][name] = value
        return SpectraConfig.from_dict(sections)

    @staticmethod
    def _scatterer_to_row(s):
        """A Scatterer -> an element-table row dict (all columns, strings)."""
        def fmt(v):
            """Blank for None, else str(value)."""
            return "" if v is None else str(v)
        pos = (" ".join(str(c) for p in s.positions for c in p)
               if s.positions else "")
        return {"symbol": s.symbol, "sigma_bound_b": fmt(s.sigma_bound_b),
                "awr": fmt(s.awr), "b_coh_fm": fmt(s.b_coh_fm),
                "sigma_inc_b": fmt(s.sigma_inc_b),
                "multiplicity": str(s.multiplicity),
                "dos_unit": s.dos_unit or "meV", "dos_file": s.dos_file or "",
                "positions": pos}

    def load_config(self, cfg):
        """Populate every widget from a SpectraConfig (for Open / round-trip).

        Refuses (before touching a widget) an indirect-geometry map, which
        the panel cannot represent: it runs maps for direct geometry only.
        """
        m, p, g, ins = cfg.material, cfg.physics, cfg.grid, cfg.instrument
        if ins.geometry != "direct" and ins.output_mode == "map":
            raise SpectraConfigError(
                f"this config asks for a {ins.geometry}-geometry 2-D map, and the "
                f"panel runs maps for direct geometry only; run it with "
                f"`irma spectra map <config> -o <map.csv>`")
        load_phonopy_fields(self, m)
        self.temperature.set(m.temperature_K)
        self.lattice.set(",".join(str(x) for x in m.lattice) if m.lattice else "")
        self.element_table.set_rows([self._scatterer_to_row(s) for s in m.scatterers])
        if not self.element_table.rows:
            # A config with no scatterers must leave the same single blank row
            # a fresh panel ships, not a header with nothing under it.
            self.element_table.add_row()

        # input-source gate is derived: mode 0 + DOS files -> DOS-files branch,
        # everything else -> the Phonopy branch (mode 1/2, or DOS-from-phonopy).
        self.input_source.set("dos_files" if (p.inelastic_mode == 0
                                              and p.dos_source == "file") else "phonopy")
        self.inelastic_mode.set(self._MODE_BY_INT[p.inelastic_mode])
        self.max_phonon_order.set(p.max_phonon_order)
        cutoff = float(p.min_phonon_energy_meV)
        self.min_phonon_energy.set("" if cutoff == 0.0 else f"{cutoff:g}")
        self.n_directions.set(p.n_directions)
        self.mp_directions.set(p.multiphonon_directions)
        self.jobs.set("" if p.jobs is None else p.jobs)
        self.elastic.set("on" if p.elastic else "off")
        self.elastic_kind.set(p.elastic_kind)
        self.incoherent_elastic_dw.set(p.incoherent_elastic_mode)
        self.include_gain.set(p.include_energy_gain)
        self.gain_side.set(_GAIN_SIDE_REV[p.gain_side])
        self.kinematic.set(p.kinematic_kf_ki)

        self.e_min.set(g.e_min_meV); self.e_max.set(g.e_max_meV)
        self.de.set(g.de_meV); self.dq.set(g.dq_max_invA)
        self.q_max.set("" if g.q_max_invA is None else g.q_max_invA)

        # The GUI has two geometry tabs (indirect / direct); a legacy 'vision'
        # config maps onto the indirect tab with the VISION banks, since VISION
        # IS indirect geometry with Ef=3.5 and the 45/135 banks.
        if ins.geometry == "direct":
            self.geom_nb.select(1)
            self.dir_ei.set(ins.e_fixed_meV)
            if ins.angles_deg:
                self.dir_angles.set(",".join(str(a) for a in ins.angles_deg))
            # direct resolution model + its params
            self.dir_res_model.set(_RES_MODEL_REV.get(ins.resolution_model,
                                                      "width polynomial"))
            if ins.chopper_spec:
                self.chop_instrument.set(ins.chopper_spec.get("instrument", "ARCS"))
                self._sync_chop_packages()
                self.chop_package.set(ins.chopper_spec.get("package", ""))
                self.chop_frequency.set(ins.chopper_spec.get("frequency", 600.0))
            self._sync_res_model()
            # direct output mode + its fields. Set AFTER the chopper sync so an
            # explicit stored coverage wins over the instrument-default pre-fill;
            # a None stored coverage keeps the pre-fill (chopper) or blank (poly).
            self.dir_output.set(_OUTPUT_REV.get(ins.output_mode, "fixed cuts"))
            self.dir_cut_by.set(_CUT_BY_REV.get(ins.cut_by, "detector angles"))
            if ins.map_coverage_deg:
                self.map_coverage.set(",".join(str(a) for a in ins.map_coverage_deg))
            elif not (ins.resolution_model == "chopper" and ins.chopper_spec):
                # No stored coverage + generic model -> full map: clear the
                # field rather than inherit the previously loaded config's
                # band (chopper re-seeded it from the instrument above).
                self.map_coverage.set("")
            self.dir_map_mask.set(bool(ins.map_mask))
            self._sync_output()
            self._sync_cut_by()
        else:
            self.geom_nb.select(0)
            self.ind_ef.set(ins.e_fixed_meV)
            angles = ins.angles_deg or [45.0, 135.0]   # vision auto-fills its banks
            self.ind_angles.set(",".join(str(a) for a in angles))
        # q_cuts / cut_dq are geometry-independent (mirroring build_config):
        # populate them for any geometry so an indirect/vision config
        # carrying them round-trips.
        self.q_cuts.set("" if not ins.q_cuts
                        else ",".join(str(q) for q in ins.q_cuts))
        self.cut_dq.set("" if ins.cut_dq_invA is None else ins.cut_dq_invA)
        # Resolution shape + width live on BOTH geometry tabs (indirect; direct
        # 'width polynomial' branch); populate both so a tab switch is
        # consistent. build_config reads whichever is active.
        self.ind_resolution_shape.set(ins.resolution_shape)
        self.dir_resolution_shape.set(ins.resolution_shape)
        self.ind_sigma_coeffs.set_coeffs(ins.sigma_coeffs)
        self.dir_sigma_coeffs.set_coeffs(ins.sigma_coeffs)
        self.export_components.set(bool(ins.export_components))
        self._carried = _unrepresented_fields(cfg)
        self._sync_ns_context()          # apply field/column visibility for the loaded mode

    # -------------------------------------------------------------- actions --
    def _run(self):
        """Collect the config and start the spectra subprocess."""
        if self.runner.is_running:
            messagebox.showwarning("Running", "A calculation is already in progress.")
            return
        try:
            cfg = self.build_config()
        except (SpectraConfigError, ValueError, argparse.ArgumentTypeError) as exc:
            messagebox.showerror("Config Error", str(exc))
            return
        # Direct + '2-D map' output: Run computes the dense map (no output file
        # required -- it plots in the Plot tab and is exportable there).
        if cfg.instrument.geometry == "direct" and cfg.instrument.output_mode == "map":
            self._run_map()
            return
        output = self.output.get().strip()
        if not output:
            messagebox.showerror("Error", "Please specify an output file.")
            return
        cfg_path = self._temp_path("config.yaml")
        dump(cfg, cfg_path)
        self._last_output = output
        self.plot_btn.config(state=tk.DISABLED)
        self._start(
            [sys.executable, "-u", "-m", "irma.spectra", "run",
             cfg_path, "-o", output],
            "IRMA neutron-scattering spectrum",
            f"Spectrum written to {output}.", "Spectra config error",
            header=f"geometry={cfg.instrument.geometry}  output={output}\n\n",
            running="Running spectrum...", done="Spectrum complete",
            output_path=output, on_ok=self._spectrum_done)

    def _action_buttons(self):
        return (self.run_btn,)

    def _spectrum_done(self):
        """Enable Plot and plot the spectrum that was just written."""
        if self._last_output and os.path.exists(self._last_output):
            self.plot_btn.config(state=tk.NORMAL)
            self._plot()

    # -- 2-D S(Q,E) map ------------------------------------------------------
    def _run_map(self):
        """Start the 2-D S(Q,E) map subprocess."""
        if self.runner.is_running:
            messagebox.showwarning("Running", "A calculation is already in progress.")
            return
        try:
            cfg = self.build_config()
        except (SpectraConfigError, ValueError, argparse.ArgumentTypeError) as exc:
            messagebox.showerror("Config Error", str(exc))
            return
        cfg_path = self._temp_path("config.yaml")
        dump(cfg, cfg_path)
        # drop the previous run's temp map unless it is still plotted (then it
        # is released when the new map supersedes it in _plot_map)
        prev = self._pending_map
        if prev and prev != self._map_path:
            self._discard_map_file(prev)
        fh = tempfile.NamedTemporaryFile(suffix=".npz", delete=False)
        fh.close()
        self._pending_map = fh.name
        # the map's accessible band comes from the detector COVERAGE (explicit
        # field -> selected instrument's span -> detector angles as a fallback).
        cov = cfg.instrument.map_coverage_deg
        if not cov and cfg.instrument.chopper_spec:
            cov = default_coverage(cfg.instrument.chopper_spec.get("instrument", ""))
        if not cov:
            ang = cfg.instrument.angles_deg or [45.0, 135.0]
            cov = [min(ang), max(ang)]
        th_min, th_max = float(cov[0]), float(cov[1])
        # the Plot-tab view toggle starts from the config's mask preference
        self.map_mask.set(bool(cfg.instrument.map_mask))
        # Q max: an explicit field wins; otherwise auto-cover the kinematic
        # envelope so the data fills the whole accessible arch. A hardcoded cap
        # leaves an empty band wherever the envelope reaches past it (wide angles
        # at high Ei, or the energy-gain side) -- the bug this replaces.
        if cfg.grid.q_max_invA:
            q_max = float(cfg.grid.q_max_invA)
        else:
            import numpy as _np
            from irma.spectra.forward import kinematic_envelope
            g = cfg.grid
            Eg = _np.arange(g.e_min_meV, g.e_max_meV + 0.5 * g.de_meV, g.de_meV)
            _, q_hi = kinematic_envelope(cfg.instrument.geometry,
                                         cfg.instrument.e_fixed_meV, th_min, th_max, Eg)
            # Mask to finite entries instead of np.nanmax: an all-NaN envelope
            # would make nanmax emit a RuntimeWarning (and it was called twice).
            finite_q = q_hi[_np.isfinite(q_hi)]
            q_max = float(finite_q.max()) + 0.5 if finite_q.size else 13.0
            q_max = min(q_max, 40.0)                 # cover the arch + pad; sane cap
        # q_min 0: the same full-arch principle as q_max above -- the CLI
        # accepts a zero lower bound, and a hardcoded 0.5 cut a valid low-Q
        # band out of cold/low-Ei maps while these comments promised full
        # accessible coverage (review GUI-MAP).
        argv = [sys.executable, "-u", "-m", "irma.spectra", "map",
                cfg_path, "--q-min", "0.0", "--q-max", f"{q_max:.3f}",
                "--dq-map", str(cfg.grid.dq_max_invA),
                "--angle-range", str(th_min), str(th_max),
                # the full map: the Plot tab masks at view and export time
                "--no-mask", "-o", self._pending_map]
        # a success guarantees a non-empty map file (the runner checks
        # output_path on exit 0), so on_ok can load and plot it
        self._start(
            argv, "IRMA 2-D S(Q,E) map",
            f"Map written to {self._pending_map}.", "Spectra config error",
            header="\n", running="Computing 2D map (dense grid)...",
            done="2D map complete", error_title="Map Error",
            output_path=self._pending_map,
            on_ok=lambda: self._plot_map(self._pending_map))

    def cleanup_temp_files(self):
        """Remove every panel-owned temp file: the run config and the 2-D
        map files. Idempotent; app close and panel destroy call it."""
        super().cleanup_temp_files()
        self._discard_map_file(self._map_path)
        if self._pending_map != self._map_path:
            self._discard_map_file(self._pending_map)
        self._map_path = None
        self._pending_map = None

    @staticmethod
    def _discard_map_file(path):
        """Unlink a temp 2-D map .npz (best-effort; they are panel-owned)."""
        if not path:
            return
        try:
            os.unlink(path)
        except OSError:
            pass

    def _on_destroy(self, event):
        # event.widget may be a path string during teardown; compare names
        """Release callbacks and figures when the panel is destroyed."""
        if str(event.widget) != str(self):
            return
        self.cleanup_temp_files()

    def _plot_map(self, path):
        """Load the computed map CSV and draw it."""
        if self._map_path and self._map_path != path:
            self._discard_map_file(self._map_path)    # superseded temp map
        self._map_path = path          # subsequent y-scale toggles redraw the map
        self._plot_data = None
        self._draw_map()
        self.savemap_btn.config(state=tk.NORMAL)     # map is now exportable
        self.map_mask_chk.config(state=tk.NORMAL)    # and maskable
        self._out_nb.select(1)

    def _save_map(self):
        """Export the displayed 2-D map (full, or masked to the coverage arch to
        match the 'mask to accessible' toggle) as long-form CSV or npz."""
        if not self._map_path:
            messagebox.showinfo("Save map", "Run a 2-D map first.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV (long-form Q,E,S)", "*.csv"), ("NumPy", "*.npz")])
        if not path:
            return
        import numpy as np
        from irma.spectra.forward import save_sqe_map
        d = np.load(self._map_path)
        env = ((d["envelope_E"], d["envelope_q_lo"], d["envelope_q_hi"])
               if "envelope_E" in d.files else None)
        masked = bool(self.map_mask.get())
        try:
            save_sqe_map(path, d["Q"], d["E"], d["S"], envelope=env, masked=masked)
        except Exception as exc:                     # pragma: no cover - I/O
            messagebox.showerror("Save map", f"Could not save: {exc}")
            return
        self._status(f"Map saved to {path}"
                     f" ({'masked to coverage' if masked else 'full grid'})")

    def _draw_map(self):
        """Render the 2-D map into the embedded matplotlib canvas."""
        import numpy as np
        try:
            from matplotlib.figure import Figure
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
            import matplotlib.colors as mcolors
        except ImportError:
            messagebox.showinfo("Plot", "matplotlib is not installed.")
            return
        from matplotlib import colormaps
        d = np.load(self._map_path)
        Q, E, S = d["Q"], d["E"], np.asarray(d["S"], float)
        have_env = "envelope_E" in d.files
        if self.map_mask.get() and have_env:
            from irma.spectra.forward import kinematic_mask
            m = kinematic_mask(Q, d["envelope_q_lo"], d["envelope_q_hi"])
            S = np.where(m, S, np.nan)         # blank the inaccessible (q,E) region
        for w in self._plot_holder.winfo_children():
            w.destroy()
        fig = Figure(figsize=(6, 4), dpi=100)
        ax = fig.add_subplot(111)
        cmap = colormaps["viridis"].copy()
        cmap.set_bad(cmap(0.0))               # masked cells -> the colormap floor
        Sc = np.clip(S, 1e-6, None)
        finite = Sc[np.isfinite(Sc)]
        if finite.size == 0:
            # The kinematic mask blanked every cell (the coverage band does not
            # intersect the Q grid). nanmin/nanmax would be NaN, and feeding that
            # to LogNorm/pcolormesh raises from inside this Tk callback -- show a
            # clear message instead of crashing.
            ax.text(0.5, 0.5, "no accessible (q, E) data in the masked region",
                    ha="center", va="center", transform=ax.transAxes, fontsize=10)
        else:
            norm = (mcolors.LogNorm(vmin=max(1e-5, float(finite.min())),
                                    vmax=float(finite.max()))
                    if self.yscale.get() == "log" else None)
            pcm = ax.pcolormesh(Q, E, S.T, shading="auto", norm=norm, cmap=cmap)
            if have_env:
                ax.plot(d["envelope_q_lo"], d["envelope_E"], "w-", lw=1.0, alpha=0.8)
                ax.plot(d["envelope_q_hi"], d["envelope_E"], "w-", lw=1.0, alpha=0.8,
                        label="kinematic envelope")
                ax.legend(loc="upper right", fontsize=8)
            # Keep the x-axis on the computed Q range: the envelope curve must
            # never drag it into empty space past the data (q_max-mismatch).
            ax.set_xlim(float(Q.min()), float(Q.max()))
            fig.colorbar(pcm, ax=ax, label="S(Q,E)")
        ax.set_xlabel(r"|Q| (1/$\AA$)")
        ax.set_ylabel("energy transfer (meV)")
        fig.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=self._plot_holder)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self._canvas = canvas

    def _save_config(self):
        """Save the panel state as a spectra YAML config."""
        try:
            cfg = self.build_config()
        except (SpectraConfigError, ValueError, argparse.ArgumentTypeError) as exc:
            messagebox.showerror("Config Error", str(exc))
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".yaml",
            filetypes=[("YAML", "*.yaml"), ("JSON", "*.json")])
        if path:
            dump(cfg, path)
            self._status(f"Saved config to {os.path.basename(path)}")

    def _open_config(self):
        """Load a spectra YAML config into the panel."""
        path = filedialog.askopenfilename(
            filetypes=[("Spectra config", "*.yaml *.yml *.json *.toml"),
                       ("All files", "*.*")])
        if not path:
            return
        try:
            self.load_config(load(path))
        except Exception as exc:
            messagebox.showerror("Open Config", str(exc))
            return
        if self._carried:
            # no control on the form shows these, so say so in the log
            self.log.append(
                "note: carried from the file (no control on this panel; the "
                "next Run or Save keeps them): "
                + ", ".join(f"{s}.{n}={v!r}"
                            for (s, n), v in sorted(self._carried.items()))
                + "\n")
        self._status(f"Loaded config from {os.path.basename(path)}")

    def _plot(self):
        """Plot the computed spectrum CSV."""
        if not (self._last_output and os.path.exists(self._last_output)):
            return
        try:
            self._plot_data = self._read_spectrum(self._last_output)
        except Exception as exc:  # pragma: no cover - I/O/format
            messagebox.showerror("Plot", f"Could not read spectrum: {exc}")
            return
        self._discard_map_file(self._map_path)
        self._map_path = None       # 1-D spectrum supersedes any map
        self._replot()
        self._out_nb.select(1)

    def _replot(self):
        """Re-draw on a y-scale change: the 2-D map if one is loaded, else the
        per-cut 1-D spectrum (lin/log)."""
        if self._map_path:
            self._draw_map()
            return
        if not self._plot_data:
            return
        try:
            from matplotlib.figure import Figure
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        except ImportError:
            messagebox.showinfo("Plot", "matplotlib is not installed.")
            return
        E, labels, Tpa, Ipa, Epa = self._plot_data
        for w in self._plot_holder.winfo_children():
            w.destroy()
        fig = Figure(figsize=(6, 4), dpi=100)
        ax = fig.add_subplot(111)
        show_comp = (self.export_components.get() and Ipa is not None and Epa is not None)
        single = (len(labels) == 1)
        for k, lab in enumerate(labels):
            # by default plot only each cut's TOTAL; the components toggle adds
            # its inelastic + elastic curves.
            tlab = "total" if single else lab
            ax.plot(E, Tpa[k], lw=1.2 if single else 1.0,
                    label=(tlab if (single or not show_comp) else f"{lab} total"))
            if show_comp:
                ilab = "inelastic" if single else f"{lab} inelastic"
                elab = "elastic" if single else f"{lab} elastic"
                ax.plot(E, Ipa[k], lw=0.9, alpha=0.8, label=ilab)
                ax.plot(E, Epa[k], lw=0.9, alpha=0.8, label=elab)
        ax.set_xlabel("energy transfer (meV)")
        ax.set_ylabel("intensity (barn/sr)")
        ax.set_yscale(self.yscale.get())
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3, which="both")
        fig.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=self._plot_holder)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self._canvas = canvas

    @staticmethod
    def _read_spectrum(path):
        """Return (E, labels, total, inelastic, elastic).

        Each cut (a detector angle or a constant-Q slice) is one row of the
        (n_cuts, nE) arrays; ``labels`` are display strings ('45 deg', 'Q=2').
        """
        import numpy as np
        if path.endswith((".npz", ".json")):
            if path.endswith(".npz"):
                d = np.load(path)
                keys = set(d.files)
            else:
                import json
                d = json.load(open(path))
                keys = set(d)
            getf = lambda k: np.atleast_2d(d[k])              # noqa: E731
            E = np.asarray(d["E_meV"], float)
            labels, T, I, El = [], [], [], []
            for pre, fmt, src in (("angle", lambda a: f"{float(a):g} deg", "angles_deg"),
                                  ("q", lambda q: f"Q={float(q):g}", "q_cuts")):
                if f"I_total_per_{pre}" not in keys:
                    continue
                labels += [fmt(v) for v in d[src]]
                T.append(getf(f"I_total_per_{pre}"))
                I.append(getf(f"I_inelastic_per_{pre}") if f"I_inelastic_per_{pre}" in keys else None)
                El.append(getf(f"I_elastic_per_{pre}") if f"I_elastic_per_{pre}" in keys else None)
            comp = all(x is not None for x in I) and all(x is not None for x in El)
            return (E, labels, np.vstack(T),
                    np.vstack(I) if comp else None, np.vstack(El) if comp else None)
        # csv: '# ...' comment, header 'E_meV,<channel>@<cut>,...'. A total-only
        # file has just one 'total@<cut>' column per cut; the breakdown export
        # adds 'inelastic@<cut>' and 'elastic@<cut>'.
        lines = [ln.rstrip("\n") for ln in open(path) if ln.strip()]
        body = [ln for ln in lines if not ln.startswith("#")]
        header = body[0].split(",")
        data = np.array([[float(x) for x in r.split(",")] for r in body[1:]])
        E = data[:, 0]
        cuts, chan = [], {}                   # ordered tags + {tag: {channel: col}}
        for j, col in enumerate(header[1:], start=1):
            if "@" not in col:
                continue
            ch, tag = col.split("@", 1)
            chan.setdefault(tag, {})[ch] = j
            if tag not in cuts:
                cuts.append(tag)
        labels = [(t[:-3] + " deg" if t.endswith("deg") else t) for t in cuts]
        T = np.vstack([data[:, chan[t]["total"]] for t in cuts])
        have = all("inelastic" in chan[t] and "elastic" in chan[t] for t in cuts)
        I = np.vstack([data[:, chan[t]["inelastic"]] for t in cuts]) if have else None
        El = np.vstack([data[:, chan[t]["elastic"]] for t in cuts]) if have else None
        return E, labels, T, I, El
