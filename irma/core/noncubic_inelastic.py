"""Controls and help text for the noncubic inelastic path (inelastic_mode=1/2)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class NoncubicInelasticControls:
    """User-facing controls for IRMA's noncubic inelastic MT4 path."""

    num_directions: int
    multiphonon_num_directions: int
    multiphonon_max_order: int
    # Opt-in: when True, the engine sizes the multiphonon order up to the value
    # required to converge the incoherent Poisson(2W) sum for the anisotropic
    # Debye-Waller factor (never below the deck nphon). When False (default), the
    # deck's Card 3 nphon is honored verbatim and the engine only warns if it is
    # below that requirement. Set from the Card 6g optional 3rd field.
    auto_multiphonon_order: bool = False
    # Optional one-value card immediately before Card 6g. Zero preserves the
    # established automatic per-q floors exactly; positive values exclude
    # modes with energy <= this threshold everywhere in the phonon pipeline.
    min_phonon_energy_mev: float = 0.0


MIN_PHONON_ENERGY_HELP = (
    "Minimum phonon energy in meV, 0 or blank for none. Every phonon mode "
    "with energy at or below this value is removed from all terms of the "
    "calculation: the phonon DOS, the Debye-Waller factors, the coherent and "
    "incoherent one-phonon scattering and the multiphonon expansion, on the "
    "ENDF, spectra and NCrystal paths alike. Nothing replaces the removed "
    "modes: the result is a truncated vibrational model and its provenance "
    "says so.\n\n"
    "This is not a repair for an unstable model. Imaginary modes and "
    "numerical noise near zero energy are already excluded by IRMA's "
    "automatic floors (1 ueV, 0.1 meV at Gamma) whatever this value is; a "
    "positive cutoff only ever removes real, positive modes.\n\n"
    "Mean-square displacements weight modes as 1/E^2, so a cutoff that "
    "removes a negligible share of the modes can remove a large share of the "
    "Debye-Waller exponent: on graphite at 296 K a 5 meV cutoff removes 0.13% "
    "of the modes but 29% of the displacement, and raises the Debye-Waller "
    "intensity factor along c by 74% at Q = 10 1/A. The run log and the "
    "metadata report the removed weight and the displacement change, with a "
    "warning above 1%.")


def run_noncubic_sab_inprocess(*args, **kwargs):
    """See :func:`irma.core.noncubic_engine.run_noncubic_sab_inprocess`.

    Imported on call so that importing the controls stays cheap.
    """
    from irma.core.noncubic_engine import run_noncubic_sab_inprocess as run

    return run(*args, **kwargs)


__all__ = [
    "MIN_PHONON_ENERGY_HELP",
    "NoncubicInelasticControls",
    "run_noncubic_sab_inprocess",
]
