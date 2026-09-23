"""``irma.spectra`` -- neutron-scattering forward modelling.

Turns a powder ``S(Q,E)`` (computed fresh from a phonon model via the IRMA
noncubic SAB engine, or loaded from OCLIMAX/ENDF for validation) into the 1-D
spectrum an instrument records (VISION / generic indirect / generic direct
geometry), with an optional elastic line.

This is IRMA's SECOND capability, kept separate from ENDF/TSL evaluation but
sharing core physics (``irma.core`` constants, the noncubic SAB engine, and the
coherent/incoherent elastic). ``phonopy`` and ``PyYAML`` are optional and are
imported lazily by the compute/config layers, so ``import irma.spectra`` works
without them.
"""
from irma.spectra.sqe import (
    PowderSQE, from_oclimax, from_irma_cache, from_noncubic_arrays,
    signed_sqe, sqe_interpolator, k_of_E, Q_indirect, Q_direct, Q_fit,
    kf_ki_indirect, kf_ki_direct, sample_along, sigma_of_E, gaussian_resolution,
    resolution_convolve, RESOLUTION_SHAPES, elastic_line, instrument_spectrum,
    VISION_EF_MEV, VISION_BANKS, VISION_SIGMA_COEFFS, C_E, KB,
)
from irma.spectra.chopper_resolution import (
    chopper_sigma_of_E, direct_resolution_fwhm, instrument_geometry,
    available_instruments, available_packages, INSTRUMENT_DB,
)
from irma.spectra.elastic import (
    ElasticModel, from_endf_mf7mt2, from_engine_elastic_state,
    bank_elastic_area, selftest,
)
from irma.spectra.instruments import (
    Instrument, VISION, indirect, direct, simulate,
)
from irma.spectra.forward import (
    _pick_sqe_key, SpectrumResult, build_locus_support, compute_spectrum,
    SQEMap, compute_sqe_map, kinematic_envelope, kinematic_mask, save_sqe_map,
)
from irma.spectra.config import (
    SpectraConfig, MaterialConfig, PhysicsConfig, GridConfig, InstrumentConfig,
    Scatterer, SpectraConfigError, load, dump, validate, run_spectra,
)

__all__ = [
    # containers + loaders
    "PowderSQE", "from_oclimax", "from_irma_cache", "from_noncubic_arrays",
    # low-level forward model
    "signed_sqe", "sqe_interpolator", "k_of_E", "Q_indirect", "Q_direct", "Q_fit",
    "kf_ki_indirect", "kf_ki_direct", "sample_along", "sigma_of_E",
    "gaussian_resolution", "resolution_convolve", "RESOLUTION_SHAPES",
    "elastic_line", "instrument_spectrum",
    # direct-geometry chopper resolution (auto)
    "chopper_sigma_of_E", "direct_resolution_fwhm", "instrument_geometry",
    "available_instruments", "available_packages", "INSTRUMENT_DB",
    # elastic
    "ElasticModel", "from_endf_mf7mt2", "from_engine_elastic_state",
    "bank_elastic_area", "selftest",
    # instruments
    "Instrument", "VISION", "indirect", "direct", "simulate",
    # engine key selector + forward orchestrator
    "_pick_sqe_key", "SpectrumResult", "build_locus_support",
    "compute_spectrum", "SQEMap", "compute_sqe_map", "kinematic_envelope",
    "kinematic_mask", "save_sqe_map",
    # config schema + I/O + runner
    "SpectraConfig", "MaterialConfig", "PhysicsConfig", "GridConfig",
    "InstrumentConfig", "Scatterer", "SpectraConfigError",
    "load", "dump", "validate", "run_spectra",
    # presets/constants
    "VISION_EF_MEV", "VISION_BANKS", "VISION_SIGMA_COEFFS", "C_E", "KB",
]
