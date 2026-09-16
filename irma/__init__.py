"""IRMA — (In)elastic Representation of Materials As S(alpha,beta) evaluations.

A standalone tool for generating thermal neutron scattering law S(alpha, beta)
libraries in ENDF-6 format using the phonon expansion method.

Supports:
  - Continuous phonon frequency distributions
  - Generalized coherent elastic (Bragg edges) for arbitrary crystals
  - Phonopy-backed hybrid inelastic_mode paths for noncubic MT4 generation
  - Free-gas and diffusion translational modes
  - Discrete oscillators
  - Cold hydrogen/deuterium (ortho/para)
  - Automatic alpha/beta grid generation
  - Neutron-scattering forward model (irma.spectra): powder S(Q,E) ->
    instrument-resolved 1-D spectrum for VISION / indirect / direct geometries,
    with an auto chopper-resolution model for the 8 PyChop direct instruments
"""

__version__ = "1.0.3"
__author__ = "IRMA developers"

# Pin native BLAS/OMP thread pools to 1 at package import, before numpy can be
# loaded through irma. The mode-1/2 engine parallelizes across SPAWNED worker
# processes (get_context("spawn")); a multithreaded BLAS in the parent or the
# workers oversubscribes the cores under the process pool (measured >10x
# slower). Doing it here covers the common `import irma` / CLI entry;
# setdefault preserves a deliberate caller override (a host that imports numpy
# before irma should set these itself, or accept threaded BLAS).
import os as _os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
    _os.environ.setdefault(_v, "1")

# Stable top-level public API. Importing these from `irma` directly
# (`from irma import run_leapr, DeckError`) decouples callers from the internal
# module layout, so the core modules can be refactored without breaking them.
# Exposed LAZILY (PEP 562 __getattr__) so `import irma` stays light -- the heavy
# engine is only imported when one of these names is actually accessed, and the
# light core (numpy + endf-parserpy) install never pulls an optional extra.
__all__ = ["run_leapr", "LeaprResult", "parse_leapr_input", "DeckError",
           "__version__", "__author__"]

# public name -> "module:attr" it is loaded from on first access
_PUBLIC_API = {
    "run_leapr": "irma.core.engine:run_leapr",
    "LeaprResult": "irma.core.driver:LeaprResult",
    "parse_leapr_input": "irma.core.deck:parse_leapr_input",
    "DeckError": "irma.core.deck:DeckError",
}


def __getattr__(name):
    """Lazily resolve the public API symbols (PEP 562)."""
    target = _PUBLIC_API.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib
    module_name, attr = target.split(":")
    value = getattr(importlib.import_module(module_name), attr)
    globals()[name] = value          # cache so later access skips __getattr__
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
