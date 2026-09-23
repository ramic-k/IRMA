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

# Stable top-level public API: `from irma import run_leapr, DeckError`.
from irma.core.deck import DeckError, parse_leapr_input  # noqa: E402
from irma.core.driver import LeaprResult, run_leapr  # noqa: E402

__all__ = ["run_leapr", "LeaprResult", "parse_leapr_input", "DeckError",
           "__version__", "__author__"]
