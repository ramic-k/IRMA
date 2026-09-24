#!/usr/bin/env python3
"""
IRMA Core Engine facade — thermal scattering law S(alpha, beta) calculator.

Based on the LEAPR module from NJOY2016 with extensions for generalized
elastic scattering (arbitrary crystal structures, ``iel=10`` SEF/MEF) and
phonopy-backed noncubic inelastic workflows (``inelastic_mode=1/2``; one
principal species per run, polyatomic cells supported). Output is an ENDF-6
File 7 tape (MF7/MT2 + MF7/MT4) written via endf-parserpy.

Usage::

    python -m irma <input_file> <output_file>

This module is a stable import facade: the public API (``run_leapr``,
``LeaprResult``, the deck/tokenizer symbols, the kernel and writer helpers)
is re-exported here, while the implementations live in the focused modules —
``irma.core.driver`` (the deck-to-output driver), ``irma.core.deck`` (the
tokenizer/reader), ``irma.core.crystal_cards`` (the ``iel=10`` Card 6b-6g
block), ``irma.core.kernels`` (the classic LEAPR physics), and
``irma.core.endf_writer`` (ENDF-6 serialization).

The authoritative, card-by-card input-deck reference (including the
generalized-elastic Cards 6b-6g, the per-temperature detail block, and two
complete runnable decks) is the manual page ``docs/input-reference.md``
("Input deck reference" on the published docs site). NJOY job streams are
also accepted on input: a ``leapr ... stop`` block is located automatically
and the surrounding modules are ignored.
"""

import json
import os
from pathlib import Path
# Native (BLAS/OpenMP) thread pools are pinned to a single thread, set BEFORE
# numpy is imported, for two measured reasons: (1) the mode-1/2 paths
# parallelize across processes (Card 6f ncpu), and threaded BLAS inside every
# worker oversubscribes the machine by an order of magnitude; (2) a single
# free-threaded process is severalfold slower than the pinned process pool.
# The LEAPR kernels also operate on many small arrays where pool
# synchronization costs more than the parallelism returns, and
# single-threaded reductions keep output bit-reproducible.
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

import numpy as np
from math import sqrt, pi, cos, radians
import sys


from irma.core.noncubic_inelastic import NoncubicInelasticControls

from irma.core.constants import BK, WL2EKIN, _Z_TO_SYMBOL
from irma.core.deck import (
    CARD_END, DeckError,
    parse_leapr_input, TokenReader,
    _read_temperature_detail_cards,
)
from irma.core.kernels import (
    contin, trans, discre, coldh, skold_approx,
    sigfig,
)
from irma.core.crystal import (
    AtomSite, CrystalStructure,
    compute_bragg_edges_general, coher,
)
from irma.core.crystal_cards import _parse_crystal_cards
from irma.core.endf_writer import (
    write_endf_output, _grouped_coherent_s_table,
)

# The LEAPR driver (run_leapr) and its phonopy/MT4 helpers live in
# irma.core.driver; the CLI and the tests import them from this facade.
from irma.core.driver import (
    LeaprResult,
    run_leapr,
    _noncubic_mt4_step,
    _store_directional_species_dw,
)


# ============================================================================
# Entry point
# ============================================================================

if __name__ == '__main__':
    # Delegate to the canonical CLI so running this module directly gets the
    # same friendly DeckError handling and single-sourced usage text as
    # `python -m irma`. main() returns the exit code; propagate it.
    import sys as _sys
    from irma.cli import main as _cli_main
    _sys.exit(_cli_main())
