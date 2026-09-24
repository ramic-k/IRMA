"""
IRMA Core Engine facade — thermal scattering law S(alpha, beta) calculator.

Based on the LEAPR module from NJOY2016 with extensions for generalized
elastic scattering (arbitrary crystal structures, ``iel=10`` SEF/MEF) and
phonopy-backed noncubic inelastic workflows (``inelastic_mode=1/2``; one
principal species per run, polyatomic cells supported). Output is an ENDF-6
File 7 tape (MF7/MT2 + MF7/MT4) written via endf-parserpy.

Usage::

    python -m irma <input_file> <output_file>

This module re-exports the public API (``run_leapr``, ``LeaprResult``,
``DeckError``, ``parse_leapr_input``, ``TokenReader``); the implementations
live in the focused modules —
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

from irma.core.deck import DeckError, TokenReader, parse_leapr_input
from irma.core.driver import LeaprResult, run_leapr

__all__ = ["DeckError", "LeaprResult", "TokenReader", "parse_leapr_input",
           "run_leapr"]


if __name__ == '__main__':
    # The same DeckError handling and usage text as `python -m irma`.
    import sys
    from irma.cli import main
    sys.exit(main())
