"""Cross-process sparse worker IPC round-trip.

``test_sparse_block_results.py`` pins the
``compress_block_result``/``accumulate_block_result`` protocol thoroughly
but ENTIRELY in-process: it never sends a compressed ``(tag, shape,
index, value)`` tuple across a fork boundary, so the actual contract —
that the tuple pickles/un-pickles correctly across processes and the
parent re-accumulates worker tuples in fixed block order to a bitwise
sum — had only indirect, byte-identity-gated coverage via the phonopy-
gated noncubic e2e test (which silently skips wholesale without phonopy,
Codex C9).

This is a deterministic, phonopy-free check of the IPC contract: it
compresses synthetic numpy blocks in real worker processes via a fork
``ProcessPoolExecutor`` (the exact start method the production pool uses)
and accumulates them in the parent in fixed block order, asserting
BITWISE equality with the in-process dense reduction. A plain
``pickle.loads(pickle.dumps(...))`` round-trip is also asserted so the
contract holds independent of BLAS order and core count.
"""
import multiprocessing as mp
import pickle

import numpy as np

from irma.core.noncubic_engine import (
    _SPARSE_BLOCK_TAG,
    accumulate_block_result,
    compress_block_result,
)


def _sparse_random(shape, density, seed):
    """A mostly-zero block (worker block deposits are sparse on tiny grids)."""
    rng = np.random.default_rng(seed)
    arr = np.zeros(shape, dtype=float)
    flat = arr.reshape(-1)
    n = max(1, int(flat.size * density))
    idx = rng.choice(flat.size, size=n, replace=False)
    flat[idx] = rng.standard_normal(n)              # negative values included
    return arr


# Mix of densities so both the compressed-tuple and dense-passthrough
# branches cross the fork boundary; fixed seeds keep blocks deterministic.
_SHAPE = (3, 40, 50)
_BLOCK_SPECS = [
    (_SHAPE, 0.005, 11),    # -> sparse tuple
    (_SHAPE, 0.4, 12),      # -> dense passthrough (above density cutoff)
    (_SHAPE, 0.0, 13),      # -> empty sparse tuple
    (_SHAPE, 0.02, 14),     # -> sparse tuple
]


def _make_blocks():
    return [_sparse_random(shape, dens, seed)
            for (shape, dens, seed) in _BLOCK_SPECS]


def _compress_worker(spec):
    """Top-level (picklable) worker: rebuild the block in the child and
    return its compressed result, exactly as the production pool does."""
    shape, dens, seed = spec
    return compress_block_result(_sparse_random(shape, dens, seed))


def _dense_reference():
    """In-process dense ``result += block`` reduction (the contract's
    historical baseline)."""
    ref = np.zeros(_SHAPE)
    for block in _make_blocks():
        ref += block
    return ref


def test_pickle_roundtrip_of_compressed_block_is_bitwise_identical():
    """The compressed tuple must survive a raw pickle round-trip and
    accumulate identically to the dense add — independent of any pool."""
    acc = np.zeros(_SHAPE)
    ref = np.zeros(_SHAPE)
    for spec in _BLOCK_SPECS:
        block = _sparse_random(*spec)
        ref += block
        wire = pickle.loads(pickle.dumps(compress_block_result(block.copy())))
        accumulate_block_result(acc, wire)
    assert np.array_equal(acc, ref)


def test_spawn_pool_accumulation_matches_in_process_dense_sum():
    """Compress blocks in real spawn worker processes (the production pool's
    start method), accumulate in the parent in fixed block order, and require
    bitwise equality with the in-process dense reduction (the production
    pool's contract)."""
    import concurrent.futures as cf
    ctx = mp.get_context("spawn")

    acc = np.zeros(_SHAPE)
    saw_sparse_tuple = False
    with cf.ProcessPoolExecutor(max_workers=2, mp_context=ctx) as pool:
        # Ordered map -> fixed block order, matching the production pool.
        for partial in pool.map(_compress_worker, _BLOCK_SPECS):
            if isinstance(partial, tuple) and partial and partial[0] == _SPARSE_BLOCK_TAG:
                saw_sparse_tuple = True
            accumulate_block_result(acc, partial)

    assert saw_sparse_tuple, "no compressed tuple crossed the process boundary"
    assert np.array_equal(acc, _dense_reference())
