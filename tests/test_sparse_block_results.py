"""Sparse worker block results must accumulate exactly like dense adds.

Pool workers compress block deposits below a density cutoff to
(index, value) pairs (compress_block_result) and the parent adds them
back (accumulate_block_result). The contract is bitwise equality with
the historical dense ``result += partial``: same values, same cells,
same block order. These tests pin the protocol — roundtrip equality,
the dense fallback above the cutoff, all-zero blocks, stacked
multiphonon-style shapes, and the shape-mismatch guard.
"""
import numpy as np
import pytest

from irma.core.noncubic_engine import (
    _SPARSE_BLOCK_TAG,
    _SPARSE_BLOCK_DENSITY_CUTOFF,
    accumulate_block_result,
    compress_block_result,
)


def _sparse_random(shape, density, seed):
    rng = np.random.default_rng(seed)
    arr = np.zeros(shape, dtype=float)
    flat = arr.reshape(-1)
    n = max(1, int(flat.size * density))
    idx = rng.choice(flat.size, size=n, replace=False)
    flat[idx] = rng.standard_normal(n)  # negative values included
    return arr


def test_sparse_roundtrip_is_bitwise_identical():
    block = _sparse_random((3, 40, 50), density=0.01, seed=1)
    compressed = compress_block_result(block.copy())
    assert isinstance(compressed, tuple)
    assert compressed[0] == _SPARSE_BLOCK_TAG

    dense_acc = np.zeros((3, 40, 50))
    sparse_acc = np.zeros((3, 40, 50))
    dense_acc += block
    accumulate_block_result(sparse_acc, compressed)
    assert np.array_equal(dense_acc, sparse_acc)
    # repeated accumulation in block order stays bitwise identical
    dense_acc += block
    accumulate_block_result(sparse_acc, compressed)
    assert np.array_equal(dense_acc, sparse_acc)


def test_dense_results_above_cutoff_pass_through_unchanged():
    block = _sparse_random((20, 30), density=0.5, seed=2)
    assert (np.count_nonzero(block)
            > block.size * _SPARSE_BLOCK_DENSITY_CUTOFF)
    out = compress_block_result(block)
    assert out is block                     # no copy, no tuple
    acc = np.zeros((20, 30))
    accumulate_block_result(acc, out)
    assert np.array_equal(acc, block)


def test_all_zero_block_ships_empty_and_adds_nothing():
    compressed = compress_block_result(np.zeros((2, 10, 10)))
    assert isinstance(compressed, tuple)
    assert compressed[2].size == 0          # no indices
    acc = np.full((2, 10, 10), 7.5)
    accumulate_block_result(acc, compressed)
    assert np.array_equal(acc, np.full((2, 10, 10), 7.5))


def test_mixed_sparse_and_dense_blocks_match_dense_sum():
    shapes = (5, 17, 23)
    blocks = [
        _sparse_random(shapes, density=0.005, seed=3),   # -> sparse
        _sparse_random(shapes, density=0.4, seed=4),     # -> dense
        np.zeros(shapes),                                # -> empty sparse
        _sparse_random(shapes, density=0.02, seed=5),    # -> sparse
    ]
    dense_acc = np.zeros(shapes)
    proto_acc = np.zeros(shapes)
    for block in blocks:
        dense_acc += block
        accumulate_block_result(proto_acc, compress_block_result(block.copy()))
    assert np.array_equal(dense_acc, proto_acc)


def test_shape_mismatch_raises():
    compressed = compress_block_result(np.zeros((4, 4)))
    with pytest.raises(ValueError, match="does not match accumulator"):
        accumulate_block_result(np.zeros((4, 5)), compressed)


def test_non_contiguous_accumulator_raises_for_sparse_blocks():
    """reshape(-1) on a non-C-contiguous accumulator returns a COPY, so the
    sparse fancy-index add would silently drop every deposit (F43). The
    sparse path must refuse such accumulators; the dense path still works
    in-place on the same view."""
    block = _sparse_random((4, 5), density=0.05, seed=6)
    compressed = compress_block_result(block.copy())
    assert isinstance(compressed, tuple)

    view = np.zeros((4, 10))[:, :5]      # shape matches, not C-contiguous
    assert view.shape == block.shape
    assert not view.flags['C_CONTIGUOUS']
    with pytest.raises(ValueError, match="C-contiguous"):
        accumulate_block_result(view, compressed)
    assert np.count_nonzero(view) == 0   # nothing landed in the view

    # dense blocks accumulate correctly on the very same view (+= is in-place)
    accumulate_block_result(view, block)
    assert np.array_equal(view, block)

    # a contiguous accumulator takes the sparse path as before
    acc = np.zeros((4, 5))
    accumulate_block_result(acc, compressed)
    assert np.array_equal(acc, block)


def test_nan_and_inf_cells_are_preserved():
    block = np.zeros((6, 6))
    block[1, 2] = np.nan
    block[3, 4] = np.inf
    compressed = compress_block_result(block)
    acc = np.zeros((6, 6))
    accumulate_block_result(acc, compressed)
    assert np.isnan(acc[1, 2]) and np.isinf(acc[3, 4])
    assert np.count_nonzero(np.nan_to_num(acc)) == 1
