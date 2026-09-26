"""Tiling and per-tile exponent statistics (DESIGN.md 2).

A tile is T contiguous weights along K within one row. Rows are padded to a multiple
of T; statistics count only the real weights, while index costs charge the padded
width (see policies.py). Tiles never cross rows, so any row range can be processed
independently.
"""

import mlx.core as mx
import numpy as np


def tiles(e2d, T: int):
    """Split a [R, K] array into ([R*(K//T), T] full tiles, [R, K%T] partial tiles or None)."""
    R, K = e2d.shape
    full = (K // T) * T
    main = e2d[:, :full].reshape(R * (K // T), T) if full else None
    rem = e2d[:, full:] if full < K else None
    return main, rem


def tile_counts(t: mx.array) -> tuple[mx.array, mx.array]:
    """Per-tile exponent frequencies.

    t: [nt, n] integer array (one tile per row).
    Returns k [nt] int32 (distinct values per tile) and c [nt, n] int32, the value
    counts of each tile sorted in descending order and zero padded.
    """
    nt, n = t.shape
    s = mx.sort(t, axis=1)
    start = mx.concatenate([mx.ones((nt, 1), dtype=mx.bool_), s[:, 1:] != s[:, :-1]], axis=1)
    pos = mx.arange(n, dtype=mx.int32)
    nxt = mx.cummin(mx.where(start, pos, n), axis=1, reverse=True, inclusive=False)
    run = mx.where(start, mx.minimum(nxt, n) - pos, 0)
    c = mx.sort(run, axis=1)[:, ::-1]
    k = start.astype(mx.int32).sum(axis=1)
    return k, c


def tile_counts_ref(t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Slow per-tile reference for tile_counts (tests only)."""
    nt, n = t.shape
    k = np.zeros(nt, np.int32)
    c = np.zeros((nt, n), np.int32)
    for i in range(nt):
        cnt = np.sort(np.unique(t[i], return_counts=True)[1])[::-1]
        k[i] = len(cnt)
        c[i, : len(cnt)] = cnt
    return k, c
