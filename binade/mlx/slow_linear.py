"""Slow runtime (DESIGN.md 10): rebuild each BF16 weight from Binade v1 arrays with MLX ops,
use it for one matmul and let it go. Weights at rest stay packed.

A packed module keeps the file's arrays under `weight` (weight.raw, .table, .meta, .idx,
.esc), so mlx_lm's loader fills them by name. Offsets the file leaves out (tile word
starts, escape starts) are derived once, on first use, and kept as private arrays.

Memory: decoding runs in row chunks of about CHUNK weights, each evaluated before the
next, and every packed matmul is evaluated where it is called. Decoding does not depend
on the activations, so a lazy graph over a whole forward pass would decode every layer
before the first matmul and hold all decoded weights at once; the embedding alone needs
~18 GB of intermediates when decoded in one piece (measured on Gemma 4 E4B).
"""

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from ..format import ESCAPE, TILE

PARTS = ("raw", "table", "meta", "idx", "esc")
CHUNK = 1 << 24  # weights per decode chunk


class _Chunk:
    """Tiles of whole rows grouped by width, with their word starts."""

    def __init__(self, b: np.ndarray, esc_tile: np.ndarray, start: np.ndarray):
        self.n = b.size
        groups, order = [], []
        for width in np.unique(b):
            sel = np.flatnonzero(b == width)
            order.append(sel)
            if width == 0:
                groups.append((0, sel.size, None, None))
                continue
            pos = np.arange(TILE) * width
            lo, sh = pos >> 5, pos & 31
            lanes = (
                mx.array(lo.astype(np.int32)),
                mx.array(np.minimum(lo + 1, 4 * width - 1).astype(np.int32)),
                mx.array(sh.astype(np.uint32)),
                mx.array(sh + width > 32),
            )
            groups.append((int(width), sel.size, mx.array(start[sel].astype(np.int32)), lanes))
        self.groups = groups
        self.inv = mx.array(np.argsort(np.concatenate(order), kind="stable").astype(np.int32))
        self.has_esc = bool(esc_tile.any())
        if self.has_esc:
            self.mode = mx.array(esc_tile)
            self.code = mx.array(((1 << b) - 1).astype(np.uint8))
            self.eoff = None  # set by _Plan once escape counts are known

    def codes(self, idx: mx.array) -> mx.array:
        parts = []
        for width, n, start, lanes in self.groups:
            if width == 0:
                parts.append(mx.zeros((n, TILE), dtype=mx.uint8))
                continue
            lo_w, hi_w, sh, straddle = lanes
            wv = idx[start[:, None] + mx.arange(4 * width, dtype=mx.int32)[None, :]]
            lo = mx.take(wv, lo_w, axis=1) >> sh
            hi = mx.take(wv, hi_w, axis=1) << ((32 - sh) & 31)
            parts.append((mx.where(straddle, lo | hi, lo) & ((1 << width) - 1)).astype(mx.uint8))
        return mx.concatenate(parts)[self.inv]

    def hits(self, codes: mx.array) -> mx.array:
        return self.mode[:, None] & (codes == self.code[:, None])

    def exponents(self, w: dict) -> mx.array:
        codes = self.codes(w["idx"])
        e = w["table"][codes]
        if self.has_esc:
            hit = self.hits(codes)
            pos = self.eoff[:, None] + mx.cumsum(hit.astype(mx.int32), axis=1) - 1
            e = mx.where(hit, w["esc"][mx.maximum(pos, 0)], e)
        return e


class _Plan:
    def __init__(self, w: dict):
        raw = w["raw"]
        self.K = raw.shape[-1]
        self.R = raw.size // self.K
        meta = np.array(w["meta"])
        self.per_row = meta.size // self.R
        self.rows = max(1, CHUNK // (self.per_row * TILE))  # rows per chunk
        b = (meta & 0x0F).astype(np.int64)
        esc_tile = ((meta & ESCAPE) != 0) & (b >= 1)
        start = 4 * (np.cumsum(b) - b)
        self.chunks = []
        for r0 in range(0, self.R, self.rows):
            t0, t1 = r0 * self.per_row, min(self.R, r0 + self.rows) * self.per_row
            self.chunks.append(_Chunk(b[t0:t1], esc_tile[t0:t1], start[t0:t1]))
        base = 0
        for c in self.chunks:
            if c.has_esc:
                counts = np.array(c.hits(c.codes(w["idx"])).astype(mx.int32).sum(axis=1)).astype(np.int64)
                c.eoff = mx.array((base + np.cumsum(counts) - counts).astype(np.int32))
                base += int(counts.sum())

    def rows_bf16(self, w: dict, i: int) -> mx.array:
        """BF16 rows of chunk i, [rows, K], evaluated."""
        r0 = i * self.rows
        r1 = min(self.R, r0 + self.rows)
        e = self.chunks[i].exponents(w).reshape(r1 - r0, -1)[:, : self.K].astype(mx.uint16)
        r = w["raw"].reshape(self.R, self.K)[r0:r1].astype(mx.uint16)
        out = (((r & 0x80) << 8) | (e << 7) | (r & 0x7F)).view(mx.bfloat16)
        mx.eval(out)
        return out


def _plan(module) -> _Plan:
    if "bits" in module.weight:
        raise RuntimeError("Binade format 2 (Rice) weights run on the R4 runtime (binade.mlx.r4.load), not the slow runtime")
    if module._plan is None:
        module._plan = _Plan(module.weight)
    return module._plan


def dense(module) -> mx.array:
    """The module's BF16 weight, rebuilt from its packed arrays."""
    p = _plan(module)
    rows = [p.rows_bf16(module.weight, i) for i in range(len(p.chunks))]
    return (rows[0] if len(rows) == 1 else mx.concatenate(rows)).reshape(module.weight["raw"].shape)


def _placeholders(spec: dict) -> dict:
    return {k: mx.zeros(shape, dtype=getattr(mx, dtype)) for k, (shape, dtype) in spec.items()}


class BinadeLinear(nn.Module):
    """Drop-in for nn.Linear: y = x W^T (+ b), W decoded per call."""

    def __init__(self, spec: dict, bias: bool = False):
        super().__init__()
        self.weight = _placeholders(spec)
        if bias:
            self.bias = mx.zeros((spec["raw"][0][0],), dtype=mx.bfloat16)
        self._plan = None

    def __call__(self, x: mx.array) -> mx.array:
        w = dense(self)
        y = mx.addmm(self["bias"], x, w.T) if "bias" in self else x @ w.T
        mx.eval(y)
        return y


class BinadeEmbedding(nn.Module):
    """Drop-in for nn.Embedding: lookups decode only the chunks holding the requested rows;
    the tied output projection (as_linear) decodes the whole table."""

    def __init__(self, spec: dict):
        super().__init__()
        self.weight = _placeholders(spec)
        self._plan = None

    def __call__(self, ids: mx.array) -> mx.array:
        p = _plan(self)
        flat = np.array(ids).reshape(-1)
        need = np.unique(flat // p.rows)
        block = mx.concatenate([p.rows_bf16(self.weight, int(c)) for c in need])
        offset = {int(c): j * p.rows for j, c in enumerate(need)}  # every chunk but the last is full
        local = np.array([offset[int(t // p.rows)] + int(t % p.rows) for t in flat], np.int32)
        y = block[mx.array(local)].reshape(*ids.shape, p.K)
        mx.eval(y)
        return y

    def as_linear(self, x: mx.array) -> mx.array:
        y = x @ dense(self).T
        mx.eval(y)
        return y
