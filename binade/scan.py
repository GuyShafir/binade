"""Per-tensor scan for the phase 1 histogram (DESIGN.md 5).

For one BF16 weight, read in row chunks straight from the safetensors file:
  - exponent histogram
  - per-tile policy costs for T in `tile_sizes` (palette and rank families), on three
    exponent layouts:
      real         as stored
      shuf_row     exponents permuted within each row (keeps per-row distributions)
      shuf_tensor  exponents permuted within the whole tensor (the shuffle control, DESIGN.md 5)
  - 64-weight row segments covered by the tensor's 16 most frequent exponents (Unweight-class)
  - lossy dial histograms (real layout only)
Shuffles are seeded from (seed, tensor name), so results do not depend on scan order.
"""

import time
import zlib

import mlx.core as mx
import numpy as np

from . import lossy, st
from .bf16 import exponent
from .policies import (
    CHOICES,
    OBJECTIVES,
    PALETTE_POLICIES,
    POLICIES,
    RANK_POLICIES,
    allowed,
    best_codes,
    candidates,
    rank_candidates,
    select,
)
from .tile import tile_counts, tiles

VARIANTS = ("real", "shuf_row", "shuf_tensor")
CHUNK = 1 << 25  # elements per GPU batch
MLX_SHUFFLE_MAX = 1 << 27  # larger tensors are shuffled in place with numpy (no index array)
SEGMENT = 64  # Unweight checks its palette per 64-weight tile row


def _onehot_sum(x: mx.array, n: int) -> mx.array:
    return (x[:, None] == mx.arange(n, dtype=x.dtype)[None, :]).astype(mx.int64).sum(axis=0)


class TileAccum:
    """Exact integer sums of per-tile quantities for one (layout, T)."""

    def __init__(self, T: int):
        self.T = T
        self.ntiles = 0
        self.khist = np.zeros(T + 1, np.int64)
        # bits[objective][policy] = [idx, pal, esc]
        self.bits = {o: {p: np.zeros(3, np.int64) for p in POLICIES} for o in OBJECTIVES}
        self.choice = {o: np.zeros(len(CHOICES), np.int64) for o in OBJECTIVES}

    def add(self, e2d: mx.array) -> None:
        """Palette-family policies, k histogram and tile count from exponents [R, K]."""
        T = self.T
        for t in tiles(e2d, T):
            if t is None:
                continue
            nt, n = t.shape
            k, c = tile_counts(t)
            cands = candidates(k, c, n, T)
            sums, choices = [], []
            for o in OBJECTIVES:
                for p in PALETTE_POLICIES:
                    choice, i, pa, e = select(cands, allowed(p, k), o)
                    sums += [i.astype(mx.int64).sum(), pa.astype(mx.int64).sum(), e.astype(mx.int64).sum()]
                    if p == "best":
                        choices.append(_onehot_sum(best_codes(choice, k), len(CHOICES)))
            sums = mx.stack(sums)
            kh = _onehot_sum(k, n + 1)
            mx.eval(sums, kh, *choices)
            s = np.array(sums).reshape(len(OBJECTIVES), len(PALETTE_POLICIES), 3)
            for a, o in enumerate(OBJECTIVES):
                for b, p in enumerate(PALETTE_POLICIES):
                    self.bits[o][p] += s[a, b]
                self.choice[o] += np.array(choices[a])
            self.khist[: n + 1] += np.array(kh)
            self.ntiles += nt

    def add_ranks(self, r2d: mx.array) -> None:
        """Rank-family policies from per-tensor frequency ranks [R, K] (no palette, so p16 == actual)."""
        T = self.T
        for t in tiles(r2d, T):
            if t is None:
                continue
            nt, n = t.shape
            cands = rank_candidates(t, n, T)
            ones = mx.ones((nt,), dtype=mx.int32)
            sums = []
            for p in RANK_POLICIES:
                _, i, pa, e = select(cands, allowed(p, ones), "actual")
                sums += [i.astype(mx.int64).sum(), pa.astype(mx.int64).sum(), e.astype(mx.int64).sum()]
            s = np.array(mx.stack(sums)).reshape(len(RANK_POLICIES), 3)
            for b, p in enumerate(RANK_POLICIES):
                for o in OBJECTIVES:
                    self.bits[o][p] += s[b]

    def merge(self, other: "TileAccum") -> None:
        self.ntiles += other.ntiles
        self.khist += other.khist
        for o in OBJECTIVES:
            self.choice[o] += other.choice[o]
            for p in POLICIES:
                self.bits[o][p] += other.bits[o][p]

    def bits_total(self, numel: int, policy: str, objective: str = "actual", with_off: bool = False) -> int:
        """Bit accounting (DESIGN.md 4): raw + idx + pal + esc + meta (+ off)."""
        idx, pal, esc = (int(x) for x in self.bits[objective][policy])
        return 8 * numel + idx + pal + esc + 8 * self.ntiles + (32 * self.ntiles if with_off else 0)


class LossyAccum:
    """Histograms of per-tile lossy errors, per (T, m, metric)."""

    def __init__(self, tile_sizes):
        self.h = {T: {m: {k: np.zeros(lossy.NBINS, np.int64) for k in lossy.METRICS} for m in lossy.MANTISSA_BITS} for T in tile_sizes}
        self.max = {T: {m: {k: 0.0 for k in lossy.METRICS} for m in lossy.MANTISSA_BITS} for T in tile_sizes}

    def add(self, w16: mx.array) -> None:
        for m in lossy.MANTISSA_BITS:
            d, a = lossy.elementwise(w16, m)
            for T in self.h:
                vals = lossy.tile_errors(d, a, T)
                mx.eval(*vals)
                for key, v in zip(lossy.METRICS, vals):
                    v = np.array(v)
                    self.h[T][m][key] += lossy.histogram(v)
                    if np.any(~np.isnan(v)):
                        self.max[T][m][key] = max(self.max[T][m][key], float(np.nanmax(v)))

    def merge(self, other: "LossyAccum") -> None:
        for T in self.h:
            for m in self.h[T]:
                for k in lossy.METRICS:
                    self.h[T][m][k] += other.h[T][m][k]
                    self.max[T][m][k] = max(self.max[T][m][k], other.max[T][m][k])


def tensor_seed(seed: int, name: str) -> int:
    return (seed * 1_000_003 + zlib.crc32(name.encode())) % (1 << 31)


def scan_tensor(info: st.TensorInfo, tile_sizes=(32, 64, 128), seed: int = 0, do_lossy: bool = True, chunk: int = CHUNK) -> dict:
    t0 = time.perf_counter()
    R, K = info.rows, info.shape[-1]
    step = max(1, chunk // K)
    spans = [(r, min(R, r + step)) for r in range(0, R, step)]
    acc = {v: {T: TileAccum(T) for T in tile_sizes} for v in VARIANTS}
    lacc = LossyAccum(tile_sizes) if do_lossy else None
    exps = np.empty((R, K), np.uint8)
    hist = np.zeros(256, np.int64)

    # pass 1: real layout (palette family), lossy dial, histogram
    for r0, r1 in spans:
        w = mx.array(st.read(info, (r0, r1)))
        e = exponent(w)
        for T in tile_sizes:
            acc["real"][T].add(e)
        if lacc:
            lacc.add(w)
        e = np.array(e)
        exps[r0:r1] = e
        hist += np.bincount(e.ravel(), minlength=256)

    # frequency ranks (0 = most common) and the top-16 palette, from the tensor histogram
    order = np.argsort(-hist, kind="stable")
    rank = np.empty(256, np.uint8)
    rank[order] = np.arange(256)
    rank = mx.array(rank)
    in16 = np.zeros(256, np.bool_)
    in16[order[:16][hist[order[:16]] > 0]] = True
    in16 = mx.array(in16)

    # pass 2: real layout (rank family), row shuffle (both families), Unweight segments
    key = mx.random.key(tensor_seed(seed, info.name))
    seg_in_weights, segments = 0, 0
    for r0, r1 in spans:
        e = mx.array(exps[r0:r1])
        key, sub = mx.random.split(key)
        perm = mx.argsort(mx.random.randint(0, (1 << 31) - 1, shape=e.shape, key=sub), axis=1)
        er = mx.take_along_axis(e, perm, axis=1)
        for T in tile_sizes:
            acc["real"][T].add_ranks(rank[e])
            acc["shuf_row"][T].add(er)
            acc["shuf_row"][T].add_ranks(rank[er])
        for seg in tiles(in16[e], SEGMENT):
            if seg is not None:
                seg_in_weights += int(seg.all(axis=1).sum().item()) * seg.shape[1]
                segments += seg.shape[0]

    # pass 3: tensor shuffle (overwrites exps), both families
    flat = exps.reshape(-1)
    if flat.size <= MLX_SHUFFLE_MAX:
        key, sub = mx.random.split(key)
        flat[:] = np.array(mx.array(flat)[mx.random.permutation(flat.size, key=sub)])
    else:
        np.random.default_rng(tensor_seed(seed, info.name)).shuffle(flat)
    for r0, r1 in spans:
        e = mx.array(exps[r0:r1])
        for T in tile_sizes:
            acc["shuf_tensor"][T].add(e)
            acc["shuf_tensor"][T].add_ranks(rank[e])

    # the rank table itself: 8 bits per distinct exponent, once per tensor
    distinct = int((hist > 0).sum())
    for v in VARIANTS:
        for T in tile_sizes:
            for o in OBJECTIVES:
                for p in RANK_POLICIES:
                    acc[v][T].bits[o][p][1] += 8 * distinct

    return {
        "name": info.name,
        "shape": list(info.shape),
        "rows": R,
        "k": K,
        "numel": info.numel,
        "hist": hist,
        "seg_in_weights": seg_in_weights,
        "segments": segments,
        "tiles": acc,
        "lossy": lacc,
        "seconds": time.perf_counter() - t0,
    }
