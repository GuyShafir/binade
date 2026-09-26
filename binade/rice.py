"""Binade format v2 exponent coding: Rice codes over per-tensor frequency ranks.
numpy reference encoder and decoder.

Same rank table as v1 (format.py). Tiles are T = 128 weights along K in one row; the
last tile of a row holds n = K - 128 (T - 1) <= 128 weights and codes only those. Per
tile, the packer picks the Rice parameter k in 0..3 minimizing the tile's bits (ties to
the smaller k); q = rank >> k.
  meta   k | qmax << 2, qmax = the tile's largest q (at most 63, which k >= 2 always meets)
  bits   per tile, contiguous:
           low plane    n * k bits, bit b of weight i's rank at i * k + b
           unary planes j = 0 .. qmax - 1: one bit per weight with q >= j, in weight
                        order, 1 if q > j. A weight with q == qmax has no closing 0.
         A tile costs n k + sum(min(q + 1, qmax)) bits, the phase 1 rank_rice accounting.
         Rows start on a 32-bit word boundary. Bits are packed LSB first into
         little-endian uint32 words.
  roff   uint32 per row, the row's first word, so rows decode independently.

Plane j's length is the number of weights alive after plane j - 1 (a prefix count, the
decode of DESIGN.md 7), so tiles in a row decode in sequence. That suits a
load-time transcode to R4 (binade/mlx/rice_kernel.py), not the matvec kernel.
"""

from dataclasses import dataclass

import numpy as np

from .format import TILE

KMAX = 3
QMAX = 63
_BIG = np.iinfo(np.int64).max


@dataclass
class RiceEncoded:
    table: np.ndarray  # uint8 [256]
    meta: np.ndarray  # uint8 [ntiles]
    bits: np.ndarray  # uint32 [words]
    roff: np.ndarray  # uint32 [rows], first word of each row, relative to bits[0]

    def nbits(self) -> int:
        """Exponent-side payload in bits (excludes the raw sign+mantissa bytes)."""
        return 8 * self.table.size + 8 * self.meta.size + 32 * self.bits.size + 32 * self.roff.size


def tiles_per_row(K: int, T: int = TILE) -> int:
    return -(-K // T)


def _valid(R: int, K: int, T: int) -> np.ndarray:
    """[R * ceil(K/T), T] mask of real (not padding) weights."""
    Tn = tiles_per_row(K, T)
    row = (np.arange(Tn * T) < K).reshape(Tn, T)
    return np.broadcast_to(row, (R, Tn, T)).reshape(-1, T)


def choose(r: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-tile (k, qmax, bits) for ranks r [nt, T] (padding weights hold rank 0)."""
    n = valid.sum(axis=1, dtype=np.int64)
    best_k = best_q = best_bits = None
    for k in range(KMAX + 1):
        q = r >> k
        qmax = q.max(axis=1).astype(np.int64)
        bits = k * n + (np.minimum(q + 1, qmax[:, None]) * valid).sum(axis=1, dtype=np.int64)
        bits = np.where(qmax <= QMAX, bits, _BIG)
        if best_bits is None:
            best_k, best_q, best_bits = np.zeros_like(qmax), qmax, bits
        else:
            better = bits < best_bits
            best_k = np.where(better, k, best_k)
            best_q = np.where(better, qmax, best_q)
            best_bits = np.where(better, bits, best_bits)
    return best_k, best_q, best_bits


def encode_ranks(ranks: np.ndarray, T: int = TILE) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """[R, K] ranks -> (meta [R * ceil(K/T)] uint8, words uint32, roff [R] uint32)."""
    R, K = ranks.shape
    Tn = tiles_per_row(K, T)
    r = np.zeros((R, Tn * T), np.int32)
    r[:, :K] = ranks
    r = r.reshape(-1, T)
    valid = _valid(R, K, T)
    n = valid.sum(axis=1, dtype=np.int64)
    k, qmax, tbits = choose(r, valid)

    tb = tbits.reshape(R, Tn)
    row_words = (tb.sum(axis=1) + 31) >> 5
    roff = np.cumsum(row_words) - row_words
    start = (32 * roff[:, None] + np.cumsum(tb, axis=1) - tb).reshape(-1)
    bits = np.zeros(32 * int(row_words.sum()), np.uint8)

    for kk in range(1, KMAX + 1):
        sel = np.flatnonzero(k == kk)
        if sel.size:
            t_i, w_i = np.nonzero(valid[sel])
            v = r[sel[t_i], w_i]
            p = start[sel[t_i]] + w_i * kk
            for b in range(kk):
                bits[p + b] = (v >> b) & 1

    q = r >> k[:, None].astype(np.int32)
    tiles = np.flatnonzero(qmax > 0)
    offs = start[tiles] + k[tiles] * n[tiles]
    qq, vv, qm = q[tiles], valid[tiles], qmax[tiles]
    j = 0
    while tiles.size:
        alive = vv & (qq >= j)
        idx = np.cumsum(alive, axis=1) - 1
        t_i, w_i = np.nonzero(alive)
        bits[offs[t_i] + idx[t_i, w_i]] = qq[t_i, w_i] > j
        offs = offs + alive.sum(axis=1)
        j += 1
        keep = qm > j
        tiles, offs, qq, vv, qm = tiles[keep], offs[keep], qq[keep], vv[keep], qm[keep]

    words = np.packbits(bits, bitorder="little").view("<u4")
    meta = (k | (qmax << 2)).astype(np.uint8)
    return meta, words, roff.astype(np.uint32)


def encode(exps: np.ndarray, table: np.ndarray, rank: np.ndarray, T: int = TILE) -> RiceEncoded:
    """Encode [R, K] uint8 exponents (any row range of a tensor) with the tensor's rank table."""
    meta, words, roff = encode_ranks(rank[exps], T)
    return RiceEncoded(table=table, meta=meta, bits=words, roff=roff)


def decode_ranks(meta: np.ndarray, words: np.ndarray, roff: np.ndarray, K: int, T: int = TILE) -> tuple[np.ndarray, np.ndarray]:
    """Decode rows -> (ranks [R, K] int32, end [R] int64, each row's bit after its last tile).

    roff indexes `words`; meta holds the rows' tiles in order."""
    R = roff.size
    Tn = tiles_per_row(K, T)
    meta = meta.reshape(R, Tn)
    k = (meta & 3).astype(np.int64)
    qmax = (meta >> 2).astype(np.int64)
    bits = np.unpackbits(np.ascontiguousarray(words, dtype="<u4").view(np.uint8), bitorder="little")
    cursor = 32 * roff.astype(np.int64)
    ranks = np.zeros((R, Tn, T), np.int32)
    for c in range(Tn):
        n = min(T, K - c * T)
        kc, qc = k[:, c], qmax[:, c]
        low = np.zeros((R, T), np.int32)
        for kk in range(1, KMAX + 1):
            sel = np.flatnonzero(kc == kk)
            if sel.size:
                p = cursor[sel][:, None] + np.arange(n) * kk
                v = np.zeros(p.shape, np.int32)
                for b in range(kk):
                    v |= bits[p + b].astype(np.int32) << b
                low[sel, :n] = v
        cursor = cursor + kc * n

        q = np.zeros((R, T), np.int32)
        act = np.flatnonzero(qc > 0)
        cur = cursor[act]
        alive = np.zeros((act.size, T), bool)
        alive[:, :n] = True
        j = 0
        while act.size:
            idx = np.cumsum(alive, axis=1) - 1
            a_i, w_i = np.nonzero(alive)
            b = bits[cur[a_i] + idx[a_i, w_i]]
            q[act[a_i], w_i] += b
            cur = cur + alive.sum(axis=1)
            alive = np.zeros_like(alive)
            alive[a_i, w_i] = b.astype(bool)
            j += 1
            done = qc[act] <= j
            cursor[act[done]] = cur[done]
            act, cur, alive = act[~done], cur[~done], alive[~done]
        ranks[:, c] = (q << kc[:, None].astype(np.int32)) | low
    return ranks.reshape(R, Tn * T)[:, :K], cursor


def decode(enc: RiceEncoded, R: int, K: int, T: int = TILE) -> np.ndarray:
    """Inverse of encode: -> [R, K] uint8 exponents. Checks every row ends where the next starts."""
    ranks, end = decode_ranks(enc.meta, enc.bits, enc.roff, K, T)
    nxt = np.append(enc.roff[1:].astype(np.int64), enc.bits.size)
    if np.any((end + 31) >> 5 != nxt):
        raise ValueError("bit stream length does not match meta")
    return enc.table[ranks]
