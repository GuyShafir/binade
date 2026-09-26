"""Binade format v1: exponent coding (DESIGN.md 6). numpy reference encoder and decoder.

One table per tensor lists its exponents by descending frequency (rank 0 = most common).
Tiles are T = 128 weights along K in one row. Per tile, a width b in 0..8 and a mode:
  fixed   every weight stores its rank in b bits, b = bit length of the tile's largest rank
  escape  b in 1..4; ranks below 2^b - 1 are stored directly, code 2^b - 1 marks an
          escape whose exponent byte is appended to the tensor's escape stream
meta = b | escape << 4. Codes are packed LSB first into little-endian 32-bit words;
a tile of width b is exactly 4*b words. Padding weights (K up to a multiple of T) code
rank 0 and are dropped on decode.
"""

from dataclasses import dataclass

import numpy as np

TILE = 128
ESCAPE = 0x10
_BITLEN = np.array([v.bit_length() for v in range(256)], np.uint8)


@dataclass
class Encoded:
    table: np.ndarray  # uint8 [256]
    meta: np.ndarray  # uint8 [ntiles]
    idx: np.ndarray  # uint32 [4 * sum(b)]
    esc: np.ndarray  # uint8 [n_escapes]

    def bits(self) -> int:
        """Exponent-side payload in bits (excludes the raw sign+mantissa bytes)."""
        return 8 * self.table.size + 8 * self.meta.size + 32 * self.idx.size + 8 * self.esc.size


def rank_table(hist: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(table [256] exponent by rank, rank [256] rank of each exponent) from a histogram.

    Ties break toward the smaller exponent (stable sort), matching binade.scan."""
    order = np.argsort(-hist, kind="stable").astype(np.uint8)
    table = np.where(hist[order] > 0, order, 0).astype(np.uint8)
    rank = np.empty(256, np.uint8)
    rank[order] = np.arange(256, dtype=np.uint8)
    return table, rank


def choose(r: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-tile (b, escape) minimizing index + escape bits (policy rank_best).

    r: [nt, T] uint8 ranks. Ties go to fixed, then to the smaller escape width,
    the same order binade.policies uses. This is the decision a search would replace."""
    T = r.shape[1]
    fixed_b = _BITLEN[r.max(axis=1)]
    costs = [T * fixed_b.astype(np.int64)]
    for b in (1, 2, 3, 4):
        costs.append(T * b + 8 * (r >= (1 << b) - 1).sum(axis=1, dtype=np.int64))
    pick = np.argmin(np.stack(costs), axis=0)
    b = np.where(pick == 0, fixed_b, pick).astype(np.uint8)
    return b, pick > 0


def pack_codes(codes: np.ndarray, b: int) -> np.ndarray:
    """[n, T] codes < 2^b -> [n, T*b/32] uint32, code i at bits i*b .. i*b+b-1, LSB first."""
    n, T = codes.shape
    bits = ((codes[:, :, None] >> np.arange(b, dtype=np.uint8)) & 1).astype(np.uint8)
    packed = np.packbits(bits.reshape(n, T * b), axis=1, bitorder="little")
    return np.ascontiguousarray(packed).view("<u4").reshape(n, T * b // 32)


def unpack_codes(words: np.ndarray, b: int, T: int = TILE) -> np.ndarray:
    """Inverse of pack_codes: [n, T*b/32] uint32 -> [n, T] uint8."""
    n = words.shape[0]
    bits = np.unpackbits(np.ascontiguousarray(words, dtype="<u4").view(np.uint8), axis=1, bitorder="little")
    bits = bits.reshape(n, T, b).astype(np.uint16)
    return (bits << np.arange(b, dtype=np.uint16)).sum(axis=2).astype(np.uint8)


def _tiles(exps: np.ndarray, fill: int, T: int) -> np.ndarray:
    """[R, K] -> [R * ceil(K/T), T], padding each row with `fill`."""
    R, K = exps.shape
    pad = -K % T
    if pad:
        exps = np.concatenate([exps, np.full((R, pad), fill, exps.dtype)], axis=1)
    return exps.reshape(-1, T)


def encode(exps: np.ndarray, table: np.ndarray, rank: np.ndarray, T: int = TILE) -> Encoded:
    """Encode [R, K] uint8 exponents (any row range of a tensor) with the tensor's rank table."""
    e = _tiles(exps, int(table[0]), T)
    r = rank[e]
    b, esc_mode = choose(r)
    code_max = (1 << b.astype(np.int64)) - 1  # escape code per tile (for escape mode)
    escaped = esc_mode[:, None] & (r >= code_max[:, None])
    codes = np.where(escaped, code_max[:, None], r).astype(np.uint8)
    words = np.zeros(4 * int(b.astype(np.int64).sum()), np.uint32)
    start = 4 * (np.cumsum(b, dtype=np.int64) - b)
    for w in np.unique(b):
        if w == 0:
            continue
        sel = np.flatnonzero(b == w)
        words[(start[sel][:, None] + np.arange(4 * w)).ravel()] = pack_codes(codes[sel], int(w)).ravel()
    meta = (b | np.where(esc_mode, ESCAPE, 0)).astype(np.uint8)
    return Encoded(table=table, meta=meta, idx=words, esc=e[escaped])


def decode_tiles(meta: np.ndarray, idx: np.ndarray, esc: np.ndarray, table: np.ndarray, T: int = TILE) -> tuple[np.ndarray, int]:
    """Decode consecutive tiles -> ([nt, T] uint8 exponents, escape bytes consumed).

    `idx` must start at the first tile's words; `esc` at its first escape byte and may run
    past the last tile, so a tensor can be decoded in row chunks with running cursors."""
    b = (meta & 0x0F).astype(np.int64)
    esc_mode = (meta & ESCAPE) != 0
    codes = np.zeros((b.size, T), np.uint8)
    start = 4 * (np.cumsum(b) - b)
    for w in np.unique(b):
        if w == 0:
            continue
        sel = np.flatnonzero(b == w)
        codes[sel] = unpack_codes(idx[start[sel][:, None] + np.arange(4 * w)], int(w), T)
    escaped = esc_mode[:, None] & (codes == ((1 << b) - 1)[:, None])
    n = int(escaped.sum())
    if n > esc.size:
        raise ValueError(f"escape stream too short: need {n}, have {esc.size}")
    e = table[codes]
    e[escaped] = esc[:n]
    return e, n


def decode(enc: Encoded, R: int, K: int, T: int = TILE) -> np.ndarray:
    """Inverse of encode: -> [R, K] uint8 exponents."""
    e, n = decode_tiles(enc.meta, enc.idx, enc.esc, enc.table, T)
    if n != enc.esc.size or 32 * enc.idx.size != T * int((enc.meta & 0x0F).astype(np.int64).sum()):
        raise ValueError("stream length does not match meta")
    return e.reshape(R, -1)[:, :K]
