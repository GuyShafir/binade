"""BF16 bit helpers (DESIGN.md 1). Every function accepts numpy or mlx arrays.

BF16 layout: bit 15 sign, bits 14..7 exponent, bits 6..0 mantissa.
`raw` packs sign into bit 7 and mantissa into bits 6..0; it is stored verbatim.
Only the exponent byte is compressed.
"""

import numpy as np


def _ns(a):
    """(module, uint8, uint16) matching the array type of `a`."""
    if isinstance(a, np.ndarray):
        return np, np.uint8, np.uint16
    import mlx.core as mx

    return mx, mx.uint8, mx.uint16


def exponent(w):
    """uint16 BF16 bit patterns -> uint8 exponent field."""
    _, u8, _ = _ns(w)
    return ((w >> 7) & 0xFF).astype(u8)


def raw(w):
    """uint16 BF16 bit patterns -> uint8 sign (bit 7) | mantissa (bits 6..0)."""
    _, u8, _ = _ns(w)
    return (((w >> 8) & 0x80) | (w & 0x7F)).astype(u8)


def split(w):
    return raw(w), exponent(w)


def join(r, e):
    """Inverse of split: (raw, exp) uint8 -> uint16 BF16 bit patterns."""
    _, _, u16 = _ns(r)
    r = r.astype(u16)
    e = e.astype(u16)
    return ((r & 0x80) << 8) | (e << 7) | (r & 0x7F)


def round_mantissa(w, m):
    """Keep the top m of 7 mantissa bits, round half to even, exponent field unchanged.

    A round-up that would carry into the exponent saturates at the largest m-bit
    mantissa instead, so the exponent stream stays lossless (DESIGN.md 11).
    Inf and NaN (exponent 255) pass through untouched.
    """
    if not 1 <= m <= 7:
        raise ValueError(f"m must be in 1..7, got {m}")
    if m == 7:
        return w
    ns, _, _ = _ns(w)
    s = 7 - m
    mant = w & 0x7F
    q = (mant + ((1 << (s - 1)) - 1) + ((mant >> s) & 1)) >> s
    q = ns.minimum(q, (1 << m) - 1)
    out = (w & 0xFF80) | (q << s)
    return ns.where(((w >> 7) & 0xFF) == 0xFF, w, out)
