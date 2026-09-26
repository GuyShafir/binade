import struct

import mlx.core as mx
import numpy as np
import pytest

from binade import bf16

ALL = np.arange(1 << 16, dtype=np.uint32).astype(np.uint16)


def test_split_join_exhaustive_numpy():
    r, e = bf16.split(ALL)
    assert r.dtype == np.uint8 and e.dtype == np.uint8
    assert np.array_equal(bf16.join(r, e), ALL)


def test_split_join_exhaustive_mlx():
    w = mx.array(ALL)
    r, e = bf16.split(w)
    assert r.dtype == mx.uint8 and e.dtype == mx.uint8
    assert np.array_equal(np.array(bf16.join(r, e)), ALL)
    assert np.array_equal(np.array(r), bf16.raw(ALL))
    assert np.array_equal(np.array(e), bf16.exponent(ALL))


def test_fields_match_float_decomposition():
    for x in (1.0, -2.5, 3.0e-3, -7.0e20, 0.0):
        u32 = struct.unpack("<I", struct.pack("<f", x))[0]
        w = np.array([u32 >> 16], dtype=np.uint16)
        sign, exp, mant = u32 >> 31, (u32 >> 23) & 0xFF, (u32 >> 16) & 0x7F
        assert bf16.exponent(w)[0] == exp
        assert bf16.raw(w)[0] == (sign << 7) | mant


def _round_ref(w: int, m: int) -> int:
    """Independent reference: Python's round() is half-to-even; saturate at the binade top."""
    exp = (w >> 7) & 0xFF
    if exp == 0xFF or m == 7:
        return w
    s = 7 - m
    q = min(round((w & 0x7F) / (1 << s)), (1 << m) - 1)
    return (w & 0xFF80) | (q << s)


@pytest.mark.parametrize("m", [1, 2, 3, 4, 5, 6, 7])
def test_round_mantissa_exhaustive(m):
    ref = np.array([_round_ref(int(w), m) for w in ALL], dtype=np.uint16)
    got_np = bf16.round_mantissa(ALL, m)
    got_mx = np.array(bf16.round_mantissa(mx.array(ALL), m))
    assert np.array_equal(got_np, ref)
    assert np.array_equal(got_mx, ref)
    # exponent field never changes, low 7-m bits are cleared for finite values
    assert np.array_equal(bf16.exponent(got_np), bf16.exponent(ALL))
    finite = bf16.exponent(ALL) != 0xFF
    assert not np.any(got_np[finite] & ((1 << (7 - m)) - 1))


def test_round_mantissa_keeps_nan_and_inf():
    specials = np.array([0x7F80, 0xFF80, 0x7FC0, 0x7F81, 0xFFFF], dtype=np.uint16)
    assert np.array_equal(bf16.round_mantissa(specials, 2), specials)
