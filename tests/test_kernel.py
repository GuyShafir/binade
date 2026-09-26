"""Binade v1 Metal matvec: bit-identical to MLX's own BF16 matvec on the unpacked weight."""

import mlx.core as mx
import numpy as np
import pytest

from binade import bf16
from binade.format import encode, rank_table
from binade.mlx.kernel import BinadeMatrix
from binade.mlx.cuda_kernels import available as cuda_available

pytestmark = pytest.mark.skipif(cuda_available(), reason="format 1 and R4 Metal kernels (CUDA kernels: test_r4, test_r4_moe)")


def _pack(w16: np.ndarray) -> BinadeMatrix:
    raw, exps = bf16.split(w16)
    table, rank = rank_table(np.bincount(exps.ravel(), minlength=256))
    enc = encode(exps, table, rank)
    return BinadeMatrix(raw, enc.table, enc.meta, enc.idx, enc.esc)


def _weights(rng, N, K, kind):
    f = rng.standard_normal((N, K)).astype(np.float32) * 0.02
    if kind == "row scales":
        f *= np.exp(rng.standard_normal((N, 1)) * 2).astype(np.float32)
    if kind == "outliers":
        f[:, ::128] *= 1e4  # one escape per tile
    w16 = (f.view(np.uint32) >> 16).astype(np.uint16)
    if kind == "wide exponents":  # every exponent byte except inf/nan: tiles at b = 7 or 8
        e = rng.integers(1, 255, size=(N, K), dtype=np.uint16)
        w16 = (w16 & 0x807F) | (e << 7)
    if kind == "flat":  # constant exponent: b = 0 tiles
        w16 = (w16 & 0x807F) | np.uint16(120 << 7)
    if kind == "many escapes":  # 40 escapes per tile: 2-bit escape tiles past the 32-lane window
        e = rng.choice(np.array([119, 120, 121], np.uint16), size=(N, K))
        rare = rng.permuted(np.tile(np.arange(K) % 128 < 40, (N, 1)), axis=1)
        e = np.where(rare, 60 + rng.integers(0, 45, size=(N, K), dtype=np.uint16), e)
        w16 = (w16 & 0x807F) | (e << 7)
    return w16


# MLX splits K across SIMD groups when K >= 16 N (a different summation order), so bit
# equality is tested where MLX uses the same single-group-per-row order as the kernel
@pytest.mark.parametrize("N,K", [(4, 128), (8, 256), (36, 384), (64, 896), (128, 1024), (4096, 256)])
@pytest.mark.parametrize("kind", ["gaussian", "row scales", "outliers", "wide exponents", "flat", "many escapes"])
@pytest.mark.parametrize(
    "variant",
    [
        dict(tm=4),
        dict(tm=2, prologue=True),
        dict(tm=1, prologue=True, fast3=True),
        dict(tm=4, prologue=True, fast3=True, bm=2),
        dict(tm=1, prologue=True, fast3=True, regtab=True, tgmeta=True, unroll=2),
        dict(tm=1, prologue=True, fast3=True, xf32=True, swar=True),
        dict(tm=4, xf32=True, swar=True),
        dict(tm=1, prologue=True, fast3=True, specesc=True),
        dict(tm=4, specesc=True),
        dict(tm=1, prologue=True, fast3=True, ebase=True),
        dict(tm=1, prologue=True, fast3=True, ebase=True, specesc=True),
        dict(tm=4, prologue=True, ebase=True, specesc=True),
        dict(tm=1, prologue=True, fast3=True, tgesc=True),
        dict(tm=4, prologue=True, tgesc=True),
        dict(tm=1, prologue=True, fast3=True, batch=2),
        dict(tm=1, prologue=True, fast3=True, batch=4),
        dict(tm=1, prologue=True, fast3=True, batch=7),
    ],
)
def test_matvec_bit_identical_to_mlx(N, K, kind, variant):
    rng = np.random.default_rng(N * 7 + K)
    w16 = _weights(rng, N, K, kind)
    m = _pack(w16)
    W = mx.array(w16).view(mx.bfloat16)
    x = mx.array(rng.standard_normal((1, K)).astype(np.float32)).astype(mx.bfloat16)
    ref = x @ W.T
    got = m.matvec(x, **variant)
    mx.eval(ref, got)
    a, b = np.array(ref.view(mx.uint16)).ravel(), np.array(got.view(mx.uint16))
    assert np.array_equal(a, b), f"{np.count_nonzero(a != b)} of {N} outputs differ"


def test_many_escapes_reach_the_fallback():
    rng = np.random.default_rng(5)
    m = _pack(_weights(rng, 64, 896, "many escapes"))
    meta = np.array(m.meta)
    esc_tiles = (meta & 0x10) != 0
    assert esc_tiles.mean() > 0.9 and np.all((meta[esc_tiles] & 0x0F) <= 2)
    assert np.array(m.esc).size - 32 > 32 * esc_tiles.sum()  # more than 32 escapes per escape tile on average


@pytest.mark.parametrize("N,K", [(4, 128), (36, 384), (64, 896), (4096, 256)])
@pytest.mark.parametrize("kind", ["gaussian", "row scales", "outliers", "wide exponents", "flat", "many escapes"])
@pytest.mark.parametrize("tm", [1, 2, 4])
@pytest.mark.parametrize("shuffle,pf", [(False, 0), (True, 0), (False, 1), (False, 2), (False, 4)])
def test_r4_bit_identical_to_mlx(N, K, kind, tm, shuffle, pf):
    if pf and tm != 1:
        pytest.skip("prefetch variant is tm = 1")
    from binade.mlx.kernel import R4Matrix

    rng = np.random.default_rng(N * 3 + K)
    w16 = _weights(rng, N, K, kind)
    raw, exps = bf16.split(w16)
    table, rank = rank_table(np.bincount(exps.ravel(), minlength=256))
    enc = encode(exps, table, rank)
    m = R4Matrix(raw, enc.table, enc.meta, enc.idx, enc.esc)
    W = mx.array(w16).view(mx.bfloat16)
    x = mx.array(rng.standard_normal((1, K)).astype(np.float32)).astype(mx.bfloat16)
    ref, got = x @ W.T, m.matvec(x, tm=tm, shuffle=shuffle, pf=pf)
    mx.eval(ref, got)
    assert np.array_equal(np.array(ref.view(mx.uint16)).ravel(), np.array(got.view(mx.uint16)))
