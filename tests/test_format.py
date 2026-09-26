import mlx.core as mx
import numpy as np
import pytest

from binade import bf16
from binade.format import TILE, Encoded, choose, decode, encode, pack_codes, rank_table, unpack_codes
from binade.scan import TileAccum


@pytest.mark.parametrize("b", range(1, 9))
def test_pack_unpack_codes(b):
    rng = np.random.default_rng(b)
    codes = rng.integers(0, 1 << b, size=(37, TILE), dtype=np.uint8)
    words = pack_codes(codes, b)
    assert words.shape == (37, 4 * b) and words.dtype == np.uint32
    assert np.array_equal(unpack_codes(words, b), codes)
    # LSB-first layout: code i starts at bit i*b of the tile's bit stream
    stream = np.unpackbits(words[0].view(np.uint8), bitorder="little")
    for i in (0, 1, 5, TILE - 1):
        assert sum(int(stream[i * b + j]) << j for j in range(b)) == codes[0, i]


def _roundtrip(exps):
    hist = np.bincount(exps.ravel(), minlength=256)
    table, rank = rank_table(hist)
    enc = encode(exps, table, rank)
    assert np.array_equal(decode(enc, *exps.shape), exps)
    return enc


def _zoo(rng):
    geo = lambda shape: np.clip(125 - rng.geometric(0.45, size=shape) + 1, 0, 255).astype(np.uint8)
    yield "geometric", geo((64, 1000))
    yield "all same", np.full((8, 256), 120, np.uint8)
    yield "all distinct", np.tile(np.arange(256, dtype=np.uint8), (4, 2))
    one = geo((16, 512))
    one[:, ::TILE] = 3  # one outlier per tile
    yield "outlier per tile", one
    yield "zeros and inf/nan", np.where(rng.random((16, 384)) < 0.5, 0, 255).astype(np.uint8)
    yield "uniform", rng.integers(0, 256, size=(16, 300), dtype=np.uint8)
    yield "single row", geo((1, 5))
    yield "single weight", np.array([[77]], np.uint8)
    yield "K below tile", geo((40, 100))


def test_roundtrip_adversarial():
    for name, exps in _zoo(np.random.default_rng(0)):
        enc = _roundtrip(exps)
        assert enc.meta.size == exps.shape[0] * -(-exps.shape[1] // TILE), name


def test_all_bf16_patterns_roundtrip():
    """Every BF16 bit pattern, including NaN, Inf, zeros and subnormals, survives split/encode/decode/join."""
    w = np.random.default_rng(1).permutation(np.arange(1 << 16, dtype=np.uint32).astype(np.uint16)).reshape(256, 256)
    raw, exps = bf16.split(w)
    enc = _roundtrip(exps)
    assert np.array_equal(bf16.join(raw, decode(enc, 256, 256)), w)


def test_escape_ranks_at_the_escape_code():
    """A rank equal to 2^b - 1 must escape, since that code is reserved."""
    # counts 46, 40, 40, 1, 1 -> ranks 0..4. fixed needs b=3 (384 bits); escape b=2 costs
    # 256 + 2 escapes * 8 = 272 and must escape rank 3 (== code 3) as well as rank 4.
    exps = np.array([[103, 104] + [100] * 46 + [101] * 40 + [102] * 40], np.uint8)
    enc = _roundtrip(exps)
    assert enc.meta[0] == 0x12
    assert list(enc.esc) == [103, 104]


def test_choose_matches_phase1_rank_best():
    """Packed index + escape bits equal binade.scan's rank_best accounting for the same data."""
    rng = np.random.default_rng(2)
    f = (rng.standard_normal((96, 640)) * np.exp(rng.standard_normal((96, 1)))).astype(np.float32) * 0.02
    w = (f.view(np.uint32) >> 16).astype(np.uint16)
    exps = bf16.exponent(w)
    hist = np.bincount(exps.ravel(), minlength=256)
    table, rank = rank_table(hist)
    enc = encode(exps, table, rank)
    acc = TileAccum(TILE)
    acc.add_ranks(mx.array(rank[exps]))
    idx_bits, _, esc_bits = acc.bits["actual"]["rank_best"]
    assert 32 * enc.idx.size == idx_bits and 8 * enc.esc.size == esc_bits


def test_encode_in_row_chunks_concatenates():
    rng = np.random.default_rng(3)
    exps = np.clip(125 - rng.geometric(0.45, size=(50, 300)) + 1, 0, 255).astype(np.uint8)
    table, rank = rank_table(np.bincount(exps.ravel(), minlength=256))
    whole = encode(exps, table, rank)
    parts = [encode(exps[r : r + 7], table, rank) for r in range(0, 50, 7)]
    joined = Encoded(table, *(np.concatenate([getattr(p, k) for p in parts]) for k in ("meta", "idx", "esc")))
    for k in ("meta", "idx", "esc"):
        assert np.array_equal(getattr(whole, k), getattr(joined, k))


def test_choose_prefers_fixed_on_ties():
    r = np.zeros((1, TILE), np.uint8)  # all rank 0: fixed b=0 costs 0
    b, esc = choose(r)
    assert b[0] == 0 and not esc[0]
