import mlx.core as mx
import numpy as np
import pytest

from binade import bf16
from binade.format import TILE, rank_table
from binade.policies import rank_candidates
from binade.rice import KMAX, RiceEncoded, _valid, choose, decode, decode_ranks, encode, encode_ranks


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
    yield "K = 704 (26B experts)", geo((24, 704))


def test_roundtrip_adversarial():
    for name, exps in _zoo(np.random.default_rng(0)):
        enc = _roundtrip(exps)
        assert enc.meta.size == exps.shape[0] * -(-exps.shape[1] // TILE), name
        assert enc.roff.size == exps.shape[0], name


def test_all_bf16_patterns_roundtrip():
    """Every BF16 bit pattern, including NaN, Inf, zeros and subnormals, survives split/encode/decode/join."""
    w = np.random.default_rng(1).permutation(np.arange(1 << 16, dtype=np.uint32).astype(np.uint16)).reshape(256, 256)
    raw, exps = bf16.split(w)
    enc = _roundtrip(exps)
    assert np.array_equal(bf16.join(raw, decode(enc, 256, 256)), w)


def test_layout_by_hand():
    """One partial tile, ranks [0, 1, 5, 2]: k = 1 costs 4 low + 6 unary bits, the cheapest."""
    meta, words, roff = encode_ranks(np.array([[0, 1, 5, 2]]))
    assert meta.tolist() == [1 | 2 << 2] and roff.tolist() == [0]
    # low plane r & 1 = 0 1 1 0; plane 0 (all alive) q > 0 = 0 0 1 1; plane 1 (weights 2, 3) q > 1 = 1 0
    stream = [0, 1, 1, 0, 0, 0, 1, 1, 1, 0]
    assert words.tolist() == [sum(b << i for i, b in enumerate(stream))]


def test_rows_start_on_words():
    rng = np.random.default_rng(2)
    ranks = rng.geometric(0.4, size=(9, 300)) - 1
    meta, words, roff = encode_ranks(ranks)
    # each row decodes alone from its own offset
    for r in range(9):
        got, _ = decode_ranks(meta[3 * r : 3 * r + 3], words, roff[r : r + 1], 300)
        assert np.array_equal(got[0], ranks[r])


def test_cost_matches_phase1_accounting():
    """For full tiles and k <= 2 the stream costs what policies.rank_candidates charged in phase 1."""
    rng = np.random.default_rng(3)
    r = np.minimum(rng.geometric(0.35, size=(500, TILE)) - 1, 255)
    r[::7, 5] = rng.integers(16, 256, size=r[::7].shape[0])  # some large ranks
    valid = _valid(500, TILE, TILE)
    k, qmax, bits = choose(r.astype(np.int32), valid)
    c = rank_candidates(mx.array(r.astype(np.uint8)), TILE, TILE)
    phase1 = np.stack([np.array(c[n][0]) for n in ("runary", "rice1", "rice2")])
    big = np.iinfo(np.int64).max
    for kk in range(3):
        legal = (r >> kk).max(axis=1) <= 63
        assert np.array_equal(np.where(k == kk, bits, big)[k == kk], phase1[kk][k == kk])
        # a legal k the packer did not pick costs no less
        assert np.all(phase1[kk][legal] >= bits[legal])
    meta, words, roff = encode_ranks(r.reshape(50, 10 * TILE))
    tb = bits.reshape(50, 10).sum(axis=1)
    assert 32 * words.size == int((32 * ((tb + 31) // 32)).sum())


def test_k_range_and_qmax_limit():
    r = np.full((1, TILE), 255)
    meta, _, _ = encode_ranks(r)
    k, qmax = meta[0] & 3, meta[0] >> 2
    assert k >= 2 and qmax == 255 >> k and qmax <= 63 and k <= KMAX


def test_chunks_concatenate():
    """Row chunks encoded separately join into one stream by shifting roff, as the packer does."""
    rng = np.random.default_rng(4)
    exps = np.clip(125 - rng.geometric(0.45, size=(30, 520)) + 1, 0, 255).astype(np.uint8)
    table, rank = rank_table(np.bincount(exps.ravel(), minlength=256))
    parts = [encode(exps[a:b], table, rank) for a, b in ((0, 7), (7, 8), (8, 30))]
    base = np.cumsum([0] + [p.bits.size for p in parts[:-1]])
    enc = RiceEncoded(
        table=table,
        meta=np.concatenate([p.meta for p in parts]),
        bits=np.concatenate([p.bits for p in parts]),
        roff=np.concatenate([p.roff + np.uint32(o) for p, o in zip(parts, base)]),
    )
    assert np.array_equal(decode(enc, 30, 520), exps)


def test_truncated_stream_is_rejected():
    exps = np.random.default_rng(5).integers(100, 130, size=(4, 256), dtype=np.uint8)
    enc = _roundtrip(exps)
    bad = RiceEncoded(enc.table, enc.meta, enc.bits[:-1], enc.roff)
    with pytest.raises((ValueError, IndexError)):
        decode(bad, 4, 256)


@pytest.mark.parametrize("R,K", [(1, 5), (40, 100), (24, 704), (64, 1000), (7, 256), (300, 3840)])
def test_gpu_decoder_matches_reference(R, K):
    from binade.mlx.rice_kernel import rice_ranks

    rng = np.random.default_rng(K)
    ranks = np.minimum(rng.geometric(0.35, size=(R, K)) - 1, 255)
    ranks[::5, ::97] = rng.integers(0, 256, size=ranks[::5, ::97].shape)
    meta, words, roff = encode_ranks(ranks)
    ref, end = decode_ranks(meta, words, roff, K)
    got, rend = rice_ranks(mx.array(meta), mx.array(words), mx.array(roff), K)
    got = np.array(got)
    assert np.array_equal(got[:, :K], ref) and np.array_equal(ref, ranks)
    assert not got[:, K:].any()
    assert np.array_equal(np.array(rend), (end + 31) >> 5)
    assert np.array_equal(np.array(rend), np.append(roff[1:], words.size))
