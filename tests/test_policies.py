"""GPU per-tile statistics and policy costs against a slow pure-Python reference."""

import math
from collections import Counter

import mlx.core as mx
import numpy as np
import pytest

from binade.policies import CHOICES, OBJECTIVES, P16_BITS, allowed, best_codes, candidates, select
from binade.policies import PALETTE_POLICIES as POLICIES
from binade.scan import TileAccum
from binade.tile import tile_counts, tile_counts_ref, tiles


def _ref_options(vals, T):
    counts = sorted(Counter(vals).values(), reverse=True)
    k, n = len(counts), len(vals)
    plain = None
    if k <= 16:
        b = 0 if k == 1 else math.ceil(math.log2(k))
        plain = ("plain", b, (T * b, 8 * k, 0))

    def esc(b):
        npal = min(k, 2**b - 1)
        return (f"esc{b}", b, (T * b, 8 * npal, 8 * (n - sum(counts[:npal]))))

    verb = ("verb", 8, (8 * T, 0, 0))
    unary = ("unary", None, (sum(c * (j + 1) for j, c in enumerate(counts)) - counts[-1] + ((T - n) if k >= 2 else 0), 8 * k, 0))
    return k, {
        "bump": [plain if k <= 16 else verb],
        "escape": [plain] if k == 1 else [esc(b) for b in (1, 2, 3, 4)],
        "verbatim": [verb],
        "best": [o for o in [plain, esc(1), esc(2), esc(3), esc(4), verb] if o],
        "unary": [unary],
    }


def ref_tile(vals, T, policy, objective):
    """(idx, pal, esc, choice label) for one tile, straight from the definitions in DESIGN.md 3."""
    _, opts = _ref_options(vals, T)
    cost = lambda o: o[2][0] + o[2][2] + (o[2][1] if objective == "actual" else P16_BITS)
    name, b, (i, p, e) = min(opts[policy], key=cost)  # first minimum, same tie order as the GPU
    label = f"plain{b}" if name == "plain" else name
    return i, (p if objective == "actual" else P16_BITS), e, label


def _tiles_zoo(rng, n):
    """Random and adversarial tiles of width n (uint8 exponents)."""
    geo = np.clip(126 - rng.geometric(0.5, size=(300, n)) + 1, 0, 255)
    rows = [
        np.full((4, n), 120),  # k = 1
        np.tile(np.arange(n) % 2 + 100, (3, 1)),  # k = 2
        np.tile((np.arange(n) % min(n, 256)), (3, 1)),  # all distinct (k = n)
        np.tile(np.r_[np.full(n - 1, 110), 3], (3, 1)),  # one outlier
        np.tile(np.r_[np.full(max(n - 17, 0), 110), np.arange(min(n, 17))][:n], (3, 1)),  # k = 17
        np.tile(np.arange(n) % 16 + 90, (2, 1)),  # k = 16 exactly
        np.tile(np.r_[np.zeros(n // 2), np.full(n - n // 2, 255)], (2, 1)),  # zeros and inf/nan exponents
        rng.integers(118, 126, size=(200, n)),
        geo,
    ]
    return np.concatenate(rows).astype(np.uint8)


@pytest.mark.parametrize("n", [1, 3, 17, 32, 64, 128])
def test_tile_counts_matches_reference(n):
    t = _tiles_zoo(np.random.default_rng(n), n)
    k, c = tile_counts(mx.array(t))
    kr, cr = tile_counts_ref(t)
    assert np.array_equal(np.array(k), kr)
    assert np.array_equal(np.array(c), cr)


@pytest.mark.parametrize("T,n", [(32, 32), (64, 64), (128, 128), (128, 64), (64, 5)])
@pytest.mark.parametrize("objective", OBJECTIVES)
def test_policy_costs_match_reference(T, n, objective):
    t = _tiles_zoo(np.random.default_rng(T + n), n)
    k, c = tile_counts(mx.array(t))
    cands = candidates(k, c, n, T)
    for policy in POLICIES:
        choice, i, p, e = select(cands, allowed(policy, k), objective)
        got = np.stack([np.array(i), np.array(p), np.array(e)], axis=1)
        ref = [ref_tile(list(row), T, policy, objective) for row in t]
        assert np.array_equal(got, np.array([r[:3] for r in ref])), policy
        if policy == "best":
            labels = [CHOICES[x] for x in np.array(best_codes(choice, k))]
            assert labels == [r[3] for r in ref]


def test_tile_accum_handles_partial_tiles():
    """K not a multiple of T: sums over full and partial tiles equal the per-tile reference."""
    rng = np.random.default_rng(7)
    R, K, T = 6, 150, 64  # two full tiles and one 22-wide partial tile per row
    e = np.clip(125 - rng.geometric(0.45, size=(R, K)) + 1, 0, 255).astype(np.uint8)
    acc = TileAccum(T)
    acc.add(mx.array(e))
    assert acc.ntiles == R * 3
    ref_tiles = [list(e[r, j : j + T]) for r in range(R) for j in range(0, K, T)]
    for objective in OBJECTIVES:
        for policy in POLICIES:
            ref = np.sum([ref_tile(v, T, policy, objective)[:3] for v in ref_tiles], axis=0)
            assert np.array_equal(acc.bits[objective][policy], ref), (policy, objective)
    kh = np.bincount([len(set(v)) for v in ref_tiles], minlength=T + 1)
    assert np.array_equal(acc.khist, kh)
    assert acc.choice["actual"].sum() == acc.ntiles


def test_tiles_split():
    e = mx.arange(2 * 10).reshape(2, 10)
    main, rem = tiles(e, 4)
    assert main.shape == (4, 4) and rem.shape == (2, 2)
    assert np.array_equal(np.array(main[2]), [10, 11, 12, 13])
    assert tiles(e, 5)[1] is None


def ref_rank_tile(r, T, policy):
    """(idx, pal, esc) for one tile of frequency ranks, from the rank-family definitions."""
    r = [int(x) for x in r]
    n, rmax = len(r), max(r)
    fixed = (T * rmax.bit_length(), 0, 0)
    esc = [(T * b, 0, 8 * sum(x >= 2**b - 1 for x in r)) for b in (1, 2, 3, 4)]
    verb = (8 * T, 0, 0)
    def rice(k):
        q = [x >> k for x in r]
        qmax = rmax >> k
        return (T * k + sum(min(x + 1, qmax) for x in q) + ((T - n) if qmax > 0 else 0), 0, 0)

    unary = (sum(x + 1 for x in r) - sum(x == rmax for x in r) + ((T - n) if rmax > 0 else 0), 0, 0)
    assert unary == rice(0)
    opts = {
        "rank_fixed": [fixed],
        "rank_best": [fixed, *esc, verb],
        "rank_unary": [unary],
        "rank_rice": [rice(0), rice(1), rice(2)],
    }[policy]
    return min(opts, key=lambda o: o[0] + o[2])


@pytest.mark.parametrize("T,n", [(32, 32), (64, 64), (128, 128), (128, 64), (64, 5)])
def test_rank_costs_match_reference(T, n):
    rng = np.random.default_rng(T * n)
    r = np.concatenate(
        [
            np.zeros((3, n)),
            np.clip(rng.geometric(0.45, size=(300, n)) - 1, 0, 255),
            rng.integers(0, 40, size=(50, n)),
            np.tile(np.r_[np.zeros(n - 1), 200], (2, 1)),
        ]
    ).astype(np.uint8)
    from binade.policies import RANK_POLICIES, rank_candidates

    cands = rank_candidates(mx.array(r), n, T)
    ones = mx.ones((len(r),), dtype=mx.int32)
    for policy in RANK_POLICIES:
        for objective in OBJECTIVES:
            _, i, p, e = select(cands, allowed(policy, ones), objective)
            got = np.stack([np.array(i), np.array(p), np.array(e)], axis=1)
            ref = np.array([ref_rank_tile([int(x) for x in row], T, policy) for row in r])
            assert np.array_equal(got, ref), (policy, objective)


def test_scan_tensor_end_to_end(tmp_path):
    """scan_tensor on a small file: real-layout sums equal the per-tile references."""
    from binade import st
    from binade.policies import PALETTE_POLICIES, RANK_POLICIES
    from binade.scan import scan_tensor

    rng = np.random.default_rng(3)
    R, K, T = 40, 200, 64
    f = (rng.standard_normal((R, K)) * np.exp(rng.standard_normal((R, 1)))).astype(np.float32) * 0.02
    w = mx.array(f).astype(mx.bfloat16)
    mx.save_safetensors(str(tmp_path / "model.safetensors"), {"layers.0.mlp.up_proj.weight": w})
    info = st.list_tensors(tmp_path)[0]
    rec = scan_tensor(info, (T,), seed=1, do_lossy=False, chunk=R * K // 3)
    e = np.array((w.view(mx.uint16) >> 7) & 0xFF).astype(np.uint8)
    hist = np.bincount(e.ravel(), minlength=256)
    assert np.array_equal(rec["hist"], hist)
    order = np.argsort(-hist, kind="stable")
    rank = np.empty(256, np.int64)
    rank[order] = np.arange(256)
    tl = [(r, j) for r in range(R) for j in range(0, K, T)]
    acc = rec["tiles"]["real"][T]
    for policy in PALETTE_POLICIES:
        ref = np.sum([ref_tile(list(e[r, j : j + T]), T, policy, "actual")[:3] for r, j in tl], axis=0)
        assert np.array_equal(acc.bits["actual"][policy], ref), policy
    distinct = int((hist > 0).sum())
    for policy in RANK_POLICIES:
        ref = np.sum([ref_rank_tile(list(rank[e[r, j : j + T]]), T, policy) for r, j in tl], axis=0)
        ref[1] += 8 * distinct
        assert np.array_equal(acc.bits["actual"][policy], ref), policy
    # shuffles keep the multiset of exponents: tile count and total weights match
    for v in ("shuf_row", "shuf_tensor"):
        a = rec["tiles"][v][T]
        assert a.ntiles == acc.ntiles and a.khist.sum() == acc.khist.sum()
    in16 = np.isin(e, order[:16][hist[order[:16]] > 0])
    segs = [(r, j) for r in range(R) for j in range(0, K, 64)]
    assert rec["segments"] == len(segs)
    assert rec["seg_in_weights"] == sum(in16[r, j : j + 64].size for r, j in segs if in16[r, j : j + 64].all())
