"""Per-tile encodings of the exponent stream and their bit costs (DESIGN.md 3, 4.5).

Candidates for one tile with k distinct exponents, counts c_0 >= c_1 >= ..., n real
weights and padded width T. Costs are in bits and exclude the raw byte, the meta byte
and the offset word, which every tile pays whatever the encoding.

  plain   b = ceil(log2 k) <= 4, palette of k entries        idx T*b   pal 8k
  escN    b = N in 1..4, palette = top 2^b-1 exponents,      idx T*b   pal 8*min(k, 2^b-1)
          index 2^b-1 escapes to a verbatim byte                       esc 8*(n - covered)
  verb    exponent stored raw (meta b = 8)                   idx 8T
  unary   palette of k entries by frequency, truncated       idx sum_j c_j*(j+1) - c_{k-1}
          unary code of the rank (exploratory, DESIGN.md 5)   pal 8k

Policies take the cheapest allowed candidate per tile:
  bump      plain if k <= 16, else verb
  escape    plain if k == 1, else the cheapest escN
  verbatim  verb
  best      cheapest of plain (k <= 16), esc1..esc4, verb
  unary     unary

Palette accounting: "actual" charges 8 bits per stored entry. "p16" charges the v0
on-disk layout, a zero-padded [ntiles, 16] uint8 palette (128 bits for every tile),
and re-optimizes the per-tile choice under that cost.

Rank family (no per-tile palette). One table per tensor maps exponents to their
frequency rank r (0 = most common); tiles code ranks:
  rfixed  b = bit length of the tile's largest rank      idx T*b        (ENEC-class)
  rescN   b = N in 1..4, ranks >= 2^b-1 escape           idx T*b        esc 8*count
  rverb   exponent stored raw                            idx 8T
  runary  truncated unary of the rank, the tile's        idx sum(r+1) - count(r == rmax)
          largest rank in the meta byte (the delta-from-mode fallback of DESIGN.md 5)
  riceK   Rice code, K in 1..2: K low bits of r verbatim, truncated unary of q = r >> K
                                                         idx n*K + sum(q+1) - count(q == qmax)
Rank policies: rank_fixed = rfixed, rank_best = cheapest of rfixed, resc1..4, rverb,
rank_unary = runary, rank_rice = cheapest of runary, rice1, rice2 (parameter in the
meta byte). The per-tensor table (8 bits per distinct exponent) is charged once per
tensor by the caller.
"""

import mlx.core as mx

CANDIDATES = ("plain", "esc1", "esc2", "esc3", "esc4", "verb", "unary")
RANK_CANDIDATES = ("rfixed", "resc1", "resc2", "resc3", "resc4", "rverb", "runary", "rice1", "rice2")
PALETTE_POLICIES = ("bump", "escape", "verbatim", "best", "unary")
RANK_POLICIES = ("rank_fixed", "rank_best", "rank_unary", "rank_rice")
POLICIES = PALETTE_POLICIES + RANK_POLICIES
OBJECTIVES = ("actual", "p16")
# best-of choice codes: plain b=0..4 -> 0..4, esc1..esc4 -> 5..8, verb -> 9
CHOICES = ("plain0", "plain1", "plain2", "plain3", "plain4", "esc1", "esc2", "esc3", "esc4", "verb")
P16_BITS = 128
_BIG = 1 << 24
_BLUT = mx.array([0, 0, 1, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 99], dtype=mx.int32)


def plain_width(k: mx.array) -> mx.array:
    """ceil(log2 k) for k <= 16; 99 marks k > 16."""
    return _BLUT[mx.minimum(k, 17)]


def candidates(k: mx.array, c: mx.array, n: int, T: int) -> dict:
    """name -> (idx, pal, esc) bit costs, each [nt] int32. See module docstring."""
    nt = k.shape[0]
    zero = mx.zeros((nt,), dtype=mx.int32)
    cum = mx.cumsum(c, axis=1)
    out = {"plain": (T * plain_width(k), 8 * k, zero)}
    for b in (1, 2, 3, 4):
        npal = mx.minimum(k, (1 << b) - 1)
        covered = mx.take_along_axis(cum, (npal - 1)[:, None], axis=1)[:, 0]
        out[f"esc{b}"] = (mx.full((nt,), T * b, dtype=mx.int32), 8 * npal, 8 * (n - covered))
    out["verb"] = (mx.full((nt,), 8 * T, dtype=mx.int32), zero, zero)
    ranks = mx.arange(1, c.shape[1] + 1, dtype=mx.int32)
    last = mx.take_along_axis(c, (k - 1)[:, None], axis=1)[:, 0]
    # padding weights in a partial tile are coded as the mode: 1 bit each when k >= 2
    un = (c * ranks).sum(axis=1) - last + (T - n) * (k >= 2).astype(mx.int32)
    out["unary"] = (un, 8 * k, zero)
    return out


_BITLEN = mx.array([v.bit_length() for v in range(256)], dtype=mx.int32)


def rank_candidates(r: mx.array, n: int, T: int) -> dict:
    """Rank-family (idx, pal, esc) bit costs per tile. r: [nt, n] uint8 frequency ranks."""
    nt = r.shape[0]
    zero = mx.zeros((nt,), dtype=mx.int32)
    ri = r.astype(mx.int32)
    rmax = ri.max(axis=1)
    out = {"rfixed": (T * _BITLEN[rmax], zero, zero)}
    for b in (1, 2, 3, 4):
        cnt = (ri >= (1 << b) - 1).astype(mx.int32).sum(axis=1)
        out[f"resc{b}"] = (mx.full((nt,), T * b, dtype=mx.int32), zero, 8 * cnt)
    out["rverb"] = (mx.full((nt,), 8 * T, dtype=mx.int32), zero, zero)
    # truncated unary of q = r >> K plus K verbatim low bits; padding weights are rank 0
    for K, name in ((0, "runary"), (1, "rice1"), (2, "rice2")):
        q = ri >> K
        qmax = rmax >> K
        at_max = (q == qmax[:, None]).astype(mx.int32).sum(axis=1)
        bits = T * K + (q + 1).sum(axis=1) - at_max + (T - n) * (qmax > 0).astype(mx.int32)
        out[name] = (bits, zero, zero)
    return out


def allowed(policy: str, k: mx.array) -> dict:
    """Candidates a policy may use, in tie-break order, with per-tile validity.

    For rank policies `k` only supplies the tile count."""
    everywhere = mx.ones(k.shape, dtype=mx.bool_)
    if policy == "rank_fixed":
        return {"rfixed": everywhere}
    if policy == "rank_best":
        return {"rfixed": everywhere, **{f"resc{b}": everywhere for b in (1, 2, 3, 4)}, "rverb": everywhere}
    if policy == "rank_unary":
        return {"runary": everywhere}
    if policy == "rank_rice":
        return {"runary": everywhere, "rice1": everywhere, "rice2": everywhere}
    esc = {f"esc{b}": everywhere for b in (1, 2, 3, 4)}
    if policy == "bump":
        return {"plain": k <= 16, "verb": everywhere}
    if policy == "escape":
        return {"plain": k == 1, **esc}
    if policy == "verbatim":
        return {"verb": everywhere}
    if policy == "best":
        return {"plain": k <= 16, **esc, "verb": everywhere}
    if policy == "unary":
        return {"unary": everywhere}
    raise ValueError(policy)


def select(cands: dict, allow: dict, objective: str):
    """Cheapest allowed candidate per tile.

    Returns (choice [nt] index into `allow`, idx, pal, esc) with pal charged under
    the objective (actual entries, or 128 bits per tile for p16).
    """
    names = list(allow)
    idx = mx.stack([cands[n][0] for n in names])
    pal = mx.stack([cands[n][1] for n in names])
    esc = mx.stack([cands[n][2] for n in names])
    valid = mx.stack([allow[n] for n in names])
    no_palette = all(n in RANK_CANDIDATES for n in names)
    cost = idx + esc + (pal if objective == "actual" or no_palette else P16_BITS)
    cost = mx.where(valid, cost, _BIG)
    choice = mx.argmin(cost, axis=0)
    pick = choice[None, :]
    i = mx.take_along_axis(idx, pick, axis=0)[0]
    e = mx.take_along_axis(esc, pick, axis=0)[0]
    if objective == "actual" or no_palette:
        p = mx.take_along_axis(pal, pick, axis=0)[0]
    else:
        p = mx.full(i.shape, P16_BITS, dtype=mx.int32)
    return choice, i, p, e


def best_codes(choice: mx.array, k: mx.array) -> mx.array:
    """Map a `best` choice (plain, esc1..4, verb) to CHOICES codes."""
    return mx.where(choice == 0, plain_width(k), choice + 4)
