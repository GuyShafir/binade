# xp phase 1: Qwen/Qwen3.5-9B

Source `Qwen/Qwen3.5-9B` @ `c202236`. Script `scripts/histogram.py` @ `bb6988f`, seed 0, 2026-09-26, 6.2 min, peak RSS 2.0 GB.

Scope: 207 linear tensors, 7.12B weights, 73.8% of all parameters. Bits per weight include the 8-bit raw byte (sign + mantissa) and the 8-bit per-tile meta byte; palettes are charged at actual entries and the `.off` array is excluded unless a column says otherwise.

## Verdict

Locality gain at T=128 (best-of, tensor-shuffled minus real): **0.011 bits/weight** (11.677 - 11.667). Band `stop`: stop; the unary/Rice fallback the design calls for in this band (DESIGN.md 5) is reported below as rank_unary and rank_rice.

Position at T=128: xp best-of 11.667 vs DFloat11-class Huffman 10.646. Cheapest fixed-width code measured: rank_best at T=128, 11.265. Cheapest code measured without a Huffman table: rank_rice at T=128, 10.711.

## Best-of, linear tensors

| T | real | row-shuffled | tensor-shuffled | locality gain | real + .off | real, v0 padded palette + .off |
|---|---|---|---|---|---|---|
| 32 | 12.768 | 12.769 | 12.797 | 0.029 | 13.768 | 16.264 |
| 64 | 12.100 | 12.101 | 12.115 | 0.015 | 12.600 | 13.711 |
| 128 | 11.667 | 11.667 | 11.677 | 0.011 | 11.917 | 12.466 |

## Baselines, linear tensors

| code | bits/weight |
|---|---|
| BF16 | 16.000 |
| Huffman, one code per transformer block (DFloat11-class) | 10.646 |
| Huffman, one code for all linear tensors | 10.647 |
| Huffman, one code per tensor | 10.621 |
| Entropy bound, per-tensor distributions | 10.581 |
| 7-exponent window per tensor, 3-bit code, outliers as BF16 (ZipServ-class) | 11.432 |
| 16-entry palette per tensor, 4-bit index, verbatim 64-weight segments (Unweight-class) | 12.043 |
| Palette per tensor, b = ceil(log2 distinct) | 13.101 |
| xp best-of (per-tile palettes), T=128 | 11.667 |
| xp best-of, T=128, + .off | 11.917 |
| rank_fixed (per-tensor ranks, fixed width per tile; ENEC-class), T=128 | 11.929 |
| rank_best (per-tensor ranks, width or escape per tile), T=128 | 11.265 |
| rank_unary (per-tensor ranks, truncated unary), T=128 | 10.818 |
| rank_rice (per-tensor ranks, Rice k in 0..2 per tile), T=128 | 10.711 |
| rank_best, T=128, + .off | 11.515 |
| rank_rice, T=128, + .off | 10.961 |

## All policies, linear tensors (real / tensor-shuffled)

| policy | T=32 | T=64 | T=128 |
|---|---|---|---|
| bump | 12.983 / 13.005 | 12.401 / 12.436 | 12.342 / 12.378 |
| escape | 12.773 / 12.801 | 12.103 / 12.118 | 11.672 / 11.682 |
| verbatim | 16.250 / 16.250 | 16.125 / 16.125 | 16.062 / 16.062 |
| best | 12.768 / 12.797 | 12.100 / 12.115 | 11.667 / 11.677 |
| unary | 12.459 / 12.498 | 11.739 / 11.775 | 11.310 / 11.344 |
| rank_fixed | 11.637 / 11.642 | 11.754 / 11.763 | 11.929 / 11.939 |
| rank_best | 11.394 / 11.397 | 11.304 / 11.306 | 11.265 / 11.266 |
| rank_unary | 10.973 / 10.973 | 10.869 / 10.870 | 10.818 / 10.818 |
| rank_rice | 10.830 / 10.833 | 10.749 / 10.752 | 10.711 / 10.712 |

## Per class, best-of at T=128

| class | tensors | weights | real | tensor-shuffled | gain | Huffman (class code) |
|---|---|---|---|---|---|---|
| attn | 36 | 0.528B | 11.677 | 11.705 | 0.028 | 10.678 |
| linattn | 72 | 1.611B | 11.668 | 11.678 | 0.010 | 10.628 |
| mlp | 99 | 4.983B | 11.665 | 11.674 | 0.009 | 10.622 |
| embed | 2 | 2.034B | 11.669 | 11.687 | 0.018 | 10.610 |

## Distinct exponents per tile (k_t), linear tensors

| T | layout | k=1 | k<=2 | k<=4 | k<=8 | k<=16 | mean k |
|---|---|---|---|---|---|---|---|
| 32 | real | 0.0% | 0.0% | 0.7% | 95.2% | 100.0% | 6.77 |
| 32 | row-shuffled | 0.0% | 0.0% | 0.7% | 95.2% | 100.0% | 6.77 |
| 32 | tensor-shuffled | 0.0% | 0.0% | 0.6% | 94.5% | 100.0% | 6.83 |
| 64 | real | 0.0% | 0.0% | 0.0% | 71.5% | 100.0% | 7.93 |
| 64 | row-shuffled | 0.0% | 0.0% | 0.0% | 71.4% | 100.0% | 7.93 |
| 64 | tensor-shuffled | 0.0% | 0.0% | 0.0% | 69.0% | 100.0% | 8.00 |
| 128 | real | 0.0% | 0.0% | 0.0% | 28.9% | 100.0% | 9.10 |
| 128 | row-shuffled | 0.0% | 0.0% | 0.0% | 28.8% | 100.0% | 9.10 |
| 128 | tensor-shuffled | 0.0% | 0.0% | 0.0% | 25.9% | 100.0% | 9.19 |

## Best-of choices, share of tiles, linear tensors, T=128

| layout | plain0 | plain1 | plain2 | plain3 | plain4 | esc1 | esc2 | esc3 | esc4 | verb |
|---|---|---|---|---|---|---|---|---|---|---|
| real | 0.0% | 0.0% | 0.0% | 28.8% | 0.0% | 0.0% | 1.0% | 70.2% | 0.0% | 0.0% |
| tensor-shuffled | 0.0% | 0.0% | 0.0% | 25.8% | 0.0% | 0.0% | 0.6% | 73.5% | 0.0% | 0.0% |

## Lossy dial, linear tensors, T=128 (report only)

Per-tile relative error after keeping m mantissa bits (round half to even, exponent exact). Percentiles are upper edges of log bins 1.2% wide; max is exact.

| m | rel Frob p50 | p90 | p99 | p99.9 | max | max-rel p50 | p99 | max |
|---|---|---|---|---|---|---|---|---|
| 2 | 5.96e-02 | 6.53e-02 | 7.00e-02 | 7.50e-02 | 1.19e-01 | 1.22e-01 | 1.22e-01 | 1.22e-01 |
| 3 | 2.82e-02 | 3.09e-02 | 3.35e-02 | 3.59e-02 | 5.77e-02 | 5.89e-02 | 5.89e-02 | 5.88e-02 |
| 4 | 1.38e-02 | 1.53e-02 | 1.66e-02 | 1.78e-02 | 2.96e-02 | 3.05e-02 | 3.05e-02 | 3.03e-02 |
| 5 | 7.08e-03 | 7.94e-03 | 8.61e-03 | 9.23e-03 | 1.46e-02 | 1.55e-02 | 1.55e-02 | 1.54e-02 |
| 6 | 4.07e-03 | 4.52e-03 | 4.90e-03 | 5.19e-03 | 7.48e-03 | 7.76e-03 | 7.76e-03 | 7.75e-03 |
| 7 | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 |

## Embedding

2.034B weights (21.1% of parameters). Best-of T=128: 11.669 real, 11.687 tensor-shuffled; Huffman one code 10.610.

## Classification

| class | sub | tensors | params | share | dtypes |
|---|---|---|---|---|---|
| linear | mlp | 99 | 4,982,833,152 | 51.62% | BF16 |
| linear | linattn | 72 | 1,610,612,736 | 16.68% | BF16 |
| linear | attn | 36 | 528,482,304 | 5.47% | BF16 |
| embed | lm_head | 1 | 1,017,118,720 | 10.54% | BF16 |
| embed | embed_tokens | 1 | 1,017,118,720 | 10.54% | BF16 |
| mm | mm | 333 | 456,010,480 | 4.72% | BF16 |
| other | other | 121 | 40,633,856 | 0.42% | BF16,F32 |
| other | norm | 112 | 294,400 | 0.00% | BF16,F32 |

## Definitions

- Tile: T consecutive weights along K in one row; k_t = distinct exponents in the tile.
- bump: b = ceil(log2 k_t) index bits with a k_t-entry palette, verbatim if k_t > 16. escape: b in 1..4, top 2^b-1 exponents in the palette, the last code escapes to an 8-bit exponent. best: cheapest of bump, escape (any b) and verbatim per tile. unary: k_t-entry palette sorted by frequency, truncated unary code of the rank (candidate for the fallback of DESIGN.md 5).
- Rank family (no per-tile palette): one table per tensor orders exponents by frequency. rank_fixed: each tile stores ranks at the bit length of its largest rank (ENEC-class). rank_best: that, or b in 1..4 with ranks >= 2^b-1 escaping to 8 bits, or verbatim. rank_unary: truncated unary code of the rank, the tile's largest rank in the meta byte. rank_rice: per tile, the cheapest Rice code (k low bits verbatim, truncated unary quotient) for k in 0..2.
- v0 padded palette: the v0 `.pal` layout (DESIGN.md 3), [ntiles, 16] uint8, 128 bits per tile; per-tile choice re-optimized for that cost.
- `.off` (32 bits per tile): a tile's length follows from its meta byte only for pure fixed-width tiles (bump, rank_fixed, verbatim). Escape, unary and Rice tiles are variable-length, so random access needs `.off`: add 32/T bits per weight (0.25 at T=128).
- Row-shuffled: exponents permuted within each row. Tensor-shuffled: permuted within the whole tensor (the shuffle control, DESIGN.md 5). Locality gain = tensor-shuffled minus real.
- Baselines code the exponent byte only and add the 8-bit raw byte. DFloat11-class: one Huffman code per transformer block (DFloat11 builds one codebook per block). ZipServ-class: per tensor, the 7 contiguous exponents covering the most weights at a fixed 3-bit code, every other weight also stored as a full 16-bit value. Unweight-class: per tensor, the 16 most frequent exponents at a 4-bit index; a 64-weight row segment holding any other exponent keeps 8-bit exponents; one flag bit per segment. These are models of each scheme's exponent coding, not reimplementations.
