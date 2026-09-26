# xp phase 1: google/gemma-4-12B-it

Source `google/gemma-4-12B-it` @ `707f0a3`. Script `scripts/histogram.py` @ `bb6988f`, seed 0, 2026-09-26, 7.7 min, peak RSS 2.0 GB.

Scope: 328 linear tensors, 10.90B weights, 91.1% of all parameters. Bits per weight include the 8-bit raw byte (sign + mantissa) and the 8-bit per-tile meta byte; palettes are charged at actual entries and the `.off` array is excluded unless a column says otherwise.

## Verdict

Locality gain at T=128 (best-of, tensor-shuffled minus real): **0.019 bits/weight** (11.698 - 11.679). Band `stop`: stop; the unary/Rice fallback the design calls for in this band (DESIGN.md 5) is reported below as rank_unary and rank_rice.

Position at T=128: xp best-of 11.679 vs DFloat11-class Huffman 10.666. Cheapest fixed-width code measured: rank_best at T=128, 11.276. Cheapest code measured without a Huffman table: rank_rice at T=128, 10.740.

## Best-of, linear tensors

| T | real | row-shuffled | tensor-shuffled | locality gain | real + .off | real, v0 padded palette + .off |
|---|---|---|---|---|---|---|
| 32 | 12.783 | 12.786 | 12.826 | 0.044 | 13.783 | 16.269 |
| 64 | 12.111 | 12.113 | 12.135 | 0.024 | 12.611 | 13.722 |
| 128 | 11.679 | 11.680 | 11.698 | 0.019 | 11.929 | 12.478 |

## Baselines, linear tensors

| code | bits/weight |
|---|---|
| BF16 | 16.000 |
| Huffman, one code per transformer block (DFloat11-class) | 10.666 |
| Huffman, one code for all linear tensors | 10.765 |
| Huffman, one code per tensor | 10.647 |
| Entropy bound, per-tensor distributions | 10.611 |
| 7-exponent window per tensor, 3-bit code, outliers as BF16 (ZipServ-class) | 11.464 |
| 16-entry palette per tensor, 4-bit index, verbatim 64-weight segments (Unweight-class) | 12.051 |
| Palette per tensor, b = ceil(log2 distinct) | 13.038 |
| xp best-of (per-tile palettes), T=128 | 11.679 |
| xp best-of, T=128, + .off | 11.929 |
| rank_fixed (per-tensor ranks, fixed width per tile; ENEC-class), T=128 | 11.920 |
| rank_best (per-tensor ranks, width or escape per tile), T=128 | 11.276 |
| rank_unary (per-tensor ranks, truncated unary), T=128 | 10.868 |
| rank_rice (per-tensor ranks, Rice k in 0..2 per tile), T=128 | 10.740 |
| rank_best, T=128, + .off | 11.526 |
| rank_rice, T=128, + .off | 10.990 |

## All policies, linear tensors (real / tensor-shuffled)

| policy | T=32 | T=64 | T=128 |
|---|---|---|---|
| bump | 13.016 / 13.062 | 12.442 / 12.507 | 12.364 / 12.415 |
| escape | 12.787 / 12.830 | 12.115 / 12.140 | 11.685 / 11.705 |
| verbatim | 16.250 / 16.250 | 16.125 / 16.125 | 16.062 / 16.062 |
| best | 12.783 / 12.826 | 12.111 / 12.135 | 11.679 / 11.698 |
| unary | 12.503 / 12.573 | 11.775 / 11.838 | 11.339 / 11.396 |
| rank_fixed | 11.628 / 11.638 | 11.742 / 11.756 | 11.920 / 11.935 |
| rank_best | 11.397 / 11.401 | 11.312 / 11.314 | 11.276 / 11.279 |
| rank_unary | 11.023 / 11.024 | 10.920 / 10.920 | 10.868 / 10.869 |
| rank_rice | 10.862 / 10.867 | 10.779 / 10.783 | 10.740 / 10.743 |

## Per class, best-of at T=128

| class | tensors | weights | real | tensor-shuffled | gain | Huffman (class code) |
|---|---|---|---|---|---|---|
| attn | 184 | 2.406B | 11.678 | 11.718 | 0.040 | 10.751 |
| mlp | 144 | 8.493B | 11.679 | 11.692 | 0.013 | 10.768 |
| embed | 1 | 1.007B | 11.664 | 11.667 | 0.003 | 10.605 |

## Distinct exponents per tile (k_t), linear tensors

| T | layout | k=1 | k<=2 | k<=4 | k<=8 | k<=16 | mean k |
|---|---|---|---|---|---|---|---|
| 32 | real | 0.0% | 0.0% | 0.6% | 94.3% | 100.0% | 6.86 |
| 32 | row-shuffled | 0.0% | 0.0% | 0.6% | 94.2% | 100.0% | 6.87 |
| 32 | tensor-shuffled | 0.0% | 0.0% | 0.5% | 92.8% | 100.0% | 6.98 |
| 64 | real | 0.0% | 0.0% | 0.0% | 68.7% | 100.0% | 8.03 |
| 64 | row-shuffled | 0.0% | 0.0% | 0.0% | 68.3% | 100.0% | 8.04 |
| 64 | tensor-shuffled | 0.0% | 0.0% | 0.0% | 63.9% | 100.0% | 8.17 |
| 128 | real | 0.0% | 0.0% | 0.0% | 27.1% | 100.0% | 9.15 |
| 128 | row-shuffled | 0.0% | 0.0% | 0.0% | 26.7% | 100.0% | 9.16 |
| 128 | tensor-shuffled | 0.0% | 0.0% | 0.0% | 22.8% | 100.0% | 9.29 |

## Best-of choices, share of tiles, linear tensors, T=128

| layout | plain0 | plain1 | plain2 | plain3 | plain4 | esc1 | esc2 | esc3 | esc4 | verb |
|---|---|---|---|---|---|---|---|---|---|---|
| real | 0.0% | 0.0% | 0.0% | 26.8% | 0.0% | 0.0% | 1.4% | 71.8% | 0.0% | 0.0% |
| tensor-shuffled | 0.0% | 0.0% | 0.0% | 22.7% | 0.0% | 0.0% | 1.0% | 76.3% | 0.0% | 0.0% |

## Lossy dial, linear tensors, T=128 (report only)

Per-tile relative error after keeping m mantissa bits (round half to even, exponent exact). Percentiles are upper edges of log bins 1.2% wide; max is exact.

| m | rel Frob p50 | p90 | p99 | p99.9 | max | max-rel p50 | p99 | max |
|---|---|---|---|---|---|---|---|---|
| 2 | 5.96e-02 | 6.53e-02 | 7.08e-02 | 7.59e-02 | 1.18e-01 | 1.22e-01 | 1.22e-01 | 1.22e-01 |
| 3 | 2.82e-02 | 3.13e-02 | 3.39e-02 | 3.63e-02 | 5.74e-02 | 5.89e-02 | 5.89e-02 | 5.88e-02 |
| 4 | 1.38e-02 | 1.53e-02 | 1.68e-02 | 1.80e-02 | 2.97e-02 | 3.05e-02 | 3.05e-02 | 3.03e-02 |
| 5 | 7.08e-03 | 7.94e-03 | 8.71e-03 | 9.44e-03 | 1.46e-02 | 1.55e-02 | 1.55e-02 | 1.54e-02 |
| 6 | 4.07e-03 | 4.57e-03 | 4.95e-03 | 5.31e-03 | 7.54e-03 | 7.76e-03 | 7.76e-03 | 7.75e-03 |
| 7 | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 |

## Embedding

1.007B weights (8.4% of parameters). No `lm_head` tensor: the embedding is tied and also serves as the output projection, one full read per decoded token. Best-of T=128: 11.664 real, 11.667 tensor-shuffled; Huffman one code 10.605.

## Classification

| class | sub | tensors | params | share | dtypes |
|---|---|---|---|---|---|
| linear | mlp | 144 | 8,493,465,600 | 71.02% | BF16 |
| linear | attn | 184 | 2,406,481,920 | 20.12% | BF16 |
| embed | embed_tokens | 1 | 1,006,632,960 | 8.42% | BF16 |
| mm | mm | 11 | 52,379,904 | 0.44% | BF16 |
| other | norm | 289 | 769,792 | 0.01% | BF16 |
| other | other | 48 | 48 | 0.00% | BF16 |

## Definitions

- Tile: T consecutive weights along K in one row; k_t = distinct exponents in the tile.
- bump: b = ceil(log2 k_t) index bits with a k_t-entry palette, verbatim if k_t > 16. escape: b in 1..4, top 2^b-1 exponents in the palette, the last code escapes to an 8-bit exponent. best: cheapest of bump, escape (any b) and verbatim per tile. unary: k_t-entry palette sorted by frequency, truncated unary code of the rank (candidate for the fallback of DESIGN.md 5).
- Rank family (no per-tile palette): one table per tensor orders exponents by frequency. rank_fixed: each tile stores ranks at the bit length of its largest rank (ENEC-class). rank_best: that, or b in 1..4 with ranks >= 2^b-1 escaping to 8 bits, or verbatim. rank_unary: truncated unary code of the rank, the tile's largest rank in the meta byte. rank_rice: per tile, the cheapest Rice code (k low bits verbatim, truncated unary quotient) for k in 0..2.
- v0 padded palette: the v0 `.pal` layout (DESIGN.md 3), [ntiles, 16] uint8, 128 bits per tile; per-tile choice re-optimized for that cost.
- `.off` (32 bits per tile): a tile's length follows from its meta byte only for pure fixed-width tiles (bump, rank_fixed, verbatim). Escape, unary and Rice tiles are variable-length, so random access needs `.off`: add 32/T bits per weight (0.25 at T=128).
- Row-shuffled: exponents permuted within each row. Tensor-shuffled: permuted within the whole tensor (the shuffle control, DESIGN.md 5). Locality gain = tensor-shuffled minus real.
- Baselines code the exponent byte only and add the 8-bit raw byte. DFloat11-class: one Huffman code per transformer block (DFloat11 builds one codebook per block). ZipServ-class: per tensor, the 7 contiguous exponents covering the most weights at a fixed 3-bit code, every other weight also stored as a full 16-bit value. Unweight-class: per tensor, the 16 most frequent exponents at a 4-bit index; a 64-weight row segment holding any other exponent keeps 8-bit exponents; one flag bit per segment. These are models of each scheme's exponent coding, not reimplementations.
