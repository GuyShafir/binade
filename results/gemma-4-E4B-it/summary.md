# xp phase 1: google/gemma-4-E4B-it

Source `google/gemma-4-E4B-it` @ `ee0ef60`. Script `scripts/histogram.py` @ `bb6988f`, seed 0, 2026-09-26, 3.1 min, peak RSS 1.5 GB.

Scope: 379 linear tensors, 4.03B weights, 50.4% of all parameters. Bits per weight include the 8-bit raw byte (sign + mantissa) and the 8-bit per-tile meta byte; palettes are charged at actual entries and the `.off` array is excluded unless a column says otherwise.

## Verdict

Locality gain at T=128 (best-of, tensor-shuffled minus real): **0.027 bits/weight** (11.704 - 11.677). Band `stop`: stop; the unary/Rice fallback the design calls for in this band (DESIGN.md 5) is reported below as rank_unary and rank_rice.

Position at T=128: xp best-of 11.677 vs DFloat11-class Huffman 10.680. Cheapest fixed-width code measured: rank_best at T=128, 11.292. Cheapest code measured without a Huffman table: rank_rice at T=128, 10.737.

## Best-of, linear tensors

| T | real | row-shuffled | tensor-shuffled | locality gain | real + .off | real, v0 padded palette + .off |
|---|---|---|---|---|---|---|
| 32 | 12.787 | 12.792 | 12.839 | 0.052 | 13.787 | 16.269 |
| 64 | 12.112 | 12.114 | 12.143 | 0.031 | 12.612 | 13.720 |
| 128 | 11.677 | 11.679 | 11.704 | 0.027 | 11.927 | 12.476 |

## Baselines, linear tensors

| code | bits/weight |
|---|---|
| BF16 | 16.000 |
| Huffman, one code per transformer block (DFloat11-class) | 10.680 |
| Huffman, one code for all linear tensors | 10.688 |
| Huffman, one code per tensor | 10.648 |
| Entropy bound, per-tensor distributions | 10.611 |
| 7-exponent window per tensor, 3-bit code, outliers as BF16 (ZipServ-class) | 11.501 |
| 16-entry palette per tensor, 4-bit index, verbatim 64-weight segments (Unweight-class) | 12.054 |
| Palette per tensor, b = ceil(log2 distinct) | 13.007 |
| xp best-of (per-tile palettes), T=128 | 11.677 |
| xp best-of, T=128, + .off | 11.927 |
| rank_fixed (per-tensor ranks, fixed width per tile; ENEC-class), T=128 | 11.948 |
| rank_best (per-tensor ranks, width or escape per tile), T=128 | 11.292 |
| rank_unary (per-tensor ranks, truncated unary), T=128 | 10.867 |
| rank_rice (per-tensor ranks, Rice k in 0..2 per tile), T=128 | 10.737 |
| rank_best, T=128, + .off | 11.542 |
| rank_rice, T=128, + .off | 10.987 |

## All policies, linear tensors (real / tensor-shuffled)

| policy | T=32 | T=64 | T=128 |
|---|---|---|---|
| bump | 13.006 / 13.057 | 12.434 / 12.512 | 12.366 / 12.438 |
| escape | 12.791 / 12.843 | 12.115 / 12.147 | 11.682 / 11.709 |
| verbatim | 16.250 / 16.250 | 16.125 / 16.125 | 16.062 / 16.062 |
| best | 12.787 / 12.839 | 12.112 / 12.143 | 11.677 / 11.704 |
| unary | 12.492 / 12.567 | 11.768 / 11.839 | 11.335 / 11.402 |
| rank_fixed | 11.661 / 11.672 | 11.781 / 11.796 | 11.948 / 11.963 |
| rank_best | 11.415 / 11.422 | 11.328 / 11.335 | 11.292 / 11.298 |
| rank_unary | 11.022 / 11.024 | 10.919 / 10.920 | 10.867 / 10.868 |
| rank_rice | 10.859 / 10.864 | 10.777 / 10.781 | 10.737 / 10.741 |

## Per class, best-of at T=128

| class | tensors | weights | real | tensor-shuffled | gain | Huffman (class code) |
|---|---|---|---|---|---|---|
| attn | 168 | 0.642B | 11.672 | 11.715 | 0.043 | 10.691 |
| mlp | 126 | 3.303B | 11.677 | 11.697 | 0.020 | 10.674 |
| ple_proj | 85 | 0.083B | 11.717 | 11.924 | 0.207 | 11.110 |
| embed | 1 | 0.671B | 11.669 | 11.672 | 0.004 | 10.598 |

## Distinct exponents per tile (k_t), linear tensors

| T | layout | k=1 | k<=2 | k<=4 | k<=8 | k<=16 | mean k |
|---|---|---|---|---|---|---|---|
| 32 | real | 0.0% | 0.0% | 0.6% | 94.5% | 100.0% | 6.83 |
| 32 | row-shuffled | 0.0% | 0.0% | 0.6% | 94.4% | 100.0% | 6.84 |
| 32 | tensor-shuffled | 0.0% | 0.0% | 0.5% | 92.6% | 100.0% | 6.95 |
| 64 | real | 0.0% | 0.0% | 0.0% | 69.2% | 100.0% | 8.00 |
| 64 | row-shuffled | 0.0% | 0.0% | 0.0% | 68.8% | 100.0% | 8.01 |
| 64 | tensor-shuffled | 0.0% | 0.0% | 0.0% | 63.4% | 100.0% | 8.17 |
| 128 | real | 0.0% | 0.0% | 0.0% | 26.9% | 100.0% | 9.16 |
| 128 | row-shuffled | 0.0% | 0.0% | 0.0% | 26.5% | 100.0% | 9.18 |
| 128 | tensor-shuffled | 0.0% | 0.0% | 0.0% | 21.0% | 100.0% | 9.37 |

## Best-of choices, share of tiles, linear tensors, T=128

| layout | plain0 | plain1 | plain2 | plain3 | plain4 | esc1 | esc2 | esc3 | esc4 | verb |
|---|---|---|---|---|---|---|---|---|---|---|
| real | 0.0% | 0.0% | 0.0% | 26.8% | 0.0% | 0.0% | 0.9% | 72.3% | 0.0% | 0.0% |
| tensor-shuffled | 0.0% | 0.0% | 0.0% | 21.0% | 0.1% | 0.0% | 0.5% | 78.4% | 0.0% | 0.0% |

## Lossy dial, linear tensors, T=128 (report only)

Per-tile relative error after keeping m mantissa bits (round half to even, exponent exact). Percentiles are upper edges of log bins 1.2% wide; max is exact.

| m | rel Frob p50 | p90 | p99 | p99.9 | max | max-rel p50 | p99 | max |
|---|---|---|---|---|---|---|---|---|
| 2 | 5.89e-02 | 6.53e-02 | 7.08e-02 | 7.67e-02 | 1.18e-01 | 1.22e-01 | 1.22e-01 | 1.22e-01 |
| 3 | 2.82e-02 | 3.13e-02 | 3.39e-02 | 3.67e-02 | 5.73e-02 | 5.89e-02 | 5.89e-02 | 5.88e-02 |
| 4 | 1.38e-02 | 1.53e-02 | 1.68e-02 | 1.82e-02 | 2.95e-02 | 3.05e-02 | 3.05e-02 | 3.03e-02 |
| 5 | 7.08e-03 | 7.94e-03 | 8.71e-03 | 9.55e-03 | 1.49e-02 | 1.55e-02 | 1.55e-02 | 1.54e-02 |
| 6 | 4.07e-03 | 4.57e-03 | 4.95e-03 | 5.31e-03 | 7.56e-03 | 7.76e-03 | 7.76e-03 | 7.75e-03 |
| 7 | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 |

## Embedding

0.671B weights (8.4% of parameters). No `lm_head` tensor: the embedding is tied and also serves as the output projection, one full read per decoded token. Best-of T=128: 11.669 real, 11.672 tensor-shuffled; Huffman one code 10.598.

## Classification

| class | sub | tensors | params | share | dtypes |
|---|---|---|---|---|---|
| linear | mlp | 126 | 3,303,014,400 | 41.31% | BF16 |
| linear | attn | 168 | 642,252,800 | 8.03% | BF16 |
| linear | ple_proj | 85 | 82,575,360 | 1.03% | BF16 |
| embed | embed_tokens | 1 | 671,088,640 | 8.39% | BF16 |
| ple | ple | 1 | 2,818,572,288 | 35.25% | BF16 |
| mm | mm | 1411 | 478,088,384 | 5.98% | BF16 |
| other | norm | 296 | 565,504 | 0.01% | BF16 |
| other | other | 42 | 42 | 0.00% | BF16 |

## Definitions

- Tile: T consecutive weights along K in one row; k_t = distinct exponents in the tile.
- bump: b = ceil(log2 k_t) index bits with a k_t-entry palette, verbatim if k_t > 16. escape: b in 1..4, top 2^b-1 exponents in the palette, the last code escapes to an 8-bit exponent. best: cheapest of bump, escape (any b) and verbatim per tile. unary: k_t-entry palette sorted by frequency, truncated unary code of the rank (candidate for the fallback of DESIGN.md 5).
- Rank family (no per-tile palette): one table per tensor orders exponents by frequency. rank_fixed: each tile stores ranks at the bit length of its largest rank (ENEC-class). rank_best: that, or b in 1..4 with ranks >= 2^b-1 escaping to 8 bits, or verbatim. rank_unary: truncated unary code of the rank, the tile's largest rank in the meta byte. rank_rice: per tile, the cheapest Rice code (k low bits verbatim, truncated unary quotient) for k in 0..2.
- v0 padded palette: the v0 `.pal` layout (DESIGN.md 3), [ntiles, 16] uint8, 128 bits per tile; per-tile choice re-optimized for that cost.
- `.off` (32 bits per tile): a tile's length follows from its meta byte only for pure fixed-width tiles (bump, rank_fixed, verbatim). Escape, unary and Rice tiles are variable-length, so random access needs `.off`: add 32/T bits per weight (0.25 at T=128).
- Row-shuffled: exponents permuted within each row. Tensor-shuffled: permuted within the whole tensor (the shuffle control, DESIGN.md 5). Locality gain = tensor-shuffled minus real.
- Baselines code the exponent byte only and add the 8-bit raw byte. DFloat11-class: one Huffman code per transformer block (DFloat11 builds one codebook per block). ZipServ-class: per tensor, the 7 contiguous exponents covering the most weights at a fixed 3-bit code, every other weight also stored as a full 16-bit value. Unweight-class: per tensor, the 16 most frequent exponents at a 4-bit index; a 64-weight row segment holding any other exponent keeps 8-bit exponents; one flag bit per segment. These are models of each scheme's exponent coding, not reimplementations.
