# xp phase 1: models compared

Linear tensors only. Bits per weight include the 8-bit raw byte (sign + mantissa) and the 8-bit per-tile meta byte.

## Locality (DESIGN.md 5 go/no-go), best-of at each model's best T

| model | linear weights | T | xp best-of | tensor-shuffled | locality gain | band |
|---|---|---|---|---|---|---|
| unsloth/Llama-3.1-8B-Instruct | 6.98B | 128 | 11.672 | 11.685 | 0.013 | stop |
| Qwen/Qwen3.5-9B | 7.12B | 128 | 11.667 | 11.677 | 0.011 | stop |
| google/gemma-4-12B-it | 10.90B | 128 | 11.679 | 11.698 | 0.019 | stop |
| google/gemma-4-E4B-it | 4.03B | 128 | 11.677 | 11.704 | 0.027 | stop |

## Size against prior-art exponent codings (our accounting of each on the same tensors; xp columns without / with a stored 32-bit offset per tile)

Unweight's Huffman mode is not modeled; it reports about 10.9 to 11.0 bits per weight on Llama-3.1-8B MLP. xp v1 files store no offsets.

| model | per-block Huffman (DFloat11 granularity) | 7-exponent window, 3-bit code, outliers as BF16 (ZipServ layout) | 16-exponent palette, fixed 4-bit index (Unweight 4-bit mode) | per-tile palettes (best-of) | xp v1 (rank_best) | Rice over ranks (rank_rice) |
|---|---|---|---|---|---|---|
| unsloth/Llama-3.1-8B-Instruct | 10.648 | 11.455 | 12.044 | 11.672 / 11.922 | 11.274 / 11.524 | 10.716 / 10.966 |
| Qwen/Qwen3.5-9B | 10.646 | 11.432 | 12.043 | 11.667 / 11.917 | 11.265 / 11.515 | 10.711 / 10.961 |
| google/gemma-4-12B-it | 10.666 | 11.464 | 12.051 | 11.679 / 11.929 | 11.276 / 11.526 | 10.740 / 10.990 |
| google/gemma-4-E4B-it | 10.680 | 11.501 | 12.054 | 11.677 / 11.927 | 11.292 / 11.542 | 10.737 / 10.987 |

## Packed checkpoints (xp v1; linear and embedding packed, everything else BF16)

| model | BF16 GB | xp GB | ratio | packed tensors | linear bits/weight (file) | round-trip mismatches |
|---|---|---|---|---|---|---|
| google/gemma-4-12B-it | 23.92 | 16.89 | 70.6% | 329 | 11.276 | 0 in 677 tensors |
| google/gemma-4-E4B-it | 15.99 | 13.23 | 82.7% | 380 | 11.292 | 0 in 2130 tensors |

## Generation, stock BF16 vs xp slow runtime (MLX ops decode per call, no kernel)

| model | prompt logits identical | greedy ids identical | BF16 tok/s | xp slow runtime tok/s |
|---|---|---|---|---|
| google/gemma-4-12B-it | True | True (256 tokens) | 10.08 | 0.10 |
| google/gemma-4-E4B-it | True | True (256 tokens) | 39.65 | 0.30 |

Sources:

- unsloth/Llama-3.1-8B-Instruct @ `4699cc7`: histogram.py @ `bb6988f`, 2026-09-26, details in `results/Llama-3.1-8B-Instruct/summary.md`
- Qwen/Qwen3.5-9B @ `c202236`: histogram.py @ `bb6988f`, 2026-09-26, details in `results/Qwen3.5-9B/summary.md`
- google/gemma-4-12B-it @ `707f0a3`: histogram.py @ `bb6988f`, 2026-09-26, details in `results/gemma-4-12B-it/summary.md`
- google/gemma-4-E4B-it @ `ee0ef60`: histogram.py @ `bb6988f`, 2026-09-26, details in `results/gemma-4-E4B-it/summary.md`
- gemma-4-12B-it/roundtrip.json: packed @ `040b9ce`, checked @ `040b9ce`
- gemma-4-E4B-it/roundtrip.json: packed @ `07601d3`, checked @ `07601d3`
- gemma-4-12B-it/generate_check.json: @ `6fbfc7e`
- gemma-4-E4B-it/generate_check.json: @ `040b9ce`
