# Binade kernel test: batch-1 matvec on Gemma 4 12B shapes

`scripts/kernel_bench.py` @ `e652168`, MLX 0.32.2, Apple M4 Max (40-core GPU, 546 GB/s peak). Real 12B weights; each shape cycles through distinct copies of at least 1 GB of BF16 so every matvec streams from DRAM; 7 paired rounds, medians of per-round ratios. Efficiency = a method's GB/s over BF16's (weight bytes it reads). binade_v1 reads the v1 file arrays in place; binade_r4 reads the 4-bit runtime layout transcoded from v1 at load. Both produce outputs bit-identical to MLX's BF16 matvec.

| projection | shape | per token | BF16 GB/s | 8-bit eff / speedup | Binade v1 eff / speedup | Binade R4 eff / speedup | bit-identical (v1, R4) |
|---|---|---|---|---|---|---|---|
| lm_head (tied embedding) | 262144x3840 | 1 | 500 | 99.2% / 1.87x | 74.5% / 1.06x | 102.7% / 1.36x | True, True |
| down_proj | 3840x15360 | 48 | 495 | 99.7% / 1.88x | 72.2% / 1.02x | 100.4% / 1.33x | True, True |
| gate_up_proj | 15360x3840 | 96 | 493 | 99.4% / 1.87x | 73.6% / 1.03x | 100.9% / 1.34x | True, True |
| o_proj | 3840x8192 | 8 | 498 | 100.1% / 1.88x | 72.0% / 1.02x | 99.2% / 1.31x | True, True |
| q_proj | 8192x3840 | 8 | 495 | 98.8% / 1.86x | 72.5% / 1.03x | 100.5% / 1.33x | True, True |
| o_proj | 3840x4096 | 40 | 493 | 99.6% / 1.87x | 71.3% / 1.00x | 98.3% / 1.30x | True, True |
| q_proj | 4096x3840 | 40 | 495 | 98.2% / 1.85x | 71.4% / 1.00x | 98.2% / 1.30x | True, True |
| k_proj | 2048x3840 | 40 | 490 | 97.0% / 1.83x | 65.8% / 0.92x | 93.9% / 1.24x | True, True |
| v_proj | 2048x3840 | 40 | 485 | 97.2% / 1.83x | 66.9% / 0.93x | 92.9% / 1.23x | True, True |
| k_proj | 512x3840 | 8 | 466 | 91.5% / 1.72x | 51.5% / 0.73x | 73.6% / 0.98x | True, True |

Per token (every projection of the 12B once plus lm_head; sum of per-shape median times):

| method | ms | bytes vs BF16 | GB/s | % of peak | efficiency vs BF16 | speedup |
|---|---|---|---|---|---|---|
| bf16 | 48.2 | 1.000 | 494 | 91% | 100.0% | 1.00x |
| q8 | 25.7 | 0.531 | 491 | 90% | 99.4% | 1.87x |
| binade_v1 | 47.3 | 0.712 | 359 | 66% | 72.6% | 1.02x |
| binade_r4 | 36.2 | 0.755 | 496 | 91% | 100.3% | 1.33x |
