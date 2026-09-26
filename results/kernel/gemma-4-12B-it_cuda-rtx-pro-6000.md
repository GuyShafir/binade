# Batch-1 matvec on Gemma 4 12B shapes: NVIDIA RTX PRO 6000 Blackwell Server Edition

`scripts/matvec_bench.py` @ `629104b`, MLX 0.32.2, NVIDIA RTX PRO 6000 Blackwell Server Edition (1597 GB/s peak). Real 12B weights; each shape cycles through distinct copies of at least 1 GB of BF16; 7 paired rounds, medians of per-round ratios. Efficiency = a method's GB/s over BF16's. binade_r4: the R4 matvec as the runtime runs it.

| projection | shape | per token | BF16 GB/s | q8 eff / speedup | binade_r4 eff / speedup | R4 bit-identical |
|---|---|---|---|---|---|---|
| lm_head (tied embedding) | 262144x3840 | 1 | 1523 | 99.0% / 1.86x | 97.1% / 1.29x | True |
| down_proj | 3840x15360 | 48 | 1531 | 94.0% / 1.77x | 84.6% / 1.12x | True |
| gate_up_proj | 15360x3840 | 96 | 1585 | 91.0% / 1.71x | 90.2% / 1.19x | True |
| o_proj | 3840x8192 | 8 | 1660 | 90.6% / 1.71x | 80.4% / 1.07x | True |
| q_proj | 8192x3840 | 8 | 1676 | 89.2% / 1.68x | 80.3% / 1.06x | True |
| o_proj | 3840x4096 | 40 | 1576 | 108.7% / 2.05x | 76.0% / 1.01x | True |
| q_proj | 4096x3840 | 40 | 1560 | 107.9% / 2.03x | 77.6% / 1.03x | True |
| k_proj | 2048x3840 | 40 | 1352 | 95.6% / 1.80x | 73.6% / 0.98x | True |
| v_proj | 2048x3840 | 40 | 1354 | 95.5% / 1.80x | 73.7% / 0.98x | True |
| k_proj | 512x3840 | 8 | 1058 | 54.0% / 1.02x | 38.1% / 0.51x | True |

Per token (every projection once plus lm_head):

| method | ms | bytes vs BF16 | GB/s | % of peak | efficiency vs BF16 | speedup |
|---|---|---|---|---|---|---|
| bf16 | 15.33 | 1.000 | 1553 | 97% | 100.0% | 1.00x |
| q8 | 8.65 | 0.531 | 1462 | 92% | 94.1% | 1.77x |
| binade_r4 | 13.48 | 0.755 | 1333 | 83% | 85.8% | 1.14x |
