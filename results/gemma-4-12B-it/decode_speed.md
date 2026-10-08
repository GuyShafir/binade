# Decode speed, fresh process per model: google/gemma-4-12B-it

`scripts/decode_speed.py` @ `8f3fc48`: 3 rounds of BF16 then Binade R4 then MLX 8-bit, each in its own process, 60 s cool-down between processes, 256 greedy tokens timed after a 16-token warm-up (mlx_lm generate_step).

| model | tok/s per run | median tok/s | median step ms | peak GB |
|---|---|---|---|---|
| bf16 | 14.80, 19.15, 19.19 | 19.15 | 52.1 | 24.0 |
| binade_r4 | 24.66, 24.45, 24.75 | 24.66 | 40.6 | 20.0 |
| q8 | 35.06, 34.58, 35.02 | 35.02 | 28.5 | 14.7 |

Binade R4 / BF16: **1.29x**. Token ids identical across all runs of BF16 and Binade: True.
MLX 8-bit (lossy) / BF16: 1.83x; its 256 greedy tokens identical to the exact models': False.
