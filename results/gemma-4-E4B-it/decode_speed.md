# Decode speed, fresh process per model: google/gemma-4-E4B-it

`scripts/decode_speed.py` @ `8f3fc48`: 3 rounds of BF16 then Binade R4 then MLX 8-bit, each in its own process, 60 s cool-down between processes, 256 greedy tokens timed after a 16-token warm-up (mlx_lm generate_step).

| model | tok/s per run | median tok/s | median step ms | peak GB |
|---|---|---|---|---|
| bf16 | 42.46, 41.55, 42.21 | 42.21 | 24.1 | 15.0 |
| binade_r4 | 50.67, 54.03, 52.56 | 52.56 | 19.0 | 14.0 |
| q8 | 75.03, 71.55, 75.59 | 75.03 | 13.0 | 15.0 |

Binade R4 / BF16: **1.25x**. Token ids identical across all runs of BF16 and Binade: True.
MLX 8-bit (lossy) / BF16: 1.78x; its 256 greedy tokens identical to the exact models': False.
