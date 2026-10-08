# Decode speed, fresh process per model: google/gemma-4-31B-it

`scripts/decode_speed.py` @ `4d2376c`: 3 rounds of BF16 then Binade R4 then MLX 8-bit, each in its own process, 60 s cool-down between processes, 256 greedy tokens timed after a 16-token warm-up (mlx_lm generate_step).

| model | tok/s per run | median tok/s | median step ms | peak GB |
|---|---|---|---|---|
| bf16 | 22.99, 22.91, 22.91 | 22.91 | 43.6 | 62.1 |
| binade_r4 | 25.31, 25.29, 25.24 | 25.29 | 39.5 | 49.2 |
| q8 | 39.42, 39.43, 39.44 | 39.43 | 25.3 | 35.4 |

Binade R4 / BF16: **1.10x**. Token ids identical across all runs of BF16 and Binade: True.
MLX 8-bit (lossy) / BF16: 1.72x; its 256 greedy tokens identical to the exact models': False.
