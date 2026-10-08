# Decode speed, fresh process per model: Qwen/Qwen3.5-9B

`scripts/decode_speed.py` @ `4d2376c`: 3 rounds of BF16 then Binade R4 then MLX 8-bit, each in its own process, 60 s cool-down between processes, 256 greedy tokens timed after a 16-token warm-up (mlx_lm generate_step).

| model | tok/s per run | median tok/s | median step ms | peak GB |
|---|---|---|---|---|
| bf16 | 82.74, 82.24, 82.25 | 82.25 | 12.2 | 18.4 |
| binade_r4 | 90.93, 90.77, 90.82 | 90.82 | 11.0 | 16.5 |
| q8 | 133.01, 132.99, 133.01 | 133.01 | 7.5 | 13.6 |

Binade R4 / BF16: **1.10x**. Token ids identical across all runs of BF16 and Binade: True.
MLX 8-bit (lossy) / BF16: 1.62x; its 256 greedy tokens identical to the exact models': False.
