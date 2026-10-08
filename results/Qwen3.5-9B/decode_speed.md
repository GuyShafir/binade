# Decode speed, fresh process per model: Qwen/Qwen3.5-9B

`scripts/decode_speed.py` @ `8f3fc48`: 3 rounds of BF16 then Binade R4 then MLX 8-bit, each in its own process, 60 s cool-down between processes, 256 greedy tokens timed after a 16-token warm-up (mlx_lm generate_step).

| model | tok/s per run | median tok/s | median step ms | peak GB |
|---|---|---|---|---|
| bf16 | 26.90, 26.63, 26.91 | 26.90 | 36.9 | 18.0 |
| binade_r4 | 35.64, 31.51, 34.42 | 34.42 | 28.3 | 15.6 |
| q8 | 48.74, 47.35, 50.07 | 48.74 | 20.3 | 13.6 |

Binade R4 / BF16: **1.28x**. Token ids identical across all runs of BF16 and Binade: True.
MLX 8-bit (lossy) / BF16: 1.81x; its 256 greedy tokens identical to the exact models': False.
