# Decode speed, fresh process per model: google/gemma-4-26B-A4B-it

`scripts/decode_speed.py` @ `8f3fc48`: 3 rounds of Binade R4 then MLX 8-bit, each in its own process, 60 s cool-down between processes, 256 greedy tokens timed after a 16-token warm-up (mlx_lm generate_step).

| model | tok/s per run | median tok/s | median step ms | peak GB |
|---|---|---|---|---|
| binade_r4 | 58.82, 60.19, 58.17 | 58.82 | 16.6 | 39.1 |
| q8 | 78.02, 76.19, 77.05 | 77.05 | 13.1 | 28.7 |

Token ids identical across all runs of the exact models: True.
MLX 8-bit (lossy) / Binade R4: 1.31x; its 256 greedy tokens identical to the exact models': False.
