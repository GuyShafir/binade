# Decode speed, fresh process per model: google/gemma-4-26B-A4B-it

`scripts/decode_speed.py` @ `4d2376c`: 3 rounds of Binade R4 then MLX 8-bit, each in its own process, 60 s cool-down between processes, 256 greedy tokens timed after a 16-token warm-up (mlx_lm generate_step).

| model | tok/s per run | median tok/s | median step ms | peak GB |
|---|---|---|---|---|
| binade_r4 | 44.81, 46.49, 46.58 | 46.49 | 21.4 | 40.0 |
| q8 | 58.14, 54.33, 55.18 | 55.18 | 18.0 | 28.3 |

Token ids identical across all runs of the exact models: True.
MLX 8-bit (lossy) / Binade R4: 1.19x; its 256 greedy tokens identical to the exact models': False.
