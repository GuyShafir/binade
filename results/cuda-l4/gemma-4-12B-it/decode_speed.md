# Decode speed, fresh process per model: google/gemma-4-12B-it

`scripts/decode_speed.py` @ `4d2376c`: 3 rounds of Binade R4 then MLX 8-bit, each in its own process, 60 s cool-down between processes, 256 greedy tokens timed after a 16-token warm-up (mlx_lm generate_step).

| model | tok/s per run | median tok/s | median step ms | peak GB |
|---|---|---|---|---|
| binade_r4 | 11.12, 11.15, 11.13 | 11.13 | 89.8 | 20.0 |
| q8 | 17.88, 17.88, 17.88 | 17.88 | 55.9 | 14.7 |

Token ids identical across all runs of the exact models: True.
MLX 8-bit (lossy) / Binade R4: 1.61x; its 256 greedy tokens identical to the exact models': False.
