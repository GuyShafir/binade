# Decode speed, fresh process per model: google/gemma-4-26B-A4B-it

`scripts/decode_speed.py` @ `864d60d`: 3 rounds of Binade R4, each in its own process, 60 s cool-down between processes, 256 greedy tokens timed after a 16-token warm-up (mlx_lm generate_step).

| model | tok/s per run | median tok/s | median step ms | peak GB |
|---|---|---|---|---|
| binade_r4 | 65.42, 65.13, 65.47 | 65.42 | 15.3 | 39.1 |

Token ids identical across all runs: True.
