# Decode speed, fresh process per model: google/gemma-4-12B-it

`scripts/decode_speed.py` @ `864d60d`: 3 rounds of BF16 then Binade R4, each in its own process, 60 s cool-down between processes, 256 greedy tokens timed after a 16-token warm-up (mlx_lm generate_step).

| model | tok/s per run | median tok/s | median step ms | peak GB |
|---|---|---|---|---|
| bf16 | 19.31, 19.36, 19.35 | 19.35 | 51.7 | 24.0 |
| binade_r4 | 24.91, 24.83, 24.87 | 24.87 | 40.2 | 18.8 |

Binade R4 / BF16: **1.29x**. Token ids identical across all runs of both models: True.
