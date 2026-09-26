# Decode speed, fresh process per model: google/gemma-4-12B-it

`scripts/decode_speed.py` @ `629104b`: 3 rounds of BF16 then Binade R4, each in its own process, 60 s cool-down between processes, 256 greedy tokens timed after a 16-token warm-up (mlx_lm generate_step).

| model | tok/s per run | median tok/s | median step ms | peak GB |
|---|---|---|---|---|
| bf16 | 55.83, 55.60, 55.60 | 55.60 | 18.0 | 24.1 |
| binade_r4 | 58.94, 58.86, 58.92 | 58.92 | 17.0 | 20.0 |

Binade R4 / BF16: **1.06x**. Token ids identical across all runs of both models: True.
