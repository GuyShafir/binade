# Decode speed, fresh process per model: google/gemma-4-E4B-it

`scripts/decode_speed.py` @ `25c9e4a`: 2 rounds of BF16 then Binade R4, each in its own process, 60 s cool-down between processes, 256 greedy tokens timed after a 16-token warm-up (mlx_lm generate_step).

| model | tok/s per run | median tok/s | median step ms | peak GB |
|---|---|---|---|---|
| bf16 | 44.96, 45.50 | 45.23 | 22.1 | 15.0 |
| binade_r4 | 56.03, 56.17 | 56.10 | 17.8 | 11.4 |

Binade R4 / BF16: **1.24x**. Token ids identical across all runs of BF16 and Binade: True.
