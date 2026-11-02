# Decode speed, fresh process per model: google/gemma-4-E2B-it

`scripts/decode_speed.py` @ `7581266`: 2 rounds of BF16 then Binade R4, each in its own process, 60 s cool-down between processes, 256 greedy tokens timed after a 16-token warm-up (mlx_lm generate_step).

| model | tok/s per run | median tok/s | median step ms | peak GB |
|---|---|---|---|---|
| bf16 | 84.48, 83.70 | 84.09 | 11.9 | 9.3 |
| binade_r4 | 98.91, 99.00 | 98.95 | 10.1 | 6.9 |

Binade R4 / BF16: **1.18x**. Token ids identical across all runs of BF16 and Binade: True.
