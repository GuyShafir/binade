# Decode speed, fresh process per model: google/gemma-4-E4B-it

`scripts/decode_speed.py` @ `d0315fd`: 2 rounds of BF16 then Binade R4, each in its own process, 60 s cool-down between processes, 256 greedy tokens timed after a 16-token warm-up (mlx_lm generate_step).

| model | tok/s per run | median tok/s | median step ms | peak GB |
|---|---|---|---|---|
| bf16 | 45.81, 44.63 | 45.22 | 22.1 | 15.0 |
| binade_r4 | 55.64, 56.01 | 55.83 | 17.9 | 11.4 |

Binade R4 / BF16: **1.23x**. Token ids identical across all runs of BF16 and Binade: True.
