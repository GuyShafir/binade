# Decode speed, fresh process per model: google/gemma-4-26B-A4B-it

`scripts/decode_speed.py` @ `9e51bc3`: 3 rounds of BF16 then Binade R4, each in its own process, 60 s cool-down between processes, 256 greedy tokens timed after a 16-token warm-up (mlx_lm generate_step).

| model | tok/s per run | median tok/s | median step ms | peak GB |
|---|---|---|---|---|
| bf16 | 134.37, 132.33, 132.30 | 132.33 | 7.6 | 51.9 |
| binade_r4 | 135.09, 133.01, 133.03 | 133.03 | 7.5 | 40.0 |

Binade R4 / BF16: **1.01x**. Token ids identical across all runs of BF16 and Binade: True.
