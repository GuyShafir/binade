# Decode speed, fresh process per model: google/gemma-4-26B-A4B-it

`scripts/decode_speed.py` @ `629104b`: 3 rounds of BF16 then Binade R4, each in its own process, 60 s cool-down between processes, 256 greedy tokens timed after a 16-token warm-up (mlx_lm generate_step).

| model | tok/s per run | median tok/s | median step ms | peak GB |
|---|---|---|---|---|
| bf16 | 133.25, 132.76, 132.76 | 132.76 | 7.5 | 51.9 |
| binade_r4 | 126.76, 126.88, 124.57 | 126.76 | 7.9 | 40.0 |

Binade R4 / BF16: **0.95x**. Token ids identical across all runs of both models: True.
