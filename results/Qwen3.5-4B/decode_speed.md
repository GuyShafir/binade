# Decode speed, fresh process per model: Qwen/Qwen3.5-4B

`scripts/decode_speed.py` @ `7581266`: 2 rounds of BF16 then Binade R4, each in its own process, 60 s cool-down between processes, 256 greedy tokens timed after a 16-token warm-up (mlx_lm generate_step).

| model | tok/s per run | median tok/s | median step ms | peak GB |
|---|---|---|---|---|
| bf16 | 51.93, 52.35 | 52.14 | 19.2 | 8.5 |
| binade_r4 | 65.98, 66.07 | 66.02 | 15.1 | 6.9 |

Binade R4 / BF16: **1.27x**. Token ids identical across all runs of BF16 and Binade: True.
