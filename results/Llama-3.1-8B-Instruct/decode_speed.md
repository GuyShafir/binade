# Decode speed, fresh process per model: unsloth/Llama-3.1-8B-Instruct

`scripts/decode_speed.py` @ `8f3fc48`: 3 rounds of BF16 then Binade R4 then MLX 8-bit, each in its own process, 60 s cool-down between processes, 256 greedy tokens timed after a 16-token warm-up (mlx_lm generate_step).

| model | tok/s per run | median tok/s | median step ms | peak GB |
|---|---|---|---|---|
| bf16 | 30.75, 28.45, 7.60 | 28.45 | 35.0 | 16.2 |
| binade_r4 | 40.55, 38.29, 37.62 | 38.29 | 25.9 | 13.2 |
| q8 | 50.71, 10.61, 53.97 | 50.71 | 18.5 | 10.6 |

Binade R4 / BF16: **1.35x**. Token ids identical across all runs of BF16 and Binade: True.
MLX 8-bit (lossy) / BF16: 1.78x; its 256 greedy tokens identical to the exact models': False.
