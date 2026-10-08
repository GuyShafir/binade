# Decode speed, fresh process per model: unsloth/Llama-3.1-8B-Instruct

`scripts/decode_speed.py` @ `4d2376c`: 3 rounds of BF16 then Binade R4 then MLX 8-bit, each in its own process, 60 s cool-down between processes, 256 greedy tokens timed after a 16-token warm-up (mlx_lm generate_step).

| model | tok/s per run | median tok/s | median step ms | peak GB |
|---|---|---|---|---|
| bf16 | 77.81, 89.60, 89.54 | 89.54 | 11.0 | 16.2 |
| binade_r4 | 93.78, 93.45, 93.59 | 93.59 | 10.5 | 14.4 |
| q8 | 145.07, 145.27, 144.90 | 145.07 | 6.7 | 10.6 |

Binade R4 / BF16: **1.05x**. Token ids identical across all runs of BF16 and Binade: False.
MLX 8-bit (lossy) / BF16: 1.62x; its 256 greedy tokens identical to the exact models': False.
