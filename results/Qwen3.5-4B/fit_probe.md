# Does it fit? Qwen/Qwen3.5-4B on Apple M4 Max

`scripts/fit_probe.py` @ `7581266`, MLX 0.32.2, Apple M4 Max (51.54 GB). Each variant in its own process: load, then 40 greedy tokens.

| variant | loads and runs | resident after load (GB) | peak (GB) | decode tok/s | error |
|---|---|---|---|---|---|
| BF16 | yes | 8.41 | 8.54 | 52.18 | |
| Binade (R4 runtime) | yes | 6.35 | 6.88 | 65.90 | |
| MLX 8-bit affine (lossy) | yes | 4.47 | 5.74 | 90.70 | |
