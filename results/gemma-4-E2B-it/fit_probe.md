# Does it fit? google/gemma-4-E2B-it on Apple M4 Max

`scripts/fit_probe.py` @ `7581266`, MLX 0.32.2, Apple M4 Max (51.54 GB). Each variant in its own process: load, then 40 greedy tokens.

| variant | loads and runs | resident after load (GB) | peak (GB) | decode tok/s | error |
|---|---|---|---|---|---|
| BF16 | yes | 9.26 | 9.34 | 86.32 | |
| Binade (R4 runtime) | yes | 6.59 | 6.93 | 100.16 | |
| MLX 8-bit affine (lossy) | yes | 4.92 | 10.45 | 137.71 | |
