# Does it fit? google/gemma-4-E4B-it on Apple M4 Max

`scripts/fit_probe.py` @ `d0315fd`, MLX 0.32.2, Apple M4 Max (51.54 GB). Each variant in its own process: load, then 40 greedy tokens.

| variant | loads and runs | resident after load (GB) | peak (GB) | decode tok/s | error |
|---|---|---|---|---|---|
| BF16 | yes | 14.93 | 15.00 | 44.98 | |
| Binade (R4 runtime) | yes | 10.79 | 11.35 | 55.65 | |
| MLX 8-bit affine (lossy) | yes | 7.93 | 14.97 | 76.93 | |
