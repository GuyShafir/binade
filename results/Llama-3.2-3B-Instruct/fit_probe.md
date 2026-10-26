# Does it fit? unsloth/Llama-3.2-3B-Instruct on Apple M4 Max

`scripts/fit_probe.py` @ `bae3547`, MLX 0.32.2, Apple M4 Max (51.54 GB). Each variant in its own process: load, then 128 greedy tokens.

| variant | loads and runs | resident after load (GB) | peak (GB) | decode tok/s | error |
|---|---|---|---|---|---|
| BF16 | yes | 6.43 | 6.52 | 69.58 | |
| Binade (R4 runtime) | yes | 4.85 | 5.37 | 89.07 | |
| MLX 8-bit affine (lossy) | yes | 3.41 | 4.20 | 124.55 | |
