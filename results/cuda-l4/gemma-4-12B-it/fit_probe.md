# Does it fit? google/gemma-4-12B-it on NVIDIA L4

`scripts/fit_probe.py` @ `7bc8183`, MLX 0.32.2, NVIDIA L4 (24.15 GB). Each variant in its own process: load, then 40 greedy tokens.

| variant | loads and runs | resident after load (GB) | peak (GB) | decode tok/s | error |
|---|---|---|---|---|---|
| BF16 | no | | | | RuntimeError: cudaMallocAsync(&data, size, stream) failed: out of memory |
| Binade (R4 runtime) | yes | 17.97 | 20.02 | 11.27 | |
| MLX 8-bit affine (lossy) | yes | 12.65 | 14.67 | 18.03 | |
