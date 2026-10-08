# Does it fit? google/gemma-4-26B-A4B-it on NVIDIA A100-SXM4-40GB

`scripts/fit_probe.py` @ `7bc8183`, MLX 0.32.2, NVIDIA A100-SXM4-40GB (42.95 GB). Each variant in its own process: load, then 40 greedy tokens.

| variant | loads and runs | resident after load (GB) | peak (GB) | decode tok/s | error |
|---|---|---|---|---|---|
| BF16 | no | | | | RuntimeError: cudaMallocAsync(&data, size, stream) failed: out of memory |
| Binade (R4 runtime) | yes | 38.48 | 40.03 | 45.64 | |
| MLX 8-bit affine (lossy) | yes | 26.81 | 28.29 | 60.44 | |
