# Exactness at scale: Qwen/Qwen3.5-9B

`scripts/exactness_scale.py` @ `3fc1324`, MLX 0.32.2, Apple M4 Max. 32 windows of 512 tokens from the WikiText-2 test set (raw), logits at every position in one forward pass; 32 prompts of 48 tokens from later text, 128 greedy tokens each. Each variant in its own process.

| variant | positions with bit-identical logits | argmax agreement | perplexity | generations identical (tokens and log-probs) | tokens identical | peak GB |
|---|---|---|---|---|---|---|
| BF16 (reference) | 16352 / 16352 | 100.0000% | 8.5578 | 32 / 32 | 100.00% | 19.3 |
| Binade (R4 runtime) | 16352 / 16352 | 100.0000% | 8.5578 | 32 / 32 | 100.00% | 16.6 |
| Binade, fused R4 GEMM for all prompt lengths | 16352 / 16352 | 100.0000% | 8.5578 | 32 / 32 | 100.00% | 14.9 |
| MLX 8-bit affine (lossy) | 0 / 16352 | 98.1959% | 8.5476 | 0 / 32 | 53.34% | 13.6 |
