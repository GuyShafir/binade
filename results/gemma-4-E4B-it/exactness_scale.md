# Exactness at scale: google/gemma-4-E4B-it

`scripts/exactness_scale.py` @ `3fc1324`, MLX 0.32.2, Apple M4 Max. 32 windows of 512 tokens from the WikiText-2 test set (raw), logits at every position in one forward pass; 32 prompts of 48 tokens from later text, 128 greedy tokens each. Each variant in its own process.

| variant | positions with bit-identical logits | argmax agreement | perplexity | generations identical (tokens and log-probs) | tokens identical | peak GB |
|---|---|---|---|---|---|---|
| BF16 (reference) | 16352 / 16352 | 100.0000% | 76.4143 | 32 / 32 | 100.00% | 16.5 |
| Binade (R4 runtime) | 16352 / 16352 | 100.0000% | 76.4143 | 32 / 32 | 100.00% | 15.1 |
| Binade, fused R4 GEMM for all prompt lengths | 16352 / 16352 | 100.0000% | 76.4143 | 32 / 32 | 100.00% | 14.0 |
| MLX 8-bit affine (lossy) | 0 / 16352 | 96.4225% | 75.6951 | 0 / 32 | 34.38% | 15.0 |
