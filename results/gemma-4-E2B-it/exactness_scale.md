# Exactness at scale: google/gemma-4-E2B-it

`scripts/exactness_scale.py` @ `7581266`, MLX 0.32.2, Apple M4 Max. 32 windows of 512 tokens from the WikiText-2 test set (raw), logits at every position in one forward pass; 32 prompts of 48 tokens from later text, 128 greedy tokens each. Each variant in its own process.

| variant | positions with bit-identical logits | argmax agreement | perplexity | generations identical (tokens and log-probs) | tokens identical | peak GB |
|---|---|---|---|---|---|---|
| BF16 (reference) | 16352 / 16352 | 100.0000% | 176.2488 | 32 / 32 | 100.00% | 10.0 |
| Binade (R4 runtime) | 16352 / 16352 | 100.0000% | 176.2488 | 32 / 32 | 100.00% | 8.3 |
| Binade, fused R4 GEMM for all prompt lengths | 16352 / 16352 | 100.0000% | 176.2488 | 32 / 32 | 100.00% | 7.3 |
| MLX 8-bit affine (lossy) | 0 / 16352 | 96.3735% | 173.2360 | 0 / 32 | 70.46% | 10.4 |
