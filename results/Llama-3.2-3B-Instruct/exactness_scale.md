# Exactness at scale: unsloth/Llama-3.2-3B-Instruct

`scripts/exactness_scale.py` @ `75147d5 (dirty)`, MLX 0.32.2, Apple M4 Max. 32 windows of 512 tokens from the WikiText-2 test set (raw), logits at every position in one forward pass; 32 prompts of 48 tokens from later text, 128 greedy tokens each. Each variant in its own process.

| variant | positions with bit-identical logits | argmax agreement | perplexity | generations identical (tokens and log-probs) | tokens identical | peak GB |
|---|---|---|---|---|---|---|
| BF16 (reference) | 16352 / 16352 | 100.0000% | 13.0650 | 32 / 32 | 100.00% | 6.8 |
| Binade (R4 runtime) | 16352 / 16352 | 100.0000% | 13.0650 | 32 / 32 | 100.00% | 6.2 |
| Binade, fused R4 GEMM for all prompt lengths | 16352 / 16352 | 100.0000% | 13.0650 | 32 / 32 | 100.00% | 5.4 |
| MLX 8-bit affine (lossy) | 0 / 16352 | 98.0247% | 13.1063 | 0 / 32 | 52.32% | 4.2 |
