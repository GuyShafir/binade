# Exactness at scale: unsloth/Llama-3.1-8B-Instruct

`scripts/exactness_scale.py` @ `8f9f0e7`, MLX 0.32.2, Apple M4 Max. 32 windows of 512 tokens from the WikiText-2 test set (raw), logits at every position in one forward pass; 32 prompts of 48 tokens from later text, 128 greedy tokens each. Each variant in its own process.

| variant | positions with bit-identical logits | argmax agreement | perplexity | generations identical (tokens and log-probs) | tokens identical | peak GB |
|---|---|---|---|---|---|---|
| BF16 (reference) | 16352 / 16352 | 100.0000% | 8.8884 | 32 / 32 | 100.00% | 16.8 |
| Binade (R4 runtime) | 16352 / 16352 | 100.0000% | 8.8884 | 32 / 32 | 100.00% | 13.7 |
| Binade fused, two accumulators (valid order, not MLX's) | 0 / 16352 | 98.8258% | 8.8866 | 0 / 32 | 62.40% | 12.8 |
