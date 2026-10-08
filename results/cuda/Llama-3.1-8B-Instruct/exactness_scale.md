# Exactness at scale: unsloth/Llama-3.1-8B-Instruct

`scripts/exactness_scale.py` @ `4d2376c`, MLX 0.32.2, NVIDIA RTX PRO 6000 Blackwell Server Edition. 32 windows of 512 tokens from the WikiText-2 test set (raw), logits at every position in one forward pass; 32 prompts of 48 tokens from later text, 128 greedy tokens each. Each variant in its own process.

| variant | positions with bit-identical logits | argmax agreement | perplexity | generations identical (tokens and log-probs) | tokens identical | peak GB |
|---|---|---|---|---|---|---|
| BF16 (reference) | 16352 / 16352 | 100.0000% | 8.8888 | 32 / 32 | 100.00% | 18.2 |
| Binade (R4 runtime) | 16352 / 16352 | 100.0000% | 8.8888 | 32 / 32 | 100.00% | 23.9 |
| MLX 8-bit affine (lossy) | 0 / 16352 | 98.1287% | 8.8942 | 0 / 32 | 38.87% | 10.6 |
