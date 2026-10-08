# Exactness at scale: google/gemma-4-26B-A4B-it

`scripts/exactness_scale.py` @ `7bc8183`, MLX 0.32.2, NVIDIA A100-SXM4-40GB. 32 windows of 512 tokens from the WikiText-2 test set (raw), logits at every position in one forward pass; 0 prompts of 48 tokens from later text, 128 greedy tokens each. Each variant in its own process.

| variant | positions with bit-identical logits | argmax agreement | perplexity | generations identical (tokens and log-probs) | tokens identical | peak GB |
|---|---|---|---|---|---|---|
| BF16 (reference, streamed from disk) | 16352 / 16352 | 100.0000% | 16731.1307 | 0 / 0 | n/a | 4.3 |
| Binade (R4 runtime) | 16352 / 16352 | 100.0000% | 16731.1307 | 0 / 0 | n/a | 40.3 |
| MLX 8-bit affine (lossy) | 0 / 16352 | 75.9846% | 17619.8175 | 0 / 0 | n/a | 28.3 |
