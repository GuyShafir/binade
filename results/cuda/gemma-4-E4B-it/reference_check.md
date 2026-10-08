# MLX against the reference implementation: google/gemma-4-E4B-it

`scripts/reference_check.py` @ `d796fce`, NVIDIA RTX PRO 6000 Blackwell Server Edition, transformers 5.18.0, MLX 0.32.2. One window of 512 WikiText-2 test tokens; the same token ids on both sides.

| dtype | perplexity (transformers / MLX) | argmax agreement | per-token loss difference (mean / 99th pct / max, nats) | correlation |
|---|---|---|---|---|
| float32 | 43.22 / 43.22 | 99.80% | 0.0020 / 0.020 / 0.043 | 1.000000 |
| bfloat16 | 43.40 / 43.21 | 98.83% | 0.0725 / 0.571 / 1.380 | 0.999606 |
