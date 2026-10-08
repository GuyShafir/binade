# MLX against the reference implementation: google/gemma-4-26B-A4B-it

`scripts/reference_check.py` @ `ef2972f`, CPU, transformers 5.18.0, MLX 0.32.2. One window of 512 WikiText-2 test tokens; the same token ids on both sides.

| dtype | perplexity (transformers / MLX) | argmax agreement | per-token loss difference (mean / 99th pct / max, nats) | correlation |
|---|---|---|---|---|
| float32 | 6631.63 / 6630.75 | 99.80% | 0.0002 / 0.001 / 0.005 | 1.000000 |
