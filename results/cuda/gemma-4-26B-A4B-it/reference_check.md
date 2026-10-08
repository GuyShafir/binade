# MLX against the reference implementation: google/gemma-4-26B-A4B-it

`scripts/reference_check.py` @ `d796fce`, NVIDIA RTX PRO 6000 Blackwell Server Edition, transformers 5.18.0, MLX 0.32.2. One window of 512 WikiText-2 test tokens; the same token ids on both sides.

| dtype | perplexity (transformers / MLX) | argmax agreement | per-token loss difference (mean / 99th pct / max, nats) | correlation |
|---|---|---|---|---|
| bfloat16 | 6085.05 / 7504.68 | 74.56% | 1.1374 / 7.659 / 17.808 | 0.965645 |
