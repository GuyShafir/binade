# MLX against the reference implementation: google/gemma-4-12B-it

`scripts/reference_check.py` @ `d796fce`, NVIDIA RTX PRO 6000 Blackwell Server Edition, transformers 5.18.0, MLX 0.32.2. One window of 512 WikiText-2 test tokens; the same token ids on both sides.

| dtype | perplexity (transformers / MLX) | argmax agreement | per-token loss difference (mean / 99th pct / max, nats) | correlation |
|---|---|---|---|---|
| float32 | 715.20 / 713.03 | 99.41% | 0.0145 / 0.171 / 0.880 | 0.999962 |
| bfloat16 | 725.75 / 724.33 | 94.13% | 0.2298 / 2.025 / 7.881 | 0.996164 |
