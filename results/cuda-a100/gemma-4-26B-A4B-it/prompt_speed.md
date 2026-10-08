# Prompt speed: google/gemma-4-26B-A4B-it

`scripts/prompt_speed.py` @ `6fb9c96`, MLX 0.32.2, NVIDIA A100-SXM4-40GB. Time to the first token's logits over a prompt of WikiText-2 test text (KV cache filled in chunks of 512 tokens, as mlx_lm's generate_step); median of 3 after a warm-up pass, each model in its own process. Seconds (prompt tokens per second).

| model | 40 tokens | 128 tokens | 512 tokens | 2048 tokens | peak GB |
|---|---|---|---|---|---|
| Binade (R4 runtime) | 0.101 (396) | 0.123 (1040) | 0.302 (1697) | 1.137 (1801) | 40.6 |
| MLX 8-bit affine (lossy) | 0.084 (476) | 0.169 (757) | 0.548 (935) | 2.152 (952) | 28.3 |
