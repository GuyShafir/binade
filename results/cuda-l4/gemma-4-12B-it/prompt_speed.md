# Prompt speed: google/gemma-4-12B-it

`scripts/prompt_speed.py` @ `759684d`, MLX 0.32.2, NVIDIA L4. Time to the first token's logits over a prompt of WikiText-2 test text (KV cache filled in chunks of 512 tokens, as mlx_lm's generate_step); median of 3 after a warm-up pass, each model in its own process. Seconds (prompt tokens per second).

| model | 40 tokens | 128 tokens | 512 tokens | 2048 tokens | peak GB |
|---|---|---|---|---|---|
| Binade (R4 runtime) | 0.257 (156) | 0.327 (391) | 0.674 (759) | 2.436 (841) | 20.8 |
| MLX 8-bit affine (lossy) | 0.148 (269) | 0.198 (648) | 0.509 (1005) | 1.910 (1072) | 14.7 |
