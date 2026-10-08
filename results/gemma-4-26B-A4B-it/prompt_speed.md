# Prompt speed: google/gemma-4-26B-A4B-it

`scripts/prompt_speed.py` @ `8f9f0e7`, MLX 0.32.2, Apple M4 Max. Time to the first token's logits over a prompt of WikiText-2 test text (KV cache filled in chunks of 2048, as mlx_lm's generate_step); median of 3 after a warm-up pass, each model in its own process. Seconds (prompt tokens per second).

| model | 40 tokens | 512 tokens | 2048 tokens | peak GB |
|---|---|---|---|---|
| Binade (R4 runtime) | 0.215 (186) | 0.636 (805) | 1.982 (1033) | 39.4 |
| MLX 8-bit affine (lossy) | 0.097 (414) | 0.349 (1468) | 1.355 (1512) | 28.7 |
