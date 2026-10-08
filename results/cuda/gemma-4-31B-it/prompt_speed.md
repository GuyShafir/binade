# Prompt speed: google/gemma-4-31B-it

`scripts/prompt_speed.py` @ `4d2376c`, MLX 0.32.2, NVIDIA RTX PRO 6000 Blackwell Server Edition. Time to the first token's logits over a prompt of WikiText-2 test text (KV cache filled in chunks of 2048, as mlx_lm's generate_step); median of 3 after a warm-up pass, each model in its own process. Seconds (prompt tokens per second).

| model | 40 tokens | 512 tokens | 2048 tokens | peak GB |
|---|---|---|---|---|
| BF16 | 0.099 (405) | 0.153 (3339) | 0.528 (3879) | 70.3 |
| Binade (R4 runtime) | 0.213 (188) | 0.276 (1857) | 0.692 (2960) | 52.1 |
| MLX 8-bit affine (lossy) | 0.092 (434) | 0.215 (2380) | 0.836 (2450) | 43.7 |
