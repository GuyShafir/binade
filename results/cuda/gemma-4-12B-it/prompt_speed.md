# Prompt speed: google/gemma-4-12B-it

`scripts/prompt_speed.py` @ `01575bd`, MLX 0.32.2, NVIDIA RTX PRO 6000 Blackwell Server Edition. Time to the first token's logits over a prompt of WikiText-2 test text (KV cache filled in chunks of 2048 tokens, as mlx_lm's generate_step); median of 3 after a warm-up pass, each model in its own process. Seconds (prompt tokens per second).

| model | 40 tokens | 128 tokens | 512 tokens | 2048 tokens | peak GB |
|---|---|---|---|---|---|
| BF16 | 0.045 (887) | 0.047 (2712) | 0.068 (7580) | 0.222 (9228) | 32.1 |
| Binade (R4 runtime) | 0.059 (675) | 0.072 (1788) | 0.100 (5129) | 0.243 (8425) | 29.3 |
| MLX 8-bit affine (lossy) | 0.049 (815) | 0.055 (2342) | 0.099 (5166) | 0.329 (6223) | 22.6 |
