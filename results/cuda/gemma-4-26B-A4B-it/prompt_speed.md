# Prompt speed: google/gemma-4-26B-A4B-it

`scripts/prompt_speed.py` @ `01575bd`, MLX 0.32.2, NVIDIA RTX PRO 6000 Blackwell Server Edition. Time to the first token's logits over a prompt of WikiText-2 test text (KV cache filled in chunks of 2048 tokens, as mlx_lm's generate_step); median of 3 after a warm-up pass, each model in its own process. Seconds (prompt tokens per second).

| model | 40 tokens | 128 tokens | 512 tokens | 2048 tokens | peak GB |
|---|---|---|---|---|---|
| BF16 | 0.041 (965) | 0.052 (2472) | 0.071 (7200) | 0.160 (12814) | 57.0 |
| Binade (R4 runtime) | 0.053 (749) | 0.066 (1932) | 0.088 (5799) | 0.208 (9852) | 43.8 |
| MLX 8-bit affine (lossy) | 0.042 (960) | 0.084 (1523) | 0.282 (1814) | 1.125 (1821) | 36.0 |
