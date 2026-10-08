# Prompt speed: unsloth/Llama-3.1-8B-Instruct

`scripts/prompt_speed.py` @ `4d2376c`, MLX 0.32.2, NVIDIA RTX PRO 6000 Blackwell Server Edition. Time to the first token's logits over a prompt of WikiText-2 test text (KV cache filled in chunks of 2048, as mlx_lm's generate_step); median of 3 after a warm-up pass, each model in its own process. Seconds (prompt tokens per second).

| model | 40 tokens | 512 tokens | 2048 tokens | peak GB |
|---|---|---|---|---|
| BF16 | 0.026 (1517) | 0.039 (13023) | 0.098 (20877) | 22.5 |
| Binade (R4 runtime) | 0.046 (868) | 0.057 (9049) | 0.116 (17659) | 23.5 |
| MLX 8-bit affine (lossy) | 0.028 (1432) | 0.055 (9355) | 0.163 (12580) | 15.9 |
