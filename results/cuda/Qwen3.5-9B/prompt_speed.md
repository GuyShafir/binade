# Prompt speed: Qwen/Qwen3.5-9B

`scripts/prompt_speed.py` @ `4d2376c`, MLX 0.32.2, NVIDIA RTX PRO 6000 Blackwell Server Edition. Time to the first token's logits over a prompt of WikiText-2 test text (KV cache filled in chunks of 2048, as mlx_lm's generate_step); median of 3 after a warm-up pass, each model in its own process. Seconds (prompt tokens per second).

| model | 40 tokens | 512 tokens | 2048 tokens | peak GB |
|---|---|---|---|---|
| BF16 | 0.061 (653) | 0.609 (841) | 2.541 (806) | 23.4 |
| Binade (R4 runtime) | 0.086 (463) | 0.635 (806) | 3.123 (656) | 19.0 |
| MLX 8-bit affine (lossy) | 0.063 (636) | 0.593 (863) | 2.575 (795) | 15.0 |
