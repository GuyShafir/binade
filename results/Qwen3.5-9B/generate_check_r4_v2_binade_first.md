# Binade greedy decode check (r4 runtime): Qwen/Qwen3.5-9B

`scripts/generate_check.py` @ `2096638 (dirty)`, 256 greedy tokens, prompt of 35 tokens.

**PASS**: prompt logits identical over the full vocabulary: True; a second prompt pass repeats them bit for bit (BF16, Binade): True, True; token ids identical: True.

Checkpoint tensors without a model parameter, dropped by both loaders (listed in generate_check.json): 0 (BF16); 0 packed weights and 0 other tensors (Binade).

| model | load s | prompt pass s (first, second) | decode tok/s (after the first token) | tok/s incl. prompt | peak GB |
|---|---|---|---|---|---|
| BF16 (mlx_lm) | 3.2 | 0.10, 0.09 | 27.44 | 27.12 | 18.0 |
| Binade format 2, R4 runtime (R4 kernel for single-row steps) | 7.6 | 0.35, 0.24 | 36.02 | 34.67 | 15.6 |

Decode speedup vs BF16: 1.31x.

Output (both):

> Thinking Process:
> 
> 1.  **Analyze the Request:**
>     *   **Topic:** Lossless compression of neural network weights.
>     *   **Goal:** Explain why it speeds up inference.
>     *   **Constraint:** Three short paragraphs.
>     *   **Target Hardware:** Memory-bandwidth-bound hardware (e.g., CPUs, GPUs, NPUs where memory access is the bottleneck, not compute).
> 
> 2.  **Identify Key Concepts:**
>     *   Neural Network Weights: Large data structures stored in memory.
>     *   Lossless Compression: Reducing size without losing information (reversible).
>     *   Inference: Running the model to make predictions
