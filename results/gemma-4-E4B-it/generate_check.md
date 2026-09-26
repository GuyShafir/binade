# xp v1 greedy decode check: google/gemma-4-E4B-it

`scripts/generate_check.py` @ `040b9ce`, 256 greedy tokens, prompt of 40 tokens.

**PASS**: prompt logits identical over the full vocabulary: True; token ids identical: True.

Checkpoint tensors without a model parameter, dropped by both loaders (listed in generate_check.json): 54 (BF16); 36 packed weights and 18 other tensors (xp).

| model | prefill s | decode tok/s | peak GB |
|---|---|---|---|
| BF16 (mlx_lm) | 2.49 | 39.65 | 15.0 |
| xp v1, slow runtime (MLX ops decode per call, no kernel) | 8.19 | 0.30 | 15.9 |

Output (both):

> <|channel>thought
> Here's a plan to structure the answer:
> 1.  **Identify the core problem:** Inference on memory-bandwidth-bound hardware is limited by how fast weights can be moved from memory (DRAM) to the processing unit (e.g., GPU/TPU).
> 2.  **Explain the solution (Lossless Compression):** Compression reduces the size of the weights.
> 3.  **Explain the benefit (Speedup):** Smaller weights mean fewer data transfers, directly reducing the time spent waiting for memory access.
> 4.  **Structure the paragraphs:** Three short, focused paragraphs.
> 
> *Self-Correction/Refinement during drafting:* Ensure
