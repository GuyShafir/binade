# Binade greedy decode check (r4 runtime): google/gemma-4-E4B-it

`scripts/generate_check.py` @ `2096638 (dirty)`, 256 greedy tokens, prompt of 40 tokens.

**PASS**: prompt logits identical over the full vocabulary: True; a second prompt pass repeats them bit for bit (BF16, Binade): True, True; token ids identical: True.

Checkpoint tensors without a model parameter, dropped by both loaders (listed in generate_check.json): 54 (BF16); 36 packed weights and 18 other tensors (Binade).

| model | load s | prompt pass s (first, second) | decode tok/s (after the first token) | tok/s incl. prompt | peak GB |
|---|---|---|---|---|---|
| BF16 (mlx_lm) | 3.7 | 0.89, 0.06 | 16.38 | 16.37 | 15.0 |
| Binade format 1, R4 runtime (R4 kernel for single-row steps) | 11.0 | 0.38, 0.16 | 52.95 | 51.10 | 14.0 |

Decode speedup vs BF16: 3.23x.

Output (both):

> <|channel>thought
> Here's a plan to structure the answer:
> 1.  **Identify the core problem:** Inference on memory-bandwidth-bound hardware is limited by how fast weights can be moved from memory (DRAM) to the processing unit (e.g., GPU/TPU).
> 2.  **Explain the solution (Lossless Compression):** Compression reduces the size of the weights.
> 3.  **Explain the benefit (Speedup):** Smaller weights mean fewer data transfers, directly reducing the time spent waiting for memory access.
> 4.  **Structure the paragraphs:** Three short, focused paragraphs.
> 
> *Self-Correction/Refinement during drafting:* Ensure
