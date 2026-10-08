# Binade greedy decode check (r4 runtime): google/gemma-4-12B-it

`scripts/generate_check.py` @ `24743c8`, 256 greedy tokens, prompt of 40 tokens.

**PASS**: prompt logits identical over the full vocabulary: True; a second prompt pass repeats them bit for bit (BF16, Binade): True, True; token ids identical: True.

Checkpoint tensors without a model parameter, dropped by both loaders (listed in generate_check.json): 9 (BF16); 0 packed weights and 9 other tensors (Binade).

| model | load s | prompt pass s (first, second) | decode tok/s (after the first token) | tok/s incl. prompt | peak GB |
|---|---|---|---|---|---|
| BF16 (mlx_lm) | 4.9 | 0.03, 0.03 | 54.95 | 54.31 | 24.1 |
| Binade format 2, R4 runtime (R4 kernel for single-row steps) | 21.8 | 7.14, 0.06 | 58.83 | 21.99 | 23.9 |

Decode speedup vs BF16: 1.07x.

Output (both):

> <|channel>thought
> 
> *   Topic: Lossless compression of neural network weights.
>     *   Context: Speeding up inference on memory-bandwidth-bound hardware.
>     *   Constraint: Exactly three short paragraphs.
> 
>     *   *What is memory-bandwidth-bound?* It means the bottleneck is how fast data can move from memory (RAM/VRAM) to the processor (GPU/TPU/NPU), not how fast the processor can do the math.
>     *   *What is lossless compression?* Reducing the size of the weights without losing any information (e.g., using Huffman coding, Gzip, or specialized weight-sharing).
>     *   *How does it help?* Smal
