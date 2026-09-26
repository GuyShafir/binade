# Binade greedy decode check (r4 runtime): google/gemma-4-12B-it

`scripts/generate_check.py` @ `f02d6f9`, 256 greedy tokens, prompt of 40 tokens.

**PASS**: prompt logits identical over the full vocabulary: True; a second prompt pass repeats them bit for bit (BF16, Binade): True, True; token ids identical: True.

Checkpoint tensors without a model parameter, dropped by both loaders (listed in generate_check.json): 9 (BF16); 0 packed weights and 9 other tensors (Binade).

| model | load s | prompt pass s (first, second) | decode tok/s (after the first token) | tok/s incl. prompt | peak GB |
|---|---|---|---|---|---|
| BF16 (mlx_lm) | 4.6 | 0.92, 0.16 | 19.37 | 19.10 | 24.0 |
| Binade format 2, R4 runtime (Metal kernel for single-row steps) | 10.2 | 0.37, 0.34 | 24.90 | 24.01 | 20.0 |

Decode speedup vs BF16: 1.29x.

Output (both):

> <|channel>thought
> 
> *   Topic: Lossless compression of neural network weights.
>     *   Context: Speeding up inference on memory-bandwidth-bound hardware.
>     *   Constraint: Exactly three short paragraphs.
> 
>     *   *What is memory-bandwidth-bound?* It means the bottleneck is how fast data can move from memory (RAM/VRAM) to the processor (GPU/TPU/NPU), not how fast the processor can do the math.
>     *   *What is lossless compression?* Reducing the size of the weights without losing any information (e.g., Huffman coding, Gzip, etc.).
>     *   *How does it help?* Smaller weights = less data to move
