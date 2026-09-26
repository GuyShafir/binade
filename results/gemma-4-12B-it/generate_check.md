# xp v1 greedy decode check: google/gemma-4-12B-it

`scripts/generate_check.py` @ `6fbfc7e`, 256 greedy tokens, prompt of 40 tokens.

**PASS**: prompt logits identical over the full vocabulary: True; token ids identical: True.

Checkpoint tensors without a model parameter, dropped by both loaders (listed in generate_check.json): 9 (BF16); 0 packed weights and 9 other tensors (xp).

| model | prefill s | decode tok/s | peak GB |
|---|---|---|---|
| BF16 (mlx_lm) | 5.59 | 10.08 | 24.0 |
| xp v1, slow runtime (MLX ops decode per call, no kernel) | 18.54 | 0.10 | 22.3 |

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
