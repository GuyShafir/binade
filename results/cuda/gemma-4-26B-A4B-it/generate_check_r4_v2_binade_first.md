# Binade greedy decode check (r4 runtime): google/gemma-4-26B-A4B-it

`scripts/generate_check.py` @ `629104b`, 256 greedy tokens, prompt of 40 tokens.

**PASS**: prompt logits identical over the full vocabulary: True; a second prompt pass repeats them bit for bit (BF16, Binade): True, True; token ids identical: True.

Checkpoint tensors without a model parameter, dropped by both loaders (listed in generate_check.json): 0 (BF16); 0 packed weights and 0 other tensors (Binade).

| model | load s | prompt pass s (first, second) | decode tok/s (after the first token) | tok/s incl. prompt | peak GB |
|---|---|---|---|---|---|
| BF16 (mlx_lm) | 8.4 | 0.05, 0.04 | 129.34 | 126.29 | 50.7 |
| Binade format 2, R4 runtime (Metal kernel for single-row steps) | 33.5 | 10.28, 0.13 | 127.97 | 21.96 | 40.0 |

Decode speedup vs BF16: 0.99x.

Output (both):

> <|channel>thought
> 
> *   Topic: Why lossless compression of neural network weights speeds up inference on memory-bandwidth-bound hardware.
> *   Constraint 1: Three short paragraphs.
> *   Constraint 2: Explain the mechanism/reasoning.
> 
>     *   *What is memory-bandwidth-bound hardware?* Hardware where the bottleneck is moving data from memory (DRAM) to the processor (ALU/GPU cores), not the computation itself.
>     *   *What is lossless compression?* Reducing the size of data without losing any information (e.g., Huffman coding, LZ77, or bit-plane truncation/packing).
>     *   *How does compression he
