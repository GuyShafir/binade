# Binade greedy decode check (r4 runtime): google/gemma-4-26B-A4B-it

`scripts/generate_check.py` @ `f02d6f9`, 64 greedy tokens, prompt of 40 tokens.

**PASS**: prompt logits identical over the full vocabulary: True; a second prompt pass repeats them bit for bit (BF16, Binade): True, True; token ids identical: True.

Checkpoint tensors without a model parameter, dropped by both loaders (listed in generate_check.json): 0 (BF16); 0 packed weights and 0 other tensors (Binade).

| model | load s | prompt pass s (first, second) | decode tok/s (after the first token) | tok/s incl. prompt | peak GB |
|---|---|---|---|---|---|
| BF16 (mlx_lm), decoder layers streamed from disk | n/a | n/a | n/a | n/a | n/a |
| Binade format 2, R4 runtime (Metal kernel for single-row steps) | 17.5 | 0.40, 0.36 | 64.84 | 47.79 | 38.6 |

BF16 does not fit in memory, so the reference reads each decoder layer's weights from the checkpoint when it runs (binade.mlx.loader.load_streamed_model): its timings measure disk reads, and it serves only as the bit-exact reference.

Output (both):

> <|channel>thought
> 
> *   Topic: Why lossless compression of neural network weights speeds up inference on memory-bandwidth-bound hardware.
> *   Constraint 1: Three short paragraphs.
> *   Constraint 2: Explain the mechanism/reasoning.
> 
>     *   *What is memory-bandwidth-bound hardware?
