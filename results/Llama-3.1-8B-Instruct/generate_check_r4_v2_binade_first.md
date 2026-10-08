# Binade greedy decode check (r4 runtime): unsloth/Llama-3.1-8B-Instruct

`scripts/generate_check.py` @ `2096638`, 256 greedy tokens, prompt of 60 tokens.

**PASS**: prompt logits identical over the full vocabulary: True; a second prompt pass repeats them bit for bit (BF16, Binade): True, True; token ids identical: True.

Checkpoint tensors without a model parameter, dropped by both loaders (listed in generate_check.json): 0 (BF16); 0 packed weights and 0 other tensors (Binade).

| model | load s | prompt pass s (first, second) | decode tok/s (after the first token) | tok/s incl. prompt | peak GB |
|---|---|---|---|---|---|
| BF16 (mlx_lm) | 3.0 | 0.23, 0.09 | 29.95 | 29.56 | 16.2 |
| Binade format 2, R4 runtime (R4 kernel for single-row steps) | 7.4 | 0.25, 0.18 | 39.25 | 38.01 | 13.2 |

Decode speedup vs BF16: 1.31x.

Output (both):

> Lossless compression of neural network weights can speed up inference on memory-bandwidth-bound hardware by reducing the amount of data that needs to be transferred between the memory and the processing units. When a neural network is deployed on hardware with limited memory bandwidth, the time it takes to access and process the weights can become a significant bottleneck. By compressing the weights, the amount of data that needs to be transferred is reduced, allowing the processing units to access the weights more quickly.
> 
> This reduction in data transfer time can have a significant impact on
