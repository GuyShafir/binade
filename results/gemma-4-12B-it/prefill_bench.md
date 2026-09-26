# Prefill matmuls: BF16 against R4, Gemma 4 12B shapes

`scripts/prefill_bench.py` @ `f02d6f9`, MLX 0.32.2, real 12B weights, median of 5 after a warm-up. Per prompt: every projection of the model once (tied lm_head included), summed; prompt tokens per second of matmul work in brackets.

| prompt rows | BF16 | R4 fused | R4 rebuild | R4 runtime | runtime / BF16 |
|---|---|---|---|---|---|
| 16 | 184 ms (87 tok/s) | 192 ms (84 tok/s) | 240 ms (67 tok/s) | 175 ms (91 tok/s) | 1.05x |
| 32 | 160 ms (200 tok/s) | 251 ms (127 tok/s) | 253 ms (126 tok/s) | 236 ms (135 tok/s) | 0.68x |
| 64 | 176 ms (363 tok/s) | 392 ms (163 tok/s) | 272 ms (235 tok/s) | 270 ms (237 tok/s) | 0.65x |
| 128 | 274 ms (468 tok/s) | 627 ms (204 tok/s) | 373 ms (343 tok/s) | 376 ms (341 tok/s) | 0.73x |
| 512 | 893 ms (573 tok/s) | 2147 ms (238 tok/s) | 987 ms (519 tok/s) | 991 ms (517 tok/s) | 0.90x |
| 2048 | 3330 ms (615 tok/s) | 8940 ms (229 tok/s) | 3416 ms (599 tok/s) | 3420 ms (599 tok/s) | 0.97x |

Per shape (ms; runtime path; exact: fused output bit-identical to BF16 wherever MLX runs its plain GEMM):

| projection | shape | count | rows | BF16 | fused | rebuild | runtime | path | exact |
|---|---|---|---|---|---|---|---|---|---|
| lm_head (tied embedding) | 262144x3840 | 1 | 16 | 8.88 | 7.63 | 16.63 | 7.65 | fused | True |
| lm_head (tied embedding) | 262144x3840 | 1 | 32 | 8.92 | 13.48 | 16.80 | 14.04 | fused | True |
| lm_head (tied embedding) | 262144x3840 | 1 | 64 | 8.99 | 23.91 | 16.76 | 16.79 | rebuild | True |
| lm_head (tied embedding) | 262144x3840 | 1 | 128 | 17.42 | 43.85 | 25.23 | 25.22 | rebuild | True |
| lm_head (tied embedding) | 262144x3840 | 1 | 512 | 68.64 | 168.82 | 76.81 | 76.52 | rebuild | True |
| lm_head (tied embedding) | 262144x3840 | 1 | 2048 | 273.53 | 704.73 | 281.13 | 282.01 | rebuild | True |
| down_proj | 3840x15360 | 48 | 16 | 1.11 | 0.99 | 0.91 | 0.91 | rebuild |  |
| down_proj | 3840x15360 | 48 | 32 | 0.53 | 1.18 | 1.03 | 1.05 | rebuild |  |
| down_proj | 3840x15360 | 48 | 64 | 0.75 | 2.06 | 1.27 | 1.26 | rebuild |  |
| down_proj | 3840x15360 | 48 | 128 | 1.34 | 2.99 | 1.88 | 1.89 | rebuild |  |
| down_proj | 3840x15360 | 48 | 512 | 4.45 | 11.36 | 4.93 | 4.93 | rebuild | True |
| down_proj | 3840x15360 | 48 | 2048 | 16.47 | 50.68 | 16.97 | 16.98 | rebuild | True |
| gate_up_proj | 15360x3840 | 96 | 16 | 0.80 | 0.75 | 1.25 | 0.75 | fused | True |
| gate_up_proj | 15360x3840 | 96 | 32 | 0.80 | 1.13 | 1.26 | 1.17 | fused | True |
| gate_up_proj | 15360x3840 | 96 | 64 | 0.79 | 1.80 | 1.28 | 1.25 | rebuild | True |
| gate_up_proj | 15360x3840 | 96 | 128 | 1.21 | 2.95 | 1.71 | 1.71 | rebuild | True |
| gate_up_proj | 15360x3840 | 96 | 512 | 4.24 | 10.23 | 4.71 | 4.72 | rebuild | True |
| gate_up_proj | 15360x3840 | 96 | 2048 | 16.29 | 41.90 | 16.69 | 16.70 | rebuild | True |
| k_proj | 2048x3840 | 40 | 16 | 0.16 | 0.32 | 0.20 | 0.20 | rebuild |  |
| k_proj | 2048x3840 | 40 | 32 | 0.18 | 0.35 | 0.23 | 0.25 | rebuild |  |
| k_proj | 2048x3840 | 40 | 64 | 0.21 | 0.36 | 0.26 | 0.27 | rebuild |  |
| k_proj | 2048x3840 | 40 | 128 | 0.29 | 0.55 | 0.33 | 0.35 | rebuild |  |
| k_proj | 2048x3840 | 40 | 512 | 0.79 | 1.54 | 0.82 | 0.83 | rebuild | True |
| k_proj | 2048x3840 | 40 | 2048 | 2.40 | 5.80 | 2.45 | 2.45 | rebuild | True |
| o_proj | 3840x4096 | 40 | 16 | 0.22 | 0.33 | 0.31 | 0.31 | rebuild |  |
| o_proj | 3840x4096 | 40 | 32 | 0.27 | 0.39 | 0.35 | 0.35 | rebuild |  |
| o_proj | 3840x4096 | 40 | 64 | 0.34 | 0.61 | 0.41 | 0.41 | rebuild |  |
| o_proj | 3840x4096 | 40 | 128 | 0.45 | 1.01 | 0.56 | 0.56 | rebuild |  |
| o_proj | 3840x4096 | 40 | 512 | 1.25 | 2.90 | 1.36 | 1.40 | rebuild | True |
| o_proj | 3840x4096 | 40 | 2048 | 4.47 | 11.46 | 4.60 | 4.59 | rebuild | True |
| q_proj | 4096x3840 | 40 | 16 | 0.39 | 0.33 | 0.48 | 0.31 | fused | True |
| q_proj | 4096x3840 | 40 | 32 | 0.39 | 0.36 | 0.52 | 0.37 | fused | True |
| q_proj | 4096x3840 | 40 | 64 | 0.38 | 0.57 | 0.50 | 0.51 | rebuild | True |
| q_proj | 4096x3840 | 40 | 128 | 0.53 | 0.97 | 0.65 | 0.66 | rebuild | True |
| q_proj | 4096x3840 | 40 | 512 | 1.28 | 2.86 | 1.41 | 1.45 | rebuild | True |
| q_proj | 4096x3840 | 40 | 2048 | 4.57 | 11.46 | 4.68 | 4.69 | rebuild | True |
| v_proj | 2048x3840 | 40 | 16 | 0.17 | 0.33 | 0.21 | 0.21 | rebuild |  |
| v_proj | 2048x3840 | 40 | 32 | 0.18 | 0.37 | 0.23 | 0.23 | rebuild |  |
| v_proj | 2048x3840 | 40 | 64 | 0.22 | 0.35 | 0.27 | 0.28 | rebuild |  |
| v_proj | 2048x3840 | 40 | 128 | 0.29 | 0.56 | 0.34 | 0.36 | rebuild |  |
| v_proj | 2048x3840 | 40 | 512 | 0.77 | 1.54 | 0.82 | 0.83 | rebuild | True |
| v_proj | 2048x3840 | 40 | 2048 | 2.41 | 5.81 | 2.45 | 2.46 | rebuild | True |
| k_proj | 512x3840 | 8 | 16 | 0.14 | 0.31 | 0.16 | 0.16 | rebuild |  |
| k_proj | 512x3840 | 8 | 32 | 0.14 | 0.36 | 0.17 | 0.18 | rebuild |  |
| k_proj | 512x3840 | 8 | 64 | 0.15 | 0.35 | 0.19 | 0.18 | rebuild |  |
| k_proj | 512x3840 | 8 | 128 | 0.18 | 0.34 | 0.20 | 0.21 | rebuild |  |
| k_proj | 512x3840 | 8 | 512 | 0.30 | 0.54 | 0.33 | 0.33 | rebuild |  |
| k_proj | 512x3840 | 8 | 2048 | 0.77 | 1.65 | 0.80 | 0.81 | rebuild | True |
| o_proj | 3840x8192 | 8 | 16 | 0.29 | 0.58 | 0.53 | 0.52 | rebuild |  |
| o_proj | 3840x8192 | 8 | 32 | 0.39 | 0.72 | 0.64 | 0.64 | rebuild |  |
| o_proj | 3840x8192 | 8 | 64 | 0.46 | 1.16 | 0.73 | 0.73 | rebuild |  |
| o_proj | 3840x8192 | 8 | 128 | 0.79 | 2.09 | 1.04 | 1.05 | rebuild |  |
| o_proj | 3840x8192 | 8 | 512 | 2.39 | 6.11 | 2.64 | 2.64 | rebuild | True |
| o_proj | 3840x8192 | 8 | 2048 | 8.86 | 25.78 | 9.10 | 9.17 | rebuild | True |
| q_proj | 8192x3840 | 8 | 16 | 0.53 | 0.55 | 0.78 | 0.54 | fused | True |
| q_proj | 8192x3840 | 8 | 32 | 0.54 | 0.63 | 0.78 | 0.66 | fused | True |
| q_proj | 8192x3840 | 8 | 64 | 0.53 | 1.08 | 0.78 | 0.78 | rebuild | True |
| q_proj | 8192x3840 | 8 | 128 | 0.78 | 1.67 | 1.02 | 1.03 | rebuild | True |
| q_proj | 8192x3840 | 8 | 512 | 2.37 | 5.59 | 2.62 | 2.66 | rebuild | True |
| q_proj | 8192x3840 | 8 | 2048 | 8.90 | 22.46 | 9.07 | 9.08 | rebuild | True |
