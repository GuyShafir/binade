# Prefill matmuls: BF16 against R4, Gemma 4 12B shapes

`scripts/prefill_bench.py` @ `69e207a`, MLX 0.32.2, real 12B weights, median of 9 after a warm-up. Per prompt: every projection of the model once (tied lm_head included), summed; prompt tokens per second of matmul work in brackets.

| prompt rows | BF16 | R4 fused | R4 rebuild | R4 runtime | runtime / BF16 |
|---|---|---|---|---|---|
| 16 | 202 ms (79 tok/s) | 164 ms (97 tok/s) | 257 ms (62 tok/s) | 155 ms (103 tok/s) | 1.30x |
| 32 | 174 ms (184 tok/s) | 213 ms (150 tok/s) | 273 ms (117 tok/s) | 194 ms (165 tok/s) | 0.90x |
| 64 | 194 ms (330 tok/s) | 298 ms (215 tok/s) | 289 ms (222 tok/s) | 253 ms (253 tok/s) | 0.77x |
| 128 | 295 ms (435 tok/s) | 430 ms (298 tok/s) | 395 ms (324 tok/s) | 379 ms (337 tok/s) | 0.78x |
| 512 | 909 ms (563 tok/s) | 1191 ms (430 tok/s) | 1019 ms (502 tok/s) | 1019 ms (502 tok/s) | 0.89x |
| 2048 | 3457 ms (592 tok/s) | 4623 ms (443 tok/s) | 3610 ms (567 tok/s) | 3626 ms (565 tok/s) | 0.95x |

Per shape (ms; runtime path; exact: fused output bit-identical to BF16 wherever MLX runs its plain GEMM):

| projection | shape | count | rows | BF16 | fused | rebuild | runtime | path | exact |
|---|---|---|---|---|---|---|---|---|---|
| lm_head (tied embedding) | 262144x3840 | 1 | 16 | 8.89 | 4.76 | 16.85 | 4.74 | fused | True |
| lm_head (tied embedding) | 262144x3840 | 1 | 32 | 8.92 | 7.06 | 16.87 | 7.09 | fused | True |
| lm_head (tied embedding) | 262144x3840 | 1 | 64 | 9.01 | 11.52 | 16.95 | 11.59 | fused | True |
| lm_head (tied embedding) | 262144x3840 | 1 | 128 | 17.44 | 22.51 | 25.38 | 22.53 | fused | True |
| lm_head (tied embedding) | 262144x3840 | 1 | 512 | 68.59 | 88.81 | 76.47 | 76.61 | rebuild | True |
| lm_head (tied embedding) | 262144x3840 | 1 | 2048 | 273.44 | 355.45 | 282.63 | 292.76 | rebuild | True |
| down_proj | 3840x15360 | 48 | 16 | 1.13 | 0.98 | 0.92 | 0.92 | rebuild |  |
| down_proj | 3840x15360 | 48 | 32 | 0.53 | 1.22 | 1.07 | 1.09 | rebuild |  |
| down_proj | 3840x15360 | 48 | 64 | 0.81 | 1.71 | 1.34 | 1.33 | rebuild |  |
| down_proj | 3840x15360 | 48 | 128 | 1.40 | 2.35 | 1.92 | 1.92 | rebuild |  |
| down_proj | 3840x15360 | 48 | 512 | 4.47 | 5.86 | 4.97 | 4.98 | rebuild | True |
| down_proj | 3840x15360 | 48 | 2048 | 17.08 | 22.92 | 17.96 | 18.14 | rebuild | True |
| gate_up_proj | 15360x3840 | 96 | 16 | 0.94 | 0.51 | 1.37 | 0.52 | fused | True |
| gate_up_proj | 15360x3840 | 96 | 32 | 0.91 | 0.73 | 1.41 | 0.73 | fused | True |
| gate_up_proj | 15360x3840 | 96 | 64 | 0.91 | 1.05 | 1.37 | 1.05 | fused | True |
| gate_up_proj | 15360x3840 | 96 | 128 | 1.34 | 1.75 | 1.80 | 1.64 | fused | True |
| gate_up_proj | 15360x3840 | 96 | 512 | 4.28 | 5.62 | 4.87 | 4.86 | rebuild | True |
| gate_up_proj | 15360x3840 | 96 | 2048 | 17.17 | 23.01 | 17.76 | 17.78 | rebuild | True |
| k_proj | 2048x3840 | 40 | 16 | 0.18 | 0.33 | 0.24 | 0.25 | rebuild |  |
| k_proj | 2048x3840 | 40 | 32 | 0.21 | 0.39 | 0.27 | 0.27 | rebuild |  |
| k_proj | 2048x3840 | 40 | 64 | 0.25 | 0.51 | 0.30 | 0.31 | rebuild |  |
| k_proj | 2048x3840 | 40 | 128 | 0.32 | 0.53 | 0.37 | 0.38 | rebuild |  |
| k_proj | 2048x3840 | 40 | 512 | 0.81 | 1.09 | 0.90 | 0.91 | rebuild | True |
| k_proj | 2048x3840 | 40 | 2048 | 2.43 | 3.16 | 2.57 | 2.49 | rebuild | True |
| o_proj | 3840x4096 | 40 | 16 | 0.24 | 0.35 | 0.33 | 0.36 | rebuild |  |
| o_proj | 3840x4096 | 40 | 32 | 0.26 | 0.43 | 0.39 | 0.39 | rebuild |  |
| o_proj | 3840x4096 | 40 | 64 | 0.31 | 0.55 | 0.44 | 0.45 | rebuild |  |
| o_proj | 3840x4096 | 40 | 128 | 0.46 | 0.77 | 0.66 | 0.67 | rebuild |  |
| o_proj | 3840x4096 | 40 | 512 | 1.33 | 1.72 | 1.47 | 1.47 | rebuild | True |
| o_proj | 3840x4096 | 40 | 2048 | 4.53 | 6.19 | 4.92 | 4.87 | rebuild | True |
| q_proj | 4096x3840 | 40 | 16 | 0.40 | 0.31 | 0.49 | 0.33 | fused | True |
| q_proj | 4096x3840 | 40 | 32 | 0.39 | 0.39 | 0.49 | 0.40 | fused | True |
| q_proj | 4096x3840 | 40 | 64 | 0.40 | 0.53 | 0.50 | 0.53 | fused | True |
| q_proj | 4096x3840 | 40 | 128 | 0.54 | 0.73 | 0.73 | 0.74 | fused | True |
| q_proj | 4096x3840 | 40 | 512 | 1.36 | 1.73 | 1.48 | 1.48 | rebuild | True |
| q_proj | 4096x3840 | 40 | 2048 | 4.62 | 6.24 | 4.98 | 4.95 | rebuild | True |
| v_proj | 2048x3840 | 40 | 16 | 0.19 | 0.34 | 0.24 | 0.24 | rebuild |  |
| v_proj | 2048x3840 | 40 | 32 | 0.21 | 0.39 | 0.26 | 0.27 | rebuild |  |
| v_proj | 2048x3840 | 40 | 64 | 0.25 | 0.55 | 0.30 | 0.30 | rebuild |  |
| v_proj | 2048x3840 | 40 | 128 | 0.32 | 0.53 | 0.37 | 0.38 | rebuild |  |
| v_proj | 2048x3840 | 40 | 512 | 0.84 | 1.11 | 0.90 | 0.92 | rebuild | True |
| v_proj | 2048x3840 | 40 | 2048 | 2.43 | 3.17 | 2.49 | 2.53 | rebuild | True |
| k_proj | 512x3840 | 8 | 16 | 0.15 | 0.32 | 0.19 | 0.19 | rebuild |  |
| k_proj | 512x3840 | 8 | 32 | 0.17 | 0.39 | 0.21 | 0.21 | rebuild |  |
| k_proj | 512x3840 | 8 | 64 | 0.18 | 0.51 | 0.21 | 0.22 | rebuild |  |
| k_proj | 512x3840 | 8 | 128 | 0.21 | 0.50 | 0.23 | 0.25 | rebuild |  |
| k_proj | 512x3840 | 8 | 512 | 0.32 | 0.52 | 0.36 | 0.36 | rebuild |  |
| k_proj | 512x3840 | 8 | 2048 | 0.85 | 1.10 | 0.88 | 0.88 | rebuild | True |
| o_proj | 3840x8192 | 8 | 16 | 0.30 | 0.57 | 0.54 | 0.55 | rebuild |  |
| o_proj | 3840x8192 | 8 | 32 | 0.39 | 0.76 | 0.66 | 0.66 | rebuild |  |
| o_proj | 3840x8192 | 8 | 64 | 0.51 | 1.00 | 0.75 | 0.77 | rebuild |  |
| o_proj | 3840x8192 | 8 | 128 | 0.84 | 1.34 | 1.11 | 1.12 | rebuild |  |
| o_proj | 3840x8192 | 8 | 512 | 2.47 | 3.22 | 2.73 | 2.73 | rebuild | True |
| o_proj | 3840x8192 | 8 | 2048 | 9.21 | 12.40 | 9.70 | 9.68 | rebuild | True |
| q_proj | 8192x3840 | 8 | 16 | 0.55 | 0.42 | 0.79 | 0.44 | fused | True |
| q_proj | 8192x3840 | 8 | 32 | 0.57 | 0.51 | 0.81 | 0.51 | fused | True |
| q_proj | 8192x3840 | 8 | 64 | 0.56 | 0.75 | 0.79 | 0.74 | fused | True |
| q_proj | 8192x3840 | 8 | 128 | 0.83 | 1.10 | 1.09 | 1.11 | fused | True |
| q_proj | 8192x3840 | 8 | 512 | 2.44 | 3.17 | 2.72 | 2.67 | rebuild | True |
| q_proj | 8192x3840 | 8 | 2048 | 9.29 | 12.56 | 9.71 | 9.73 | rebuild | True |
