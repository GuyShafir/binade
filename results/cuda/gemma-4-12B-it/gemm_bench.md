# CUDA matrix products: google/gemma-4-12B-it

`scripts/cuda_gemm_bench.py` @ `6f1607b`, MLX 0.32.2, NVIDIA RTX PRO 6000 Blackwell Server Edition. Per product, median of 5 runs of 10 independent products: BF16 (cuBLAS), rebuild (R4 to BF16, then cuBLAS) and the fused R4 GEMM with cuBLAS's plan; speeds relative to BF16; exact = fused output bit-identical to BF16's.

| weight | N x K | rows | cuBLAS plan (split-K, scheme) | BF16 ms | rebuild | fused | exact |
|---|---|---|---|---|---|---|---|
| q_proj | 4096 x 3840 | 40 | (1, 0) | 0.032 | 0.066 (0.49x) | 0.042 (0.77x) | True |
| q_proj | 4096 x 3840 | 128 | (5, 4) | 0.030 | 0.074 (0.40x) | 0.037 (0.81x) | True |
| q_proj | 4096 x 3840 | 256 | (2, 4) | 0.038 | 0.075 (0.50x) | 0.051 (0.73x) | True |
| q_proj | 4096 x 3840 | 512 | (1, 0) | 0.059 | 0.086 (0.69x) | 0.075 (0.79x) | True |
| q_proj | 4096 x 3840 | 1024 | (1, 0) | 0.101 | 0.129 (0.78x) | 0.130 (0.78x) | True |
| q_proj | 4096 x 3840 | 2048 | (1, 0) | 0.165 | 0.195 (0.85x) | 0.240 (0.69x) | True |
| o_proj | 3840 x 4096 | 40 | (1, 0) | 0.031 | 0.066 (0.47x) | 0.041 (0.75x) | True |
| o_proj | 3840 x 4096 | 128 | (6, 4) | 0.030 | 0.075 (0.41x) | 0.038 (0.81x) | True |
| o_proj | 3840 x 4096 | 256 | (3, 4) | 0.040 | 0.077 (0.52x) | 0.052 (0.76x) | True |
| o_proj | 3840 x 4096 | 512 | (1, 0) | 0.060 | 0.087 (0.69x) | 0.076 (0.79x) | True |
| o_proj | 3840 x 4096 | 1024 | (1, 0) | 0.100 | 0.128 (0.78x) | 0.128 (0.78x) | True |
| o_proj | 3840 x 4096 | 2048 | (1, 0) | 0.163 | 0.193 (0.84x) | 0.234 (0.70x) | True |
| gate_proj | 15360 x 3840 | 40 | (1, 0) | 0.052 | 0.248 (0.21x) | 0.083 (0.63x) | True |
| gate_proj | 15360 x 3840 | 128 | (1, 0) | 0.087 | 0.260 (0.33x) | 0.074 (1.16x) | True |
| gate_proj | 15360 x 3840 | 256 | (1, 0) | 0.099 | 0.265 (0.37x) | 0.122 (0.81x) | True |
| gate_proj | 15360 x 3840 | 512 | (1, 0) | 0.157 | 0.323 (0.49x) | 0.227 (0.69x) | True |
| gate_proj | 15360 x 3840 | 1024 | (1, 0) | 0.303 | 0.458 (0.66x) | 0.452 (0.67x) | True |
| gate_proj | 15360 x 3840 | 2048 | (1, 0) | 0.589 | 0.742 (0.79x) | 0.920 (0.64x) | True |
| down_proj | 3840 x 15360 | 40 | (2, 4) | 0.077 | 0.233 (0.33x) | 0.108 (0.71x) | True |
| down_proj | 3840 x 15360 | 128 | (6, 2) | 0.072 | 0.241 (0.30x) | 0.096 (0.75x) | True |
| down_proj | 3840 x 15360 | 256 | (3, 4) | 0.097 | 0.245 (0.40x) | 0.141 (0.69x) | True |
| down_proj | 3840 x 15360 | 512 | (3, 4) | 0.170 | 0.294 (0.58x) | 0.253 (0.67x) | True |
| down_proj | 3840 x 15360 | 1024 | (3, 1) | 0.312 | 0.441 (0.71x) | 0.558 (0.56x) | True |
| down_proj | 3840 x 15360 | 2048 | (4, 1) | 0.595 | 0.721 (0.83x) | 1.110 (0.54x) | True |
