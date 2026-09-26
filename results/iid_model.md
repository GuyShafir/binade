# The iid model: fixed-width tile coding on Gaussian BF16 weights

`scripts/iid_model.py` @ `13d47cb`, one 4096x3840 tensor per model, seed 0. Bits per weight including the 8-bit sign+mantissa byte; same scanner code as `results/*/summary.md`.

## iid Gaussian

Exponent entropy 2.545 bits (10.545 per weight), Huffman 2.591 (10.591), 25 distinct exponents.

| policy | T=32 real / shuffled | T=64 real / shuffled | T=128 real / shuffled |
|---|---|---|---|
| bump | 12.957 / 12.957 | 12.320 / 12.321 | 12.239 / 12.238 |
| escape | 12.765 / 12.764 | 12.086 / 12.086 | 11.646 / 11.646 |
| best | 12.761 / 12.761 | 12.084 / 12.084 | 11.640 / 11.640 |
| unary | 12.426 / 12.425 | 11.697 / 11.697 | 11.270 / 11.270 |
| rank_fixed | 11.559 / 11.558 | 11.655 / 11.654 | 11.847 / 11.846 |
| rank_best | 11.356 / 11.356 | 11.257 / 11.257 | 11.215 / 11.215 |
| rank_unary | 10.910 / 10.910 | 10.808 / 10.808 | 10.756 / 10.756 |
| rank_rice | 10.788 / 10.788 | 10.710 / 10.710 | 10.674 / 10.674 |

## Gaussian, per-row log-normal scale (sigma 0.3)

Exponent entropy 2.640 bits (10.640 per weight), Huffman 2.673 (10.673), 28 distinct exponents.

| policy | T=32 real / shuffled | T=64 real / shuffled | T=128 real / shuffled |
|---|---|---|---|
| bump | 12.966 / 13.089 | 12.370 / 12.561 | 12.303 / 12.490 |
| escape | 12.744 / 12.890 | 12.088 / 12.165 | 11.663 / 11.719 |
| best | 12.739 / 12.888 | 12.085 / 12.161 | 11.657 / 11.715 |
| unary | 12.427 / 12.625 | 11.708 / 11.890 | 11.278 / 11.452 |
| rank_fixed | 11.724 / 11.735 | 11.848 / 11.866 | 11.991 / 12.011 |
| rank_best | 11.428 / 11.451 | 11.336 / 11.356 | 11.297 / 11.313 |
| rank_unary | 11.062 / 11.067 | 10.960 / 10.963 | 10.909 / 10.911 |
| rank_rice | 10.872 / 10.882 | 10.790 / 10.797 | 10.749 / 10.753 |
