# Exponent information in the row and column index: Qwen/Qwen3.5-9B

Script `scripts/mutual_info.py` @ `bb6988f`. Linear tensors, bits per exponent, size-weighted over tensors, Miller-Madow corrected. I(col) bounds the entropy any column permutation or column partition can remove; I(row) the same for rows.

| class | tensors | H(exp) | H(exp given row) | H(exp given col) | I(exp; row) | I(exp; col) |
|---|---|---|---|---|---|---|
| linear | 207 | 2.5806 | 2.5619 | 2.5718 | 0.0187 | 0.0087 |
| attn | 36 | 2.6235 | 2.5781 | 2.6028 | 0.0454 | 0.0206 |
| linattn | 72 | 2.5876 | 2.5667 | 2.5794 | 0.0209 | 0.0082 |
| mlp | 99 | 2.5738 | 2.5586 | 2.5661 | 0.0151 | 0.0077 |
