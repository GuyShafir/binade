# Exponent information in the row and column index: google/gemma-4-12B-it

Script `scripts/mutual_info.py` @ `bb6988f`. Linear tensors, bits per exponent, size-weighted over tensors, Miller-Madow corrected. I(col) bounds the entropy any column permutation or column partition can remove; I(row) the same for rows.

| class | tensors | H(exp) | H(exp given row) | H(exp given col) | I(exp; row) | I(exp; col) |
|---|---|---|---|---|---|---|
| linear | 328 | 2.6110 | 2.5802 | 2.5942 | 0.0308 | 0.0167 |
| attn | 184 | 2.6433 | 2.5906 | 2.6166 | 0.0527 | 0.0268 |
| mlp | 144 | 2.6018 | 2.5772 | 2.5879 | 0.0246 | 0.0139 |
