# Exponent information in the row and column index: google/gemma-4-E4B-it

Script `scripts/mutual_info.py` @ `bb6988f`. Linear tensors, bits per exponent, size-weighted over tensors, Miller-Madow corrected. I(col) bounds the entropy any column permutation or column partition can remove; I(row) the same for rows.

| class | tensors | H(exp) | H(exp given row) | H(exp given col) | I(exp; row) | I(exp; col) |
|---|---|---|---|---|---|---|
| linear | 379 | 2.6113 | 2.5779 | 2.5916 | 0.0334 | 0.0197 |
| attn | 168 | 2.6302 | 2.5826 | 2.6017 | 0.0476 | 0.0285 |
| mlp | 126 | 2.6029 | 2.5760 | 2.5878 | 0.0269 | 0.0152 |
| ple_proj | 85 | 2.7982 | 2.6175 | 2.6671 | 0.1807 | 0.1311 |
