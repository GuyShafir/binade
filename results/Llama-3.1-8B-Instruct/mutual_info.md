# Exponent information in the row and column index: unsloth/Llama-3.1-8B-Instruct

Script `scripts/mutual_info.py` @ `bb6988f`. Linear tensors, bits per exponent, size-weighted over tensors, Miller-Madow corrected. I(col) bounds the entropy any column permutation or column partition can remove; I(row) the same for rows.

| class | tensors | H(exp) | H(exp given row) | H(exp given col) | I(exp; row) | I(exp; col) |
|---|---|---|---|---|---|---|
| linear | 224 | 2.5822 | 2.5583 | 2.5736 | 0.0239 | 0.0087 |
| attn | 128 | 2.6384 | 2.5815 | 2.6122 | 0.0568 | 0.0261 |
| mlp | 96 | 2.5688 | 2.5527 | 2.5644 | 0.0161 | 0.0045 |
