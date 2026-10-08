# Binade format 2 round trip: unsloth/Llama-3.1-8B-Instruct

Packed with `binade/pack.py` @ `4d2376c`, checked with `scripts/roundtrip_test.py` @ `4d2376c`.

**PASS**: 291 tensors compared (226 packed), 0 mismatched elements, 0 missing, 0 extra.

Checkpoint 16.06 GB -> 10.77 GB (67.0%). Pack 10.4 min, check 0.5 min.

| class | tensors | weights | bits/weight (file) |
|---|---|---|---|
| embed | 2 | 1.051B | 10.728 |
| linear | 224 | 6.979B | 10.725 |
