# Binade format 2 round trip: google/gemma-4-26B-A4B-it

Packed with `binade/pack.py` @ `74a5799`, checked with `scripts/roundtrip_test.py` @ `74a5799`.

**PASS**: 1013 tensors compared (266 packed), 0 mismatched elements, 0 missing, 0 extra.

Checkpoint 51.61 GB -> 35.05 GB (67.9%). Pack 23.2 min, check 1.0 min.

| class | tensors | weights | bits/weight (file) |
|---|---|---|---|
| embed | 1 | 0.738B | 10.733 |
| linear | 265 | 24.483B | 10.748 |
