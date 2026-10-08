# Binade format 2 round trip: google/gemma-4-31B-it

Packed with `binade/pack.py` @ `4d2376c`, checked with `scripts/roundtrip_test.py` @ `4d2376c`.

**PASS**: 1188 tensors compared (411 packed), 0 mismatched elements, 0 missing, 0 extra.

Checkpoint 62.55 GB -> 42.30 GB (67.6%). Pack 38.4 min, check 1.4 min.

| class | tensors | weights | bits/weight (file) |
|---|---|---|---|
| embed | 1 | 1.409B | 10.720 |
| linear | 410 | 29.287B | 10.724 |
