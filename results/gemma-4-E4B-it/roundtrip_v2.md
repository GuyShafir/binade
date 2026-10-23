# Binade format 2 round trip: google/gemma-4-E4B-it

Packed with `binade/pack.py` @ `df77a49`, checked with `scripts/roundtrip_test.py` @ `df77a49`.

**PASS**: 2130 tensors compared (380 packed), 0 mismatched elements, 0 missing, 0 extra.

Checkpoint 15.99 GB -> 12.91 GB (80.7%). Pack 4.8 min, check 0.4 min.

| class | tensors | weights | bits/weight (file) |
|---|---|---|---|
| embed | 1 | 0.671B | 10.714 |
| linear | 379 | 4.028B | 10.754 |
