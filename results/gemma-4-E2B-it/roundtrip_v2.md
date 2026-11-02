# Binade format 2 round trip: google/gemma-4-E2B-it

Packed with `binade/pack.py` @ `d0315fd`, checked with `scripts/roundtrip_test.py` @ `d0315fd`.

**PASS**: 2011 tensors compared (568 packed), 0 mismatched elements, 0 missing, 0 extra.

Checkpoint 10.25 GB -> 6.84 GB (66.8%). Pack 4.8 min, check 0.2 min.

| class | tensors | weights | bits/weight (file) |
|---|---|---|---|
| mm | 250 | 0.475B | 10.207 |
| embed | 1 | 0.403B | 10.742 |
| ple | 1 | 2.349B | 10.705 |
| linear | 316 | 1.896B | 10.767 |
