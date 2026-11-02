# Binade format 2 round trip: google/gemma-4-E4B-it

Packed with `binade/pack.py` @ `d0315fd`, checked with `scripts/roundtrip_test.py` @ `d0315fd`.

**PASS**: 2130 tensors compared (631 packed), 0 mismatched elements, 0 missing, 0 extra.

Checkpoint 15.99 GB -> 10.70 GB (66.9%). Pack 7.5 min, check 0.4 min.

| class | tensors | weights | bits/weight (file) |
|---|---|---|---|
| mm | 250 | 0.478B | 10.210 |
| embed | 1 | 0.671B | 10.714 |
| ple | 1 | 2.819B | 10.713 |
| linear | 379 | 4.028B | 10.754 |
