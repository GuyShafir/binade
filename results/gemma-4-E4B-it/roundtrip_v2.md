# Binade format 2 round trip: google/gemma-4-E4B-it

Packed with `binade/pack.py` @ `90326ad`, checked with `scripts/roundtrip_test.py` @ `90326ad`.

**PASS**: 2130 tensors compared (381 packed), 0 mismatched elements, 0 missing, 0 extra.

Checkpoint 15.99 GB -> 11.05 GB (69.1%). Pack 6.9 min, check 0.3 min.

| class | tensors | weights | bits/weight (file) |
|---|---|---|---|
| embed | 1 | 0.671B | 10.714 |
| ple | 1 | 2.819B | 10.713 |
| linear | 379 | 4.028B | 10.754 |
