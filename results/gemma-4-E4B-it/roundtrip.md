# xp v1 round trip: google/gemma-4-E4B-it

Packed with `xp/pack.py` @ `07601d3`, checked with `scripts/roundtrip_test.py` @ `07601d3`.

**PASS**: 2130 tensors compared (380 packed), 0 mismatched elements, 0 missing, 0 extra.

Checkpoint 15.99 GB -> 13.23 GB (82.7%). Pack 1.2 min, check 1.1 min.

| class | tensors | weights | bits/weight (file) |
|---|---|---|---|
| embed | 1 | 0.671B | 11.270 |
| linear | 379 | 4.028B | 11.292 |
