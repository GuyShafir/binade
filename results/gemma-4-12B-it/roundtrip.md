# xp v1 round trip: google/gemma-4-12B-it

Packed with `xp/pack.py` @ `040b9ce`, checked with `scripts/roundtrip_test.py` @ `040b9ce`.

**PASS**: 677 tensors compared (329 packed), 0 mismatched elements, 0 missing, 0 extra.

Checkpoint 23.92 GB -> 16.89 GB (70.6%). Pack 2.8 min, check 2.4 min.

| class | tensors | weights | bits/weight (file) |
|---|---|---|---|
| embed | 1 | 1.007B | 11.242 |
| linear | 328 | 10.900B | 11.276 |
