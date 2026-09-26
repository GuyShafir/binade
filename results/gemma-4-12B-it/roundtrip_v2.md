# Binade format 2 round trip: google/gemma-4-12B-it

Packed with `binade/pack.py` @ `bd1ba8b`, checked with `scripts/roundtrip_test.py` @ `bd1ba8b`.

**PASS**: 677 tensors compared (329 packed), 0 mismatched elements, 0 missing, 0 extra.

Checkpoint 23.92 GB -> 16.10 GB (67.3%). Pack 10.9 min, check 0.5 min.

| class | tensors | weights | bits/weight (file) |
|---|---|---|---|
| embed | 1 | 1.007B | 10.707 |
| linear | 328 | 10.900B | 10.750 |
