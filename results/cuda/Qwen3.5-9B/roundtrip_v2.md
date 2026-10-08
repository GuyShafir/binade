# Binade format 2 round trip: Qwen/Qwen3.5-9B

Packed with `binade/pack.py` @ `4d2376c`, checked with `scripts/roundtrip_test.py` @ `4d2376c`.

**PASS**: 775 tensors compared (209 packed), 0 mismatched elements, 0 missing, 0 extra.

Checkpoint 19.31 GB -> 13.27 GB (68.7%). Pack 11.8 min, check 0.5 min.

| class | tensors | weights | bits/weight (file) |
|---|---|---|---|
| embed | 2 | 2.034B | 10.726 |
| linear | 207 | 7.122B | 10.720 |
