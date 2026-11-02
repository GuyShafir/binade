# Binade format 2 round trip: Qwen/Qwen3.5-4B

Packed with `binade/pack.py` @ `d0315fd`, checked with `scripts/roundtrip_test.py` @ `7581266`.

**PASS**: 738 tensors compared (307 packed), 0 mismatched elements, 0 missing, 0 extra.

Checkpoint 9.32 GB -> 6.27 GB (67.3%). Pack 5.4 min, check 0.2 min.

| class | tensors | weights | bits/weight (file) |
|---|---|---|---|
| embed | 1 | 0.636B | 10.740 |
| linear | 207 | 3.673B | 10.738 |
| mm | 99 | 0.332B | 10.799 |
