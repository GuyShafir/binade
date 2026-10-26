# Binade format 2 round trip: unsloth/Llama-3.2-3B-Instruct

Packed with `binade/pack.py` @ `75147d5`, checked with `scripts/roundtrip_test.py` @ `75147d5 (dirty)`.

**PASS**: 254 tensors compared (197 packed), 0 mismatched elements, 0 missing, 0 extra.

Checkpoint 6.43 GB -> 4.31 GB (67.1%). Pack 3.0 min, check 0.1 min.

| class | tensors | weights | bits/weight (file) |
|---|---|---|---|
| embed | 1 | 0.394B | 10.737 |
| linear | 196 | 2.819B | 10.730 |
