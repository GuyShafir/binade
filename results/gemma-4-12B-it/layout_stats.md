# Tile layouts: google/gemma-4-12B-it

`scripts/layout_stats.py` @ `13cd53c`, all packed tensors (linear and embedding), 11.91B weights.

| layout | tiles by width | tiles with escapes | escaped weights | bits/weight |
|---|---|---|---|---|
| v1 (file) | b=2: 0.18%, b=3: 99.60%, b=4: 0.22%, b=5: 0.00% | 83.50% (escape mode) | 2.628% | see roundtrip.md |
| R4 (memory) | 4 bits for all | 3.10% | 0.0261% | 12.071 (raw, codes, flags, escapes, table, row starts) |
