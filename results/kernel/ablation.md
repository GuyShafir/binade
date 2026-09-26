# Where the v1 in-place kernel loses time (timing-only ablations)

`scripts/kernel_ablate.py` @ `13cd53c`, Gemma 4 12B weights, DRAM-resident, 7 paired rounds against MLX's BF16 matvec; efficiency on v1's bytes. Ablated variants compute wrong outputs on purpose.

| projection | shape | v1 kernel | escape load replaced by a constant | escape handling skipped |
|---|---|---|---|---|
| gate_up_proj | 15360x3840 | 73.0% | 89.5% | 95.3% |
| down_proj | 3840x15360 | 70.0% | 84.1% | 82.6% |
| q_proj | 4096x3840 | 71.8% | 84.1% | 85.7% |
