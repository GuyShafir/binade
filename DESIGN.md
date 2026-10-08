# Binade design

How Binade stores BF16 weights, how the formats were chosen, and how the runtime reproduces the reference model's outputs bit for bit. Result files under `results/` hold the measurements referred to here.

## 1. BF16 layout

A BF16 weight is bit 15 sign, bits 14..7 exponent, bits 6..0 mantissa. Binade splits it into a raw byte, stored verbatim, and the exponent, which is compressed:

```
raw = ((w >> 8) & 0x80) | (w & 0x7F)    # uint8: sign in bit 7, mantissa in bits 6..0
exp = (w >> 7) & 0xFF                    # uint8
w   = (raw & 0x80) << 8 | exp << 7 | (raw & 0x7F)
```

Every BF16 bit pattern, including NaN, infinities, zeros and subnormals, round-trips (`tests/test_bf16.py`). Only linear and embedding weights are packed; everything else stays BF16.

## 2. Tiles

A linear weight is `[N, K]` (output rows, reduction dimension K), row-major. A tile is T contiguous weights along K within one row. Binade uses T = 128, the best T for almost every coding studied (`results/*/summary.md`). A row's last tile may be partial when K is not a multiple of 128. Fused expert stacks `[E, N, K]` are treated as `[E * N, K]`.

## 3. Per-tile palette encodings (the starting hypothesis)

The initial design gave every tile its own palette: `k_t` distinct exponents, a palette of up to `2^b_t` of them, and `b_t`-bit indices, with three ways to handle tiles whose exponents do not fit:

- **bump**: `b_t = ceil(log2 k_t)`, verbatim if `k_t > 16`;
- **escape**: `b_t` in 1..4, the top `2^b_t - 1` exponents in the palette, the last code escaping to a byte in a side stream;
- **verbatim**: 8-bit exponents.

"Best-of" picks the cheapest per tile. A v0 file layout stored a zero-padded `[ntiles, 16]` palette per tensor (reported as "v0 padded palette"). This design pays off only if exponents cluster along K, which section 5 tests.

## 4. Bit accounting

Bits per weight count everything a file stores for a weight: the 8-bit raw byte, index bits, palette or table bytes, per-tile meta bytes, escaped bytes and any offsets, divided by the number of weights. Baselines are computed by the same scanner on the same tensors:

- per-block Huffman over exponents (the granularity of DFloat11);
- a 7-exponent window with 3-bit codes and outliers stored as BF16 (the ZipServ layout);
- a 16-exponent palette with fixed 4-bit indices (Unweight's 4-bit mode).

These are our accounting of each layout, not the other systems' reported numbers.

## 5. Locality test

For every linear tensor the scanner computes each encoding's cost on the real exponents and after shuffling them within the tensor, which keeps the distribution and destroys any locality. The difference is the locality gain. A row-shuffled control shuffles within each row. Decision bands, fixed before measuring:

- gain of at least 0.7 bits per weight: per-tile palettes are worth building;
- 0.2 to 0.7: a speed story at about 11.5 bits, smaller formats exist;
- below 0.2: stop, and model a Rice-style code over frequency ranks before anything else.

All four models measured (Gemma 4 12B and E4B, Llama-3.1-8B, Qwen3.5-9B) land below 0.2: 0.011 to 0.027 bits per weight (`results/comparison.md`). An iid Gaussian model predicts this (`results/iid_model.md`), and the mutual information between an exponent and its row or column index bounds what any reordering could gain (`results/*/mutual_info.md`).

## 6. Format 1: per-tensor ranks, fixed width with escapes

One table per tensor lists its exponents by descending frequency (rank 0 is the most common). Each weight stores its rank. Per tile, a width `b` in 0..8 and a mode:

- **fixed**: every weight stores its rank in b bits, b = bit length of the tile's largest rank;
- **escape** (b in 1..4): ranks below `2^b - 1` are stored directly; code `2^b - 1` escapes to the exponent byte in `.esc`.

The packer takes the cheaper per tile.

| name | dtype | shape | content |
|---|---|---|---|
| `{name}.raw` | uint8 | `[N, K]` | sign + mantissa |
| `{name}.table` | uint8 | `[256]` | exponent at each frequency rank, zero padded |
| `{name}.meta` | uint8 | `[ntiles]` | bits 0..3 = b, bit 4 = escape mode |
| `{name}.idx` | uint32 | `[4 * sum_t b_t]` | 128 b-bit codes per tile, LSB first within little-endian words; tile t is exactly 4 b_t words |
| `{name}.esc` | uint8 | `[n_escapes]` | escaped exponents, tile order |

No offsets are stored: tile t starts at word 4 x (sum of b over earlier tiles), and its escapes after the escape codes of earlier tiles. `quantization_config.version` is "1". Cost: 11.265 to 11.292 bits per weight on the four models.

## 7. Format 2: Rice over the same ranks (default)

Same tiles, table and raw bytes. Each tile's ranks are Rice coded with a parameter k in 0..3 chosen per tile, q = rank >> k:

- a low plane of `n k` bits (weight i's low bits at `i k`), n being the tile's weight count (the last tile of a row codes only its real weights);
- unary planes j = 0 .. qmax - 1, each holding one bit for every weight whose q is at least j, in weight order: 1 if q > j. A weight with q = qmax has no closing 0.

A tile costs `n k + sum(min(q + 1, qmax))` bits. Rows start on a 32-bit word, so rows decode independently.

| name | dtype | shape | content |
|---|---|---|---|
| `{name}.raw` | uint8 | `[N, K]` | sign + mantissa |
| `{name}.table` | uint8 | `[256]` | as format 1 |
| `{name}.meta` | uint8 | `[ntiles]` | k in bits 0..1, qmax in bits 2..7 (at most 63; k >= 2 always meets that) |
| `{name}.bits` | uint32 | `[words]` | tile streams, LSB first within little-endian words, rows word aligned |
| `{name}.roff` | uint32 | `[rows]` | first word of each row |

Plane j's length is the number of weights still alive after plane j - 1, so a row's tiles decode in order. The decoder reports where each row ends and rejects a stream that does not end where the next row starts. `quantization_config.version` is "2". Cost: 10.746 bits per weight on Gemma 4 12B and 10.747 on 26B-A4B, a Gemma 4 12B checkpoint of 16.10 GB against 23.92 GB in BF16 (`results/gemma-4-12B-it/roundtrip_v2.md`).

Rice's plane-by-plane decode depends on data loaded in the previous plane, which is slow inside a matrix-vector kernel (section 9). Format 2 is therefore a file format only: at load, a GPU kernel decodes it (one SIMD group or warp per row, one prefix sum per plane) into the R4 runtime layout.

## 8. R4 runtime layout

Both file formats are transcoded at load into R4:

- every weight's rank in 4 bits, four per lane in one uint16, in the lane order of the matvec kernel (lane i owns weights 4i..4i+3 of each 128-weight tile);
- code 15 escapes: the exponent comes from a per-tensor escape stream (0.026% of Gemma 4 12B weights, in 3.1% of tiles);
- a flag per tile that has escapes, and each row's first escape.

R4 takes 12.07 bits per weight in memory (`results/gemma-4-12B-it/layout_stats.md`). Its addresses are all known in advance except an escape's, so a kernel can prefetch tiles ahead.

## 9. Kernels and exactness

The runtime does not only restore the weights exactly; its outputs are bit-identical to the framework's own BF16 kernels for the same inputs. Each kernel reproduces the floating-point summation order of the MLX kernel it replaces:

- **Batch-1 matvec (Metal `gemv`)**: one SIMD group per output row, lane i accumulating weights 4i..4i+3 of every 128-weight block in float, K in order, then a shuffle-down reduction. A partial last block adds 0 x 0 on idle lanes as MLX's tail does. MLX uses this layout for 64 < K < 16 N; elsewhere the runtime rebuilds the BF16 weight and calls MLX.
- **Expert gather (Metal `gemv_gather`)**: the same per-row order, one matvec per (token, expert) slot. mlx-lm's MoE uses it for every decode step.
- **GEMM (Metal steel GEMM and sorted `gather_mm`)**: MLX accumulates each output in float `simdgroup_multiply_accumulate` steps of 8 along K, in order, whatever its tile sizes, when K is a multiple of its K block and no split-K applies. The R4 GEMM runs the same steps, so its tiling is free: in a 64 x 64 threadgroup tile, four SIMD groups each own 32 x 32 outputs, and K is staged in chunks of 32 while two threads per weight row decode it (smaller tiles for short inputs, `r4.gemm_tile`).
- **CUDA (MLX's CUDA backend)**: `gemv_impl` runs one warp per row, thread t owning n consecutive weights of every block of 32 n (n = 4, 2 or 1 by K), then `cooperative_groups::reduce`. The CUDA kernels mirror it and call the same reduce.
- **Multi-row products on CUDA (cuBLASLt, closed)**: without split-K each output is tensor-core `mma.sync m16n8k16` steps along K, in order, from zero, then one rounding to BF16. With split-K into S partitions, partitions are ceil(K / S) rounded up to a multiple of 64; each is summed the same way and its partial rounded to BF16 (`OUTPUT_TYPE`) or kept in float32 (`COMPUTE_TYPE`), and partials are added in float32 in partition order; with `INPLACE`, partition s adds its float32 sum to the BF16 output so far and rounds again. One algorithm (cuBLASLt's 31, on the A100 and L4) sums each output in two accumulators over alternating 32-wide slices of K, added at the end. `binade/mlx/cublas_plan.py` asks cuBLASLt which algorithm, split and reduction it picks for MLX's descriptors; the fused R4 GEMMs follow that plan for up to 128 rows when the algorithm is one verified on that GPU (`ALGO_SLICES`: RTX PRO 6000, L4, A100), and the runtime rebuilds BF16 for cuBLAS otherwise. MoE prompts mirror MLX's CUTLASS grouped GEMM (per expert, in-order m16n8k16 steps, no split-K).

For multi-row inputs (prompts), the runtime picks per call, every path exact:

- the fused R4 GEMM for up to 128 rows, or whenever a BF16 rebuild would crowd GPU memory;
- otherwise the weight rebuilt in BF16 and MLX's own matmul.

The dispatch conditions follow MLX 0.32.2 (`mlx/backend/metal/matmul.cpp`, `mlx/backend/cuda/gemms/gemv.cu`). Exactness is specific to that version and GPU generation, and the tests check it. Random test data cannot confirm a summation order, because float differences rarely survive rounding to BF16. The tests therefore also use rows of 2^25, ones that vanish against it, and -2^25, whose result depends on the order of the sum (`tests/test_r4_moe.py`).

## 10. Packer, unpack and runtimes

- `binade pack` streams a BF16 safetensors model one tensor at a time and writes packed tensors next to copied config and tokenizer files. `quantization_config.quant_method` is "binade", because mlx-lm reads that key.
- `binade unpack` rebuilds the BF16 checkpoint. `scripts/roundtrip_test.py` checks every tensor bit for bit.
- The slow runtime (`binade/mlx/slow_linear.py`, format 1 only) rebuilds each weight with MLX array operations for every call. It is the kernel-free reference.
- The R4 runtime (`binade/mlx/r4.py`) is the fast path: `binade.mlx.r4.load(path)` returns an mlx-lm model and tokenizer.
- The loader wraps mlx-lm's model class rather than patching it. It runs the model's own `sanitize` on placeholders to find which module each packed weight belongs to, and drops checkpoint tensors that have no model parameter (recorded in the result files).

## 11. Lossy dial (reported, not implemented)

For each tile and m in 2..7 mantissa bits kept (round to nearest even, exponent exact), the scanner reports relative Frobenius and elementwise errors. It is the starting point for a lossy mode, not part of the formats.
