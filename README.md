# Binade

Bit-exact compression of BF16 LLM weights for Apple Silicon (MLX), with an experimental CUDA port.

A BF16 weight keeps its sign and mantissa as one raw byte. Its 8-bit exponent (the weight's binade) is replaced by its frequency rank in a per-tensor table, stored on disk in Rice codes at about 10.75 bits per weight instead of 16. At load, the ranks become a 4-bit runtime layout that Metal kernels read directly during inference.

The outputs are bit-identical to the original model, not merely close. The kernels reproduce the summation order of MLX's own BF16 kernels, so prompt logits and generated tokens match the uncompressed model exactly.

## Results

Apple M4 Max, 48 GB, MLX 0.32.2. Every number comes from a script in `scripts/`; its output is in `results/`.

| | BF16 | Binade | |
|---|---|---|---|
| Gemma 4 12B checkpoint | 23.92 GB | 16.10 GB | 10.746 bits per packed weight, bit-exact round trip ([roundtrip_v2](results/gemma-4-12B-it/roundtrip_v2.md)) |
| Gemma 4 12B greedy decode | 19.35 tok/s | 24.87 tok/s | **1.29x**, identical tokens ([decode_speed](results/gemma-4-12B-it/decode_speed.md)) |
| Gemma 4 12B exactness | | | prompt logits over the full vocabulary and 256 greedy tokens identical ([generate_check](results/gemma-4-12B-it/generate_check_r4_v2_binade_first.md)) |
| Gemma 4 26B-A4B checkpoint | 51.61 GB | 35.05 GB | BF16 does not load on a 48 GB Mac ([roundtrip_v2](results/gemma-4-26B-A4B-it/roundtrip_v2.md)) |
| Gemma 4 26B-A4B decode | n/a | 65.4 tok/s | 40-token prompt in 0.40 s, logits and 64 tokens identical to BF16 ([decode_speed](results/gemma-4-26B-A4B-it/decode_speed.md), [generate_check](results/gemma-4-26B-A4B-it/generate_check_r4_v2_streamed_binade_first.md)) |

The 26B exactness check compares against a BF16 reference that reads one layer at a time from disk, since BF16 does not fit in memory.

- **Matvec kernel.** Batch-1 matvec on the 12B's shapes runs at 100% of BF16's bandwidth efficiency, 1.33x BF16 per token ([kernel test](results/kernel/gemma-4-12B-it.md)).
- **Prompt processing (12B matmuls).** 1.05x BF16 at 16 prompt rows, 0.65 to 0.73x at 32 to 128, and 0.97x at 2048 ([prefill_bench](results/gemma-4-12B-it/prefill_bench.md)).
- **CUDA (NVIDIA RTX PRO 6000 Blackwell, MLX's CUDA backend).** Exact against BF16 on the 12B and the 26B. Decode is 1.06x (12B) and 0.95x (26B) with an unoptimized first kernel port ([results/cuda](results/cuda)).

## How it works

- **No exponent locality.** Neighbouring weights do not share exponents: per-tile palettes gain only 0.011 to 0.027 bits per weight over a shuffled tensor on four models ([comparison](results/comparison.md)). Binade therefore codes frequency ranks from one table per tensor.
- **Two file formats.** Format 1 uses fixed width per 128-weight tile with escapes (11.27 bits per weight). Format 2, the default, uses Rice codes (10.75 bits per weight).
- **Runtime layout.** At load a GPU kernel turns either format into R4: every rank in 4 bits, with rare escapes. R4 takes 12.07 bits per weight in memory, and its matvec streams at full bandwidth.
- **Exactness.** The matvec, the MoE expert gather and the GEMM mirror MLX's own kernels step for step. Where they cannot, the runtime rebuilds the BF16 weight and calls MLX, which is exact by construction.

[DESIGN.md](DESIGN.md) has the formats, the runtime layout and the exactness argument.

## Install

Python 3.11 or newer on Apple Silicon (or Linux with CUDA 12). Exactness depends on the MLX version, so the dependencies are pinned (MLX 0.32.2, mlx-lm 0.31.3).

```
git clone https://github.com/GuyShafir/binade && cd binade
pip install -e ".[dev]"                          # Apple Silicon
pip install -e ".[dev]" "mlx[cuda]==0.32.2"      # Linux + CUDA 12
```

## Use

Pack a model from the Hugging Face cache or a local directory:

```
hf download google/gemma-4-12B-it
binade pack google/gemma-4-12B-it --out models/gemma-4-12B-it-binade
```

Run it with mlx-lm:

```python
from mlx_lm import generate
from binade.mlx.r4 import load

model, tokenizer, _ = load("models/gemma-4-12B-it-binade")
messages = [{"role": "user", "content": "Why is the sky blue?"}]
prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
print(generate(model, tokenizer, prompt, max_tokens=200))
```

`binade unpack <packed-dir> --out DIR` rebuilds the original BF16 checkpoint.

## Verify and reproduce

- `scripts/roundtrip_test.py`: every tensor of a packed model against the original, bit for bit.
- `scripts/generate_check.py`: greedy decoding against the BF16 model (`--reference streamed` for models that do not fit).
- `scripts/decode_speed.py`, `scripts/matvec_bench.py`, `scripts/prefill_bench.py`: speed.
- `scripts/histogram.py`, `scripts/mutual_info.py`, `scripts/iid_model.py`, `scripts/compare.py`: the exponent statistics.
- `pytest`: unit and kernel tests, including order-sensitive data that exposes any change in summation order.

Result files record the commit of the development repository that produced them; that history is not published. Early result files use the project's working name, `xp`.

## Limitations

- **Tied to MLX's internals.** Exactness relies on MLX 0.32.2's kernels and dispatch rules. A different MLX version or GPU generation can change the summation order, and the tests would catch it. The fused GEMM is enabled only on M3 and M4 GPUs.
- **Validated models.** Validated end to end on Gemma 4 12B, E4B (slow runtime) and 26B-A4B. Llama-3.1-8B and Qwen3.5-9B were used for the exponent statistics only.
- **E-models.** Gemma 4 E-models keep their per-layer embedding tables in BF16.
- **Prompt processing.** Prefill at 32 to 128 prompt rows runs at 0.65 to 0.73x BF16. On long prompts, a model without memory headroom for rebuilding weights runs the fused GEMM, which is about 2.7x slower than MLX's.
- **Quantization is faster.** Lossy 8-bit quantization is faster and smaller; Binade's case is exactness.
- **CUDA.** The CUDA kernels are a first port.

## Related work

Binade codes the same exponent redundancy as:
- DFloat11 (Huffman, arXiv 2504.11651);
- ZipServ (fixed-width exponent windows fused into GEMM, arXiv 2603.17435);
- Unweight (per-tensor palettes, Cloudflare);
- ENEC (frequency ranks, arXiv 2604.03298);
- NeuZip (arXiv 2410.20650) and ZipNN (arXiv 2411.05239).

We found no public implementation of lossless compressed inference on Metal.

## License

Apache-2.0 ([LICENSE](LICENSE)). Model weights keep their own licenses.
