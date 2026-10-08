#!/usr/bin/env python3
"""Exactness at scale: full logits on WikiText-2 windows and greedy generations, Binade (and
optionally 8-bit quantization) against BF16.

    python scripts/exactness_scale.py <original model-dir | cached-hf-repo-id> <packed-dir> [--q8] [--fused] [--reordered] [--reference resident|streamed] [--windows 32] [--window-tokens 512] [--prompts 32] [--gen-tokens 128]

Each variant runs in its own process: bf16 (stock mlx-lm), binade (R4 runtime), with --fused
binade_fused (the R4 runtime with the fused R4 GEMM for every input it can mirror at any
length: on Metal every input MLX runs its plain GEMM on, as on a model without memory
headroom; on CUDA every input whose cuBLAS plan it reproduces), with --reordered binade_reordered (binade_fused with every fused kernel summing even
and odd K steps in two accumulators: the same weights and products in a valid order that is
not MLX's, the ablation of exactness) and, with --q8, MLX's 8-bit affine quantization
(group 64) of the same model. On the same token
windows of the WikiText-2 test set (raw), every variant computes the logits at every
position in one forward pass (the multi-row paths), and on the same prompts it greedy-decodes
(the single-row paths). Compared against BF16: positions whose full logit vector is
bit-identical (a hash of the BF16 bits), argmax agreement, perplexity, and generations whose
tokens and per-step log-probabilities are all identical. For a model whose BF16 does not fit
in memory, --reference streamed reads each BF16 decoder layer from the checkpoint when it runs
(binade.mlx.loader.load_streamed_model) and runs last; at about one checkpoint read per
forward pass it suits the windows, not generation (--prompts 0). Writes exactness_scale.json
/ .md to results/<model>/ (or --out).
"""

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
DATASET = ("Salesforce/wikitext", "wikitext-2-raw-v1/test-00000-of-00001.parquet")

CHILD = r'''
import hashlib, json, sys, time
import numpy as np
import mlx.core as mx
from mlx_lm.generate import generate_step
variant, path, data, out, reference = sys.argv[1:6]
if variant == "bf16" and reference == "streamed":
    from binade.mlx.loader import load_reference_streamed as load
elif variant == "bf16":
    from binade.mlx.loader import load_reference as load
elif variant == "q8":
    from binade.mlx.loader import load_quantized_reference as load
else:
    import binade.mlx.r4 as r4
    if variant in ("binade_fused", "binade_reordered"):
        r4.FUSED_MAX_ROWS = r4.CUDA_FUSED_MAX_ROWS = 1 << 30
    if variant == "binade_reordered":
        import binade.mlx.kernel
        binade.mlx.kernel.TWO_ACCUMULATORS = True
    load = r4.load
model, tok, _ = load(path)
d = np.load(data)
windows, prompts, gen = d["windows"], d["prompts"], int(d["gen"])
def h(row):  # 64-bit hash of one position's logits (bit patterns)
    return int.from_bytes(hashlib.blake2b(row.tobytes(), digest_size=8).digest(), "little")
W, L = windows.shape
wh = np.zeros((W, L - 1), np.uint64); wa = np.zeros((W, L - 1), np.int32); wl = np.zeros((W, L - 1), np.float32)
t0 = time.perf_counter()
C = 64  # positions per chunk of float32 statistics: a 26B model leaves little GPU memory beside its logits
for i in range(W):
    x = mx.array(windows[i][None])
    logits = model(x)[0, :-1]
    mx.eval(logits)
    for c in range(0, L - 1, C):
        lg = logits[c : c + C].astype(mx.float32)
        tgt = mx.take_along_axis(lg, mx.array(windows[i][c + 1 : c + C + 1, None]), axis=-1)[:, 0] - mx.logsumexp(lg, axis=-1)
        am = mx.argmax(logits[c : c + C], axis=-1)
        mx.eval(tgt, am)
        wa[i, c : c + C] = np.array(am); wl[i, c : c + C] = np.array(tgt)
    bits = np.array(logits.view(mx.uint16))
    wh[i] = [h(r) for r in bits]
    del logits, bits
t1 = time.perf_counter()
P = prompts.shape[0]
gi = np.zeros((P, gen), np.int32); gh = np.zeros((P, gen), np.uint64)
for i in range(P):
    for j, (t, lp) in enumerate(generate_step(mx.array(prompts[i]), model, max_tokens=gen)):
        gi[i, j] = int(t); gh[i, j] = h(np.array(lp.astype(mx.float32)))
t2 = time.perf_counter()
np.savez(out, wh=wh, wa=wa, wl=wl, gi=gi, gh=gh, t_windows=t1 - t0, t_gen=t2 - t1, peak_gb=mx.get_peak_memory() / 1e9)
print(json.dumps({"variant": variant, "windows_s": t1 - t0, "gen_s": t2 - t1}))
'''


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("original")
    ap.add_argument("packed", type=Path)
    ap.add_argument("--q8", action="store_true")
    ap.add_argument("--fused", action="store_true", help="also run binade_fused")
    ap.add_argument("--reordered", action="store_true", help="also run binade_reordered")
    ap.add_argument("--reference", choices=("resident", "streamed"), default="resident")
    ap.add_argument("--windows", type=int, default=32)
    ap.add_argument("--window-tokens", type=int, default=512)
    ap.add_argument("--prompts", type=int, default=32)
    ap.add_argument("--prompt-tokens", type=int, default=48)
    ap.add_argument("--gen-tokens", type=int, default=128)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    from mlx_lm.utils import load_tokenizer

    from binade import st

    src = st.resolve(args.original)
    source = st.provenance(src)
    out = args.out or REPO / "results" / source.get("repo_id", src.name).split("/")[-1]
    out.mkdir(parents=True, exist_ok=True)
    text = "".join(pq.read_table(hf_hub_download(DATASET[0], DATASET[1], repo_type="dataset")).column("text").to_pylist())
    tok = load_tokenizer(src)
    ids = np.array(tok.encode(text, add_special_tokens=False), np.int32)
    bos = [tok.bos_token_id] if tok.bos_token_id is not None else []
    L, W, P = args.window_tokens, args.windows, args.prompts
    body = L - len(bos)
    if (W + P) * body > ids.size:
        raise SystemExit(f"corpus has {ids.size} tokens, need {(W + P) * body}")
    windows = np.array([bos + list(ids[i * body : (i + 1) * body]) for i in range(W)], np.int32)
    start = W * body  # prompts come from text the windows did not use
    prompts = np.array([bos + list(ids[start + i * body : start + i * body + args.prompt_tokens - len(bos)]) for i in range(P)], np.int32)
    variants = ["bf16", "binade"] + (["binade_fused"] if args.fused else []) + (["binade_reordered"] if args.reordered else []) + (["q8"] if args.q8 else [])
    order = variants[1:] + ["bf16"] if args.reference == "streamed" else variants  # a streamed reference after the others (memory)
    res, runs = {}, {}
    with tempfile.TemporaryDirectory() as tmp:
        data = Path(tmp) / "data.npz"
        np.savez(data, windows=windows, prompts=prompts, gen=args.gen_tokens)
        for v in order:
            path = args.packed if v.startswith("binade") else src
            r = subprocess.run([sys.executable, "-c", CHILD, v, str(path), str(data), str(Path(tmp) / f"{v}.npz"), args.reference], capture_output=True, text=True, cwd=REPO)
            if r.returncode:
                raise SystemExit(f"{v} failed:\n{r.stderr[-3000:]}")
            print(r.stdout.strip().splitlines()[-1], flush=True)
            z = np.load(Path(tmp) / f"{v}.npz")
            runs[v] = {k: z[k] for k in z.files}
    ref = runs["bf16"]
    for v in variants:
        z = runs[v]
        same_pos = z["wh"] == ref["wh"]
        same_gen = np.all(z["gi"] == ref["gi"], axis=1) & np.all(z["gh"] == ref["gh"], axis=1)
        diverge = [int(np.argmax(a != b)) for a, b in zip(z["gi"], ref["gi"]) if np.any(a != b)]
        res[v] = {
            "positions": int(same_pos.size),
            "positions_bit_identical": int(same_pos.sum()),
            "argmax_agreement": float(np.mean(z["wa"] == ref["wa"])),
            "perplexity": float(np.exp(-np.mean(z["wl"].astype(np.float64)))),
            "max_abs_target_logprob_diff": float(np.max(np.abs(z["wl"] - ref["wl"]))),
            "generations": int(same_gen.size),
            "generations_identical": int(same_gen.sum()),
            "tokens_identical": float(np.mean(z["gi"] == ref["gi"])) if P else None,
            "mean_first_divergence": float(np.mean(diverge)) if diverge else None,
            "windows_s": float(z["t_windows"]),
            "gen_s": float(z["t_gen"]),
            "peak_gb": float(z["peak_gb"]),
        }
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=no", "--", ".", ":!results"], cwd=REPO, capture_output=True, text=True).stdout.strip())
    import mlx.core as mx

    summary = {
        "source": source,
        "packed": st.display_path(args.packed),
        "git": {"commit": commit, "dirty": dirty},
        "mlx": mx.__version__,
        "device": mx.device_info().get("device_name"),
        "dataset": {"repo": DATASET[0], "file": DATASET[1]},
        "args": {k: str(v) for k, v in st.display_args(vars(args)).items()},
        "variants": res,
    }
    (out / "exactness_scale.json").write_text(json.dumps(summary, indent=1) + "\n")
    names = {"bf16": "BF16 (reference, streamed from disk)" if args.reference == "streamed" else "BF16 (reference)", "binade": "Binade (R4 runtime)", "binade_fused": "Binade, fused R4 GEMM for all prompt lengths", "binade_reordered": "Binade fused, two accumulators (valid order, not MLX's)", "q8": "MLX 8-bit affine (lossy)"}
    lines = [
        f"# Exactness at scale: {source.get('repo_id', src.name)}",
        "",
        f"`scripts/exactness_scale.py` @ `{commit[:7]}{' (dirty)' if dirty else ''}`, MLX {mx.__version__}, {summary['device']}. "
        f"{W} windows of {L} tokens from the WikiText-2 test set (raw), logits at every position in one forward pass; "
        f"{P} prompts of {args.prompt_tokens} tokens from later text, {args.gen_tokens} greedy tokens each. Each variant in its own process.",
        "",
        "| variant | positions with bit-identical logits | argmax agreement | perplexity | generations identical (tokens and log-probs) | tokens identical | peak GB |",
        "|---|---|---|---|---|---|---|",
    ]
    for v in variants:
        r = res[v]
        lines.append(
            f"| {names[v]} | {r['positions_bit_identical']} / {r['positions']} | {r['argmax_agreement']:.4%} | {r['perplexity']:.4f} | "
            f"{r['generations_identical']} / {r['generations']} | {'n/a' if r['tokens_identical'] is None else format(r['tokens_identical'], '.2%')} | {r['peak_gb']:.1f} |"
        )
    (out / "exactness_scale.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
