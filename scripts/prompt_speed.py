#!/usr/bin/env python3
"""Prompt (prefill) speed, one model per fresh process: time to the first token's logits.

    python scripts/prompt_speed.py <original model-dir | cached-hf-repo-id> <packed-dir> [--lengths 40,512,2048] [--reps 3] [--models bf16,binade_r4,q8] [--prefill-step 2048]

prefill_bench.py times the matrix products alone; this times what a user waits for before the
first token: a prompt of L tokens from the WikiText-2 test set run as mlx_lm's generate_step
runs it (chunks of up to --prefill-step tokens filling the KV cache, then the last token's
logits; mlx_lm's default chunk is 2048), for each length in --lengths. A smaller chunk bounds
activation memory, which a GPU with little room beside the weights needs. Each model runs
in its own process: it loads, runs every length once to warm up, then --reps timed passes
per length (median reported). --models binade_r4 times Binade alone, for models whose BF16
weights do not fit in memory (the fused R4 GEMM then serves every prompt length). Writes
prompt_speed.json / .md to results/<model>/ (or --out).
"""

import argparse
import json
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
DATASET = ("Salesforce/wikitext", "wikitext-2-raw-v1/test-00000-of-00001.parquet")

CHILD = r'''
import json, sys, time
import numpy as np
import mlx.core as mx
which, path, data, reps, step = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
if which == "bf16":
    from binade.mlx.loader import load_reference as load
elif which == "q8":
    from binade.mlx.loader import load_quantized_reference as load
else:
    from binade.mlx.r4 import load
from mlx_lm.models.cache import make_prompt_cache
model, tok, _ = load(path)
mx.eval(model.parameters())
d = np.load(data)
out = {}
def run(ids):  # as mlx_lm's generate_step: the prompt in chunks of `step` filling the cache, then the last token's logits
    cache = make_prompt_cache(model)
    x = mx.array(ids)
    t = time.perf_counter()
    for i in range(0, len(ids) - 1, step):
        model(x[i : min(i + step, len(ids) - 1)][None], cache=cache)
        mx.eval([c.state for c in cache])
    mx.eval(model(x[-1:][None], cache=cache))
    return time.perf_counter() - t
for key in sorted(d.files, key=lambda k: int(k[1:])):
    run(d[key])
for key in sorted(d.files, key=lambda k: int(k[1:])):
    out[key[1:]] = [run(d[key]) for _ in range(reps)]
print(json.dumps({"s": out, "peak_gb": mx.get_peak_memory() / 1e9}))
'''


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("original")
    ap.add_argument("packed", type=Path)
    ap.add_argument("--lengths", default="40,512,2048")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--models", default="bf16,binade_r4", help="comma list of bf16, binade_r4, q8")
    ap.add_argument("--prefill-step", type=int, default=2048, help="prompt chunk size (mlx_lm's prefill_step_size)")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    models = args.models.split(",")
    if not models or set(models) - {"bf16", "binade_r4", "q8"}:
        raise SystemExit(f"--models: {args.models}")
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
    ids = tok.encode(text, add_special_tokens=False)
    bos = [tok.bos_token_id] if tok.bos_token_id is not None else []
    lengths = [int(x) for x in args.lengths.split(",")]
    res = {}
    with tempfile.TemporaryDirectory() as tmp:
        data = Path(tmp) / "prompts.npz"
        np.savez(data, **{f"L{L}": np.array(bos + ids[: L - len(bos)], np.int32) for L in lengths})
        for m in models:
            path = args.packed if m == "binade_r4" else src
            which = "r4" if m == "binade_r4" else m
            r = subprocess.run([sys.executable, "-c", CHILD, which, str(path), str(data), str(args.reps), str(args.prefill_step)], capture_output=True, text=True, cwd=REPO)
            line = [l for l in r.stdout.splitlines() if l.startswith("{")]
            if r.returncode or not line:
                raise SystemExit(f"{m} failed:\n{r.stderr[-3000:]}")
            res[m] = json.loads(line[-1])
            print(m, {L: f"{statistics.median(v):.3f} s" for L, v in res[m]["s"].items()}, flush=True)
    med = {m: {L: statistics.median(v) for L, v in r["s"].items()} for m, r in res.items()}
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=no", "--", ".", ":!results"], cwd=REPO, capture_output=True, text=True).stdout.strip())
    import mlx.core as mx

    summary = {"source": source, "packed": st.display_path(args.packed), "git": {"commit": commit, "dirty": dirty}, "mlx": mx.__version__, "device": mx.device_info().get("device_name"), "args": {k: str(v) for k, v in st.display_args(vars(args)).items()}, "runs": res, "median_s": med}
    (out / "prompt_speed.json").write_text(json.dumps(summary, indent=1) + "\n")
    names = {"bf16": "BF16", "binade_r4": "Binade (R4 runtime)", "q8": "MLX 8-bit affine (lossy)"}
    lines = [
        f"# Prompt speed: {source.get('repo_id', src.name)}",
        "",
        f"`scripts/prompt_speed.py` @ `{commit[:7]}{' (dirty)' if dirty else ''}`, MLX {mx.__version__}, {summary['device']}. "
        f"Time to the first token's logits over a prompt of WikiText-2 test text (KV cache filled in chunks of {args.prefill_step} tokens, as mlx_lm's generate_step); median of {args.reps} after a warm-up pass, each model in its own process. Seconds (prompt tokens per second).",
        "",
        "| model | " + " | ".join(f"{L} tokens" for L in lengths) + " | peak GB |",
        "|---|" + "---|" * len(lengths) + "---|",
    ]
    for m in models:
        lines.append(f"| {names[m]} | " + " | ".join(f"{med[m][str(L)]:.3f} ({L / med[m][str(L)]:.0f})" for L in lengths) + f" | {res[m]['peak_gb']:.1f} |")
    (out / "prompt_speed.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
