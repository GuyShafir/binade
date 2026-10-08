#!/usr/bin/env python3
"""Decode speed, one model per fresh process: stock BF16 vs Binade R4 on the same prompt.

    python scripts/decode_speed.py <original model-dir | cached-hf-repo-id> <packed-dir> [--runs 2] [--tokens 256] [--cooldown 60] [--models bf16,binade_r4]

Runs `--runs` rounds of (BF16, Binade R4), each model in its own Python process so neither
inherits the other's memory state, with a cool-down between processes. Each process loads
its model, runs the prompt, warms up 16 tokens, then times `--tokens` greedy decode steps
through mlx_lm's generate_step. Writes decode_speed.json / .md to results/<model>/.
--models binade_r4 times Binade alone, for models whose BF16 weights do not fit in memory;
q8 adds MLX's 8-bit affine quantization (group 64) of the same model, the usual lossy option.
"""

import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PROMPT = "Explain in three short paragraphs why lossless compression of neural network weights can speed up inference on memory-bandwidth-bound hardware."

CHILD = r'''
import json, sys, time
import mlx.core as mx
from mlx_lm.generate import generate_step
which, path, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
if which == "bf16":
    from binade.mlx.loader import load_reference as load
    model, tok, _ = load(path)
elif which == "q8":
    from binade.mlx.loader import load_quantized_reference as load
    model, tok, _ = load(path)
else:
    from binade.mlx.r4 import load
    model, tok, _ = load(path)
mx.eval(model.parameters())
prompt = mx.array(tok.apply_chat_template([{"role": "user", "content": sys.argv[4]}], add_generation_prompt=True))
stamps, ids = [], []
for t, _ in generate_step(prompt, model, max_tokens=n + 16):
    ids.append(int(t)); stamps.append(time.perf_counter())
dt = [b - a for a, b in zip(stamps[16:-1], stamps[17:])]
print(json.dumps({"tok_s": len(dt) / sum(dt), "median_ms": 1e3 * sorted(dt)[len(dt) // 2], "ids": ids, "peak_gb": mx.get_peak_memory() / 1e9}))
'''


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("original")
    ap.add_argument("packed", type=Path)
    ap.add_argument("--runs", type=int, default=2)
    ap.add_argument("--tokens", type=int, default=256)
    ap.add_argument("--cooldown", type=float, default=60)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--models", default="bf16,binade_r4", help="comma list of bf16, binade_r4, q8")
    args = ap.parse_args()
    models = args.models.split(",")
    if not models or set(models) - {"bf16", "binade_r4", "q8"}:
        raise SystemExit(f"--models: {args.models}")
    from binade import st

    src = st.resolve(args.original)
    source = st.provenance(src)
    out = args.out or REPO / "results" / source.get("repo_id", src.name).split("/")[-1]
    runs = {m: [] for m in models}
    first = True
    for r in range(args.runs):
        for label, which, path in [x for x in (("bf16", "bf16", src), ("binade_r4", "r4", args.packed), ("q8", "q8", src)) if x[0] in runs]:
            if not first:
                time.sleep(args.cooldown)
            first = False
            res = subprocess.run([sys.executable, "-c", CHILD, which, str(path), str(args.tokens), PROMPT], capture_output=True, text=True, cwd=REPO)
            line = [l for l in res.stdout.splitlines() if l.startswith("{")]
            if res.returncode or not line:
                raise SystemExit(f"{label} run failed:\n{res.stderr[-2000:]}")
            runs[label].append(json.loads(line[-1]))
            print(f"round {r + 1} {label}: {runs[label][-1]['tok_s']:.2f} tok/s, median step {runs[label][-1]['median_ms']:.1f} ms", flush=True)
    exact = [m for m in runs if m != "q8"]  # q8 is lossy: its tokens are compared, not required to match
    first_ids = runs[exact[0]][0]["ids"]
    same = all(x["ids"] == first_ids for m in exact for x in runs[m])
    q8_same = all(x["ids"] == first_ids for x in runs["q8"]) if "q8" in runs else None
    med = {k: statistics.median(x["tok_s"] for x in v) for k, v in runs.items()}
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=no", "--", ".", ":!results"], cwd=REPO, capture_output=True, text=True).stdout.strip())
    summary = {"source": source, "git": {"commit": commit, "dirty": dirty}, "args": {k: str(v) for k, v in st.display_args(vars(args)).items()},
               "median_tok_s": med, "speedup": med["binade_r4"] / med["bf16"] if {"bf16", "binade_r4"} <= set(med) else None, "identical_ids_all_runs": same,
               "q8_ids_identical_to_reference": q8_same,
               "runs": {k: [{kk: vv for kk, vv in x.items() if kk != "ids"} for x in v] for k, v in runs.items()}}
    (out / "decode_speed.json").write_text(json.dumps(summary, indent=1) + "\n")
    lines = [
        f"# Decode speed, fresh process per model: {source.get('repo_id', src.name)}",
        "",
        f"`scripts/decode_speed.py` @ `{commit[:7]}{' (dirty)' if dirty else ''}`: {args.runs} rounds of {' then '.join({'bf16': 'BF16', 'binade_r4': 'Binade R4', 'q8': 'MLX 8-bit'}[m] for m in runs)}, each in its own process, "
        f"{args.cooldown:g} s cool-down between processes, {args.tokens} greedy tokens timed after a 16-token warm-up (mlx_lm generate_step).",
        "",
        "| model | tok/s per run | median tok/s | median step ms | peak GB |",
        "|---|---|---|---|---|",
    ]
    for k, v in runs.items():
        lines.append(f"| {k} | {', '.join(f'{x['tok_s']:.2f}' for x in v)} | {med[k]:.2f} | {statistics.median(x['median_ms'] for x in v):.1f} | {max(x['peak_gb'] for x in v):.1f} |")
    if summary["speedup"] is not None:
        lines += ["", f"Binade R4 / BF16: **{summary['speedup']:.2f}x**. Token ids identical across all runs of BF16 and Binade: {same}."]
    else:
        lines += ["", f"Token ids identical across all runs of the exact models: {same}."]
    if "q8" in runs:
        ref = "bf16" if "bf16" in med else "binade_r4"
        lines += [f"MLX 8-bit (lossy) / {'BF16' if ref == 'bf16' else 'Binade R4'}: {med['q8'] / med[ref]:.2f}x; its 256 greedy tokens identical to the exact models': {q8_same}."]
    (out / "decode_speed.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
