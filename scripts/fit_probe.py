#!/usr/bin/env python3
"""Does a model load and decode on this GPU? BF16, Binade and 8-bit, each in a fresh process.

    python scripts/fit_probe.py <original model-dir | cached-hf-repo-id> <packed-dir> [--variants bf16,binade_r4,q8] [--tokens 40] [--timeout 1500]

For each variant: load the model, evaluate its weights, greedy-decode --tokens tokens from a
fixed chat prompt; record load time, resident memory after loading, peak memory and decode
speed, or the error that stopped it (for instance out of GPU memory). Shows which variants a
smaller GPU can hold at all. Writes fit_probe.json / .md to results/<model>/ (or --out).
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PROMPT = "Explain in three short paragraphs why lossless compression of neural network weights can speed up inference."

CHILD = r'''
import json, sys, time
import mlx.core as mx
from mlx_lm.generate import generate_step
which, path, n, prompt = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
res = {"variant": which}
t0 = time.perf_counter()
try:
    if which == "bf16":
        from binade.mlx.loader import load_reference as load
    elif which == "q8":
        from binade.mlx.loader import load_quantized_reference as load
    else:
        from binade.mlx.r4 import load
    model, tok, _ = load(path)
    mx.eval(model.parameters())
    res["load_s"] = time.perf_counter() - t0
    res["resident_gb"] = mx.get_active_memory() / 1e9
    p = mx.array(tok.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True))
    stamps = []
    for t, _ in generate_step(p, model, max_tokens=n):
        stamps.append(time.perf_counter())
    dt = [b - a for a, b in zip(stamps[8:-1], stamps[9:])]
    res.update(tok_s=len(dt) / sum(dt), peak_gb=mx.get_peak_memory() / 1e9, ok=True)
except Exception as e:
    res.update(ok=False, error=f"{type(e).__name__}: {str(e)[:300]}")
print(json.dumps(res), flush=True)
'''


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("original")
    ap.add_argument("packed", type=Path)
    ap.add_argument("--variants", default="bf16,binade_r4,q8")
    ap.add_argument("--tokens", type=int, default=40)
    ap.add_argument("--timeout", type=float, default=1500, help="seconds per variant")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    import mlx.core as mx

    from binade import st

    src = st.resolve(args.original)
    source = st.provenance(src)
    out = args.out or REPO / "results" / source.get("repo_id", src.name).split("/")[-1]
    out.mkdir(parents=True, exist_ok=True)
    info = mx.device_info()
    total = info.get("total_memory") or info.get("memory_size")
    res = []
    for v in args.variants.split(","):
        path = args.packed if v == "binade_r4" else src
        try:
            r = subprocess.run([sys.executable, "-c", CHILD, v, str(path), str(args.tokens), PROMPT], capture_output=True, text=True, cwd=REPO, timeout=args.timeout)
            line = [l for l in r.stdout.splitlines() if l.startswith("{")]
            row = json.loads(line[-1]) if line else {"variant": v, "ok": False, "error": (r.stderr.strip().splitlines() or ["no output"])[-1][:300]}
        except subprocess.TimeoutExpired:
            row = {"variant": v, "ok": False, "error": f"timed out after {args.timeout:g} s"}
        res.append(row)
        print(json.dumps(row), flush=True)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=no", "--", ".", ":!results"], cwd=REPO, capture_output=True, text=True).stdout.strip())
    summary = {"source": source, "packed": st.display_path(args.packed), "git": {"commit": commit, "dirty": dirty}, "mlx": mx.__version__, "device": info.get("device_name"), "device_memory_gb": total / 1e9 if total else None, "args": {k: str(v) for k, v in st.display_args(vars(args)).items()}, "variants": res}
    (out / "fit_probe.json").write_text(json.dumps(summary, indent=1) + "\n")
    names = {"bf16": "BF16", "binade_r4": "Binade (R4 runtime)", "q8": "MLX 8-bit affine (lossy)"}
    lines = [
        f"# Does it fit? {source.get('repo_id', src.name)} on {summary['device']}",
        "",
        f"`scripts/fit_probe.py` @ `{commit[:7]}{' (dirty)' if dirty else ''}`, MLX {mx.__version__}, {summary['device']}"
        + (f" ({total / 1e9:.2f} GB)" if total else "")
        + f". Each variant in its own process: load, then {args.tokens} greedy tokens.",
        "",
        "| variant | loads and runs | resident after load (GB) | peak (GB) | decode tok/s | error |",
        "|---|---|---|---|---|---|",
    ]
    for r in res:
        ok = r.get("ok")
        lines.append(
            f"| {names.get(r['variant'], r['variant'])} | {'yes' if ok else 'no'} | {r['resident_gb']:.2f} | {r['peak_gb']:.2f} | {r['tok_s']:.2f} | |"
            if ok
            else f"| {names.get(r['variant'], r['variant'])} | no | | | | {r.get('error', '')} |"
        )
    (out / "fit_probe.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
