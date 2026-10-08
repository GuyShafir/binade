#!/usr/bin/env python3
"""Prefill (multi-row) matmul speed on the Gemma 4 12B projection shapes: BF16 against R4.

    python scripts/prefill_bench.py [--packed models/gemma-4-12B-it-binade-v2] [--rows 16,128,512,2048] [--reps 5]

For one weight per distinct projection shape, and M prompt rows, times y = x W^T four ways:
  bf16         MLX's matmul on the original BF16 weight
  r4_fused     the R4 GEMM (binade/mlx/kernel.py source_r4_gemm, tile from r4.gemm_tile), decoding weights as it loads
  r4_rebuild   BF16 weight rebuilt from R4 by the decode kernel, then MLX's matmul
  r4_runtime   what R4Linear does for M rows (fused for short inputs or when memory is tight, else rebuild)
Each is the median of --reps evaluations after a warm-up. r4_fused is checked bit for bit
against bf16 wherever the runtime would use it. "Per prompt" sums every projection of the
model once (count x time per shape) and reports prompt tokens per second of matmul work.
Writes prefill_bench.json / .md to results/gemma-4-12B-it/.
"""

import argparse
import json
import statistics
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import numpy as np

from binade import st
from binade.mlx.r4 import FUSED_MAX_ROWS, R4Linear, rebuild_fits, transcode, use_mma

REPO = Path(__file__).resolve().parents[1]
METHODS = ("bf16", "r4_fused", "r4_rebuild", "r4_runtime")
KINDS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def shapes(packed: dict) -> dict:
    """(label, (N, K)) -> [weight names], one entry per distinct projection shape plus lm_head."""
    out = {}
    for name, info in packed.items():
        if not name.endswith(".raw"):
            continue
        base = name[:-4]
        kind = next((k for k in KINDS if base.endswith(f".{k}.weight")), None)
        if kind:
            kind = "gate_up_proj" if kind in ("gate_proj", "up_proj") else kind
        elif base.endswith("embed_tokens.weight"):
            kind = "lm_head (tied embedding)"
        else:
            continue
        out.setdefault((kind, tuple(info.shape)), []).append(base)
    return out


def median_time(fn, reps: int) -> float:
    mx.eval(fn())
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        mx.eval(fn())
        ts.append(time.perf_counter() - t)
    return statistics.median(ts)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--packed", type=Path, default=REPO / "models/gemma-4-12B-it-binade-v2")
    ap.add_argument("--original", default="google/gemma-4-12B-it")
    ap.add_argument("--rows", default="16,128,512,2048")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--out", type=Path, default=REPO / "results/gemma-4-12B-it")
    args = ap.parse_args()
    rows = [int(r) for r in args.rows.split(",")]
    src = st.resolve(args.original)
    orig = {i.name: i for i in st.list_tensors(src)}
    packed = {i.name: i for i in st.list_tensors(args.packed)}
    arrays = {}
    for f in st.shard_files(args.packed):
        arrays.update(mx.load(str(f)))
    table = []
    for (kind, shape), names in shapes(packed).items():
        N, K = shape
        name = names[0]
        w = mx.array(st.read(orig[name])).view(mx.bfloat16)
        parts = ("raw", "table", "meta", "bits", "roff") if f"{name}.bits" in packed else ("raw", "table", "meta", "idx", "esc")
        m = R4Linear(transcode(SimpleNamespace(weight={p: arrays[f"{name}.{p}"] for p in parts}, _plan=None)))
        mx.eval(w, m._a["codes"])
        for M in rows:
            x = mx.random.normal((M, K), key=mx.random.key(M)).astype(mx.bfloat16)
            fns = {
                "bf16": lambda: x @ w.T,
                "r4_fused": lambda: m._gather_mm(x, mx.zeros((M,), dtype=mx.uint32), N, N),
                "r4_rebuild": lambda: x @ m.dense().T,
                "r4_runtime": lambda: m._rows(x),
            }
            t = {k: median_time(f, args.reps) for k, f in fns.items()}
            exact = None
            if use_mma(M, N, K):
                a, b = fns["bf16"](), fns["r4_fused"]()
                mx.eval(a, b)
                exact = bool(np.array_equal(np.array(a.view(mx.uint16)), np.array(b.view(mx.uint16))))
            path = "fused" if use_mma(M, N, K) and (M <= FUSED_MAX_ROWS or not rebuild_fits(2 * N * K)) else "rebuild"
            row = {"kind": kind, "N": N, "K": K, "count": len(names), "M": M, "runtime_path": path, "fused_exact": exact, **{f"{k}_ms": 1e3 * v for k, v in t.items()}}
            table.append(row)
            print(f"{kind:26s} {N}x{K} M={M:5d}: " + "  ".join(f"{k} {1e3 * v:8.2f} ms" for k, v in t.items()) + f"  runtime={row['runtime_path']} exact={exact}", flush=True)
        del m, w
        mx.clear_cache()

    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=no", "--", ".", ":!results"], cwd=REPO, capture_output=True, text=True).stdout.strip())
    per_prompt = {M: {k: sum(r[f"{k}_ms"] * r["count"] for r in table if r["M"] == M) for k in METHODS} for M in rows}
    summary = {"git": {"commit": commit, "dirty": dirty}, "packed": st.display_path(args.packed), "rows": rows, "reps": args.reps, "shapes": table, "per_prompt_ms": per_prompt}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "prefill_bench.json").write_text(json.dumps(summary, indent=1) + "\n")
    lines = [
        "# Prefill matmuls: BF16 against R4, Gemma 4 12B shapes",
        "",
        f"`scripts/prefill_bench.py` @ `{commit[:7]}{' (dirty)' if dirty else ''}`, MLX {mx.__version__}, real 12B weights, median of {args.reps} after a warm-up. "
        "Per prompt: every projection of the model once (tied lm_head included), summed; prompt tokens per second of matmul work in brackets.",
        "",
        "| prompt rows | BF16 | R4 fused | R4 rebuild | R4 runtime | runtime / BF16 |",
        "|---|---|---|---|---|---|",
    ]
    for M in rows:
        p = per_prompt[M]
        lines.append(
            f"| {M} | " + " | ".join(f"{p[k]:.0f} ms ({1e3 * M / p[k]:.0f} tok/s)" for k in METHODS) + f" | {p['bf16'] / p['r4_runtime']:.2f}x |"
        )
    lines += ["", "Per shape (ms; runtime path; exact: fused output bit-identical to BF16 wherever MLX runs its plain GEMM):", "", "| projection | shape | count | rows | BF16 | fused | rebuild | runtime | path | exact |", "|---|---|---|---|---|---|---|---|---|---|"]
    for r in table:
        lines.append(
            f"| {r['kind']} | {r['N']}x{r['K']} | {r['count']} | {r['M']} | {r['bf16_ms']:.2f} | {r['r4_fused_ms']:.2f} | {r['r4_rebuild_ms']:.2f} | {r['r4_runtime_ms']:.2f} | {r['runtime_path']} | {'' if r['fused_exact'] is None else r['fused_exact']} |"
        )
    (args.out / "prefill_bench.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:6 + len(rows)]))


if __name__ == "__main__":
    main()
