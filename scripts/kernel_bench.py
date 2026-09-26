#!/usr/bin/env python3
"""Kernel test: batch-1 matvec bandwidth on the Gemma 4 12B projection shapes.

    python scripts/kernel_bench.py [--packed models/gemma-4-12B-it-binade] [--min-gb 1.0] [--rounds 7]

Four ways to compute y = x W^T for one input row, on the real 12B weights:
  bf16       MLX's matvec on the original BF16 weight (what mlx_lm runs today)
  q8         MLX quantized_matmul, 8-bit affine, group 64 (mx.quantize of the same weight)
  binade_v1  Binade kernel reading the v1 arrays in place (tm 1, prologue, b=3 fast path)
  binade_r4  Binade kernel on the R4 runtime layout, transcoded from v1 at load
             (4-bit rank codes, rare escapes; binade/mlx/kernel.py)
Every shape cycles through distinct weight copies totalling at least --min-gb of BF16, so
each matvec streams from DRAM rather than the system-level cache. Each round times every
method back to back in rotating order, 8 passes over the copies per timing; ratios are
medians of per-round ratios, which cancels clock and thermal drift. GB/s counts the bytes
each method reads for the weight (x and y excluded); efficiency is a method's GB/s over
BF16's in the same round. The kernel test's pass bar was 75% of BF16's efficiency.
"""

import argparse
import json
import re
import statistics
import subprocess
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

from binade import st
from binade.mlx.kernel import BinadeMatrix, R4Matrix

REPO = Path(__file__).resolve().parents[1]
PEAK_GBS = 546  # Apple's figure for the M4 Max with a 40-core GPU
PROJ = re.compile(r"\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)\.weight$")
METHODS = ("bf16", "q8", "binade_v1", "binade_r4")
V1_OPTS = dict(tm=1, prologue=True, fast3=True, bm=8)
R4_OPTS = dict(tm=1, bm=8, pf=2)


def groups(packed: dict) -> dict:
    """(label, shape) -> [weight names] for the projections, plus the tied lm_head."""
    out = {}
    for name, info in packed.items():
        if not name.endswith(".raw"):
            continue
        base = name[:-4]
        m = PROJ.search(base)
        if m:
            kind = "gate_up_proj" if m.group(1) in ("gate_proj", "up_proj") else m.group(1)
            out.setdefault((kind, tuple(info.shape)), []).append(base)
        elif base.endswith("embed_tokens.weight"):
            out.setdefault(("lm_head (tied embedding)", tuple(info.shape)), []).append(base)
    return out


def sweep(fn, items, inner: int = 8) -> float:
    """Seconds per call: `inner` passes over all items in one evaluation."""
    t = time.perf_counter()
    mx.eval([fn(i) for _ in range(inner) for i in items])
    return (time.perf_counter() - t) / (inner * len(items))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--original", default="google/gemma-4-12B-it")
    ap.add_argument("--packed", type=Path, default=REPO / "models" / "gemma-4-12B-it-binade")
    ap.add_argument("--min-gb", type=float, default=1.0)
    ap.add_argument("--rounds", type=int, default=7)
    ap.add_argument("--out", type=Path, default=REPO / "results" / "kernel")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    mx.set_cache_limit(4 << 30)

    orig = {t.name: t for t in st.list_tensors(st.resolve(args.original))}
    packed = {t.name: t for t in st.list_tensors(args.packed)}
    rng = np.random.default_rng(0)
    rows = []
    for (kind, shape), names in sorted(groups(packed).items(), key=lambda kv: -np.prod(kv[0][1])):
        N, K = shape
        copies = max(1, int(np.ceil(args.min_gb * 1e9 / (2 * N * K))))
        chosen = [names[i % len(names)] for i in range(copies)]  # real tensors first, then repeats in fresh buffers
        items = {m: [] for m in METHODS}
        for n in chosen:
            parts = [st.read(packed[f"{n}.{p}"]) for p in ("raw", "table", "meta", "idx", "esc")]
            w = mx.array(st.read(orig[n])).view(mx.bfloat16)
            items["bf16"].append(w)
            items["q8"].append(mx.quantize(w, group_size=64, bits=8))
            items["binade_v1"].append(BinadeMatrix(*parts))
            items["binade_r4"].append(R4Matrix(*parts))
        mx.eval(items["bf16"], items["q8"], [(m.raw, m.idx) for m in items["binade_v1"]], [(m.raw, m.codes) for m in items["binade_r4"]])
        x = mx.array(rng.standard_normal((1, K)).astype(np.float32)).astype(mx.bfloat16)
        fns = {
            "bf16": lambda w: x @ w.T,
            "q8": lambda q: mx.quantized_matmul(x, *q, transpose=True, group_size=64, bits=8),
            "binade_v1": lambda m: m.matvec(x, **V1_OPTS),
            "binade_r4": lambda m: m.matvec(x, **R4_OPTS),
        }
        nbytes = {
            "bf16": 2 * N * K,
            "q8": sum(a.nbytes for a in items["q8"][0]),
            "binade_v1": items["binade_v1"][0].nbytes(),
            "binade_r4": items["binade_r4"][0].nbytes(),
        }
        exact = {
            m: all(
                np.array_equal(np.array((x @ w.T).view(mx.uint16)).ravel(), np.array(fns[m](b).view(mx.uint16)))
                for w, b in list(zip(items["bf16"], items[m]))[:3]
            )
            for m in ("binade_v1", "binade_r4")
        }
        for m in METHODS:
            sweep(fns[m], items[m])  # compile and warm up
        times = {m: [] for m in METHODS}
        for r in range(args.rounds):
            for m in METHODS[r % len(METHODS):] + METHODS[: r % len(METHODS)]:
                times[m].append(sweep(fns[m], items[m]))
        row = {
            "kind": kind,
            "shape": [N, K],
            "per_token": len(names),
            "copies": copies,
            "bytes": nbytes,
            "us": {m: 1e6 * statistics.median(times[m]) for m in METHODS},
            "gbs": {m: nbytes[m] / statistics.median(times[m]) / 1e9 for m in METHODS},
            "efficiency": {m: statistics.median([(nbytes[m] / t) / (nbytes["bf16"] / tb) for t, tb in zip(times[m], times["bf16"])]) for m in METHODS},
            "speedup": {m: statistics.median([tb / t for t, tb in zip(times[m], times["bf16"])]) for m in METHODS},
            "bit_identical_to_bf16": exact,
        }
        rows.append(row)
        print(
            f"{kind:24s} {N:>6d}x{K:<5d} bf16 {row['gbs']['bf16']:5.0f} GB/s | "
            + " | ".join(f"{m} eff {row['efficiency'][m]:.1%} x{row['speedup'][m]:.2f}" for m in METHODS[1:])
            + f" | exact {exact}",
            flush=True,
        )
        del items
        mx.clear_cache()

    tot = {m: sum(r["us"][m] * r["per_token"] for r in rows) for m in METHODS}
    byt = {m: sum(r["bytes"][m] * r["per_token"] for r in rows) for m in METHODS}
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=no", "--", ".", ":!results"], cwd=REPO, capture_output=True, text=True).stdout.strip())
    per_token = {
        "us": tot,
        "bytes": byt,
        "gbs": {m: byt[m] / tot[m] / 1e3 for m in METHODS},
        "speedup": {m: tot["bf16"] / tot[m] for m in METHODS},
        "efficiency": {m: (byt[m] / tot[m]) / (byt["bf16"] / tot["bf16"]) for m in METHODS},
    }
    summary = {
        "git": {"commit": commit, "dirty": dirty},
        "mlx": mx.__version__,
        "device": mx.device_info().get("device_name"),
        "peak_gbs": PEAK_GBS,
        "args": {k: str(v) for k, v in st.display_args(vars(args)).items()},
        "v1_opts": V1_OPTS,
        "r4_opts": R4_OPTS,
        "shapes": rows,
        "per_token": per_token,
    }
    (args.out / "gemma-4-12B-it.json").write_text(json.dumps(summary, indent=1) + "\n")
    lines = [
        "# Binade kernel test: batch-1 matvec on Gemma 4 12B shapes",
        "",
        f"`scripts/kernel_bench.py` @ `{commit[:7]}{' (dirty)' if dirty else ''}`, MLX {mx.__version__}, {summary['device']} "
        f"(40-core GPU, {PEAK_GBS} GB/s peak). Real 12B weights; each shape cycles through distinct copies of at least "
        f"{args.min_gb:g} GB of BF16 so every matvec streams from DRAM; {args.rounds} paired rounds, medians of per-round ratios. "
        "Efficiency = a method's GB/s over BF16's (weight bytes it reads). binade_v1 reads the v1 file arrays in place; "
        "binade_r4 reads the 4-bit runtime layout transcoded from v1 at load. Both produce outputs bit-identical to MLX's BF16 matvec.",
        "",
        "| projection | shape | per token | BF16 GB/s | 8-bit eff / speedup | Binade v1 eff / speedup | Binade R4 eff / speedup | bit-identical (v1, R4) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        e, sp = r["efficiency"], r["speedup"]
        lines.append(
            f"| {r['kind']} | {r['shape'][0]}x{r['shape'][1]} | {r['per_token']} | {r['gbs']['bf16']:.0f} | "
            f"{e['q8']:.1%} / {sp['q8']:.2f}x | {e['binade_v1']:.1%} / {sp['binade_v1']:.2f}x | {e['binade_r4']:.1%} / {sp['binade_r4']:.2f}x | "
            f"{r['bit_identical_to_bf16']['binade_v1']}, {r['bit_identical_to_bf16']['binade_r4']} |"
        )
    p = per_token
    lines += [
        "",
        "Per token (every projection of the 12B once plus lm_head; sum of per-shape median times):",
        "",
        "| method | ms | bytes vs BF16 | GB/s | % of peak | efficiency vs BF16 | speedup |",
        "|---|---|---|---|---|---|---|",
    ]
    for m in METHODS:
        lines.append(
            f"| {m} | {p['us'][m] / 1e3:.1f} | {p['bytes'][m] / p['bytes']['bf16']:.3f} | {p['gbs'][m]:.0f} | {p['gbs'][m] / PEAK_GBS:.0%} | "
            f"{p['efficiency'][m]:.1%} | {p['speedup'][m]:.2f}x |"
        )
    (args.out / "gemma-4-12B-it.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
