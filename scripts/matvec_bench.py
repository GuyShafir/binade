#!/usr/bin/env python3
"""Batch-1 matvec bandwidth on the Gemma 4 12B projection shapes, on either MLX backend.

    python scripts/matvec_bench.py --peak-gbs 1792 --tag cuda-rtx-pro-6000 [--packed models/gemma-4-12B-it-binade-v2]

The kernel test of scripts/kernel_bench.py, for the runtime as it ships (R4 through
R4Linear, which picks the Metal or CUDA kernel), so it runs on CUDA too:
  bf16       MLX's matvec on the original BF16 weight
  q8         MLX quantized_matmul, 8-bit affine, group 64 (lossy reference; skipped if unsupported)
  binade_r4  the R4 matvec on weights transcoded from the packed file
Each shape cycles through distinct weight copies of at least --min-gb of BF16, so every
matvec streams from DRAM rather than cache; paired rounds in rotating order, medians of
per-round ratios. Efficiency = a method's GB/s over BF16's (weight bytes read).
Writes results/kernel/gemma-4-12B-it_<tag>.json / .md.
"""

import argparse
import json
import statistics
import subprocess
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import numpy as np

from binade import st
from binade.mlx.r4 import R4Linear, transcode
from kernel_bench import groups, sweep

REPO = Path(__file__).resolve().parents[1]


def r4_bytes(a: dict) -> int:
    return sum(a[k].nbytes for k in ("raw", "codes", "flags", "esc", "table", "row_esc"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--original", default="google/gemma-4-12B-it")
    ap.add_argument("--packed", type=Path, default=REPO / "models" / "gemma-4-12B-it-binade-v2")
    ap.add_argument("--min-gb", type=float, default=1.0)
    ap.add_argument("--rounds", type=int, default=7)
    ap.add_argument("--peak-gbs", type=float, required=True, help="the GPU's peak memory bandwidth, for the %% of peak column")
    ap.add_argument("--tag", required=True, help="device label for the output file name")
    ap.add_argument("--out", type=Path, default=REPO / "results" / "kernel")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    mx.set_cache_limit(4 << 30)

    orig = {t.name: t for t in st.list_tensors(st.resolve(args.original))}
    packed = {t.name: t for t in st.list_tensors(args.packed)}
    arrays = {}
    for f in st.shard_files(args.packed):
        arrays.update(mx.load(str(f)))
    parts = ("raw", "table", "meta", "bits", "roff") if any(n.endswith(".bits") for n in packed) else ("raw", "table", "meta", "idx", "esc")
    x_rng = np.random.default_rng(0)
    methods = ["bf16", "q8", "binade_r4"]
    try:
        q = mx.quantize(mx.zeros((64, 128), dtype=mx.bfloat16), group_size=64, bits=8)
        mx.eval(mx.quantized_matmul(mx.zeros((1, 128), dtype=mx.bfloat16), *q, transpose=True, group_size=64, bits=8))
    except Exception:
        methods.remove("q8")
    rows = []
    for (kind, shape), names in sorted(groups(packed).items(), key=lambda kv: -np.prod(kv[0][1])):
        N, K = shape
        copies = max(1, int(np.ceil(args.min_gb * 1e9 / (2 * N * K))))
        chosen = [names[i % len(names)] for i in range(copies)]
        items = {m: [] for m in methods}
        for n in chosen:
            w = mx.array(st.read(orig[n])).view(mx.bfloat16)
            items["bf16"].append(w)
            if "q8" in methods:
                items["q8"].append(mx.quantize(w, group_size=64, bits=8))
            items["binade_r4"].append(R4Linear(transcode(SimpleNamespace(weight={p: mx.array(arrays[f"{n}.{p}"]) for p in parts}, _plan=None))))
        mx.eval(items["bf16"], items.get("q8", []), [m._a["codes"] for m in items["binade_r4"]])
        x = mx.array(x_rng.standard_normal((1, K)).astype(np.float32)).astype(mx.bfloat16)
        fns = {
            "bf16": lambda w: x @ w.T,
            "q8": lambda q: mx.quantized_matmul(x, *q, transpose=True, group_size=64, bits=8),
            "binade_r4": lambda m: m._matvec(x),
        }
        nbytes = {"bf16": 2 * N * K, "binade_r4": r4_bytes(items["binade_r4"][0]._a)}
        if "q8" in methods:
            nbytes["q8"] = sum(a.nbytes for a in items["q8"][0])
        exact = all(
            np.array_equal(np.array((x @ w.T).view(mx.uint16)).ravel(), np.array(m._matvec(x).view(mx.uint16)).ravel())
            for w, m in list(zip(items["bf16"], items["binade_r4"]))[:3]
        )
        for m in methods:
            sweep(fns[m], items[m])
        times = {m: [] for m in methods}
        for r in range(args.rounds):
            for m in methods[r % len(methods):] + methods[: r % len(methods)]:
                times[m].append(sweep(fns[m], items[m]))
        row = {
            "kind": kind,
            "shape": [N, K],
            "per_token": len(names),
            "copies": copies,
            "bytes": nbytes,
            "us": {m: 1e6 * statistics.median(times[m]) for m in methods},
            "gbs": {m: nbytes[m] / statistics.median(times[m]) / 1e9 for m in methods},
            "efficiency": {m: statistics.median([(nbytes[m] / t) / (nbytes["bf16"] / tb) for t, tb in zip(times[m], times["bf16"])]) for m in methods},
            "speedup": {m: statistics.median([tb / t for t, tb in zip(times[m], times["bf16"])]) for m in methods},
            "r4_bit_identical_to_bf16": exact,
        }
        rows.append(row)
        print(f"{kind:24s} {N:>6d}x{K:<5d} bf16 {row['gbs']['bf16']:5.0f} GB/s | " + " | ".join(f"{m} eff {row['efficiency'][m]:.1%} x{row['speedup'][m]:.2f}" for m in methods[1:]) + f" | exact {exact}", flush=True)
        del items
        mx.clear_cache()

    tot = {m: sum(r["us"][m] * r["per_token"] for r in rows) for m in methods}
    byt = {m: sum(r["bytes"][m] * r["per_token"] for r in rows) for m in methods}
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=no", "--", ".", ":!results"], cwd=REPO, capture_output=True, text=True).stdout.strip())
    device = mx.device_info().get("device_name", "?")
    per_token = {"us": tot, "bytes": byt, "gbs": {m: byt[m] / tot[m] / 1e3 for m in methods}, "speedup": {m: tot["bf16"] / tot[m] for m in methods}, "efficiency": {m: (byt[m] / tot[m]) / (byt["bf16"] / tot["bf16"]) for m in methods}}
    summary = {"git": {"commit": commit, "dirty": dirty}, "mlx": mx.__version__, "device": device, "peak_gbs": args.peak_gbs, "args": {k: str(v) for k, v in st.display_args(vars(args)).items()}, "shapes": rows, "per_token": per_token}
    stem = f"gemma-4-12B-it_{args.tag}"
    (args.out / f"{stem}.json").write_text(json.dumps(summary, indent=1) + "\n")
    lines = [
        f"# Batch-1 matvec on Gemma 4 12B shapes: {device}",
        "",
        f"`scripts/matvec_bench.py` @ `{commit[:7]}{' (dirty)' if dirty else ''}`, MLX {mx.__version__}, {device} ({args.peak_gbs:g} GB/s peak). "
        f"Real 12B weights; each shape cycles through distinct copies of at least {args.min_gb:g} GB of BF16; {args.rounds} paired rounds, medians of per-round ratios. "
        "Efficiency = a method's GB/s over BF16's. binade_r4: the R4 matvec as the runtime runs it.",
        "",
        "| projection | shape | per token | BF16 GB/s | " + " | ".join(f"{m} eff / speedup" for m in methods[1:]) + " | R4 bit-identical |",
        "|---|---|---|---|" + "---|" * (len(methods) - 1) + "---|",
    ]
    for r in rows:
        lines.append(f"| {r['kind']} | {r['shape'][0]}x{r['shape'][1]} | {r['per_token']} | {r['gbs']['bf16']:.0f} | " + " | ".join(f"{r['efficiency'][m]:.1%} / {r['speedup'][m]:.2f}x" for m in methods[1:]) + f" | {r['r4_bit_identical_to_bf16']} |")
    p = per_token
    lines += ["", "Per token (every projection once plus lm_head):", "", "| method | ms | bytes vs BF16 | GB/s | % of peak | efficiency vs BF16 | speedup |", "|---|---|---|---|---|---|---|"]
    for m in methods:
        lines.append(f"| {m} | {p['us'][m] / 1e3:.2f} | {p['bytes'][m] / p['bytes']['bf16']:.3f} | {p['gbs'][m]:.0f} | {p['gbs'][m] / args.peak_gbs:.0%} | {p['efficiency'][m]:.1%} | {p['speedup'][m]:.2f}x |")
    (args.out / f"{stem}.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
