#!/usr/bin/env python3
"""Where the v1 in-place kernel loses time: timing-only ablations on Gemma 4 12B shapes.

    python scripts/kernel_ablate.py [--packed models/gemma-4-12B-it-binade] [--rounds 7]

Variants of the v1 kernel (tm 1, prologue, b=3 fast path), measured like kernel_bench.py
(DRAM-resident copies, paired rounds against MLX's BF16 matvec). The ablated variants
compute wrong outputs on purpose and only time what remains:
  noload  escape bookkeeping kept, the escape byte load replaced by a constant
  noesc   escape handling skipped entirely
Writes results/kernel/ablation.json / .md.
"""

import argparse
import json
import statistics
import subprocess
from pathlib import Path

import mlx.core as mx
import numpy as np

from binade import st
from binade.mlx.kernel import BinadeMatrix
from kernel_bench import REPO, groups, sweep

BASE = dict(tm=1, prologue=True, fast3=True, bm=8)
VARIANTS = {"v1": {}, "noload": {"ablate": "noload"}, "noesc": {"ablate": "noesc"}}
SHAPES = [("gate_up_proj", (15360, 3840)), ("down_proj", (3840, 15360)), ("q_proj", (4096, 3840))]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--original", default="google/gemma-4-12B-it")
    ap.add_argument("--packed", type=Path, default=REPO / "models" / "gemma-4-12B-it-binade")
    ap.add_argument("--rounds", type=int, default=7)
    args = ap.parse_args()
    mx.set_cache_limit(4 << 30)
    orig = {t.name: t for t in st.list_tensors(st.resolve(args.original))}
    packed = {t.name: t for t in st.list_tensors(args.packed)}
    G = groups(packed)
    rng = np.random.default_rng(0)
    rows = []
    for kind, shape in SHAPES:
        names = G[(kind, shape)]
        N, K = shape
        copies = max(1, int(np.ceil(1e9 / (2 * N * K))))
        chosen = [names[i % len(names)] for i in range(copies)]
        W = [mx.array(st.read(orig[n])).view(mx.bfloat16) for n in chosen]
        X = [BinadeMatrix(*(st.read(packed[f"{n}.{p}"]) for p in ("raw", "table", "meta", "idx", "esc"))) for n in chosen]
        mx.eval(W, [(m.raw, m.idx) for m in X])
        x = mx.array(rng.standard_normal((1, K)).astype(np.float32)).astype(mx.bfloat16)
        fns = {"bf16": (lambda w: x @ w.T, W)}
        for v, o in VARIANTS.items():
            fns[v] = ((lambda o: (lambda m: m.matvec(x, **BASE, **o)))(o), X)
        for f, it in fns.values():
            sweep(f, it)
        times = {k: [] for k in fns}
        keys = list(fns)
        for r in range(args.rounds):
            for k in keys[r % len(keys):] + keys[: r % len(keys)]:
                times[k].append(sweep(*fns[k]))
        nb, nbf = X[0].nbytes(), 2 * N * K
        eff = {v: statistics.median([(nb / t) / (nbf / tb) for t, tb in zip(times[v], times["bf16"])]) for v in VARIANTS}
        rows.append({"kind": kind, "shape": [N, K], "efficiency": eff})
        print(kind, {k: f"{e:.1%}" for k, e in eff.items()}, flush=True)
        del W, X
        mx.clear_cache()
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=no", "--", ".", ":!results"], cwd=REPO, capture_output=True, text=True).stdout.strip())
    out = REPO / "results" / "kernel"
    (out / "ablation.json").write_text(json.dumps({"git": {"commit": commit, "dirty": dirty}, "base": BASE, "rows": rows}, indent=1) + "\n")
    lines = [
        "# Where the v1 in-place kernel loses time (timing-only ablations)",
        "",
        f"`scripts/kernel_ablate.py` @ `{commit[:7]}{' (dirty)' if dirty else ''}`, Gemma 4 12B weights, DRAM-resident, {args.rounds} paired rounds "
        "against MLX's BF16 matvec; efficiency on v1's bytes. Ablated variants compute wrong outputs on purpose.",
        "",
        "| projection | shape | v1 kernel | escape load replaced by a constant | escape handling skipped |",
        "|---|---|---|---|---|",
    ]
    for r in rows:
        e = r["efficiency"]
        lines.append(f"| {r['kind']} | {r['shape'][0]}x{r['shape'][1]} | {e['v1']:.1%} | {e['noload']:.1%} | {e['noesc']:.1%} |")
    (out / "ablation.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
