#!/usr/bin/env python3
"""CUDA matrix products of a prompt pass, one layer's weights at a time: BF16 (MLX's cuBLAS
matmul), rebuild (R4 decoded to BF16, then the same matmul) and the fused R4 GEMM run with
cuBLAS's plan for the shape (cublas_plan), each checked bit for bit against BF16.

    python scripts/cuda_gemm_bench.py <original model-dir | cached-hf-repo-id> <packed-dir> [--rows 40,128,256,512,1024,2048] [--out DIR]

Times are per product: the median of 5 runs of 10 independent products evaluated together
(launch overhead amortized), so they show kernel speed, not what a prompt pass waits for
(prompt_speed.py: there a rebuild overlaps other layers' work). The fused kernel is whichever
r4_gemm picks for the row count (register kernel up to 64 rows, 128 x 128 tiles above), at
every row count, whatever r4.CUDA_FUSED_MAX_ROWS says. Writes gemm_bench.json / .md to
results/cuda/<model>/ (or --out).
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
from binade.mlx import cublas_plan
from binade.mlx import cuda_kernels as ck
from binade.mlx.r4 import R4Linear, transcode

REPO = Path(__file__).resolve().parents[1]
SUFFIXES = ("layers.0.self_attn.q_proj.weight", "layers.0.self_attn.o_proj.weight", "layers.0.mlp.gate_proj.weight", "layers.0.mlp.down_proj.weight")


def timeit(f, n=5, chain=10):
    mx.eval([f() for _ in range(chain)])
    mx.synchronize()
    ts = []
    for _ in range(n):
        t = time.perf_counter()
        mx.eval([f() for _ in range(chain)])
        mx.synchronize()
        ts.append((time.perf_counter() - t) / chain)
    return statistics.median(ts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("original")
    ap.add_argument("packed", type=Path)
    ap.add_argument("--rows", default="40,128,256,512,1024,2048")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    src = st.resolve(args.original)
    source = st.provenance(src)
    out = args.out or REPO / "results" / "cuda" / source.get("repo_id", src.name).split("/")[-1]
    out.mkdir(parents=True, exist_ok=True)
    rows = [int(r) for r in args.rows.split(",")]
    orig = {t.name: t for t in st.list_tensors(src)}
    packed = {t.name: t for t in st.list_tensors(args.packed)}
    res = []
    for suffix in SUFFIXES:
        nm = next((n for n in orig if n.endswith(suffix) and f"{n}.raw" in packed), None)
        if nm is None:
            continue
        arrays = {}
        for f in {packed[f"{nm}.{q}"].file for q in ("raw", "table", "meta", "bits", "roff")}:
            arrays.update({k: v for k, v in mx.load(str(f)).items() if k.startswith(nm + ".")})
        m = R4Linear(transcode(SimpleNamespace(weight={q: arrays[f"{nm}.{q}"] for q in ("raw", "table", "meta", "bits", "roff")}, _plan=None)))
        w = mx.array(st.read(orig[nm])).view(mx.bfloat16)
        mx.eval(w, m._a["codes"])
        for M in rows:
            x = mx.random.normal((M, m.K), key=mx.random.key(M)).astype(mx.bfloat16)
            p = cublas_plan.plan(M, m.N, m.K)
            split, sl = cublas_plan.supported(p), cublas_plan.slices(M, m.N, m.K)
            ref = x @ w.T
            r = {"tensor": suffix, "N": m.N, "K": m.K, "M": M, "plan": p, "slices": sl, "bf16_s": timeit(lambda: x @ w.T), "rebuild_s": timeit(lambda: x @ m.dense().T)}
            if split is not None and sl is not None and ck.gemm_supported(M, m.N, m.K, sl):
                b = cublas_plan.partitions(m.K, p[0])
                f = ck.r4_gemm(m._a, x, m.N, m.K, b, split, slices=sl)
                r["exact"] = bool(np.array_equal(np.array(ref.view(mx.uint16)), np.array(f.view(mx.uint16))))
                r["fused_s"] = timeit(lambda: ck.r4_gemm(m._a, x, m.N, m.K, b, split, slices=sl))
            res.append(r)
            print(suffix, M, {k: (f"{v * 1e3:.3f} ms" if k.endswith("_s") else v) for k, v in r.items() if k not in ("tensor",)}, flush=True)
        del m, w, arrays
        mx.clear_cache()
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=no", "--", ".", ":!results"], cwd=REPO, capture_output=True, text=True).stdout.strip())
    summary = {"source": source, "packed": st.display_path(args.packed), "git": {"commit": commit, "dirty": dirty}, "mlx": mx.__version__, "device": mx.device_info().get("device_name"), "args": {k: str(v) for k, v in st.display_args(vars(args)).items()}, "products": res}
    (out / "gemm_bench.json").write_text(json.dumps(summary, indent=1) + "\n")
    lines = [
        f"# CUDA matrix products: {source.get('repo_id', src.name)}",
        "",
        f"`scripts/cuda_gemm_bench.py` @ `{commit[:7]}{' (dirty)' if dirty else ''}`, MLX {mx.__version__}, {summary['device']}. "
        "Per product, median of 5 runs of 10 independent products: BF16 (cuBLAS), rebuild (R4 to BF16, then cuBLAS) and the fused R4 GEMM with cuBLAS's plan; speeds relative to BF16; exact = fused output bit-identical to BF16's.",
        "",
        "| weight | N x K | rows | cuBLAS plan (split-K, scheme) | BF16 ms | rebuild | fused | exact |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in res:
        fused = f"{r['fused_s'] * 1e3:.3f} ({r['bf16_s'] / r['fused_s']:.2f}x)" if "fused_s" in r else "not mirrored"
        lines.append(f"| {r['tensor'].split('.')[-2]} | {r['N']} x {r['K']} | {r['M']} | {tuple(r['plan']) if r['plan'] else '-'} | {r['bf16_s'] * 1e3:.3f} | {r['rebuild_s'] * 1e3:.3f} ({r['bf16_s'] / r['rebuild_s']:.2f}x) | {fused} | {r.get('exact', '-')} |")
    (out / "gemm_bench.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
