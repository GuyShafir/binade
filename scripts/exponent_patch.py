#!/usr/bin/env python3
"""Exponent ranks of a patch of one weight matrix, as stored and with positions shuffled, for
the paper's statistics figure.

    python scripts/exponent_patch.py <original model-dir | cached-hf-repo-id> [--tensor layers.20.mlp.gate_proj.weight] [--rows 48] [--cols 192] [--seed 0] [--out DIR]

Ranks are the per-tensor frequency ranks Binade stores (0 = most frequent exponent; ties by
exponent). Three views of the same tensor: the top-left rows x cols patch; the same rows with
each row's weights shuffled along K before taking the first cols (what a within-row shuffle
does to a tile); and rows x cols weights drawn without replacement from the whole tensor (a
tensor-wide shuffle). Writes exponent_patch.json to results/<model>/ (or --out).
"""

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np

from binade import st

REPO = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("original")
    ap.add_argument("--tensor", default="layers.20.mlp.gate_proj.weight")
    ap.add_argument("--rows", type=int, default=48)
    ap.add_argument("--cols", type=int, default=192)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    src = st.resolve(args.original)
    source = st.provenance(src)
    out = args.out or REPO / "results" / source.get("repo_id", src.name).split("/")[-1]
    out.mkdir(parents=True, exist_ok=True)
    info = next(t for t in st.list_tensors(src) if t.name.endswith(args.tensor))
    w = np.asarray(st.read(info)).view(np.uint16).reshape(info.shape)
    e = ((w >> 7) & 0xFF).astype(np.int64)
    counts = np.bincount(e.ravel(), minlength=256)
    order = np.lexsort((np.arange(256), -counts))  # descending frequency, ties by exponent
    rank = np.empty(256, np.int64)
    rank[order] = np.arange(256)
    r = rank[e]
    rng = np.random.default_rng(args.seed)
    R, C = args.rows, args.cols
    real = r[:R, :C]
    row_shuffled = np.stack([rng.permutation(r[i])[:C] for i in range(R)])
    tensor_shuffled = r.ravel()[rng.choice(r.size, R * C, replace=False)].reshape(R, C)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=no", "--", ".", ":!results"], cwd=REPO, capture_output=True, text=True).stdout.strip())
    summary = {
        "source": source, "git": {"commit": commit, "dirty": dirty}, "args": {k: str(v) for k, v in st.display_args(vars(args)).items()},
        "tensor": info.name, "shape": list(info.shape),
        "rank_share": (counts[order][: int((counts > 0).sum())] / e.size).tolist(),
        "real": real.tolist(), "row_shuffled": row_shuffled.tolist(), "tensor_shuffled": tensor_shuffled.tolist(),
    }
    (out / "exponent_patch.json").write_text(json.dumps(summary) + "\n")
    print(info.name, info.shape, "ranks in patch:", np.bincount(real.ravel()).tolist())


if __name__ == "__main__":
    main()
