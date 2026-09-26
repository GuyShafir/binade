#!/usr/bin/env python3
"""The iid model behind phase 1 (DESIGN.md 5): what fixed-width tile coding
costs when BF16 weights are independent Gaussian draws, through the same code as the scans.

    python scripts/iid_model.py [--rows 4096] [--cols 3840] [--seed 0]

Tensors: iid N(0, 0.02^2), and the same with a per-row scale (log-normal, sigma 0.3) to
check that row-scale variation alone does not create tile locality. Reports exponent
entropy, Huffman, per-tile palette policies, the rank family and the tensor-shuffle
locality gain, in bits per weight including the 8-bit sign+mantissa byte.
Writes results/iid_model.json and .md.
"""

import argparse
import json
import subprocess
from pathlib import Path

import mlx.core as mx
import numpy as np

from binade import baselines
from binade.bf16 import exponent
from binade.format import rank_table
from binade.scan import TileAccum

REPO = Path(__file__).resolve().parents[1]
TS = (32, 64, 128)
POLICIES = ("bump", "escape", "best", "unary", "rank_fixed", "rank_best", "rank_unary", "rank_rice")


def bf16_bits(f: np.ndarray) -> np.ndarray:
    """float32 -> BF16 bit patterns, round to nearest even (as torch and MLX cast)."""
    u = f.astype(np.float32).view(np.uint32)
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)


def measure(w16: np.ndarray, rng) -> dict:
    e = exponent(w16)
    hist = np.bincount(e.ravel(), minlength=256)
    _, rank = rank_table(hist)
    shuffled = rng.permutation(e.ravel()).reshape(e.shape)
    out = {"entropy": baselines.entropy(hist), "huffman": baselines.huffman_bits(hist) / e.size, "distinct": int((hist > 0).sum())}
    n = e.size
    for T in TS:
        for layout, arr in (("real", e), ("shuffled", shuffled)):
            acc = TileAccum(T)
            acc.add(mx.array(arr))
            acc.add_ranks(mx.array(rank[arr]))
            for p in POLICIES:
                acc.bits["actual"][p][1] += 8 * out["distinct"] if p.startswith("rank") else 0
                out[f"T{T}_{p}_{layout}"] = acc.bits_total(n, p) / n
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rows", type=int, default=4096)
    ap.add_argument("--cols", type=int, default=3840)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    f = rng.standard_normal((args.rows, args.cols)).astype(np.float32) * 0.02
    models = {
        "iid Gaussian": bf16_bits(f),
        "Gaussian, per-row log-normal scale (sigma 0.3)": bf16_bits(f * np.exp(0.3 * rng.standard_normal((args.rows, 1))).astype(np.float32)),
    }
    res = {name: measure(w, rng) for name, w in models.items()}

    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=no", "--", ".", ":!results"], cwd=REPO, capture_output=True, text=True).stdout.strip())
    (REPO / "results" / "iid_model.json").write_text(json.dumps({"git": {"commit": commit, "dirty": dirty}, "args": vars(args), "models": res}, indent=1) + "\n")
    lines = [
        "# The iid model: fixed-width tile coding on Gaussian BF16 weights",
        "",
        f"`scripts/iid_model.py` @ `{commit[:7]}{' (dirty)' if dirty else ''}`, one {args.rows}x{args.cols} tensor per model, seed {args.seed}. "
        "Bits per weight including the 8-bit sign+mantissa byte; same scanner code as `results/*/summary.md`.",
        "",
    ]
    for name, r in res.items():
        lines += [
            f"## {name}",
            "",
            f"Exponent entropy {r['entropy']:.3f} bits ({8 + r['entropy']:.3f} per weight), Huffman {r['huffman']:.3f} ({8 + r['huffman']:.3f}), {r['distinct']} distinct exponents.",
            "",
            "| policy | " + " | ".join(f"T={T} real / shuffled" for T in TS) + " |",
            "|---|" + "---|" * len(TS),
        ]
        for p in POLICIES:
            lines.append(f"| {p} | " + " | ".join(f"{r[f'T{T}_{p}_real']:.3f} / {r[f'T{T}_{p}_shuffled']:.3f}" for T in TS) + " |")
        lines.append("")
    (REPO / "results" / "iid_model.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
