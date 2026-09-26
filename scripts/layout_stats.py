#!/usr/bin/env python3
"""Tile statistics of a packed model in the v1 file layout and the R4 runtime layout.

    python scripts/layout_stats.py <packed-dir> [--out DIR]

v1: share of tiles per width, escape-mode tiles, escaped weights. R4 (transcoded on the
GPU as the runtime does): tiles holding an escape, escaped weights, bits per weight in
memory. Linear tensors and the embedding, whole model. Writes layout_stats.json / .md.
"""

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np

from binade import st
from binade.mlx.loader import packed_specs
from binade.mlx.r4 import transcode
from binade.mlx.slow_linear import BinadeEmbedding, BinadeLinear

REPO = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("packed", type=Path)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    import mlx.core as mx

    report = json.loads((args.packed / "binade_pack.json").read_text())
    name = report["source"].get("repo_id", args.packed.name).split("/")[-1]
    out = args.out or REPO / "results" / name
    infos = {t.name: t for t in st.list_tensors(args.packed)}
    arrays = mx.load(str(st.shard_files(args.packed)[0])) if len(st.shard_files(args.packed)) == 1 else {k: v for f in st.shard_files(args.packed) for k, v in mx.load(str(f)).items()}
    widths = np.zeros(16, np.int64)
    v = {"tiles": 0, "esc_tiles": 0, "weights": 0, "escapes": 0}
    r = {"tiles": 0, "flag_tiles": 0, "escapes": 0, "bytes": 0}
    for n, spec in packed_specs(args.packed).items():
        meta = st.read(infos[f"{n}.meta"])
        widths += np.bincount(meta & 15, minlength=16)
        v["tiles"] += meta.size
        v["esc_tiles"] += int(((meta & 16) != 0).sum())
        v["weights"] += infos[f"{n}.raw"].numel
        v["escapes"] += infos[f"{n}.esc"].numel
        m = BinadeEmbedding(spec) if "embed" in n else BinadeLinear(spec)
        m.update({"weight": {p: arrays[f"{n}.{p}"] for p in spec}})
        a = transcode(m)
        flags = np.array(a["flags"])
        r["tiles"] += flags.size
        r["flag_tiles"] += int(flags.sum())
        r["escapes"] += int(a["esc"].size) - 1
        r["bytes"] += sum(int(a[k].nbytes) for k in ("raw", "codes", "flags", "esc", "table", "row_esc"))
        del m, a
        mx.clear_cache()
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=no", "--", ".", ":!results"], cwd=REPO, capture_output=True, text=True).stdout.strip())
    s = {
        "git": {"commit": commit, "dirty": dirty},
        "v1": {**v, "width_share": {str(k): int(c) / v["tiles"] for k, c in enumerate(widths) if c}, "esc_tile_share": v["esc_tiles"] / v["tiles"], "escaped_weight_share": v["escapes"] / v["weights"]},
        "r4": {**r, "flag_tile_share": r["flag_tiles"] / r["tiles"], "escaped_weight_share": r["escapes"] / v["weights"], "bits_per_weight": 8 * r["bytes"] / v["weights"]},
    }
    (out / "layout_stats.json").write_text(json.dumps(s, indent=1) + "\n")
    lines = [
        f"# Tile layouts: {report['source'].get('repo_id', name)}",
        "",
        f"`scripts/layout_stats.py` @ `{commit[:7]}{' (dirty)' if dirty else ''}`, all packed tensors (linear and embedding), {v['weights'] / 1e9:.2f}B weights.",
        "",
        "| layout | tiles by width | tiles with escapes | escaped weights | bits/weight |",
        "|---|---|---|---|---|",
        f"| v1 (file) | {', '.join(f'b={k}: {x:.2%}' for k, x in s['v1']['width_share'].items())} | {s['v1']['esc_tile_share']:.2%} (escape mode) | {s['v1']['escaped_weight_share']:.3%} | see roundtrip.md |",
        f"| R4 (memory) | 4 bits for all | {s['r4']['flag_tile_share']:.2%} | {s['r4']['escaped_weight_share']:.4%} | {s['r4']['bits_per_weight']:.3f} (raw, codes, flags, escapes, table, row starts) |",
    ]
    (out / "layout_stats.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
