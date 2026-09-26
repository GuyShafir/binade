#!/usr/bin/env python3
"""How much a weight's row or column index says about its exponent.

    python scripts/mutual_info.py <model-dir | cached-hf-repo-id> [--revision REV] [--out DIR]

I(exp; col) = H(exp) - H(exp | col) bounds what any column permutation or column
partition (for example one found by a search) can take off the exponent entropy; I(exp; row) does
the same for rows. Entropies are Miller-Madow corrected ((m-1) / (2 n ln 2) for m
observed symbols in n samples), since each column holds only a few thousand weights.
Writes mutual_info.json and mutual_info.md next to the phase 1 summary.
"""

import argparse
import json
import subprocess
import time
from pathlib import Path

from binade import st
from binade.classify import classify
from binade.info import tensor_info

REPO = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model")
    ap.add_argument("--revision")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    t0 = time.perf_counter()
    model_dir = st.resolve(args.model, args.revision)
    source = st.provenance(model_dir)
    out = args.out or REPO / "results" / source.get("repo_id", model_dir.name).split("/")[-1]
    out.mkdir(parents=True, exist_ok=True)

    per_tensor, groups = [], {}
    for info in st.list_tensors(model_dir):
        k = classify(info.name, info.shape, info.dtype)
        if k.cls != "linear":
            continue
        r = tensor_info(info)
        r.update(name=info.name, sub=k.sub)
        per_tensor.append(r)
        for g in (k.sub, "linear"):
            a = groups.setdefault(g, {"numel": 0, "H": 0.0, "H_given_row": 0.0, "H_given_col": 0.0, "tensors": 0})
            a["tensors"] += 1
            a["numel"] += r["numel"]
            for key in ("H", "H_given_row", "H_given_col"):
                a[key] += r[key] * r["numel"]
        print(f"{info.name}: I(row) {r['H'] - r['H_given_row']:.4f}  I(col) {r['H'] - r['H_given_col']:.4f} bits", flush=True)
    for a in groups.values():
        for key in ("H", "H_given_row", "H_given_col"):
            a[key] /= a["numel"]
        a["I_row"] = a["H"] - a["H_given_row"]
        a["I_col"] = a["H"] - a["H_given_col"]

    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=no", "--", ".", ":!results"], cwd=REPO, capture_output=True, text=True).stdout.strip())
    result = {"source": source, "git": {"commit": commit, "dirty": dirty}, "seconds": time.perf_counter() - t0, "groups": groups, "tensors": per_tensor}
    (out / "mutual_info.json").write_text(json.dumps(result, indent=1) + "\n")

    order = [g for g in ("linear", "attn", "linattn", "mlp", "experts", "ple_proj") if g in groups]
    lines = [
        f"# Exponent information in the row and column index: {source.get('repo_id', model_dir.name)}",
        "",
        f"Script `scripts/mutual_info.py` @ `{commit[:7]}{' (dirty)' if dirty else ''}`. Linear tensors, bits per exponent, "
        "size-weighted over tensors, Miller-Madow corrected. I(col) bounds the entropy any column permutation or column "
        "partition can remove; I(row) the same for rows.",
        "",
        "| class | tensors | H(exp) | H(exp given row) | H(exp given col) | I(exp; row) | I(exp; col) |",
        "|---|---|---|---|---|---|---|",
    ]
    for g in order:
        a = groups[g]
        lines.append(
            f"| {g} | {a['tensors']} | {a['H']:.4f} | {a['H_given_row']:.4f} | {a['H_given_col']:.4f} | {a['I_row']:.4f} | {a['I_col']:.4f} |"
        )
    (out / "mutual_info.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
