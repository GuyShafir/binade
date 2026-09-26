#!/usr/bin/env python3
"""Cross-model phase 1 table from results/*/summary.json -> results/comparison.md.

    python scripts/compare.py
"""

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def f3(x):
    return f"{x:.3f}"


def main():
    verdict_rows, size_rows, notes = [], [], []
    for path in sorted((REPO / "results").glob("*/summary.json")):
        s = json.loads(path.read_text())
        if s["scope"].get("limited"):
            continue
        lin = s["groups"]["linear"]
        b = lin["baselines"]
        v = s["verdict"]
        T = str(v["best_T"])
        t = lin["tiles"][T]
        src = s["source"]
        name = src.get("repo_id", path.parent.name)
        bpw = lambda var, pol, off=False: t[var]["bpw_with_off" if off else "bpw"]["actual"][pol]
        verdict_rows.append(
            [name, f"{lin['numel'] / 1e9:.2f}B", T, f3(bpw("real", "best")), f3(bpw("shuf_tensor", "best")), f3(v["gain"]), v["band"]]
        )
        size_rows.append(
            [
                name,
                f3(b["huffman_block"]),
                f3(b["window7"]),
                f3(b["palette16_segments"]),
                f"{f3(bpw('real', 'best'))} / {f3(bpw('real', 'best', True))}",
                f"{f3(bpw('real', 'rank_best'))} / {f3(bpw('real', 'rank_best', True))}",
                f"{f3(bpw('real', 'rank_rice'))} / {f3(bpw('real', 'rank_rice', True))}",
            ]
        )
        g = s["git"]
        notes.append(
            f"- {name} @ `{src.get('revision', '')[:7]}`: histogram.py @ "
            f"`{g['commit'][:7] if g['commit'] else 'uncommitted'}`{' (dirty)' if g['dirty'] else ''}, {s['started'][:10]}, "
            f"details in `results/{path.parent.name}/summary.md`"
        )

    def table(header, rows):
        return ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)] + ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]

    lines = [
        "# Binade phase 1: models compared",
        "",
        "Linear tensors only. Bits per weight include the 8-bit raw byte (sign + mantissa) and the 8-bit per-tile meta byte.",
        "",
        "## Locality (DESIGN.md 5 go/no-go), best-of at each model's best T",
        "",
        *table(["model", "linear weights", "T", "per-tile palettes (v0, best-of)", "tensor-shuffled", "locality gain", "band"], verdict_rows),
        "",
        "## Size against prior-art exponent codings (our accounting of each on the same tensors; Binade columns without / with a stored 32-bit offset per tile)",
        "",
        "Unweight's Huffman mode is not modeled; it reports about 10.9 to 11.0 bits per weight on Llama-3.1-8B MLP. Binade v1 files store no offsets.",
        "",
        *table(
            [
                "model",
                "per-block Huffman (DFloat11 granularity)",
                "7-exponent window, 3-bit code, outliers as BF16 (ZipServ layout)",
                "16-exponent palette, fixed 4-bit index (Unweight 4-bit mode)",
                "per-tile palettes (v0, best-of)",
                "Binade v1 (rank_best)",
                "Rice over ranks (rank_rice)",
            ],
            size_rows,
        ),
    ]
    ck, ex = [], []
    for path in sorted((REPO / "results").glob("*/roundtrip.json")):
        r = json.loads(path.read_text())
        lin = r["classes"].get("linear", {})
        ck.append(
            [
                r["source"].get("repo_id", path.parent.name),
                f"{r['input_bytes'] / 1e9:.2f}",
                f"{r['output_bytes'] / 1e9:.2f}",
                f"{r['output_bytes'] / r['input_bytes']:.1%}",
                r["tensors_packed"],
                f"{lin.get('bits_per_weight', float('nan')):.3f}",
                f"{r['mismatched_elements']} in {r['tensors_checked']} tensors",
            ]
        )
        notes.append(f"- {path.parent.name}/roundtrip.json: packed @ `{r['pack_git'][:7]}`, checked @ `{r['git']['commit'][:7]}`{' (dirty)' if r['git']['dirty'] else ''}")
    for path in sorted((REPO / "results").glob("*/generate_check.json")):
        g = json.loads(path.read_text())
        ex.append(
            [
                g["source"].get("repo_id", path.parent.name),
                g["identical_logits"],
                f"{g['identical_ids']} ({g['tokens']} tokens)",
                f"{g['bf16']['tokens_per_s']:.2f}",
                f"{g.get('binade', g.get('xp'))['tokens_per_s']:.2f}",  # key was "xp" before the rename
            ]
        )
        notes.append(f"- {path.parent.name}/generate_check.json: @ `{g['git']['commit'][:7]}`{' (dirty)' if g['git']['dirty'] else ''}")
    if ck:
        lines += [
            "",
            "## Packed checkpoints (Binade v1; linear and embedding packed, everything else BF16)",
            "",
            *table(["model", "BF16 GB", "Binade GB", "ratio", "packed tensors", "linear bits/weight (file)", "round-trip mismatches"], ck),
        ]
    if ex:
        lines += [
            "",
            "## Generation, stock BF16 vs Binade slow runtime (MLX ops decode per call, no kernel)",
            "",
            *table(["model", "prompt logits identical", "greedy ids identical", "BF16 tok/s", "Binade slow runtime tok/s"], ex),
        ]
    lines += ["", "Sources:", "", *notes]
    out = REPO / "results" / "comparison.md"
    out.write_text("\n".join(lines) + "\n")
    print(out.read_text())


if __name__ == "__main__":
    main()
