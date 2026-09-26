#!/usr/bin/env python3
"""Phase 1: exponent statistics and the locality go/no-go for Binade (DESIGN.md 5).

    python scripts/histogram.py <model-dir | cached-hf-repo-id> [--revision REV] [--out DIR]

Streams one tensor at a time from the safetensors files. Writes summary.json,
summary.md, CSVs and PNGs to results/<model>/ (or --out).
"""

import argparse
import csv
import json
import math
import platform
import resource
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import mlx.core as mx
import numpy as np

from binade import baselines, lossy, st
from binade.classify import classify
from binade.policies import CHOICES, POLICIES
from binade.scan import VARIANTS, LossyAccum, TileAccum, scan_tensor

REPO = Path(__file__).resolve().parents[1]
LINEAR_MIN_SHARE = 0.9
BANDS = (
    (0.7, "alive", "size story is alive; continue as designed"),
    (0.2, "speed", "format is a speed and Metal story; README and paper must state DFloat11 is smaller"),
    (-math.inf, "stop", "stop; the unary/Rice fallback the design calls for in this band (DESIGN.md 5) is reported below as rank_unary and rank_rice"),
)
LAYOUT_NAMES = {"real": "real", "shuf_row": "row-shuffled", "shuf_tensor": "tensor-shuffled"}


class Group:
    """Sums over a set of scanned tensors."""

    def __init__(self, tile_sizes):
        self.tensors = 0
        self.numel = 0
        self.hist = np.zeros(256, np.int64)
        self.tiles = {v: {T: TileAccum(T) for T in tile_sizes} for v in VARIANTS}
        self.block_hist = {}
        self.huff_tensor = 0
        self.ent_tensor = 0.0
        self.pal_tensor = 0
        self.pal16_segments = 0
        self.window7 = 0

    def add(self, rec, block=None):
        self.tensors += 1
        self.numel += rec["numel"]
        self.hist += rec["hist"]
        self.block_hist.setdefault(block, np.zeros(256, np.int64))
        self.block_hist[block] += rec["hist"]
        for v in VARIANTS:
            for T, acc in rec["tiles"][v].items():
                self.tiles[v][T].merge(acc)
        self.huff_tensor += baselines.huffman_bits(rec["hist"])
        self.ent_tensor += rec["numel"] * baselines.entropy(rec["hist"])
        self.pal_tensor += baselines.tensor_palette_bits(rec["hist"])
        self.pal16_segments += baselines.palette16_segment_bits(rec["seg_in_weights"], rec["numel"], rec["segments"])
        self.window7 += baselines.window7_bits(rec["hist"])

    def bpw(self, variant, T, policy="best", objective="actual", with_off=False):
        return self.tiles[variant][T].bits_total(self.numel, policy, objective, with_off) / self.numel

    def baselines(self):
        n = self.numel
        return {
            "bf16": 16.0,
            "huffman_global": 8 + baselines.huffman_bits(self.hist) / n,
            "huffman_block": 8 + sum(baselines.huffman_bits(h) for h in self.block_hist.values()) / n,
            "huffman_tensor": 8 + self.huff_tensor / n,
            "entropy_global": 8 + baselines.entropy(self.hist),
            "entropy_tensor": 8 + self.ent_tensor / n,
            "palette_tensor": 8 + self.pal_tensor / n,
            "palette16_segments": 8 + self.pal16_segments / n,
            "window7": 8 + self.window7 / n,
        }


def git_state():
    def run(*a):
        r = subprocess.run(["git", *a], cwd=REPO, capture_output=True, text=True)
        return r.stdout.strip() if r.returncode == 0 else None

    commit = run("rev-parse", "HEAD")
    dirty = bool(run("status", "--porcelain", "--untracked-files=no"))
    return {"commit": commit, "dirty": dirty}


def classification_table(infos):
    rows = {}
    for t in infos:
        k = classify(t.name, t.shape, t.dtype)
        r = rows.setdefault((k.cls, k.sub), {"cls": k.cls, "sub": k.sub, "tensors": 0, "params": 0, "dtypes": set()})
        r["tensors"] += 1
        r["params"] += t.numel
        r["dtypes"].add(t.dtype)
    total = sum(r["params"] for r in rows.values())
    order = ("linear", "embed", "ple", "mm", "other")
    out = sorted(rows.values(), key=lambda r: (order.index(r["cls"]), -r["params"]))
    for r in out:
        r["share"] = r["params"] / total
        r["dtypes"] = sorted(r["dtypes"])
    return out, total


def print_classification(table, total):
    print(f"{'class':8s} {'sub':14s} {'tensors':>8s} {'params':>16s} {'share':>8s}  dtypes")
    for r in table:
        print(f"{r['cls']:8s} {r['sub']:14s} {r['tensors']:8d} {r['params']:16,d} {r['share']:8.2%}  {','.join(r['dtypes'])}")
    print(f"{'total':8s} {'':14s} {sum(r['tensors'] for r in table):8d} {total:16,d}")


def verdict(lin, tile_sizes):
    real = {T: lin.bpw("real", T) for T in tile_sizes}
    best_T = min(tile_sizes, key=real.get)
    gain = {T: lin.bpw("shuf_tensor", T) - real[T] for T in tile_sizes}
    band, meaning = next((b, m) for lo, b, m in BANDS if gain[best_T] >= lo)
    return {"best_T": best_T, "gain": gain[best_T], "gain_by_T": gain, "band": band, "meaning": meaning}


def f3(x):
    return f"{x:.3f}"


def md_table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def kdist(acc: TileAccum):
    h = acc.khist
    tot = h.sum()
    cum = np.cumsum(h)
    at = lambda k: cum[min(k, len(h) - 1)] / tot
    return {"le1": at(1), "le2": at(2), "le4": at(4), "le8": at(8), "le16": at(16), "mean": float((np.arange(len(h)) * h).sum() / tot)}


def write_markdown(path, s, groups, lin_lossy, tile_sizes):
    v = s["verdict"]
    T0 = v["best_T"]
    best_fixed, best_any = v["cheapest_fixed_width"], v["cheapest_any"]
    lin = groups["linear"]
    src = s["source"]
    rev = src.get("revision", "")[:7]
    g = s["git"]
    lines = [
        f"# Binade phase 1: {src.get('repo_id', src['path'])}",
        "",
        f"Source `{src.get('repo_id', src['path'])}`{' @ `' + rev + '`' if rev else ''}. "
        f"Script `scripts/histogram.py` @ `{g['commit'][:7] if g['commit'] else 'uncommitted'}{' (dirty)' if g['dirty'] else ''}`, "
        f"seed {s['args']['seed']}, {s['started'][:10]}, {s['seconds'] / 60:.1f} min, peak RSS {s['peak_rss_gb']:.1f} GB.",
        "",
        f"Scope: {lin.tensors} linear tensors, {lin.numel / 1e9:.2f}B weights, "
        f"{s['scope']['linear_share_of_all']:.1%} of all parameters. "
        "Bits per weight include the 8-bit raw byte (sign + mantissa) and the 8-bit per-tile meta byte; "
        "palettes are charged at actual entries and the `.off` array is excluded unless a column says otherwise.",
        "",
        "## Verdict",
        "",
        f"Locality gain at T={T0} (best-of, tensor-shuffled minus real): **{v['gain']:.3f} bits/weight** "
        f"({f3(lin.bpw('shuf_tensor', T0))} - {f3(lin.bpw('real', T0))}). "
        f"Band `{v['band']}`: {v['meaning']}.",
        "",
        f"Position at T={T0}: per-tile palettes (v0, best-of) {f3(lin.bpw('real', T0))} vs per-block Huffman (DFloat11 granularity) "
        f"{f3(s['groups']['linear']['baselines']['huffman_block'])}. "
        f"Cheapest fixed-width code measured: {best_fixed[0]} at T={best_fixed[1]}, {f3(best_fixed[2])}. "
        f"Cheapest code measured without a Huffman table: {best_any[0]} at T={best_any[1]}, {f3(best_any[2])}.",
        "",
        "## Best-of, linear tensors",
        "",
        md_table(
            ["T", "real", "row-shuffled", "tensor-shuffled", "locality gain", "real + .off", "real, v0 padded palette + .off"],
            [
                [
                    T,
                    f3(lin.bpw("real", T)),
                    f3(lin.bpw("shuf_row", T)),
                    f3(lin.bpw("shuf_tensor", T)),
                    f3(v["gain_by_T"][T]),
                    f3(lin.bpw("real", T, with_off=True)),
                    f3(lin.bpw("real", T, objective="p16", with_off=True)),
                ]
                for T in tile_sizes
            ],
        ),
        "",
        "## Baselines, linear tensors",
        "",
    ]
    b = s["groups"]["linear"]["baselines"]
    lines.append(
        md_table(
            ["code", "bits/weight"],
            [
                ["BF16", f3(b["bf16"])],
                ["Huffman, one code per transformer block (DFloat11's codebook granularity)", f3(b["huffman_block"])],
                ["Huffman, one code for all linear tensors", f3(b["huffman_global"])],
                ["Huffman, one code per tensor", f3(b["huffman_tensor"])],
                ["Entropy bound, per-tensor distributions", f3(b["entropy_tensor"])],
                ["7 consecutive exponents per tensor at a fixed 3-bit code, other weights also stored as full BF16 (ZipServ's layout, our accounting)", f3(b["window7"])],
                ["16 most frequent exponents per tensor at a fixed 4-bit index, 64-weight segments with any other exponent kept as 8-bit exponents (Unweight's 4-bit mode, our accounting; its Huffman mode is not modeled)", f3(b["palette16_segments"])],
                ["Palette per tensor, b = ceil(log2 distinct)", f3(b["palette_tensor"])],
                [f"per-tile palettes (v0, best-of), T={T0}", f3(lin.bpw("real", T0))],
                [f"per-tile palettes (v0, best-of), T={T0}, + .off", f3(lin.bpw("real", T0, with_off=True))],
                [f"rank_fixed (per-tensor ranks, fixed width per tile; ENEC-class), T={T0}", f3(lin.bpw("real", T0, "rank_fixed"))],
                [f"rank_best (per-tensor ranks, width or escape per tile), T={T0}", f3(lin.bpw("real", T0, "rank_best"))],
                [f"rank_unary (per-tensor ranks, truncated unary), T={T0}", f3(lin.bpw("real", T0, "rank_unary"))],
                [f"rank_rice (per-tensor ranks, Rice k in 0..2 per tile), T={T0}", f3(lin.bpw("real", T0, "rank_rice"))],
                [f"rank_best, T={T0}, + .off", f3(lin.bpw("real", T0, "rank_best", with_off=True))],
                [f"rank_rice, T={T0}, + .off", f3(lin.bpw("real", T0, "rank_rice", with_off=True))],
            ],
        )
    )
    lines += ["", "## All policies, linear tensors (real / tensor-shuffled)", ""]
    lines.append(
        md_table(
            ["policy"] + [f"T={T}" for T in tile_sizes],
            [
                [p]
                + [f"{f3(lin.bpw('real', T, p))} / {f3(lin.bpw('shuf_tensor', T, p))}" for T in tile_sizes]
                for p in POLICIES
            ],
        )
    )
    lines += ["", f"## Per class, best-of at T={T0}", ""]
    rows = []
    for name in ("attn", "linattn", "mlp", "experts", "ple_proj", "embed"):
        grp = groups.get(name)
        if grp is None or grp.tensors == 0:
            continue
        rows.append(
            [
                name,
                grp.tensors,
                f"{grp.numel / 1e9:.3f}B",
                f3(grp.bpw("real", T0)),
                f3(grp.bpw("shuf_tensor", T0)),
                f3(grp.bpw("shuf_tensor", T0) - grp.bpw("real", T0)),
                f3(grp.baselines()["huffman_global"]),
            ]
        )
    lines.append(md_table(["class", "tensors", "weights", "real", "tensor-shuffled", "gain", "Huffman (class code)"], rows))
    lines += ["", "## Distinct exponents per tile (k_t), linear tensors", ""]
    rows = []
    for T in tile_sizes:
        for var in VARIANTS:
            d = kdist(lin.tiles[var][T])
            rows.append([T, LAYOUT_NAMES[var]] + [f"{d[k]:.1%}" for k in ("le1", "le2", "le4", "le8", "le16")] + [f"{d['mean']:.2f}"])
    lines.append(md_table(["T", "layout", "k=1", "k<=2", "k<=4", "k<=8", "k<=16", "mean k"], rows))
    lines += ["", f"## Best-of choices, share of tiles, linear tensors, T={T0}", ""]
    rows = []
    for var in ("real", "shuf_tensor"):
        ch = lin.tiles[var][T0].choice["actual"]
        rows.append([LAYOUT_NAMES[var]] + [f"{x / ch.sum():.1%}" for x in ch])
    lines.append(md_table(["layout"] + list(CHOICES), rows))
    lines += [
        "",
        f"## Lossy dial, linear tensors, T={T0} (report only)",
        "",
        "Per-tile relative error after keeping m mantissa bits (round half to even, exponent exact). "
        f"Percentiles are upper edges of log bins {100 * (10 ** (1 / lossy.BINS_PER_DECADE) - 1):.1f}% wide; max is exact.",
        "",
    ]
    rows = []
    for m in lossy.MANTISSA_BITS:
        pf = lossy.percentiles(lin_lossy.h[T0][m]["rel_fro"])
        pm = lossy.percentiles(lin_lossy.h[T0][m]["max_rel"])
        rows.append(
            [m]
            + [f"{pf.get(q, 0):.2e}" for q in ("p50", "p90", "p99", "p99.9")]
            + [f"{lin_lossy.max[T0][m]['rel_fro']:.2e}"]
            + [f"{pm.get(q, 0):.2e}" for q in ("p50", "p99")]
            + [f"{lin_lossy.max[T0][m]['max_rel']:.2e}"]
        )
    lines.append(md_table(["m", "rel Frob p50", "p90", "p99", "p99.9", "max", "max-rel p50", "p99", "max"], rows))
    emb = groups.get("embed")
    if emb is not None and emb.tensors:
        eb = emb.baselines()
        lines += [
            "",
            "## Embedding",
            "",
            f"{emb.numel / 1e9:.3f}B weights ({s['scope']['embed_share_of_all']:.1%} of parameters). "
            + ("No `lm_head` tensor: the embedding is tied and also serves as the output projection, one full read per decoded token. " if s["scope"]["tied_embeddings"] else "")
            + f"Best-of T={T0}: {f3(emb.bpw('real', T0))} real, {f3(emb.bpw('shuf_tensor', T0))} tensor-shuffled; "
            f"Huffman one code {f3(eb['huffman_global'])}.",
        ]
    lines += ["", "## Classification", ""]
    lines.append(
        md_table(
            ["class", "sub", "tensors", "params", "share", "dtypes"],
            [[r["cls"], r["sub"], r["tensors"], f"{r['params']:,}", f"{r['share']:.2%}", ",".join(r["dtypes"])] for r in s["classification"]],
        )
    )
    lines += [
        "",
        "## Definitions",
        "",
        "- Tile: T consecutive weights along K in one row; k_t = distinct exponents in the tile.",
        "- bump: b = ceil(log2 k_t) index bits with a k_t-entry palette, verbatim if k_t > 16. "
        "escape: b in 1..4, top 2^b-1 exponents in the palette, the last code escapes to an 8-bit exponent. "
        "best: cheapest of bump, escape (any b) and verbatim per tile. "
        "unary: k_t-entry palette sorted by frequency, truncated unary code of the rank (candidate for the fallback of DESIGN.md 5).",
        "- Rank family (no per-tile palette): one table per tensor orders exponents by frequency. "
        "rank_fixed: each tile stores ranks at the bit length of its largest rank (ENEC-class). "
        "rank_best: that, or b in 1..4 with ranks >= 2^b-1 escaping to 8 bits, or verbatim. "
        "rank_unary: truncated unary code of the rank, the tile's largest rank in the meta byte. "
        "rank_rice: per tile, the cheapest Rice code (k low bits verbatim, truncated unary quotient) for k in 0..2.",
        "- v0 padded palette: the v0 `.pal` layout (DESIGN.md 3), [ntiles, 16] uint8, 128 bits per tile; per-tile choice re-optimized for that cost.",
        "- `.off` (32 bits per tile): a tile's length follows from its meta byte only for pure fixed-width tiles (bump, rank_fixed, verbatim). "
        "Escape, unary and Rice tiles are variable-length, so random access needs `.off`: add 32/T bits per weight (0.25 at T=128).",
        "- Row-shuffled: exponents permuted within each row. Tensor-shuffled: permuted within the whole tensor (the shuffle control, DESIGN.md 5). "
        "Locality gain = tensor-shuffled minus real.",
        "- Baselines code the exponent byte only and add the 8-bit raw byte; each is our accounting of one scheme's exponent coding on the "
        "same tensors, not a reimplementation. Per-block Huffman: one Huffman code per transformer block (DFloat11 builds one codebook per "
        "block). ZipServ layout: per tensor, the 7 consecutive exponents covering the most weights at a fixed 3-bit code, every other weight "
        "also stored as a full 16-bit value. Unweight 4-bit mode: per tensor, the 16 most frequent exponents at a fixed 4-bit index; a "
        "64-weight row segment holding any other exponent keeps 8-bit exponents; one flag bit per segment. Unweight's Huffman mode "
        "(Huffman over the palette symbols) is not modeled; it reports about 10.9 to 11.0 bits per weight on Llama-3.1-8B MLP.",
    ]
    path.write_text("\n".join(lines) + "\n")


PALETTE = {"surface": "#fcfcfb", "text": "#0b0b0b", "muted": "#52514e", "grid": "#e4e3df", "s1": "#2a78d6", "s2": "#eb6834", "s3": "#1baf7a"}


def _style(ax):
    ax.set_facecolor(PALETTE["surface"])
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(PALETTE["muted"])
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=PALETTE["muted"], labelsize=8)
    ax.grid(True, color=PALETTE["grid"], linewidth=0.6)
    ax.set_axisbelow(True)


def write_plots(out, groups, tile_sizes, T0, title):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    plt.rcParams.update({"font.size": 9, "text.color": PALETTE["text"], "axes.labelcolor": PALETTE["text"]})
    subs = [(n, g) for n, g in groups.items() if n in ("attn", "linattn", "mlp", "experts", "ple_proj") and g.tensors]
    subs = sorted(subs, key=lambda x: -x[1].numel)[:3]
    colors = [PALETTE["s1"], PALETTE["s2"], PALETTE["s3"]]

    # exponent distribution per linear class
    fig, ax = plt.subplots(figsize=(7, 3.6), facecolor=PALETTE["surface"])
    _style(ax)
    for (name, g), c in zip(subs, colors):
        nz = np.flatnonzero(g.hist)
        x = np.arange(nz.min(), nz.max() + 1)
        y = np.where(g.hist[x] > 0, g.hist[x] / g.numel, np.nan)  # gaps instead of drops to zero on a log axis
        ax.plot(x - 127, y, color=c, linewidth=2, marker="o", markersize=3, label=name)
    ax.set_yscale("log")
    ax.set_xlabel("unbiased exponent (e - 127)")
    ax.set_ylabel("share of weights")
    ax.set_title(f"{title}: exponent distribution by class", loc="left", fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "exp_hist.png", dpi=150)
    plt.close(fig)

    # k_t CDF, real vs tensor-shuffled, one panel per T
    lin = groups["linear"]
    fig, axes = plt.subplots(1, len(tile_sizes), figsize=(3.2 * len(tile_sizes), 3.2), facecolor=PALETTE["surface"], sharey=True)
    for ax, T in zip(np.atleast_1d(axes), tile_sizes):
        _style(ax)
        for var, c in (("real", PALETTE["s1"]), ("shuf_tensor", PALETTE["s2"])):
            h = lin.tiles[var][T].khist
            cdf = np.cumsum(h) / h.sum()
            kmax = int(np.flatnonzero(h).max()) if h.any() else 1
            ax.step(np.arange(1, kmax + 1), cdf[1 : kmax + 1], where="post", color=c, linewidth=2, label=LAYOUT_NAMES[var])
        ax.set_title(f"T = {T}", loc="left", fontsize=9)
        ax.set_xlabel("distinct exponents per tile (k)")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    np.atleast_1d(axes)[0].set_ylabel("share of tiles with at most k")
    np.atleast_1d(axes)[0].legend(frameon=False, fontsize=8, loc="lower right")
    fig.suptitle(f"{title}: k per tile, real vs tensor-shuffled", x=0.01, ha="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "kdist.png", dpi=150)
    plt.close(fig)

    # best-of bits/weight per layer
    layers = sorted((int(n[1:]), g) for n, g in groups.items() if n.startswith("L") and g.tensors)
    if layers:
        fig, ax = plt.subplots(figsize=(7, 3.2), facecolor=PALETTE["surface"])
        _style(ax)
        xs = [i for i, _ in layers]
        for var, c in (("real", PALETTE["s1"]), ("shuf_tensor", PALETTE["s2"])):
            ax.plot(xs, [g.bpw(var, T0) for _, g in layers], color=c, linewidth=2, label=f"per-tile palettes (v0), {LAYOUT_NAMES[var]}")
        ax.plot(xs, [g.baselines()["huffman_block"] for _, g in layers], color=PALETTE["s3"], linewidth=2, label="Huffman per block (DFloat11-class)")
        ax.set_xlabel("layer")
        ax.set_ylabel(f"bits/weight, best-of, T={T0}")
        ax.set_title(f"{title}: bits per weight by layer", loc="left", fontsize=10)
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        fig.savefig(out / "layers.png", dpi=150)
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", help="snapshot directory or an HF repo id already in the local cache")
    ap.add_argument("--revision")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--tiles", default="32,64,128")
    ap.add_argument("--classes", default="linear,embed", help="classes to scan")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-lossy", action="store_true")
    ap.add_argument("--limit", type=int, help="scan only the first N selected tensors (development)")
    ap.add_argument("--force", action="store_true", help="continue even if the classification check fails")
    args = ap.parse_args()

    mx.set_cache_limit(1 << 30)
    started = datetime.now(timezone.utc)
    t_start = time.perf_counter()
    tile_sizes = tuple(int(x) for x in args.tiles.split(","))
    model_dir = st.resolve(args.model, args.revision)
    source = st.provenance(model_dir)
    name = source.get("repo_id", model_dir.name).split("/")[-1]
    out = args.out or REPO / "results" / name
    out.mkdir(parents=True, exist_ok=True)

    infos = st.list_tensors(model_dir)
    table, total = classification_table(infos)
    print_classification(table, total)
    params = lambda cls: sum(r["params"] for r in table if r["cls"] == cls)
    lin_share = params("linear") / max(1, params("linear") + params("other"))
    print(f"\nlinear share of non-embedding, non-multimodal params: {lin_share:.2%} (need >= {LINEAR_MIN_SHARE:.0%})")
    if lin_share < LINEAR_MIN_SHARE and not args.force:
        sys.exit("classification check failed: large tensors land in `other`; fix binade/classify.py or pass --force")

    classes = set(args.classes.split(","))
    todo = [t for t in infos if t.dtype == "BF16" and classify(t.name, t.shape, t.dtype).cls in classes]
    if args.limit:
        todo = todo[: args.limit]
    has_lm_head = any(t.name.endswith("lm_head.weight") for t in infos)

    groups = {"linear": Group(tile_sizes)}
    lossy_by = {}
    tensor_rows = []
    for i, info in enumerate(todo, 1):
        k = classify(info.name, info.shape, info.dtype)
        rec = scan_tensor(info, tile_sizes, seed=args.seed, do_lossy=not args.no_lossy)
        if k.cls == "linear":
            keys = [k.sub, "linear"] + ([f"L{k.layer}"] if k.layer is not None else [])
        else:
            keys = [k.cls]
        for key in keys:
            groups.setdefault(key, Group(tile_sizes)).add(rec, block=k.block)
        if rec["lossy"] is not None:
            lossy_by.setdefault(k.cls, LossyAccum(tile_sizes)).merge(rec["lossy"])
        g = Group(tile_sizes)
        g.add(rec)
        row = {
            "name": info.name, "cls": k.cls, "sub": k.sub, "layer": k.layer, "shape": "x".join(map(str, info.shape)),
            "numel": info.numel, "distinct": int((rec["hist"] > 0).sum()), "entropy": round(baselines.entropy(rec["hist"]), 4),
            "huffman_tensor": round(g.baselines()["huffman_tensor"], 4),
        }
        for T in tile_sizes:
            for p in ("best", "bump", "escape", "unary", "rank_fixed", "rank_best", "rank_unary", "rank_rice"):
                for var in VARIANTS:
                    row[f"T{T}_{p}_{var}"] = round(g.bpw(var, T, p), 4)
        tensor_rows.append(row)
        T_mid = tile_sizes[len(tile_sizes) // 2]
        print(
            f"[{i}/{len(todo)}] {info.name} {info.shape} {rec['seconds']:.1f}s "
            f"best T={T_mid}: real {row[f'T{T_mid}_best_real']:.3f} tensor-shuf {row[f'T{T_mid}_best_shuf_tensor']:.3f}",
            flush=True,
        )
        mx.clear_cache()

    lin = groups["linear"]
    if lin.tensors == 0:
        sys.exit("no linear tensors scanned")
    v = verdict(lin, tile_sizes)
    T0 = v["best_T"]
    fixed = ("bump", "escape", "best", "rank_fixed", "rank_best")
    v["cheapest_fixed_width"] = min(((p, T, lin.bpw("real", T, p)) for p in fixed for T in tile_sizes), key=lambda x: x[2])
    v["cheapest_any"] = min(((p, T, lin.bpw("real", T, p)) for p in POLICIES for T in tile_sizes), key=lambda x: x[2])
    lin_lossy = lossy_by.get("linear") or LossyAccum(tile_sizes)

    summary = {
        "phase": 1,
        "source": source,
        "git": git_state(),
        "args": st.display_args(vars(args)),
        "versions": {"python": platform.python_version(), "mlx": mx.__version__, "numpy": np.__version__},
        "started": started.isoformat(timespec="seconds"),
        "seconds": time.perf_counter() - t_start,
        "peak_rss_gb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9,  # bytes on macOS
        "peak_mlx_gb": mx.get_peak_memory() / 1e9,
        "tile_sizes": list(tile_sizes),
        "classification": table,
        "scope": {
            "total_params": total,
            "linear_share_of_all": params("linear") / total,
            "linear_share_check": lin_share,
            "embed_share_of_all": params("embed") / total,
            "tied_embeddings": params("embed") > 0 and not has_lm_head,
            "scanned_tensors": len(todo),
            "limited": bool(args.limit),
        },
        "verdict": v,
        "groups": {},
        "lossy": {},
    }
    for gname, g in groups.items():
        if g.tensors == 0:
            continue
        if gname.startswith("L"):  # per-layer detail lives in layers.csv
            summary["groups"][gname] = {
                "tensors": g.tensors,
                "numel": g.numel,
                "huffman_block": g.baselines()["huffman_block"],
                "best": {str(T): {var: g.bpw(var, T) for var in VARIANTS} for T in tile_sizes},
            }
            continue
        summary["groups"][gname] = {
            "tensors": g.tensors,
            "numel": g.numel,
            "baselines": g.baselines(),
            "tiles": {
                str(T): {
                    var: {
                        "ntiles": g.tiles[var][T].ntiles,
                        "bpw": {
                            obj: {p: g.bpw(var, T, p, obj) for p in POLICIES} for obj in ("actual", "p16")
                        },
                        "bpw_with_off": {
                            obj: {p: g.bpw(var, T, p, obj, True) for p in POLICIES} for obj in ("actual", "p16")
                        },
                        "bits": {obj: {p: g.tiles[var][T].bits[obj][p].tolist() for p in POLICIES} for obj in ("actual", "p16")},
                        "khist": g.tiles[var][T].khist.tolist(),
                        "best_choices": {obj: dict(zip(CHOICES, g.tiles[var][T].choice[obj].tolist())) for obj in ("actual", "p16")},
                    }
                    for var in VARIANTS
                }
                for T in tile_sizes
            },
            "exp_hist": {str(e): int(c) for e, c in enumerate(g.hist) if c},
        }
    for cls, la in lossy_by.items():
        summary["lossy"][cls] = {
            str(T): {
                str(m): {
                    key: {**lossy.percentiles(la.h[T][m][key]), "max": la.max[T][m][key]} for key in lossy.METRICS
                }
                for m in lossy.MANTISSA_BITS
            }
            for T in tile_sizes
        }

    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=int) + "\n")
    with open(out / "tensors.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(tensor_rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(tensor_rows)
    with open(out / "layers.csv", "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["layer", "tensors", "numel"] + [f"T{T}_best_{var}" for T in tile_sizes for var in VARIANTS])
        for lname in sorted((n for n in groups if n.startswith("L")), key=lambda n: int(n[1:])):
            g = groups[lname]
            w.writerow([lname[1:], g.tensors, g.numel] + [round(g.bpw(var, T), 4) for T in tile_sizes for var in VARIANTS])
    with open(out / "exp_hist.csv", "w", newline="") as f:
        cols = [n for n in ("linear", "attn", "linattn", "mlp", "experts", "ple_proj", "embed") if n in groups and groups[n].tensors]
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["exponent", "unbiased"] + cols)
        nz = np.flatnonzero(sum(groups[c].hist for c in cols))
        for e in nz:
            w.writerow([e, e - 127] + [int(groups[c].hist[e]) for c in cols])
    write_markdown(out / "summary.md", summary, groups, lin_lossy, tile_sizes)
    write_plots(out, groups, tile_sizes, T0, name)

    print(f"\nlocality gain at T={T0}: {v['gain']:.3f} bits/weight -> {v['band']}: {v['meaning']}")
    for T in tile_sizes:
        print(f"  T={T}: real {lin.bpw('real', T):.3f}  row-shuf {lin.bpw('shuf_row', T):.3f}  tensor-shuf {lin.bpw('shuf_tensor', T):.3f}")
    b = summary["groups"]["linear"]["baselines"]
    print(f"  Huffman per block {b['huffman_block']:.3f}  per tensor {b['huffman_tensor']:.3f}  ZipServ layout {b['window7']:.3f}  Unweight 4-bit mode {b['palette16_segments']:.3f}")
    for p in ("rank_fixed", "rank_best", "rank_unary", "rank_rice", "unary"):
        print(f"  {p} T={T0}: real {lin.bpw('real', T0, p):.3f}  tensor-shuf {lin.bpw('shuf_tensor', T0, p):.3f}")
    print(f"wrote {out} in {summary['seconds'] / 60:.1f} min, peak RSS {summary['peak_rss_gb']:.1f} GB")


if __name__ == "__main__":
    main()
