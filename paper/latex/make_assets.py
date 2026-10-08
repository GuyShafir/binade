#!/usr/bin/env python3
"""Tables (tables/*.tex) and figures (figures/*.pdf) for the paper, generated from results/.

    python paper/latex/make_assets.py

Every number in a table or figure comes from a committed result file; nothing is typed in.
"""

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
RES = REPO / "results"
OUT = Path(__file__).resolve().parent
MODELS = [("gemma-4-12B-it", "Gemma 4 12B"), ("gemma-4-E4B-it", "Gemma 4 E4B"), ("Llama-3.1-8B-Instruct", "Llama 3.1 8B"), ("Qwen3.5-9B", "Qwen3.5 9B")]
plt.rcParams.update({"font.size": 9, "font.family": "serif", "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})


def j(path):
    return json.loads((RES / path).read_text())


def write(name, text):
    (OUT / "tables" / name).write_text(text)


def table(cols, header, rows, caption=None):
    lines = [r"\begin{tabular}{" + cols + "}", r"\toprule", " & ".join(header) + r" \\", r"\midrule"]
    lines += [" & ".join(str(c) for c in r) + r" \\" for r in rows]
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def f3(x):
    return f"{x:.3f}"


def locality():
    rows = []
    for m, name in MODELS:
        s = j(f"{m}/summary.json")
        t = s["groups"]["linear"]["tiles"]["128"]
        real, row, ten = (t[v]["bpw"]["actual"]["best"] for v in ("real", "shuf_row", "shuf_tensor"))
        rows.append([name, f"{s['groups']['linear']['numel'] / 1e9:.2f}B", f3(real), f3(row), f3(ten), f3(ten - real)])
    write("locality.tex", table("lrrrrr", ["model", "linear weights", "per-tile best-of", "row-shuffled", "tensor-shuffled", "gain"], rows))


def mutual_info():
    rows = []
    for m, name in MODELS:
        md = (RES / m / "mutual_info.md").read_text().splitlines()
        line = next(l for l in md if l.startswith("| linear |"))
        c = [x.strip() for x in line.strip("|").split("|")]
        rows.append([name, c[2], c[5], c[6]])
    write("mutual_info.tex", table("lrrr", ["model", r"$H(E)$", r"$I(E;R)$", r"$I(E;C)$"], rows))


def sizes():
    rows = []
    for m, name in MODELS:
        s = j(f"{m}/summary.json")
        lin = s["groups"]["linear"]
        b, a = lin["baselines"], lin["tiles"]["128"]["real"]["bpw"]["actual"]
        rows.append([name, f3(b["entropy_tensor"]), f3(b["huffman_block"]), f3(a["rank_rice"]), f3(a["rank_best"]), f3(a["best"]), f3(b["window7"]), f3(b["palette16_segments"])])
    write("sizes.tex", table("lrrrrrrr", ["model", "entropy", "Huffman (block)", "Rice ranks", "fixed ranks", "tile palettes", "7-exp. window", "16-palette, 4-bit"], rows))


def files():
    rows = []
    for m, name in MODELS + [("gemma-4-26B-A4B-it", "Gemma 4 26B-A4B")]:
        p = RES / m / "roundtrip_v2.json"
        if not p.exists():
            continue
        r = j(f"{m}/roundtrip_v2.json")
        bits = sum(c["bits"] for c in r["classes"].values()) / sum(c["numel"] for c in r["classes"].values())
        rows.append([name, f"{r['input_bytes'] / 1e9:.2f}", f"{r['output_bytes'] / 1e9:.2f}", f"{r['output_bytes'] / r['input_bytes']:.1%}".replace("%", r"\%"), f3(bits), str(r["tensors_checked"]), str(r["mismatched_elements"])])
    write("files.tex", table("lrrrrrr", ["model", "BF16 GB", "Binade GB", "ratio", "bits/weight", "tensors", "mismatches"], rows))


def exactness():
    # Binade as it runs, Binade forced onto the fused GEMM for every prompt length, the order
    # ablation (two accumulators; its own run in order_ablation/, same BF16 reference) and 8-bit.
    labels = (("binade", "Binade"), ("binade_fused", "Binade, fused GEMM"), ("binade_reordered", "two accumulators"), ("q8", "8-bit"))
    rows = []
    for m, name in MODELS + [("gemma-4-26B-A4B-it", r"Gemma 4 26B-A4B$^\dagger$")]:  # windows only, streamed BF16 reference
        if not (RES / m / "exactness_scale.json").exists():
            continue
        v = dict(j(f"{m}/exactness_scale.json")["variants"])
        if (RES / m / "order_ablation" / "exactness_scale.json").exists():
            v.setdefault("binade_reordered", j(f"{m}/order_ablation/exactness_scale.json")["variants"]["binade_reordered"])
        first = True
        for key, label in labels:
            if key not in v:
                continue
            r = v[key]
            rows.append([name if first else "", label, f"{r['positions_bit_identical']:,} / {r['positions']:,}", f"{r['argmax_agreement']:.1%}".replace("%", r"\%"), f"{r['perplexity']:.2f} ({v['bf16']['perplexity']:.2f})", f"{r['generations_identical']} / {r['generations']}" if r["generations"] else "n/a", f"{r['tokens_identical']:.1%}".replace("%", r"\%") if r["generations"] else "n/a"])
            first = False
    write("exactness.tex", table("llrrrrr", ["model", "variant", "identical logits", "argmax agr.", "perplexity (BF16)", "identical gen.", "identical tokens"], rows))


def decode():
    rows = []
    for m, name in MODELS + [("gemma-4-26B-A4B-it", "Gemma 4 26B-A4B")]:
        d = j(f"{m}/decode_speed.json")
        med = d["median_tok_s"]
        bf = med.get("bf16")
        cell = lambda k: f"{med[k]:.1f}" if k in med else "n/a"
        rows.append([name, cell("bf16"), cell("binade_r4"), cell("q8"), f"{med['binade_r4'] / bf:.2f}" + r"$\times$" if bf else "n/a", f"{med['q8'] / bf:.2f}" + r"$\times$" if bf and "q8" in med else "n/a"])
    write("decode.tex", table("lrrrrr", ["model", "BF16", "Binade", "8-bit", "Binade/BF16", "8-bit/BF16"], rows))


def kernel():
    rows = []
    for path, label in (("kernel/gemma-4-12B-it.json", "M4 Max (Metal)"), ("kernel/gemma-4-12B-it_cuda-rtx-pro-6000.json", "RTX PRO 6000 (CUDA)")):
        p = RES / path
        if not p.exists():
            continue
        k = j(path)["per_token"]
        for m in ("bf16", "q8", "binade_v1", "binade_r4"):
            if m not in k["us"]:
                continue
            name = {"bf16": "BF16", "q8": "8-bit (lossy)", "binade_v1": "format 1 in place", "binade_r4": "R4"}[m]
            rows.append([label if m == "bf16" else "", name, f"{k['us'][m] / 1e3:.1f}", f"{k['bytes'][m] / k['bytes']['bf16']:.3f}", f"{k['gbs'][m]:.0f}", f"{k['efficiency'][m]:.1%}".replace("%", r"\%"), f"{k['speedup'][m]:.2f}" + r"$\times$"])
    write("kernel.tex", table("llrrrrr", ["GPU", "method", "ms/token", "bytes", "GB/s", "efficiency", "speedup"], rows))


def prefill():
    p = j("gemma-4-12B-it/prefill_bench.json")
    rows = [[str(M), f"{v['bf16']:.0f}", f"{v['r4_fused']:.0f}", f"{v['r4_rebuild']:.0f}", f"{v['r4_runtime']:.0f}", f"{v['bf16'] / v['r4_runtime']:.2f}" + r"$\times$"] for M, v in ((int(k), v) for k, v in p["per_prompt_ms"].items())]
    write("prefill.tex", table("rrrrrr", ["rows", "BF16 ms", "fused ms", "rebuild ms", "runtime ms", "runtime/BF16"], rows))
    fig, ax = plt.subplots(figsize=(3.4, 2.2))
    Ms = [int(k) for k in p["per_prompt_ms"]]
    for key, label, st in (("r4_runtime", "Binade runtime", "-o"), ("r4_rebuild", "rebuild + MLX GEMM", "--s"), ("r4_fused", "fused R4 GEMM", ":^")):
        ax.plot(Ms, [p["per_prompt_ms"][str(M)]["bf16"] / p["per_prompt_ms"][str(M)][key] for M in Ms], st, ms=3, label=label)
    ax.axhline(1, color="0.5", lw=0.8)
    ax.set_xscale("log", base=2); ax.set_xlabel("prompt rows"); ax.set_ylabel("speed vs BF16"); ax.legend(frameon=False, fontsize=7)
    fig.tight_layout(); fig.savefig(OUT / "figures" / "prefill.pdf", metadata={"CreationDate": None}); plt.close(fig)


def stats():
    # The statistical basis in four panels: (a) exponent distribution against an iid Gaussian,
    # (b) a patch of stored ranks against the same values shuffled, (c) what position is worth,
    # (d) bits per weight of every layout against the entropy floor.
    import math

    import numpy as np
    from matplotlib.colors import BoundaryNorm, ListedColormap

    fig = plt.figure(figsize=(6.5, 5.6))
    top = fig.add_gridspec(1, 2, wspace=0.22, width_ratios=[1, 1.05], left=0.085, right=0.985, top=0.93, bottom=0.6)
    bot = fig.add_gridspec(1, 2, wspace=1.05, width_ratios=[1, 0.95], left=0.085, right=0.985, top=0.46, bottom=0.08)
    colors = ["#1b6ca8", "#d1495b", "#66a182", "#edae49"]

    # (a) share of weights per binade, aligned at each model's most common binade
    ax = fig.add_subplot(top[0])
    top15 = None
    for (m, name), c in zip(MODELS, colors):
        rows = [r for r in csv.DictReader(open(RES / m / "exp_hist.csv")) if int(r["linear"]) > 0]
        tot = sum(int(r["linear"]) for r in rows)
        mode = max(rows, key=lambda r: int(r["linear"]))
        xs = [int(r["unbiased"]) - int(mode["unbiased"]) for r in rows]
        ys = [int(r["linear"]) / tot for r in rows]
        ax.semilogy(xs, ys, color=c, lw=1, label=name)
        if m == "gemma-4-12B-it":
            top15 = sorted(xs, key=lambda x: -ys[xs.index(x)])[:15]
    # iid Gaussian: P(binade k) = 2 [Phi(2^(k+1) / s) - Phi(2^k / s)], scale averaged over a binade
    phi = lambda z: 0.5 * (1 + math.erf(z / math.sqrt(2)))
    ks = list(range(-40, 6))
    g = np.zeros(len(ks))
    for f in np.linspace(0, 1, 32, endpoint=False):
        s = 2.0**f
        g += [2 * (phi(2.0 ** (k + 1) / s) - phi(2.0**k / s)) for k in ks]
    g /= g.sum()
    gm = ks[int(np.argmax(g))]
    keep = g > 1e-12
    ax.semilogy(np.array(ks)[keep] - gm, g[keep], "k:", lw=1.2, label="iid Gaussian")
    lo, hi = min(top15), max(top15)
    ax.axvspan(-30, lo - 0.5, color="0.9", lw=0)
    ax.axvspan(hi + 0.5, 6, color="0.9", lw=0)
    ax.text(lo - 1, 1e-2, "escapes", fontsize=6.5, ha="right", color="0.35")
    ax.text(hi + 1.1, 3e-2, "escapes", fontsize=6.5, ha="left", va="top", color="0.35", rotation=90)
    ax.set_xlim(-30, 6); ax.set_ylim(1e-10, 1)
    ax.set_xlabel("binade relative to the most common one", fontsize=7.5); ax.set_ylabel("share of weights", fontsize=7.5)
    ax.tick_params(labelsize=7); ax.legend(frameon=False, fontsize=6.5, loc="lower right", bbox_to_anchor=(0.8, 0.0))
    r4 = j("gemma-4-12B-it/layout_stats.json")["r4"]
    ax.set_title("(a) exponent distribution", fontsize=8, loc="left")

    # (b) stored ranks of a weight patch, and the same tensor with positions shuffled
    pt = j("gemma-4-12B-it/exponent_patch.json")
    sub = top[1].subgridspec(3, 1, hspace=0.5)
    cmap = ListedColormap([plt.cm.viridis(x) for x in np.linspace(0, 1, 15)] + ["#d1495b"])
    norm = BoundaryNorm(np.arange(-0.5, 16.5), 16)
    for q, (key, label) in enumerate((("real", "as stored"), ("row_shuffled", "shuffled within each row"), ("tensor_shuffled", "shuffled across the tensor"))):
        a = fig.add_subplot(sub[q])
        im = a.imshow(np.minimum(np.array(pt[key]), 15), cmap=cmap, norm=norm, aspect="auto", interpolation="nearest")
        a.set_xticks([]); a.set_yticks([])
        if q == 0:
            a.set_title("(b) ranks of one weight patch: " + label, fontsize=8, loc="left")
        else:
            a.set_title(label, fontsize=6.5, loc="left", pad=2)
    pos = a.get_position()
    cax = fig.add_axes([pos.x0, pos.y0 - 0.035, pos.width, 0.012])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal", ticks=[0, 3, 6, 9, 12, 15]); cb.ax.tick_params(labelsize=6); cb.ax.set_xticklabels(["0", "3", "6", "9", "12", "15+"])
    cb.set_label("rank (0: most frequent exponent; 15 and up: escape)", fontsize=6.5, labelpad=1)

    # (c) what position is worth: tile cost after shuffling minus as stored, against the gap
    # the locality hypothesis needed (tile palettes minus per-tensor entropy)
    ax = fig.add_subplot(bot[0])
    Ts = [32, 64, 128]
    floor = 1e-4
    need = []
    for (m, name), c in zip(MODELS, colors):
        lin = j(f"{m}/summary.json")["groups"]["linear"]
        t = lin["tiles"]
        real = [t[str(T)]["real"]["bpw"]["actual"]["best"] for T in Ts]
        for key, ls in (("shuf_tensor", "-"), ("shuf_row", "--")):
            gain = [max(t[str(T)][key]["bpw"]["actual"]["best"] - r, floor) for T, r in zip(Ts, real)]
            ax.semilogy(Ts, gain, ls, color=c, lw=1, marker="o", ms=2.5, label=name if key == "shuf_tensor" else None)
        need.append(real[-1] - lin["baselines"]["entropy_tensor"])
    iid = j("iid_model.json")["models"]
    for key, mk, lab in (("iid Gaussian", "x", "iid Gaussian"), ("Gaussian, per-row log-normal scale (sigma 0.3)", "+", "Gaussian, row scales")):
        ax.semilogy(Ts, [max(iid[key][f"T{T}_best_shuffled"] - iid[key][f"T{T}_best_real"], floor) for T in Ts], "k" + mk, ms=4, label=lab)
    ax.axhline(sum(need) / len(need), color="0.5", lw=1)
    ax.text(33, max(need) * 1.3, f"needed: {min(need):.2f} to {max(need):.2f}", fontsize=6.3, color="0.3")
    ax.axhline(floor, color="0.7", lw=0.6, ls=":")
    ax.text(40, floor * 1.4, "0.0001 or less", fontsize=5.5, color="0.4")
    ax.set_xscale("log", base=2); ax.set_xticks(Ts); ax.set_xticklabels([str(T) for T in Ts])
    ax.set_xlabel("tile size T (weights along K)", fontsize=7.5); ax.set_ylabel("bits per weight saved by position", fontsize=7.5)
    ax.tick_params(labelsize=7)
    ax.legend(frameon=False, fontsize=6, loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0.0), columnspacing=1)
    ax.set_ylim(1e-6, 4)
    ax.set_title("(c) what an exponent's position is worth", fontsize=8, loc="left")

    # (d) bits per weight of each layout, every model (Gemma 4 12B filled), against the floor
    ax = fig.add_subplot(bot[1])
    labels = [
        ("entropy_tensor", "entropy, per tensor (floor)"),
        ("huffman_block", "Huffman per block (DFloat11's granularity)"),
        ("rank_rice", "Binade format 2: Rice ranks (disk)"),
        ("rank_best", "Binade format 1: fixed-width ranks"),
        ("window7", "7-exponent window (ZipServ layout)"),
        ("best", "per-tile palettes, T = 128"),
        ("palette16_segments", "16-exponent palette, 4-bit (Unweight)"),
        ("r4", "Binade R4 runtime layout (memory)"),
    ]
    for y, (key, label) in enumerate(labels):
        for (m, name), c in zip(MODELS, colors):
            lin = j(f"{m}/summary.json")["groups"]["linear"]
            if key == "r4":
                if m != "gemma-4-12B-it":
                    continue
                v = r4["bits_per_weight"]
            elif key in lin["baselines"]:
                v = lin["baselines"][key]
            else:
                v = lin["tiles"]["128"]["real"]["bpw"]["actual"][key]
            ax.plot([v], [y], "o", ms=4 if m == "gemma-4-12B-it" else 3, mfc=c if m == "gemma-4-12B-it" else "none", mec=c, mew=0.8)
    ax.axvline(j("gemma-4-12B-it/summary.json")["groups"]["linear"]["baselines"]["entropy_tensor"], color="0.6", ls=":", lw=1)
    ax.set_yticks(range(len(labels))); ax.set_yticklabels([l for _, l in labels], fontsize=6.5)
    ax.set_xlim(10.4, 12.3); ax.tick_params(axis="x", labelsize=7)
    ax.set_xlabel("bits per weight, linear layers", fontsize=7.5)
    ax.text(12.28, -0.3, "BF16: 16 bits", fontsize=6.5, ha="right", color="0.3")
    ax.set_title("(d) bits per weight by layout", fontsize=8, loc="left")
    ax.invert_yaxis()
    fig.savefig(OUT / "figures" / "stats.pdf", metadata={"CreationDate": None}); plt.close(fig)


CUDA_MODELS = [("Llama-3.1-8B-Instruct", "Llama 3.1 8B"), ("Qwen3.5-9B", "Qwen3.5 9B"), ("gemma-4-12B-it", "Gemma 4 12B"), ("gemma-4-26B-A4B-it", "Gemma 4 26B-A4B"), ("gemma-4-31B-it", "Gemma 4 31B")]


def exactness_cuda():
    # RTX PRO 6000: exactness at scale against resident BF16, 8-bit for reference, decoding speed and peak memory
    rows = []
    for m, name in CUDA_MODELS:
        if not (RES / "cuda" / m / "exactness_scale.json").exists():
            continue
        v = j(f"cuda/{m}/exactness_scale.json")["variants"]
        d = j(f"cuda/{m}/decode_speed.json")
        b, q = v["binade"], v["q8"]
        peak = {k: max(r["peak_gb"] for r in rr) for k, rr in d["runs"].items()}
        med = d["median_tok_s"]
        rows.append([name, f"{b['positions_bit_identical']:,} / {b['positions']:,}", f"{b['generations_identical']} / {b['generations']}", f"{q['positions_bit_identical']:,}", f"{q['argmax_agreement']:.1%}".replace("%", r"\%"), f"{med['binade_r4'] / med['bf16']:.2f}" + r"$\times$", f"{peak['binade_r4']:.1f} / {peak['bf16']:.1f}"])
    write("exactness_cuda.tex", table("lrrrrrr", ["model", "identical logits", "identical gen.", "8-bit ident.", "8-bit argmax", "decode vs BF16", "peak GB (BF16)"], rows))


def fit_cuda():
    # smaller GPUs where the BF16 checkpoint does not load
    rows = []
    for d, m, name, gpu in (("cuda-l4", "gemma-4-12B-it", "Gemma 4 12B", "L4"), ("cuda-a100", "gemma-4-26B-A4B-it", "Gemma 4 26B-A4B", "A100 40\\,GB")):
        f = RES / d / m / "fit_probe.json"
        if not f.exists():
            continue
        v = {r["variant"]: r for r in j(f"{d}/{m}/fit_probe.json")["variants"]}
        e = j(f"{d}/{m}/exactness_scale.json")["variants"]["binade"]
        dec = j(f"{d}/{m}/decode_speed.json")["median_tok_s"]
        mem = j(f"{d}/{m}/fit_probe.json")["device_memory_gb"]
        fc = RES / d / m / "fused_check" / "exactness_scale.json"  # the fused GEMMs forced on every prompt length
        forced = j(f"{d}/{m}/fused_check/exactness_scale.json")["variants"]["binade_fused"] if fc.exists() else None
        ident = f"{e['positions_bit_identical']:,} / {e['positions']:,}" + (f" ({forced['positions_bit_identical']:,})" if forced else "")
        rows.append([f"{gpu} ({mem:.1f}\\,GB)", name, "out of memory" if not v["bf16"]["ok"] else "runs", f"{v['binade_r4']['peak_gb']:.1f}\\,GB, {dec['binade_r4']:.1f} tok/s", ident, f"{v['q8']['peak_gb']:.1f}\\,GB, {dec['q8']:.1f} tok/s"])
    write("fit_cuda.tex", table("llllrl", ["GPU", "model", "BF16", "Binade: peak, decode", "identical logits (fused forced)", "8-bit: peak, decode"], rows))


def reference():
    # (model dir, label, [(dtype, device dir)]): the 26B does not fit the GPU in float32, so that row ran on the CPU
    runs = (
        ("gemma-4-E4B-it", "Gemma 4 E4B (native)", (("float32", "cuda"), ("bfloat16", "cuda"))),
        ("gemma-4-12B-it", "Gemma 4 12B (aliased)", (("float32", "cuda"), ("bfloat16", "cuda"))),
        ("gemma-4-26B-A4B-it", "Gemma 4 26B-A4B (native)", (("float32", "cpu"), ("bfloat16", "cuda"))),
    )
    rows = []
    for m, name, cases in runs:
        for i, (dt, dev) in enumerate(cases):
            if not (RES / dev / m / "reference_check.json").exists():
                continue
            r = j(f"{dev}/{m}/reference_check.json")["dtypes"][dt]
            rows.append([name if i == 0 else "", dt, dev.upper() if dev == "cpu" else "CUDA", f"{r['perplexity_reference']:.2f} / {r['perplexity_mlx']:.2f}", f"{r['argmax_agreement']:.2%}".replace("%", r"\%"), f"{r['nll_abs_diff_mean']:.4f}", f"{r['nll_abs_diff_max']:.3f}"])
    write("reference.tex", table("lllrrrr", ["model", "dtype", "device", "perplexity (ref. / MLX)", "argmax agr.", "mean $|\\Delta|$", "max $|\\Delta|$"], rows))


if __name__ == "__main__":
    for f in (locality, mutual_info, sizes, files, exactness, exactness_cuda, fit_cuda, decode, kernel, prefill, stats, reference):
        f()
        print("ok", f.__name__)
