#!/usr/bin/env python3
"""The repository's social preview and README header, 1280 x 640, drawn from committed results:
the stored exponent ranks of a real Gemma 4 12B weight patch (exponent_patch.json) and the 12B
checkpoint sizes (roundtrip_v2.json), under the 16-square "b" mark.

    python scripts/social_preview.py [--out docs/social-preview.png]

The fonts (Bricolage Grotesque, JetBrains Mono, Kalam; SIL Open Font License) are downloaded from
the google/fonts repository on first use and cached in ~/.cache/binade/fonts.
"""

import argparse
import json
import urllib.request
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.font_manager import FontProperties
from matplotlib.patches import FancyBboxPatch, PathPatch
from matplotlib.textpath import TextPath
from matplotlib.transforms import Affine2D

REPO = Path(__file__).resolve().parents[1]
RES = REPO / "results" / "gemma-4-12B-it"
CACHE = Path.home() / ".cache" / "binade" / "fonts"
GF = "https://github.com/google/fonts/raw/main/ofl/"
FONTS = {
    "display": (GF + "bricolagegrotesque/BricolageGrotesque%5Bopsz,wdth,wght%5D.ttf", {"wght": 800, "opsz": 96, "wdth": 100}),
    "text": (GF + "bricolagegrotesque/BricolageGrotesque%5Bopsz,wdth,wght%5D.ttf", {"wght": 500, "opsz": 24, "wdth": 100}),
    "mono": (GF + "jetbrainsmono/JetBrainsMono%5Bwght%5D.ttf", {"wght": 700}),
    "hand": (GF + "kalam/Kalam-Regular.ttf", None),
}
W, H = 1280, 640
NIGHT, INK, MUTED, YELLOW, CORAL = "#160A2E", "#F5F1FF", "#C9BEE6", "#FDE725", "#FF6B5B"
# The logo: 16 squares of real weight exponents forming a lowercase b, (row, column, color),
# the coral one an escape value at the top of the stem.
MARK = [(0, 0, CORAL), (1, 0, "#481b6d"), (2, 0, "#440154"), (2, 1, "#46327e"), (2, 2, "#21918c"), (2, 3, "#4ac16d"),
        (3, 0, "#440154"), (3, 4, "#1fa187"), (4, 0, "#481b6d"), (4, 4, "#fde725"), (5, 0, "#3f4788"), (5, 4, "#2e6e8e"),
        (6, 0, "#440154"), (6, 1, "#365c8d"), (6, 2, "#3f4788"), (6, 3, "#481b6d")]


def font(name: str) -> Path:
    url, axes = FONTS[name]
    path = CACHE / f"{name}.ttf"
    if not path.exists():
        from fontTools.ttLib import TTFont
        from fontTools.varLib.instancer import instantiateVariableFont

        CACHE.mkdir(parents=True, exist_ok=True)
        raw = CACHE / url.rsplit("/", 1)[1]
        if not raw.exists():
            urllib.request.urlretrieve(url, raw)
        f = TTFont(raw)
        (instantiateVariableFont(f, axes) if axes else f).save(path)
    return path


def px(size: float) -> float:
    return size * 72 / 100  # pixels to points at dpi 100


def square(ax, x, y, s, color, r=0.12):
    ax.add_patch(FancyBboxPatch((x, y), s, s, boxstyle=f"round,pad=0,rounding_size={s * r}", fc=color, ec="none"))


def wordmark(ax, x, baseline, size, color):
    # "binade" whose i-dot is a coral escape square
    from fontTools.ttLib import TTFont

    path = font("display")
    tt = TTFont(path)
    cmap, em = tt.getBestCmap(), tt["head"].unitsPerEm
    adv = lambda ch: tt["hmtx"][cmap[ord(ch)]][0] / em * size
    flip = Affine2D().scale(1, -1).translate(x, baseline)
    tp = TextPath((0, 0), "bınade", size=size, prop=FontProperties(fname=path))
    ax.add_patch(PathPatch(flip.transform_path(tp), fc=color, ec="none"))
    g = tt["glyf"][cmap[0x131]]
    stem = x + adv("b") + (g.xMin + g.xMax) / 2 / em * size
    top = tt["glyf"][cmap[ord("b")]].yMax / em * size
    dot = 0.2 * size
    square(ax, stem - dot / 2, baseline - top, dot, CORAL, r=0.15)
    return x + sum(adv(c) for c in "bınade")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=REPO / "docs" / "social-preview.png")
    args = ap.parse_args()
    patch = np.array(json.loads((RES / "exponent_patch.json").read_text())["real"])
    rt = json.loads((RES / "roundtrip_v2.json").read_text())

    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.axis("off")
    fig.patch.set_facecolor(NIGHT)

    # the band of real exponent ranks: viridis for the 15 coded ranks (square-root spaced, since most
    # weights hold the first few), coral for escapes
    rows, cols = 24, 128
    cell = W / cols
    y0 = H - rows * cell
    cmap = ListedColormap([plt.cm.viridis((r / 14) ** 0.5) for r in range(15)] + [CORAL])
    ax.imshow(np.minimum(patch[:rows, :cols], 15), cmap=cmap, norm=BoundaryNorm(np.arange(-0.5, 16.5), 16),
              extent=(0, W, H, y0), interpolation="nearest", zorder=1)
    gap = px(1.6)
    ax.vlines(np.arange(cols + 1) * cell, y0, H, color=NIGHT, lw=gap, zorder=2)
    ax.hlines(y0 + np.arange(rows + 1) * cell, 0, W, color=NIGHT, lw=gap, zorder=2)
    fade = np.zeros((90, 1, 4))
    fade[..., :3] = matplotlib.colors.to_rgb(NIGHT)
    fade[..., 3] = np.linspace(1, 0, 90)[:, None] ** 1.5
    ax.imshow(fade, extent=(0, W, y0 + 90, y0 - 1), aspect="auto", zorder=3)

    # the mark and the wordmark
    s, g = 21, 3.6
    mx, my = 84, 66
    for r, c, color in MARK:
        square(ax, mx + c * (s + g), my + r * (s + g), s, color)
    wordmark(ax, mx + 5 * (s + g) + 30, my + 7 * (s + g) - g, 132, INK)

    disp, text, mono, hand = (FontProperties(fname=font(n)) for n in ("display", "text", "mono", "hand"))
    ax.text(84, 310, "Lossless LLM weights.", fontproperties=disp, fontsize=px(54), color=INK, va="baseline")
    ax.text(84, 372, "Outputs bit-identical.", fontproperties=disp, fontsize=px(54), color=YELLOW, va="baseline")

    # pip install, on a strip of label tape
    ax.text(818, 124, "pip install binade", fontproperties=mono, fontsize=px(34), color=NIGHT, rotation=3, va="center",
            bbox=dict(boxstyle="round,pad=0.45,rounding_size=0.25", fc=YELLOW, ec="none"))
    gb = lambda b: f"{b / 1e9:.2f} GB"
    ax.text(822, 226, f"Gemma 4 12B: {gb(rt['input_bytes'])} → {gb(rt['output_bytes'])}", fontproperties=text, fontsize=px(25), color=INK, va="baseline")
    ax.text(822, 264, "Apple Silicon and NVIDIA GPUs", fontproperties=text, fontsize=px(25), color=MUTED, va="baseline")

    # a handwritten note on the band
    ax.text(1010, 336, "real Gemma 4 weights,\none square per exponent", fontproperties=hand, fontsize=px(27), color=CORAL,
            rotation=-4, ha="center", va="center", linespacing=1.0)
    ax.annotate("", xy=(1150, y0 + 38), xytext=(1128, 372), arrowprops=dict(arrowstyle="-|>,head_width=0.35,head_length=0.7",
                color=CORAL, lw=2.4, connectionstyle="arc3,rad=-0.35"), zorder=4)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=100, facecolor=NIGHT, metadata={"Software": None})
    plt.close(fig)
    print(args.out)


if __name__ == "__main__":
    main()
