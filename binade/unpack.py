"""Reconstruct BF16 weights from a Binade model (DESIGN.md 10).

    binade unpack <packed-dir> --out DIR

Decodes in row chunks, so memory stays bounded: format 1 with running word and escape
cursors, format 2 (Rice) from the per-row word offsets, on the GPU by default (the numpy
reference decoder runs at about 20 M weights/s, some 10 minutes for a 12B model).
"""

import argparse
import json
import math
import shutil
from pathlib import Path

import numpy as np

from . import bf16, rice, st
from .format import TILE, decode_tiles
from .pack import ALL_SUFFIXES, CHUNK


class PackedModel:
    def __init__(self, path, gpu: bool = True):
        self.path = Path(path)
        self.gpu = gpu
        self.infos = {t.name: t for t in st.list_tensors(self.path)}
        self.packed = sorted(
            n[: -len(".raw")] for n in self.infos if n.endswith(".raw") and (f"{n[:-4]}.idx" in self.infos or f"{n[:-4]}.bits" in self.infos)
        )

    def names(self) -> list[str]:
        """Original tensor names, in file order."""
        members = {f"{p}.{s}" for p in self.packed for s in ALL_SUFFIXES}
        out = []
        for n in self.infos:
            if n.endswith(".raw") and n[:-4] in self.packed:
                out.append(n[:-4])
            elif n not in members:
                out.append(n)
        return out

    def info(self, name: str) -> st.TensorInfo:
        return self.infos[f"{name}.raw"] if name in self.packed else self.infos[name]

    def rows(self, name: str, chunk: int = CHUNK):
        """Yield (r0, r1, uint16 [r1 - r0, K]) of a packed tensor's 2-D view."""
        if f"{name}.bits" in self.infos:
            yield from self._rice_rows(name, chunk)
            return
        raw = self.infos[f"{name}.raw"]
        table, meta, idx, esc = (st.read(self.infos[f"{name}.{s}"]) for s in ("table", "meta", "idx", "esc"))
        R, K = raw.rows, raw.shape[-1]
        per_row = math.ceil(K / TILE)
        step = max(1, chunk // K)
        wpos = epos = 0
        for r0 in range(0, R, step):
            r1 = min(R, r0 + step)
            m = meta[r0 * per_row : r1 * per_row]
            nw = 4 * int((m & 0x0F).astype(np.int64).sum())
            e, n = decode_tiles(m, idx[wpos : wpos + nw], esc[epos:], table)
            wpos += nw
            epos += n
            yield r0, r1, bf16.join(st.read(raw, (r0, r1)), e.reshape(r1 - r0, -1)[:, :K])
        if wpos != idx.size or epos != esc.size:
            raise ValueError(f"{name}: consumed {wpos}/{idx.size} words, {epos}/{esc.size} escapes")

    def _rice_rows(self, name: str, chunk: int):
        raw = self.infos[f"{name}.raw"]
        table, meta, words, roff = (st.read(self.infos[f"{name}.{s}"]) for s in ("table", "meta", "bits", "roff"))
        R, K = raw.rows, raw.shape[-1]
        per_row = math.ceil(K / TILE)
        if meta.size != R * per_row or roff.size != R:
            raise ValueError(f"{name}: {meta.size} tiles and {roff.size} row offsets for {R} x {K}")
        step = max(1, chunk // K)
        nxt = np.append(roff[1:].astype(np.int64), words.size)
        if self.gpu:
            import mlx.core as mx

            from .mlx.rice_kernel import rice_ranks

            mw = mx.array(words if words.size else np.zeros(1, np.uint32))
        for r0 in range(0, R, step):
            r1 = min(R, r0 + step)
            m = meta[r0 * per_row : r1 * per_row]
            if self.gpu:
                ranks, rend = rice_ranks(mx.array(m), mw, mx.array(roff[r0:r1]), K)
                ranks, rend = np.array(ranks)[:, :K], np.array(rend)
            else:
                ranks, end = rice.decode_ranks(m, words, roff[r0:r1], K)
                rend = (end + 31) >> 5
            if np.any(rend != nxt[r0:r1]):
                raise ValueError(f"{name}: rows {r0}..{r1} do not end where the next row starts")
            yield r0, r1, bf16.join(st.read(raw, (r0, r1)), table[ranks])

    def tensor(self, name: str) -> np.ndarray:
        """Whole tensor: uint16 BF16 bit patterns for packed weights, stored dtype otherwise."""
        if name not in self.packed:
            return st.read(self.infos[name])
        return np.concatenate([u for _, _, u in self.rows(name)]).reshape(self.info(name).shape)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("packed", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    pm = PackedModel(args.packed)
    args.out.mkdir(parents=True, exist_ok=True)
    by_file = {}
    for n in pm.names():
        by_file.setdefault(pm.info(n).file.name, []).append(n)
    for fname, names in by_file.items():
        w = st.Writer(args.out / fname, sum(len(n) + 160 for n in names) + 4096)
        for n in names:
            info = pm.info(n)
            if n in pm.packed:
                w.begin(n, "BF16", info.shape)
                for _, _, u in pm.rows(n):
                    w.write(u)
                w.end()
            else:
                w.add(n, st.read(info), info.dtype)
        w.close()
    config = json.loads((args.packed / "config.json").read_text())
    config.pop("quantization_config", None)
    (args.out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    for f in args.packed.iterdir():
        if f.is_file() and not f.name.endswith(".safetensors") and f.name not in ("config.json", "binade_pack.json", "model.safetensors.index.json"):
            shutil.copy(f, args.out / f.name)


if __name__ == "__main__":
    main()
