"""Binade packer: BF16 safetensors model -> Binade model (DESIGN.md 10, 6, 7).

    binade pack <model-dir | cached-hf-repo-id> --out DIR [--format 2|1] [--revision REV] [--classes linear,embed]

Streams one tensor at a time. A packed weight `name` becomes name.raw (sign + mantissa,
original shape), name.table and the exponent coding: format 1 (binade/format.py)
name.meta, name.idx, name.esc; format 2, Rice (binade/rice.py), name.meta, name.bits,
name.roff. Everything else is copied byte for byte. Shard names follow the input, so the
input index maps one to one.
"""

import argparse
import json
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np

from . import bf16, st
from .classify import classify
from . import rice
from .format import TILE, encode, rank_table

VERSION = "2"  # default format: Rice (smaller, faster to load); "1" for the slow runtime
CHUNK = 1 << 24  # weights per read
PARTS = {"1": ("raw", "table", "meta", "idx", "esc"), "2": ("raw", "table", "meta", "bits", "roff")}
SUFFIXES = PARTS[VERSION]
ALL_SUFFIXES = tuple(dict.fromkeys(PARTS["1"] + PARTS["2"]))


def quant(version: str = VERSION) -> dict:
    return {"quant_method": "binade", "version": version, "tile": TILE}


QUANT = quant()
REPO = Path(__file__).resolve().parents[1]


def packable(info: st.TensorInfo, classes) -> bool:
    k = classify(info.name, info.shape, info.dtype)
    return info.dtype == "BF16" and len(info.shape) in (2, 3) and k.cls in classes


def spans(info: st.TensorInfo, chunk: int = CHUNK):
    step = max(1, chunk // info.shape[-1])
    return [(r, min(info.rows, r + step)) for r in range(0, info.rows, step)]


def pack_tensor(w: st.Writer, info: st.TensorInfo, chunk: int = CHUNK, version: str = VERSION) -> dict:
    hist = np.zeros(256, np.int64)
    for span in spans(info, chunk):
        hist += np.bincount(bf16.exponent(st.read(info, span)).ravel(), minlength=256)
    table, rank = rank_table(hist)
    code = encode if version == "1" else rice.encode
    parts = []
    w.begin(info.name + ".raw", "U8", info.shape)
    for span in spans(info, chunk):
        u = st.read(info, span)
        w.write(bf16.raw(u))
        parts.append(code(bf16.exponent(u), table, rank))
    w.end()
    w.add(info.name + ".table", table)
    out = {"numel": info.numel, "distinct": int((hist > 0).sum())}
    if version == "1":
        meta, idx, esc = (np.concatenate([getattr(p, k) for p in parts]) for k in ("meta", "idx", "esc"))
        w.add(info.name + ".meta", meta)
        w.add(info.name + ".idx", idx)
        w.add(info.name + ".esc", esc)
        bits = 8 * info.numel + 8 * table.size + 8 * meta.size + 32 * idx.size + 8 * esc.size
        return {**out, "ntiles": int(meta.size), "escapes": int(esc.size), "bits": bits}
    base = np.cumsum([0] + [p.bits.size for p in parts[:-1]])
    meta = np.concatenate([p.meta for p in parts])
    words = np.concatenate([p.bits for p in parts])
    roff = np.concatenate([p.roff.astype(np.uint64) + int(b) for p, b in zip(parts, base)])
    if words.size >= 1 << 32:
        raise ValueError(f"{info.name}: {words.size} words overflow uint32 row offsets")
    w.add(info.name + ".meta", meta)
    w.add(info.name + ".bits", words)
    w.add(info.name + ".roff", roff.astype(np.uint32))
    bits = 8 * info.numel + 8 * table.size + 8 * meta.size + 32 * words.size + 32 * roff.size
    return {**out, "ntiles": int(meta.size), "bits": bits}


def copy_tensor(w: st.Writer, info: st.TensorInfo, chunk: int = CHUNK) -> None:
    w.begin(info.name, info.dtype, info.shape)
    if len(info.shape) < 2:
        w.write(st.read(info))
    else:
        for span in spans(info, chunk):
            w.write(st.read(info, span))
    w.end()


def _reserve(infos, classes, metadata) -> int:
    n = sum((len(i.name) + 160) * (len(SUFFIXES) if packable(i, classes) else 1) for i in infos)  # same count in both formats
    return n + len(json.dumps(metadata)) + 4096


def pack_model(src: Path, out: Path, classes=("linear", "embed"), chunk: int = CHUNK, log=print, version: str = VERSION) -> dict:
    if version not in PARTS:
        raise ValueError(f"unknown Binade format {version!r}")
    t0 = time.perf_counter()
    out.mkdir(parents=True, exist_ok=True)
    infos = st.list_tensors(src)
    need = sum(i.nbytes for i in infos)
    free = shutil.disk_usage(out).free
    if free < need:
        raise OSError(f"{out}: {free / 1e9:.1f} GB free, packing may need up to {need / 1e9:.1f} GB")
    weight_map, tensors = {}, {}
    for shard in st.shard_files(src):
        shard_infos = [i for i in infos if i.file == shard]
        packed = {i.name: list(i.shape) for i in shard_infos if packable(i, classes)}
        metadata = {"format": "binade", "binade_version": version, "binade_tile": str(TILE), "binade_tensors": json.dumps(packed)}
        w = st.Writer(out / shard.name, _reserve(shard_infos, classes, metadata), metadata)
        for i in shard_infos:
            ts = time.perf_counter()
            if i.name in packed:
                tensors[i.name] = pack_tensor(w, i, chunk, version)
                names = [f"{i.name}.{s}" for s in PARTS[version]]
                log(f"packed {i.name} {i.shape}: {tensors[i.name]['bits'] / i.numel:.3f} bits/weight, {time.perf_counter() - ts:.1f}s")
            else:
                copy_tensor(w, i, chunk)
                names = [i.name]
            weight_map.update({n: shard.name for n in names})
        w.close()

    config = json.loads((src / "config.json").read_text())
    config["quantization_config"] = quant(version)
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    for f in src.iterdir():
        if f.is_file() and not f.name.endswith(".safetensors") and f.name not in ("config.json", "model.safetensors.index.json"):
            shutil.copy(f, out / f.name)
    files = sorted(set(weight_map.values()))
    total = sum((out / f).stat().st_size for f in files)
    (out / "model.safetensors.index.json").write_text(json.dumps({"metadata": {"total_size": total}, "weight_map": weight_map}, indent=1) + "\n")

    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip()
    report = {
        "source": st.provenance(src),
        "binade": quant(version),
        "classes": list(classes),
        "git": commit,
        "seconds": time.perf_counter() - t0,
        "input_bytes": sum((src / f.name).stat().st_size for f in st.shard_files(src)),
        "output_bytes": total,
        "packed_numel": sum(t["numel"] for t in tensors.values()),
        "packed_bits": sum(t["bits"] for t in tensors.values()),
        "tensors": tensors,
    }
    (out / "binade_pack.json").write_text(json.dumps(report, indent=1) + "\n")
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--revision")
    ap.add_argument("--classes", default="linear,embed")
    ap.add_argument("--format", default=VERSION, choices=sorted(PARTS), help="1: fixed width with escapes; 2: Rice (smaller, decoded to R4 at load)")
    args = ap.parse_args()
    r = pack_model(st.resolve(args.model, args.revision), args.out, tuple(args.classes.split(",")), version=args.format)
    print(
        f"{r['input_bytes'] / 1e9:.2f} GB -> {r['output_bytes'] / 1e9:.2f} GB "
        f"({r['output_bytes'] / r['input_bytes']:.1%}); packed weights {r['packed_bits'] / r['packed_numel']:.3f} bits/weight; "
        f"{r['seconds'] / 60:.1f} min"
    )


if __name__ == "__main__":
    main()
