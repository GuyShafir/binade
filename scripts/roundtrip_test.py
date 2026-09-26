#!/usr/bin/env python3
"""Phase 2 round trip (DESIGN.md 10): every tensor of a packed model equals the original, bit for bit.

    python scripts/roundtrip_test.py <original model-dir | cached-hf-repo-id> <packed-dir> [--revision REV]

Streams both models one tensor (and row chunk) at a time. Writes roundtrip.json and
roundtrip.md (format 1) or roundtrip_v2.* (format 2) to results/<model>/ and exits
non-zero on any mismatch.
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from binade import st
from binade.classify import classify
from binade.pack import spans
from binade.unpack import PackedModel

REPO = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("original")
    ap.add_argument("packed", type=Path)
    ap.add_argument("--revision")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    t0 = time.perf_counter()
    src = st.resolve(args.original, args.revision)
    source = st.provenance(src)
    out = args.out or REPO / "results" / source.get("repo_id", src.name).split("/")[-1]
    out.mkdir(parents=True, exist_ok=True)

    pm = PackedModel(args.packed)
    orig = {i.name: i for i in st.list_tensors(src)}
    missing = sorted(set(orig) - set(pm.names()))
    extra = sorted(set(pm.names()) - set(orig))
    mismatches, checked, classes = 0, 0, {}
    report = json.loads((args.packed / "binade_pack.json").read_text())
    for name, info in orig.items():
        if name in missing:
            continue
        if name in pm.packed:
            bad = 0
            for (r0, r1, u), span in zip(pm.rows(name), spans(info)):
                assert (r0, r1) == span
                bad += int(np.count_nonzero(st.read(info, span) != u))
            k = classify(name, info.shape, info.dtype)
            c = classes.setdefault(k.cls, {"tensors": 0, "numel": 0, "bits": 0})
            c["tensors"] += 1
            c["numel"] += info.numel
            c["bits"] += report["tensors"][name]["bits"]
        else:
            p = pm.info(name)
            same = (p.dtype, p.shape) == (info.dtype, info.shape) and np.array_equal(st.read(p), st.read(info))
            bad = 0 if same else info.numel
        if bad:
            print(f"MISMATCH {name}: {bad} elements", flush=True)
        mismatches += bad
        checked += 1

    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=no", "--", ".", ":!results"], cwd=REPO, capture_output=True, text=True).stdout.strip())
    result = {
        "source": source,
        "packed": st.display_path(args.packed),
        "git": {"commit": commit, "dirty": dirty},
        "pack_git": report["git"],
        "binade": report.get("binade", report.get("xp")),
        "tensors_checked": checked,
        "tensors_packed": len(pm.packed),
        "missing": missing,
        "extra": extra,
        "mismatched_elements": mismatches,
        "input_bytes": report["input_bytes"],
        "output_bytes": report["output_bytes"],
        "classes": {k: {**v, "bits_per_weight": v["bits"] / v["numel"]} for k, v in classes.items()},
        "pack_seconds": report["seconds"],
        "check_seconds": time.perf_counter() - t0,
    }
    version = str(result["binade"].get("version", "1"))
    stem = "roundtrip" if version == "1" else f"roundtrip_v{version}"
    (out / f"{stem}.json").write_text(json.dumps(result, indent=1) + "\n")
    ok = mismatches == 0 and not missing and not extra
    lines = [
        f"# Binade format {version} round trip: {source.get('repo_id', src.name)}",
        "",
        f"Packed with `binade/pack.py` @ `{report['git'][:7]}`, checked with `scripts/roundtrip_test.py` @ `{commit[:7]}{' (dirty)' if dirty else ''}`.",
        "",
        f"**{'PASS' if ok else 'FAIL'}**: {checked} tensors compared ({len(pm.packed)} packed), {mismatches} mismatched elements, "
        f"{len(missing)} missing, {len(extra)} extra.",
        "",
        f"Checkpoint {result['input_bytes'] / 1e9:.2f} GB -> {result['output_bytes'] / 1e9:.2f} GB "
        f"({result['output_bytes'] / result['input_bytes']:.1%}). Pack {report['seconds'] / 60:.1f} min, check {result['check_seconds'] / 60:.1f} min.",
        "",
        "| class | tensors | weights | bits/weight (file) |",
        "|---|---|---|---|",
    ]
    for k, v in result["classes"].items():
        lines.append(f"| {k} | {v['tensors']} | {v['numel'] / 1e9:.3f}B | {v['bits_per_weight']:.3f} |")
    (out / f"{stem}.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
