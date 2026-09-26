#!/usr/bin/env python3
"""Hash a cached snapshot's safetensors files and compare with the Hub's LFS sha256.

    python scripts/verify_snapshot.py <repo-id> --revision REV [--same-as OTHER_REPO] --out FILE

--same-as also compares against another repo's files (e.g. an ungated mirror against
the gated original, whose metadata is readable without access to the weights).
"""

import argparse
import hashlib
import json
from pathlib import Path

from huggingface_hub import HfApi

from binade import st


def sha256(path: Path, block: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(block):
            h.update(chunk)
    return h.hexdigest()


def hub_hashes(repo: str, revision: str | None = None) -> dict:
    info = HfApi().model_info(repo, revision=revision, files_metadata=True)
    return {s.rfilename: s.lfs.sha256 for s in info.siblings if s.rfilename.endswith(".safetensors") and s.lfs}, info.sha


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("repo")
    ap.add_argument("--revision", required=True)
    ap.add_argument("--same-as")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    snap = st.resolve(args.repo, args.revision)
    local = {p.name: sha256(p) for p in st.shard_files(snap)}
    own, own_rev = hub_hashes(args.repo, args.revision)
    result = {"repo": args.repo, "revision": own_rev, "files": local, "matches_hub": local == own}
    if args.same_as:
        ref, ref_rev = hub_hashes(args.same_as)
        result["same_as"] = {"repo": args.same_as, "revision": ref_rev, "identical": local == ref}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps(result, indent=1))
    if not result["matches_hub"] or (args.same_as and not result["same_as"]["identical"]):
        raise SystemExit("hash mismatch")


if __name__ == "__main__":
    main()
