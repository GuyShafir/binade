#!/usr/bin/env python3
"""Publish a Binade-packed model to the Hugging Face Hub, with a model card built from results/.

    python scripts/hf_publish.py <packed-dir> --repo binade/gemma-4-12B-it [--name "Gemma 4 12B"] [--public] [--dry-run]

Refuses a pack other than the one the bit-exact round trip checked: binade_pack.json's commit
and output size must match results/<model>/roundtrip_v2.json, which must report no mismatched
values. Stages the pack in models/hf-staging/<repo name> (hard links), with binade_pack.json's
local source path made home-relative and the base model's README replaced by a model card whose
numbers all come from the committed result files, plus the license files the base model's
license asks a redistribution to carry. When results/<model>/snapshot_sha256.json shows the packed
snapshot is a byte-identical copy of another repository (unsloth's ungated Llama), the card and
license follow that original. Uploads with upload_large_folder (resumable),
private unless --public, then compares the SHA-256 of every file on the Hub with the local file.
--dry-run stops after staging, to read the card first. Needs `hf auth login` with write access.
"""

import argparse
import hashlib
import json
import os
import shutil
import urllib.request
from pathlib import Path

from binade import st

REPO = Path(__file__).resolve().parents[1]
RES = REPO / "results"
GITHUB = "https://github.com/GuyShafir/binade"
# Llama 3.1's license asks a redistribution for a copy of it, the acceptable use policy it
# incorporates, and this notice in a "Notice" text file. Meta's model repository is gated, so the
# texts come from Meta's public llama-models repository.
LLAMA31 = "https://raw.githubusercontent.com/meta-llama/llama-models/main/models/llama3_1/"
LLAMA31_NOTICE = "Llama 3.1 is licensed under the Llama 3.1 Community License, Copyright \u00a9 Meta Platforms, Inc. All Rights Reserved.\n"


def j(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def roundtrip(model: str) -> Path:
    """results/<model>/roundtrip_v2.json, or a GPU machine's (results/cuda*/<model>) for a model
    packed there."""
    paths = [RES / model / "roundtrip_v2.json", *sorted(RES.glob(f"cuda*/{model}/roundtrip_v2.json"))]
    return next((p for p in paths if p.exists()), paths[0])


def result_dirs(model: str) -> list[Path]:
    """The model's results directories with a logits exactness run: the Mac's first, then each
    NVIDIA GPU's."""
    return [d for d in [RES / model, *sorted(RES.glob(f"cuda*/{model}"))] if (d / "exactness_scale.json").exists()]


def verified(packed: Path, model: str) -> dict:
    """The pack's metadata, after checking it is the pack the round trip verified."""
    meta = j(packed / "binade_pack.json")
    path = roundtrip(model)
    rt = j(path)
    name = path.relative_to(REPO)
    if meta is None or rt is None:
        raise SystemExit(f"{packed}: no binade_pack.json, or no {name}")
    if rt["mismatched_elements"] or rt["missing"] or rt["extra"]:
        raise SystemExit(f"{name} does not report an exact round trip")
    if meta["git"] != rt["pack_git"] or meta["output_bytes"] != rt["output_bytes"]:
        raise SystemExit(f"{packed} is not the pack {name} checked ({meta['git'][:7]}, {meta['output_bytes']} bytes vs {rt['pack_git'][:7]}, {rt['output_bytes']})")
    return meta


def origin(model: str, meta: dict) -> str:
    """The repository the packed weights come from: the source, or the original it is a
    byte-identical copy of (results/<model>/snapshot_sha256.json, for this revision)."""
    snap = j(RES / model / "snapshot_sha256.json") or {}
    same = snap.get("same_as") or {}
    if snap.get("revision") == meta["source"].get("revision") and snap.get("matches_hub") and same.get("identical"):
        return same["repo"]
    return meta["source"]["repo_id"]


def license_files(orig: str, license: str, stage: Path) -> list[str]:
    """Copies into the stage the files the base model's license asks a redistribution to carry."""
    from huggingface_hub import hf_hub_download, list_repo_files

    if license == "llama3.1":
        for name in ("LICENSE", "USE_POLICY.md"):
            urllib.request.urlretrieve(LLAMA31 + name, stage / name)
        (stage / "NOTICE").write_text(LLAMA31_NOTICE)
        return ["LICENSE", "USE_POLICY.md", "NOTICE"]
    names = [n for n in list_repo_files(orig) if n in ("LICENSE", "LICENSE.txt", "LICENSE.md", "NOTICE")]
    for n in names:
        shutil.copyfile(hf_hub_download(orig, n), stage / n)
    return names


def text_only(packed: Path) -> str:
    """The limits line on modalities: the pack keeps every tensor, the runtime serves text."""
    names = json.loads((packed / "model.safetensors.index.json").read_text())["weight_map"]
    kinds = (("vision", ("vision", "visual")), ("audio", ("audio",)), ("multi-token prediction", ("mtp.",)))
    other = [m for m, keys in kinds if any(k in n for n in names for k in keys)]
    if not other:
        return "- Exactness was tested on text generation."
    return (f"- The checkpoint keeps every tensor of the original, the {' and '.join(other)} weights included (`binade unpack` restores them), "
            "but the runtime loads the text model through mlx-lm, so it serves text. Exactness was tested on text generation.")


def card(base: str, orig: str, repo: str, name: str, model: str, meta: dict, packed: Path, licensed: list[str]) -> str:
    from huggingface_hub import model_info

    cd = model_info(orig).card_data or {}
    llama = cd.get("license") == "llama3.1"
    license_link = "LICENSE" if "LICENSE" in licensed else cd.get("license_link")
    rt_path = roundtrip(model)
    rt = j(rt_path)
    bits = sum(c["bits"] for c in rt["classes"].values()) / sum(c["numel"] for c in rt["classes"].values())
    gb_in, gb_out = rt["input_bytes"] / 1e9, rt["output_bytes"] / 1e9
    dirs = result_dirs(model)
    link = lambda p: f"[{p.relative_to(RES).as_posix()}]({GITHUB}/blob/main/results/{p.relative_to(RES).as_posix()})"

    front = ["---", f"license: {cd.get('license', 'other')}"]
    if license_link:
        front.append(f"license_link: {license_link}")
    # No base_model: the Hub's only relations are adapter, merge, quantized and finetune, and a
    # lossless re-encoding is none of them (the card links the base model instead)
    front += ["library_name: binade", "pipeline_tag: text-generation"]  # the runtime serves text (mlx-lm's text model)
    # no "mlx" tag: the Hub reads it as "loads with mlx-lm" and shows an MLX panel, while these files need binade
    front += ["tags:", "- binade", "- lossless-compression", "- bf16", "- apple-silicon", "- cuda", "---", ""]

    copy = f" The files packed were the copy at [{base}](https://huggingface.co/{base}), byte for byte the same weights." if orig != base else ""
    out = front + [
        f"# {name}, losslessly compressed with Binade",
        "",
        *(["**Built with Llama.**", ""] if llama else []),
        f"[{orig}](https://huggingface.co/{orig}) packed with [Binade]({GITHUB}): **{gb_out:.2f} GB instead of {gb_in:.2f} GB** "
        f"({bits:.3f} bits per weight instead of 16 in the packed tensors), every weight restored bit for bit, and **outputs bit-identical to the BF16 model**, not just close.{copy}",
        "",
        "Binade codes each weight's exponent by its frequency rank and keeps the sign and mantissa as they are. Its kernels decode the weights "
        "inside the matrix products and add the products in exactly the order the framework's own BF16 kernels do, so the logits and generated "
        f"tokens match the original. The [paper]({GITHUB}/blob/main/paper/binade.pdf) explains how.",
        "",
        "## Exactness",
        "",
        "| check | result |",
        "|---|---|",
        f"| every tensor unpacked and compared with the original checkpoint | {rt['tensors_checked']} tensors, {rt['mismatched_elements']} values differ ({link(rt_path.with_suffix('.md'))}) |",
    ]
    for i, d in enumerate(dirs):
        c = j(d / "exactness_scale.json")
        b = c["variants"]["binade"]
        fit = j(d / "fit_probe.json")
        oom = fit and not {r["variant"]: r for r in fit["variants"]}["bf16"]["ok"]
        note = " (BF16 does not fit this GPU)" if oom else ""
        if i == 0:
            streamed = c["args"].get("reference") == "streamed"
            ref = "BF16 read layer by layer from disk, since it does not fit in memory" if streamed else "BF16 in memory"
            out.append(f"| logits at {b['positions']:,} WikiText-2 positions, {c['device']}, against {ref} | {b['positions_bit_identical']:,} / {b['positions']:,} bit-identical ({link(d / 'exactness_scale.md')}) |")
            if b["generations"]:
                out.append(f"| {b['generations']} greedy generations of {c['args']['gen_tokens']} tokens, {c['device']} | {b['generations_identical']} / {b['generations']} identical, tokens and log-probabilities |")
        else:
            gens = f", {b['generations_identical']} / {b['generations']} generations" if b["generations"] else ""
            out.append(f"| the same logits test on {c['device']}{note} | {b['positions_bit_identical']:,} / {b['positions']:,} bit-identical{gens} ({link(d / 'exactness_scale.md')}) |")
    v = j(dirs[0] / "exactness_scale.json")["variants"] if dirs else {}
    if "q8" in v:
        out.append(f"| MLX 8-bit quantization, for comparison | {v['q8']['positions_bit_identical']:,} / {v['q8']['positions']:,} positions identical, argmax agreement {v['q8']['argmax_agreement']:.1%} |")
    sd = next((d for d in dirs if (d / "decode_speed.json").exists()), None)
    if sd:
        dec = j(sd / "decode_speed.json")
        med = dec["median_tok_s"]
        device = j(sd / "exactness_scale.json")["device"]
        bf16 = f"{med['bf16']:.1f} tok/s" if "bf16" in med else "does not load"
        q8 = f"{med['q8']:.1f} tok/s" if "q8" in med else "n/a"
        out += [
            "",
            "## Size and speed",
            "",
            "| | BF16 | Binade | MLX 8-bit (lossy) |",
            "|---|---|---|---|",
            f"| checkpoint | {gb_in:.2f} GB | {gb_out:.2f} GB | |",
            f"| greedy decoding, {device} | {bf16} | {med['binade_r4']:.1f} tok/s | {q8} |",
            "",
            f"Decoding speed: {link(sd / 'decode_speed.md')}. NVIDIA results: {GITHUB}/tree/main/results.",
        ]
    out += [
        "",
        "## Use",
        "",
        "```",
        "pip install binade               # Apple Silicon",
        'pip install "binade[cuda]"       # Linux + CUDA 12',
        "```",
        "",
        "```python",
        "from mlx_lm import generate",
        "from binade.mlx.r4 import load",
        "",
        f'model, tokenizer, _ = load("{repo}")',
        'messages = [{"role": "user", "content": "Why is the sky blue?"}]',
        "prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)",
        "print(generate(model, tokenizer, prompt, max_tokens=200))",
        "```",
        "",
        f"To get the original BF16 checkpoint back, bit for bit: `binade unpack {repo} --out {repo.split('/')[-1]}-bf16`.",
        "",
        "## Requirements and limits",
        "",
        "- MLX 0.32.2 and mlx-lm 0.31.3, which `pip install binade` installs. Exactness is tied to MLX's own kernels; the tests in the repository check it.",
        "- Apple Silicon runs the fused prompt kernels on M3 and M4 GPUs. On NVIDIA they are verified on the RTX PRO 6000, L4 and A100; on other GPUs prompts rebuild BF16 weights, which is exact but slower.",
        text_only(packed),
        *([f"- Exactness and speed were measured on NVIDIA only; the runtime's peak GPU memory there was "
           f"{j(dirs[0] / 'exactness_scale.json')['variants']['binade']['peak_gb']:.1f} GB."] if dirs and dirs[0] != RES / model else []),
        "- The Hub tags this repository 8-bit, and counts its parameters by bytes, because the packed bytes are stored as uint8 tensors. The weights are the full BF16 weights: `binade unpack` restores every one bit for bit.",
        "",
        "## License",
        "",
        *([
            "Built with Llama. " + LLAMA31_NOTICE.strip(),
            "",
            "This repository carries the [license](LICENSE), Meta's [acceptable use policy](USE_POLICY.md) and the [notice](NOTICE). "
            "Use of these weights follows both, as use of the original weights does. Binade itself is Apache-2.0.",
        ] if llama else [
            f"The weights keep the base model's license: {cd.get('license', 'see the base model')}"
            + (f" ([terms]({license_link}))" if license_link else "")
            + ". Binade itself is Apache-2.0.",
        ]),
        "",
        "## Citation",
        "",
        "```bibtex",
        "@misc{shafir2026binade,",
        "  title  = {Binade: Bit-Exact Compressed LLM Inference on Apple Silicon},",
        "  author = {Guy Shafir and Tulber},",
        "  year   = {2026},",
        f"  url    = {{{GITHUB}}}",
        "}",
        "```",
        "",
        f"Packed by Binade format {meta['binade']['version']} at commit `{meta['git'][:7]}` of the development repository.",
        "",
    ]
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("packed", type=Path)
    ap.add_argument("--repo", required=True, help="Hub repo id, e.g. binade/gemma-4-12B-it")
    ap.add_argument("--name", help="display name for the card (default: the base repo's name)")
    ap.add_argument("--public", action="store_true", help="create the repo public (default private)")
    ap.add_argument("--dry-run", action="store_true", help="stage and write the card, upload nothing")
    args = ap.parse_args()
    meta = json.loads((args.packed / "binade_pack.json").read_text())
    base = meta["source"]["repo_id"]
    model = base.split("/")[-1]
    meta = verified(args.packed, model)
    orig = origin(model, meta)

    stage = REPO / "models" / "hf-staging" / args.repo.split("/")[-1]
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)
    for f in sorted(args.packed.iterdir()):
        if f.name in ("README.md", "binade_pack.json") or not f.is_file():
            continue
        if f.name == "config.json":  # packs before binade 0.1.1 carry the marker here too; the Hub reads it as a quantization
            config = json.loads(f.read_text())
            config.pop("quantization_config", None)
            (stage / f.name).write_text(json.dumps(config, indent=2) + "\n")
            continue
        try:
            os.link(f, stage / f.name)
        except OSError:
            shutil.copy2(f, stage / f.name)
    meta = dict(meta)
    meta["source"] = dict(meta["source"], path=st.display_path(meta["source"]["path"]))
    (stage / "binade_pack.json").write_text(json.dumps(meta, indent=1) + "\n")
    from huggingface_hub import model_info

    licensed = license_files(orig, (model_info(orig).card_data or {}).get("license"), stage)
    (stage / "README.md").write_text(card(base, orig, args.repo, args.name or model, model, meta, args.packed, licensed))
    leaks = [f.name for f in stage.iterdir() if f.is_file() and f.suffix in (".json", ".md", ".jinja", ".txt", "") and str(Path.home()) in f.read_text(errors="ignore")]
    if leaks:
        raise SystemExit(f"local paths left in {leaks}")
    print(f"staged {stage} ({sum(f.stat().st_size for f in stage.iterdir()) / 1e9:.2f} GB, {len(list(stage.iterdir()))} files)")
    if args.dry_run:
        return

    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(args.repo, repo_type="model", private=not args.public, exist_ok=True)
    api.upload_large_folder(repo_id=args.repo, repo_type="model", folder_path=stage)
    files = sorted(f for f in stage.iterdir() if f.is_file())  # upload_large_folder keeps its state in stage/.cache
    remote = {p.path: p for p in api.get_paths_info(args.repo, [f.name for f in files], expand=True)}
    bad = []
    for f in files:
        r = remote.get(f.name)
        want = r.lfs.sha256 if r is not None and r.lfs is not None else None
        if r is None or (want is not None and want != sha256(f)) or (want is None and r.size != f.stat().st_size):
            bad.append(f.name)
    if bad:
        raise SystemExit(f"uploaded files differ from the local ones: {bad}")
    print(f"https://huggingface.co/{args.repo}: {len(remote)} files, hashes match the local pack ({'public' if args.public else 'private'})")


if __name__ == "__main__":
    main()
