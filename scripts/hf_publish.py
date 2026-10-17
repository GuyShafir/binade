#!/usr/bin/env python3
"""Publish a Binade-packed model to the Hugging Face Hub, with a model card built from results/.

    python scripts/hf_publish.py <packed-dir> --repo binade/gemma-4-12B-it [--name "Gemma 4 12B"] [--public] [--dry-run]

Refuses a pack other than the one the bit-exact round trip checked: binade_pack.json's commit
and output size must match results/<model>/roundtrip_v2.json, which must report no mismatched
values. Stages the pack in models/hf-staging/<repo name> (hard links), with binade_pack.json's
local source path made home-relative and the base model's README replaced by a model card whose
numbers all come from the committed result files. Uploads with upload_large_folder (resumable),
private unless --public, then compares the SHA-256 of every file on the Hub with the local file.
--dry-run stops after staging, to read the card first. Needs `hf auth login` with write access.
"""

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

from binade import st

REPO = Path(__file__).resolve().parents[1]
RES = REPO / "results"
GITHUB = "https://github.com/GuyShafir/binade"


def j(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def verified(packed: Path, model: str) -> dict:
    """The pack's metadata, after checking it is the pack the round trip verified."""
    meta = j(packed / "binade_pack.json")
    rt = j(RES / model / "roundtrip_v2.json")
    if meta is None or rt is None:
        raise SystemExit(f"{packed}: no binade_pack.json, or no results/{model}/roundtrip_v2.json")
    if rt["mismatched_elements"] or rt["missing"] or rt["extra"]:
        raise SystemExit(f"results/{model}/roundtrip_v2.json does not report an exact round trip")
    if meta["git"] != rt["pack_git"] or meta["output_bytes"] != rt["output_bytes"]:
        raise SystemExit(f"{packed} is not the pack results/{model}/roundtrip_v2.json checked ({meta['git'][:7]}, {meta['output_bytes']} bytes vs {rt['pack_git'][:7]}, {rt['output_bytes']})")
    return meta


def text_only(packed: Path) -> str:
    """The limits line on modalities: the pack keeps every tensor, the runtime serves text."""
    names = json.loads((packed / "model.safetensors.index.json").read_text())["weight_map"]
    other = [m for m in ("vision", "audio") if any(m in n for n in names)]
    if not other:
        return "- Exactness was tested on text generation."
    return (f"- The checkpoint keeps every tensor of the original, the {' and '.join(other)} weights included (`binade unpack` restores them), "
            "but the runtime loads the text model through mlx-lm, so it serves text. Exactness was tested on text generation.")


def card(base: str, repo: str, name: str, model: str, meta: dict, packed: Path) -> str:
    from huggingface_hub import model_info

    info = model_info(base)
    cd = info.card_data or {}
    rt = j(RES / model / "roundtrip_v2.json")
    bits = sum(c["bits"] for c in rt["classes"].values()) / sum(c["numel"] for c in rt["classes"].values())
    gb_in, gb_out = rt["input_bytes"] / 1e9, rt["output_bytes"] / 1e9
    ex, dec = j(RES / model / "exactness_scale.json"), j(RES / model / "decode_speed.json")
    link = lambda p: f"[{p}]({GITHUB}/blob/main/results/{p})"

    front = ["---", f"license: {cd.get('license', 'other')}"]
    if cd.get("license_link"):
        front.append(f"license_link: {cd['license_link']}")
    front += [f"base_model: {base}", "library_name: binade", "pipeline_tag: text-generation"]  # the runtime serves text (mlx-lm's text model)
    front += ["tags:", "- binade", "- lossless-compression", "- bf16", "- mlx", "---", ""]

    out = front + [
        f"# {name}, losslessly compressed with Binade",
        "",
        f"[{base}](https://huggingface.co/{base}) packed with [Binade]({GITHUB}): **{gb_out:.2f} GB instead of {gb_in:.2f} GB** "
        f"({bits:.3f} bits per weight instead of 16 in the packed tensors), every weight restored bit for bit, and **outputs bit-identical to the BF16 model**, not just close.",
        "",
        "Binade codes each weight's exponent by its frequency rank and keeps the sign and mantissa as they are. Its kernels decode the weights "
        "inside the matrix products and add the products in exactly the order the framework's own BF16 kernels do, so the logits and generated "
        f"tokens match the original. The [paper]({GITHUB}/blob/main/paper/binade.pdf) explains how.",
        "",
        "## Exactness",
        "",
        "| check | result |",
        "|---|---|",
        f"| every tensor unpacked and compared with the original checkpoint | {rt['tensors_checked']} tensors, {rt['mismatched_elements']} values differ ({link(model + '/roundtrip_v2.md')}) |",
    ]
    if ex:
        v = ex["variants"]
        streamed = ex["args"].get("reference") == "streamed"
        ref = "BF16 read layer by layer from disk, since it does not fit in memory" if streamed else "BF16 in memory"
        out.append(f"| logits at {v['binade']['positions']:,} WikiText-2 positions, {ex['device']}, against {ref} | {v['binade']['positions_bit_identical']:,} / {v['binade']['positions']:,} bit-identical ({link(model + '/exactness_scale.md')}) |")
        if v["binade"]["generations"]:
            out.append(f"| {v['binade']['generations']} greedy generations of {ex['args']['gen_tokens']} tokens, {ex['device']} | {v['binade']['generations_identical']} / {v['binade']['generations']} identical, tokens and log-probabilities |")
        for d in sorted(p for p in RES.glob("cuda*") if (p / model / "exactness_scale.json").exists()):
            c = j(d / model / "exactness_scale.json")
            b = c["variants"]["binade"]
            fit = j(d / model / "fit_probe.json")
            oom = fit and not {r["variant"]: r for r in fit["variants"]}["bf16"]["ok"]
            gens = f", {b['generations_identical']} / {b['generations']} generations" if b["generations"] else ""
            note = " (BF16 does not fit this GPU)" if oom else ""
            out.append(f"| the same logits test on {c['device']}{note} | {b['positions_bit_identical']:,} / {b['positions']:,} bit-identical{gens} ({link(d.name + '/' + model + '/exactness_scale.md')}) |")
        if "q8" in v:
            out.append(f"| MLX 8-bit quantization, for comparison | {v['q8']['positions_bit_identical']:,} / {v['q8']['positions']:,} positions identical, argmax agreement {v['q8']['argmax_agreement']:.1%} |")
    if dec:
        med = dec["median_tok_s"]
        device = ex["device"] if ex else "Apple Silicon"
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
            f"Decoding speed: {link(model + '/decode_speed.md')}. NVIDIA results: {GITHUB}/tree/main/results.",
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
        "",
        "## License",
        "",
        f"The weights keep the base model's license: {cd.get('license', 'see the base model')}"
        + (f" ([terms]({cd['license_link']}))" if cd.get("license_link") else "")
        + f". Binade itself is Apache-2.0.",
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

    stage = REPO / "models" / "hf-staging" / args.repo.split("/")[-1]
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)
    for f in sorted(args.packed.iterdir()):
        if f.name in ("README.md", "binade_pack.json") or not f.is_file():
            continue
        try:
            os.link(f, stage / f.name)
        except OSError:
            shutil.copy2(f, stage / f.name)
    meta = dict(meta)
    meta["source"] = dict(meta["source"], path=st.display_path(meta["source"]["path"]))
    (stage / "binade_pack.json").write_text(json.dumps(meta, indent=1) + "\n")
    (stage / "README.md").write_text(card(base, args.repo, args.name or model, model, meta, args.packed))
    leaks = [f.name for f in stage.iterdir() if f.suffix in (".json", ".md", ".jinja", ".txt") and str(Path.home()) in f.read_text(errors="ignore")]
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
