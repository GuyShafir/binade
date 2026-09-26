#!/usr/bin/env python3
"""Bit-exactness in generation (DESIGN.md 10): stock BF16 vs Binade slow runtime.

    python scripts/generate_check.py <original model-dir | cached-hf-repo-id> <packed-dir> [--tokens 256] [--runtime slow|r4]

Greedy-decodes the same chat prompt with both models, one after the other, and requires
identical prompt logits (every vocabulary entry, bit for bit) and identical token ids.
Writes generate_check.json / .md to results/<model>/ and exits non-zero on a mismatch.
--runtime slow (default) rebuilds every weight with MLX ops per call; --runtime r4 converts
the model to the R4 runtime (4-bit rank codes in memory, the Metal kernel for single-row
steps) and writes generate_check_r4.json / .md instead. A format 2 (Rice) model adds _v2 to
the file names. Decode tok/s counts the tokens after the first, so the prompt pass is
excluded.
"""

import argparse
import gc
import json
import subprocess
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx_lm.generate import generate_step, wired_limit

from binade import st
from binade.mlx.loader import load as binade_load
from binade.mlx.loader import load_reference, load_reference_streamed
from binade.mlx.r4 import load as r4_load

REPO = Path(__file__).resolve().parents[1]
PARTS = (".raw", ".table", ".meta", ".idx", ".esc", ".bits", ".roff")
PROMPT = "Explain in three short paragraphs why lossless compression of neural network weights can speed up inference on memory-bandwidth-bound hardware."


def run(model, tokenizer, n: int) -> dict:
    prompt = mx.array(tokenizer.apply_chat_template([{"role": "user", "content": PROMPT}], add_generation_prompt=True))
    mx.reset_peak_memory()
    with wired_limit(model, [mx.default_stream(mx.default_device())]):  # as generate_step runs
        t0 = time.perf_counter()
        logits = model(prompt[None])[0, -1].astype(mx.float32)
        mx.eval(logits)
        t_prefill = time.perf_counter() - t0
        t0 = time.perf_counter()
        again = model(prompt[None])[0, -1].astype(mx.float32)
        mx.eval(again)
        t_prefill_warm = time.perf_counter() - t0
    ids, stamps, t1 = [], [], time.perf_counter()
    for tok, _ in generate_step(prompt, model, max_tokens=n):
        ids.append(int(tok))
        stamps.append(time.perf_counter())
    t_gen = time.perf_counter() - t1
    return {
        "prompt_tokens": int(prompt.size),
        "logits": np.array(logits),
        "ids": ids,
        "prefill_s": t_prefill,
        "prefill_warm_s": t_prefill_warm,
        "prefill_repeatable": bool(np.array_equal(np.array(logits), np.array(again))),
        "generate_s": t_gen,
        "tokens_per_s": n / t_gen,
        "decode_tokens_per_s": (n - 1) / (stamps[-1] - stamps[0]),
        "peak_gb": mx.get_peak_memory() / 1e9,
        "text": tokenizer.decode(ids),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("original")
    ap.add_argument("packed", type=Path)
    ap.add_argument("--revision")
    ap.add_argument("--tokens", type=int, default=256)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--runtime", choices=("slow", "r4"), default="slow")
    ap.add_argument("--binade-first", action="store_true", help="run the Binade model before BF16 (order/thermal check)")
    ap.add_argument("--reference", choices=("resident", "streamed"), default="resident", help="streamed: BF16 layers read from disk per call, for models that do not fit")
    args = ap.parse_args()
    src = st.resolve(args.original, args.revision)
    source = st.provenance(src)
    out = args.out or REPO / "results" / source.get("repo_id", src.name).split("/")[-1]
    out.mkdir(parents=True, exist_ok=True)

    results = {}
    stem = "generate_check" if args.runtime == "slow" else f"generate_check_{args.runtime}"
    version = str(json.loads((args.packed / "config.json").read_text())["quantization_config"]["version"])
    if version != "1":
        stem += f"_v{version}"
    loaders = {"slow": binade_load, "r4": r4_load}
    order = [("bf16", load_reference if args.reference == "resident" else load_reference_streamed, src), ("binade", loaders[args.runtime], args.packed)]
    if args.reference == "streamed":
        stem += "_streamed"
    if args.binade_first:
        order.reverse()
        stem += "_binade_first"
    for label, loader, path in order:
        t_load = time.perf_counter()
        model, tokenizer, dropped = loader(path)
        if not (label == "bf16" and args.reference == "streamed"):
            mx.eval(model.parameters())  # both models fully loaded before timing
        t_load = time.perf_counter() - t_load
        results[label] = run(model, tokenizer, args.tokens)
        results[label]["dropped"] = dropped
        results[label]["load_s"] = t_load
        print(f"{label}: {results[label]['tokens_per_s']:.2f} tok/s, peak {results[label]['peak_gb']:.1f} GB", flush=True)
        del model
        gc.collect()
        mx.clear_cache()

    a, b = results["bf16"], results["binade"]
    label_b = {
        "slow": f"Binade format {version}, slow runtime (MLX ops decode per call, no kernel)",
        "r4": f"Binade format {version}, R4 runtime (R4 kernel for single-row steps)",
    }[args.runtime]
    same_logits = bool(np.array_equal(a["logits"], b["logits"]))
    diverge = next((i for i, (x, y) in enumerate(zip(a["ids"], b["ids"])) if x != y), None)
    ok = same_logits and a["ids"] == b["ids"] and a["prefill_repeatable"] and b["prefill_repeatable"]
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=no", "--", ".", ":!results"], cwd=REPO, capture_output=True, text=True).stdout.strip())
    summary = {
        "source": source,
        "packed": st.display_path(args.packed),
        "git": {"commit": commit, "dirty": dirty},
        "prompt": PROMPT,
        "tokens": args.tokens,
        "runtime": args.runtime,
        "reference": args.reference,
        "binade_format": version,
        "order": [label for label, _, _ in order],
        "identical_logits": same_logits,
        "max_abs_logit_diff": float(np.max(np.abs(a["logits"] - b["logits"]))),
        "identical_ids": a["ids"] == b["ids"],
        "first_divergence": diverge,
        **{k: {kk: vv for kk, vv in v.items() if kk != "logits"} for k, v in results.items()},
    }
    (out / f"{stem}.json").write_text(json.dumps(summary, indent=1) + "\n")
    lines = [
        f"# Binade greedy decode check ({args.runtime} runtime): {source.get('repo_id', src.name)}",
        "",
        f"`scripts/generate_check.py` @ `{commit[:7]}{' (dirty)' if dirty else ''}`, {args.tokens} greedy tokens, prompt of {a['prompt_tokens']} tokens.",
        "",
        f"**{'PASS' if ok else 'FAIL'}**: prompt logits identical over the full vocabulary: {same_logits}; "
        f"a second prompt pass repeats them bit for bit (BF16, Binade): {a['prefill_repeatable']}, {b['prefill_repeatable']}; "
        f"token ids identical: {a['ids'] == b['ids']}" + ("" if diverge is None else f" (first divergence at token {diverge})") + ".",
        "",
        f"Checkpoint tensors without a model parameter, dropped by both loaders (listed in generate_check.json): "
        f"{len(a['dropped'])} (BF16); {len({d.rsplit('.', 1)[0] for d in b['dropped'] if d.endswith(PARTS)})} packed weights and "
        f"{len([d for d in b['dropped'] if not d.endswith(PARTS)])} other tensors (Binade).",
        "",
        "| model | load s | prompt pass s (first, second) | decode tok/s (after the first token) | tok/s incl. prompt | peak GB |",
        "|---|---|---|---|---|---|",
        (
            f"| BF16 (mlx_lm) | {a['load_s']:.1f} | {a['prefill_s']:.2f}, {a['prefill_warm_s']:.2f} | {a['decode_tokens_per_s']:.2f} | {a['tokens_per_s']:.2f} | {a['peak_gb']:.1f} |"
            if args.reference == "resident"
            else "| BF16 (mlx_lm), decoder layers streamed from disk | n/a | n/a | n/a | n/a | n/a |"
        ),
        f"| {label_b} | {b['load_s']:.1f} | {b['prefill_s']:.2f}, {b['prefill_warm_s']:.2f} | {b['decode_tokens_per_s']:.2f} | {b['tokens_per_s']:.2f} | {b['peak_gb']:.1f} |",
        "",
        (
            f"Decode speedup vs BF16: {b['decode_tokens_per_s'] / a['decode_tokens_per_s']:.2f}x."
            if args.reference == "resident"
            else "BF16 does not fit in memory, so the reference reads each decoder layer's weights from the checkpoint when it runs "
            "(binade.mlx.loader.load_streamed_model): its timings measure disk reads, and it serves only as the bit-exact reference."
        ),
        "",
        "Output (both):",
        "",
        "> " + a["text"][:600].replace("\n", "\n> "),
    ]
    (out / f"{stem}.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
