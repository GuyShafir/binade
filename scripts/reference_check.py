#!/usr/bin/env python3
"""MLX against the reference implementation (Hugging Face transformers) on the same tokens.

    python scripts/reference_check.py <cached-hf-repo-id> --hf-python ~/hfref/bin/python [--device cuda|mps|cpu] [--mlx-device gpu|cpu] [--dtypes float32,bfloat16]

Checks that the MLX model behind the BF16 reference (mlx-lm, through binade's loader, which
aliases model types mlx-lm lacks) computes the same function as the model's reference
implementation. One window of the WikiText-2 test set; each side runs in its own process,
the transformers side under --hf-python (an environment with torch and transformers).
Reported per dtype: perplexity of each, argmax agreement, and the per-token loss
difference. Two implementations of the same model differ by rounding; in float32 that
difference is small, while a missing or wrong computation shows up as a large one. A model
too large for the GPU in float32 runs on the CPU on both sides (--device cpu --mlx-device cpu).
Writes reference_check.json / .md to results/<model>/ (or --out).
"""

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
DATASET = ("Salesforce/wikitext", "wikitext-2-raw-v1/test-00000-of-00001.parquet")

HF_CHILD = r'''
import sys, numpy as np, torch, transformers
from transformers import AutoModelForCausalLM
path, dtype, device, data, out = sys.argv[1:6]
ids = np.load(data)["ids"].tolist()
model = AutoModelForCausalLM.from_pretrained(path, dtype=getattr(torch, dtype), device_map=device, attn_implementation="eager").eval()
x = torch.tensor([ids], device=device)
with torch.no_grad():
    lg = model(input_ids=x).logits[0, :-1].float()
nll = torch.nn.functional.cross_entropy(lg, x[0, 1:], reduction="none").cpu().numpy()
np.savez(out, nll=nll, am=lg.argmax(-1).cpu().numpy(), transformers=transformers.__version__, torch=torch.__version__)
'''

MLX_CHILD = r'''
import sys, numpy as np, mlx.core as mx
from mlx.utils import tree_map
from binade.mlx.loader import load_reference
path, dtype, data, out, device = sys.argv[1:6]
if device == "cpu":
    mx.set_default_device(mx.cpu)
ids = np.load(data)["ids"].tolist()
model, _, _ = load_reference(path)
if dtype == "float32":
    model.update(tree_map(lambda p: p.astype(mx.float32) if p.dtype == mx.bfloat16 else p, model.parameters()))
lg = model(mx.array(ids)[None])[0, :-1].astype(mx.float32)
lp = lg - mx.logsumexp(lg, axis=-1, keepdims=True)
nll = -np.array(mx.take_along_axis(lp, mx.array(ids[1:])[:, None], axis=-1)[:, 0])
np.savez(out, nll=nll, am=np.array(mx.argmax(lg, axis=-1)), mlx=mx.__version__)
'''


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model")
    ap.add_argument("--hf-python", type=Path, required=True)
    ap.add_argument("--device", default="cuda", help="torch device for the reference side")
    ap.add_argument("--mlx-device", default="gpu", choices=("gpu", "cpu"))
    ap.add_argument("--dtypes", default="float32,bfloat16")
    ap.add_argument("--tokens", type=int, default=512)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    from mlx_lm.utils import load_tokenizer

    from binade import st

    src = st.resolve(args.model)
    source = st.provenance(src)
    out = args.out or REPO / "results" / source.get("repo_id", src.name).split("/")[-1]
    out.mkdir(parents=True, exist_ok=True)
    text = "".join(pq.read_table(hf_hub_download(DATASET[0], DATASET[1], repo_type="dataset")).column("text").to_pylist())
    tok = load_tokenizer(src)
    bos = [tok.bos_token_id] if tok.bos_token_id is not None else []
    ids = np.array(bos + tok.encode(text, add_special_tokens=False)[: args.tokens - len(bos)], np.int64)
    res = {}
    with tempfile.TemporaryDirectory() as tmp:
        data = Path(tmp) / "ids.npz"
        np.savez(data, ids=ids)
        for dt in args.dtypes.split(","):
            hf, ml = Path(tmp) / f"hf_{dt}.npz", Path(tmp) / f"mlx_{dt}.npz"
            for cmd in ([args.hf_python, "-c", HF_CHILD, str(src), dt, args.device, str(data), str(hf)], [sys.executable, "-c", MLX_CHILD, str(src), dt, str(data), str(ml), args.mlx_device]):
                r = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO)
                if r.returncode:
                    raise SystemExit(f"{cmd[0]} {dt} failed:\n{r.stderr[-3000:]}")
            a, b = np.load(hf), np.load(ml)
            d = np.abs(a["nll"] - b["nll"])
            res[dt] = {
                "tokens": int(d.size),
                "perplexity_reference": float(np.exp(a["nll"].mean())),
                "perplexity_mlx": float(np.exp(b["nll"].mean())),
                "argmax_agreement": float(np.mean(a["am"] == b["am"])),
                "nll_abs_diff_mean": float(d.mean()),
                "nll_abs_diff_p99": float(np.percentile(d, 99)),
                "nll_abs_diff_max": float(d.max()),
                "nll_correlation": float(np.corrcoef(a["nll"], b["nll"])[0, 1]),
                "transformers": str(a["transformers"]),
                "torch": str(a["torch"]),
                "mlx": str(b["mlx"]),
            }
            print(dt, json.dumps(res[dt]), flush=True)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=no", "--", ".", ":!results"], cwd=REPO, capture_output=True, text=True).stdout.strip())
    import mlx.core as mx

    summary = {"source": source, "git": {"commit": commit, "dirty": dirty}, "device": mx.device_info().get("device_name") if args.mlx_device == "gpu" else "CPU", "args": {k: str(v) for k, v in st.display_args(vars(args)).items()}, "dtypes": res}
    (out / "reference_check.json").write_text(json.dumps(summary, indent=1) + "\n")
    lines = [
        f"# MLX against the reference implementation: {source.get('repo_id', src.name)}",
        "",
        f"`scripts/reference_check.py` @ `{commit[:7]}{' (dirty)' if dirty else ''}`, {summary['device']}, transformers {next(iter(res.values()))['transformers']}, "
        f"MLX {next(iter(res.values()))['mlx']}. One window of {args.tokens} WikiText-2 test tokens; the same token ids on both sides.",
        "",
        "| dtype | perplexity (transformers / MLX) | argmax agreement | per-token loss difference (mean / 99th pct / max, nats) | correlation |",
        "|---|---|---|---|---|",
    ]
    for dt, r in res.items():
        lines.append(f"| {dt} | {r['perplexity_reference']:.2f} / {r['perplexity_mlx']:.2f} | {r['argmax_agreement']:.2%} | {r['nll_abs_diff_mean']:.4f} / {r['nll_abs_diff_p99']:.3f} / {r['nll_abs_diff_max']:.3f} | {r['nll_correlation']:.6f} |")
    (out / "reference_check.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
