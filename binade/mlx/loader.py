"""Load a Binade model with mlx_lm (DESIGN.md 10). Monkeypatch-free: wraps the model class.

    model, tokenizer = binade.mlx.loader.load("models/gemma-4-12B-it-binade")

Packed weights keep their HF names plus a suffix (name.raw, name.table, ...). The model's
own sanitize maps HF names to module paths, so it is run once on placeholders to find
which nn.Linear / nn.Embedding each packed weight belongs to; those modules are swapped
for BinadeLinear / BinadeEmbedding before mlx_lm loads the weights with strict checking.

mlx-lm 0.31.3 has no `gemma4_unified` (Gemma 4 12B). Its text stack uses only features
`gemma4_text` implements (checked field by field against the config), so it loads through
`gemma4`; vision and audio parts are dropped like any other tensor without a module.

Checkpoint tensors the model has no parameter for are dropped and recorded (mlx-lm 0.31.3
has no modules for the k/v projections of Gemma 4 E-models' KV-shared layers, which
Google's checkpoints still ship). A model parameter missing from the checkpoint is still
an error.
"""

import json
import re
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten
from mlx_lm.utils import _get_classes, load_model, load_tokenizer

from .. import st
from ..pack import PARTS
from .moe import BinadeSwitchGLU
from .slow_linear import BinadeEmbedding, BinadeLinear

DTYPES = {"U8": "uint8", "U32": "uint32", "BF16": "bfloat16"}
MODEL_TYPE_ALIASES = {"gemma4_unified": "gemma4"}
# Gemma 4 experts: sanitize leaves packed parts of experts.gate_up_proj / .down_proj unsplit;
# they load into the BinadeSwitchGLU that replaces experts.switch_glu
_EXPERT = re.compile(r"\.experts\.(gate_up_proj|down_proj)\.(raw|table|meta|idx|esc|bits|roff)$")


def packed_specs(path: Path) -> dict:
    """HF weight name -> {part: (shape, mlx dtype name)} for every packed weight (format 1
    or 2, told apart by their parts)."""
    infos = {t.name: t for t in st.list_tensors(path)}
    out = {}
    for n in infos:
        if not n.endswith(".raw"):
            continue
        name = n[: -len(".raw")]
        parts = PARTS["1"] if f"{name}.idx" in infos else PARTS["2"] if f"{name}.bits" in infos else None
        if parts:
            out[name] = {p: (list(infos[f"{name}.{p}"].shape), DTYPES[infos[f"{name}.{p}"].dtype]) for p in parts}
    return out


def _module_paths(sanitize, names) -> dict:
    """HF weight name -> module path, via the model's sanitize on placeholder arrays."""
    marks = {f"{n}.raw": mx.zeros((1,), dtype=mx.uint8) for n in names}
    ids = {id(a): n[: -len(".raw")] for n, a in marks.items()}
    out = {}
    for key, a in sanitize(dict(marks)).items():
        name = ids.get(id(a))
        e = _EXPERT.search(key)
        if name is not None and e:
            out[name] = key[: e.start()] + ".experts.switch_glu#" + e.group(1)
            continue
        if name is None or not key.endswith(".weight.raw"):
            raise ValueError(f"sanitize changed packed weight {key}; not supported")
        out[name] = key[: -len(".weight.raw")]
    return out


def _get(root, path: str):
    obj = root
    for c in path.split("."):
        obj = obj[int(c)] if isinstance(obj, list) else getattr(obj, c)
    return obj


def _set(root, path: str, module) -> None:
    parent_path, _, leaf = path.rpartition(".")
    parent = _get(root, parent_path) if parent_path else root
    if isinstance(parent, list):
        parent[int(leaf)] = module
    else:
        setattr(parent, leaf, module)


def swap_modules(model: nn.Module, specs: dict, sanitize=lambda w: w) -> list[str]:
    """Replace the modules behind packed weights; returns packed names with no module."""
    unused = []
    experts = {}
    for name, path in _module_paths(sanitize, specs).items():
        if "#" in path:
            path, role = path.split("#")
            experts.setdefault(path, {})[role] = specs[name]
            continue
        try:
            old = _get(model, path)
        except (AttributeError, IndexError, KeyError):
            unused.append(name)
            continue
        if isinstance(old, nn.Embedding):
            new = BinadeEmbedding(specs[name])
        elif isinstance(old, nn.Linear):
            new = BinadeLinear(specs[name], bias="bias" in old)
        else:
            raise TypeError(f"{path}: cannot pack a {type(old).__name__}")
        _set(model, path, new)
    for path, parts in experts.items():
        old = _get(model, path)
        if set(parts) != {"gate_up_proj", "down_proj"} or type(old).__name__ != "SwitchGLU":
            raise TypeError(f"{path}: cannot pack experts {sorted(parts)} into a {type(old).__name__}")
        _set(model, path, BinadeSwitchGLU(parts["gate_up_proj"], parts["down_proj"], old.activation))
    return unused


def model_classes(specs: dict | None = None, dropped: list | None = None):
    """get_model_classes for mlx_lm.load_model: swaps packed modules (if specs) and drops
    checkpoint tensors the model has no parameter for (appended to `dropped`)."""

    def classes(config):
        model_type = MODEL_TYPE_ALIASES.get(config["model_type"], config["model_type"])
        model_class, args_class = _get_classes({**config, "model_type": model_type})
        base_sanitize = getattr(model_class, "sanitize", None)

        class Wrapped(model_class):
            def __init__(self, args):
                super().__init__(args)
                if specs:
                    swap_modules(self, specs, (lambda w: base_sanitize(self, w)) if base_sanitize else (lambda w: w))

            def sanitize(self, weights):
                w = base_sanitize(self, weights) if base_sanitize else weights
                w = {_EXPERT.sub(r".experts.switch_glu.\1.\2", k): v for k, v in w.items()}
                params = {k for k, _ in tree_flatten(self.parameters())}
                if dropped is not None:
                    dropped.extend(sorted(k for k in w if k not in params))
                return {k: v for k, v in w.items() if k in params}

        return Wrapped, args_class

    return classes


def load_packed_model(path, lazy: bool = True):
    """(model, config) for a Binade-packed model directory."""
    path = Path(path)
    config = json.loads((path / "config.json").read_text())
    q = config.get("quantization_config", {})
    if q.get("quant_method") != "binade" or str(q.get("version")) not in PARTS:
        raise ValueError(f"{path}: not a Binade model ({q})")
    dropped = []
    model, config = load_model(path, lazy=lazy, strict=True, get_model_classes=model_classes(packed_specs(path), dropped))
    config["binade_dropped"] = dropped
    return model, config


def load(path, lazy: bool = True, cache_limit_gb: float | None = 8):
    """(model, tokenizer, dropped checkpoint tensors) for a Binade-packed model directory.

    Sets MLX's buffer cache limit (global) unless cache_limit_gb is None: the slow runtime
    frees and reallocates large decode buffers every call, and with the default unbounded
    cache MLX keeps them until the system compresses memory (E4B decode step: 30 to 55 s
    unbounded, 3.0 s at 8 GB, 4.6 s at 1 GB, measured on the 48 GB M4 Max)."""
    if cache_limit_gb is not None:
        mx.set_cache_limit(int(cache_limit_gb * (1 << 30)))
    model, config = load_packed_model(path, lazy)
    tok = load_tokenizer(Path(path), eos_token_ids=config.get("eos_token_id", None))
    return model, tok, config["binade_dropped"]


def load_reference(path, lazy: bool = True):
    """Stock BF16 model through the same path minus packing: (model, tokenizer, dropped)."""
    dropped = []
    model, config = load_model(Path(path), lazy=lazy, strict=True, get_model_classes=model_classes(None, dropped))
    return model, load_tokenizer(Path(path), eos_token_ids=config.get("eos_token_id", None)), dropped


def _stream(layer, reload) -> None:
    """Make `layer` drop its weights after each call: outputs and cache are evaluated, then
    the parameters are replaced by fresh lazy arrays that hold no data until the next call."""
    base = type(layer)

    def call(self, *args, **kwargs):
        out = base.__call__(self, *args, **kwargs)
        cache = args[2] if len(args) > 2 else kwargs.get("cache")
        live = tree_flatten(out) + tree_flatten(cache.state if cache is not None else [])
        mx.eval([a for _, a in live if isinstance(a, mx.array)])
        self.update(reload())
        mx.clear_cache()
        return out

    layer.__class__ = type(f"Streamed{base.__name__}", (base,), {"__call__": call})


def load_streamed_model(path):
    """(model, config, dropped): stock BF16 model for checking models that do not fit in
    memory. Each decoder layer reads its weights from the checkpoint when called and
    releases them afterwards, so only one layer (plus embeddings and norms) is resident.
    Every forward pass reads the checkpoint once; meant for a prompt and a few tokens."""
    path = Path(path)
    dropped = []
    model, config = load_model(path, lazy=True, strict=True, get_model_classes=model_classes(None, dropped))
    n_dropped = len(dropped)
    files = st.shard_files(path)
    layers = model.layers
    layers_path = next(name for name, m in model.named_modules() if m is layers[0]).rsplit(".", 1)[0]

    def reload(i):
        pat = re.compile(rf"\.layers\.{i}\.")
        w = {}
        for f in files:
            w.update({k: v for k, v in mx.load(str(f)).items() if pat.search(k)})
        w = model.sanitize(w)
        del dropped[n_dropped:]
        pre = f"{layers_path}.{i}."
        return tree_unflatten([(k[len(pre) :], v) for k, v in w.items() if k.startswith(pre)])

    for i, layer in enumerate(layers):
        _stream(layer, lambda i=i: reload(i))
        layer.update(reload(i))  # drop the arrays load_model attached; nothing was read yet
    return model, config, dropped


def load_reference_streamed(path):
    """(model, tokenizer, dropped) for load_streamed_model."""
    model, config, dropped = load_streamed_model(path)
    return model, load_tokenizer(Path(path), eos_token_ids=config.get("eos_token_id", None)), dropped
