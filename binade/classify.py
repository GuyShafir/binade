"""Tensor classification by name and shape (DESIGN.md 5).

Classes:
  linear  decoder projections that enter a matmul: attn, linattn (linear-attention
          blocks such as Qwen3.5 Gated DeltaNet), mlp, experts, ple_proj
  embed   token embedding and lm_head (Gemma ties lm_head to embed_tokens, so the
          embedding also enters a matmul once per decoded token)
  ple     Gemma per-layer embedding tables (gather only)
  mm      vision/audio encoders and their projections (not on the text decode path)
  other   norms, routers, scalars, anything unrecognized
"""

import re
from dataclasses import dataclass

MM_MARKERS = (
    "vision_tower",
    "audio_tower",
    "vision_embedder",
    "embed_vision",
    "embed_audio",
    "vision_model",
    "multi_modal_projector",
    "mm_projector",
    ".visual.",
)
ATTN = {"q_proj", "k_proj", "v_proj", "o_proj", "qkv_proj", "out_proj"}
LINATTN = {"linear_attn"}
MLP = {"gate_proj", "up_proj", "down_proj", "gate_up_proj", "w1", "w2", "w3"}
PLE_PROJ = {"per_layer_input_gate", "per_layer_projection", "per_layer_model_projection"}
MIN_DIM = 256
_BLOCK = re.compile(r"^(.*?\blayers\.(\d+))\.")


@dataclass(frozen=True)
class Klass:
    cls: str
    sub: str
    layer: int | None  # decoder layer index; None outside the main stack (e.g. mtp heads)
    block: str | None = None  # module prefix through the layer index, e.g. "mtp.layers.0"


def classify(name: str, shape: tuple[int, ...], dtype: str) -> Klass:
    m = _BLOCK.search(name)
    block = m.group(1) if m else None
    layer = int(m.group(2)) if m and not name.startswith("mtp.") else None
    parts = name.split(".")
    if any(s in f".{name}." for s in MM_MARKERS):
        return Klass("mm", "mm", layer, block)
    if "embed_tokens_per_layer" in parts:
        return Klass("ple", "ple", None)
    if name.endswith("embed_tokens.weight"):
        return Klass("embed", "embed_tokens", None)
    if name.endswith("lm_head.weight"):
        return Klass("embed", "lm_head", None)
    matrix = dtype == "BF16" and len(shape) in (2, 3) and min(shape[-2:]) >= MIN_DIM
    # a weight is either `<module>.weight` or a fused expert stack without the suffix
    is_weight = parts[-1] == "weight" or (len(shape) == 3 and "experts" in parts)
    if matrix and is_weight:
        comps = set(parts)
        sub = None
        if "experts" in comps:
            sub = "experts"
        elif comps & LINATTN:
            sub = "linattn"
        elif comps & ATTN:
            sub = "attn"
        elif comps & MLP:
            sub = "mlp"
        elif comps & PLE_PROJ:
            sub = "ple_proj"
        if sub:
            return Klass("linear", sub, layer, block)
    sub = "norm" if any("norm" in p for p in parts) else "other"
    return Klass("other", sub, layer, block)
