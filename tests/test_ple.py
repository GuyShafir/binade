"""Gemma per-layer embedding tables (the `ple` class, gather only): packed by default, served
from their Rice coding in memory by the R4 runtime, lookups and model outputs bit-identical."""

import json

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_flatten
from mlx_lm.models import gemma4_text
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.utils import load_model

from binade import st
from binade.mlx.loader import load_packed_model, packed_specs
from binade.mlx.r4 import RiceEmbedding, convert
from binade.pack import pack_model


def _tiny_gemma(tmp_path, per_layer: int = 64):
    config = {
        "model_type": "gemma4_text",
        "hidden_size": 256,
        "num_hidden_layers": 2,
        "intermediate_size": 640,
        "num_attention_heads": 4,
        "num_key_value_heads": 1,
        "head_dim": 32,
        "global_head_dim": 64,
        "num_kv_shared_layers": 0,
        "vocab_size": 512,
        "vocab_size_per_layer_input": 512,
        "hidden_size_per_layer_input": per_layer,
        "sliding_window": 64,
        "rms_norm_eps": 1e-6,
        "tie_word_embeddings": True,
    }
    mx.random.seed(0)
    model = gemma4_text.Model(gemma4_text.ModelArgs.from_dict(config))
    weights = {k: (v * 40).astype(mx.bfloat16) for k, v in tree_flatten(model.parameters())}
    src = tmp_path / f"src_{per_layer}"
    src.mkdir()
    mx.save_safetensors(str(src / "model.safetensors"), weights)
    (src / "config.json").write_text(json.dumps(config))
    return src


def _bits(a: mx.array) -> np.ndarray:
    return np.array(a.view(mx.uint16))


@pytest.mark.parametrize("per_layer", [64, 72])  # K = 128 (one full tile) and 144 (a partial last tile)
def test_rice_embedding_lookups(tmp_path, per_layer):
    src = _tiny_gemma(tmp_path, per_layer)
    out = tmp_path / "packed"
    report = pack_model(src, out, log=lambda *_: None)
    name = next(n for n in report["tensors"] if "embed_tokens_per_layer" in n)
    orig = {i.name: i for i in st.list_tensors(src)}[name]
    arrays = mx.load(str(out / "model.safetensors"))
    emb = RiceEmbedding({p: arrays[f"{name}.{p}"] for p in packed_specs(out)[name]})
    assert emb.K == 2 * per_layer and emb.N == 512
    assert np.array_equal(_bits(emb.dense()), st.read(orig))
    ids = mx.array([[3, 511, 0, 3], [77, 300, 42, 9]])
    got = emb(ids)
    assert got.shape == (2, 4, emb.K)
    assert np.array_equal(_bits(got), st.read(orig)[np.array(ids)])


@pytest.mark.parametrize("version", ["1", "2"])
def test_model_with_ple_matches_bf16(tmp_path, version):
    """Prompt pass and a decode step through the per-layer inputs are bit-identical; format 2
    serves the table from RiceEmbedding, format 1 (no Rice stream) from R4Embedding."""
    src = _tiny_gemma(tmp_path)
    out = tmp_path / "packed"
    report = pack_model(src, out, log=lambda *_: None, version=version)
    assert any("embed_tokens_per_layer" in n for n in report["tensors"]) and "ple" in report["classes"]
    ref, _ = load_model(src)
    model, _ = load_packed_model(out)
    convert(model)
    kinds = [type(m).__name__ for _, m in model.named_modules()]
    assert kinds.count("RiceEmbedding") == (1 if version == "2" else 0)
    assert kinds.count("R4Embedding") == (1 if version == "2" else 2)

    prompt = mx.array([[1, 5, 77, 300, 42, 511, 0, 9]])
    ca, cb = make_prompt_cache(ref), make_prompt_cache(model)
    a, b = ref(prompt, cache=ca), model(prompt, cache=cb)
    mx.eval(a, b)
    assert np.array_equal(_bits(a), _bits(b))
    step = mx.array([[271]])
    a, b = ref(step, cache=ca), model(step, cache=cb)
    mx.eval(a, b)
    assert np.array_equal(_bits(a), _bits(b))


def test_ple_left_out_when_not_requested(tmp_path):
    src = _tiny_gemma(tmp_path)
    out = tmp_path / "packed"
    report = pack_model(src, out, classes=("linear", "embed"), log=lambda *_: None)
    assert not any("embed_tokens_per_layer" in n for n in report["tensors"])
    model, _ = load_packed_model(out)
    convert(model)
    assert "RiceEmbedding" not in [type(m).__name__ for _, m in model.named_modules()]
