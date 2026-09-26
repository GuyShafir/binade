"""Slow runtime: decoded weights and model outputs are bit-identical to BF16."""

import json

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_flatten
from mlx_lm.models import llama
from mlx_lm.utils import load_model

from binade import st
from binade.mlx.loader import load_packed_model, packed_specs
from binade.mlx.slow_linear import BinadeEmbedding, BinadeLinear, dense
from binade.pack import pack_model


def _tiny_llama(tmp_path, tied: bool):
    config = {
        "model_type": "llama",
        "hidden_size": 256,
        "num_hidden_layers": 2,
        "intermediate_size": 640,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "rms_norm_eps": 1e-5,
        "vocab_size": 512,
        "rope_theta": 10000.0,
        "tie_word_embeddings": tied,
    }
    mx.random.seed(0)
    model = llama.Model(llama.ModelArgs.from_dict(config))
    weights = {k: (v * 40).astype(mx.bfloat16) for k, v in tree_flatten(model.parameters())}  # widen the exponent range
    src = tmp_path / f"src_{tied}"
    src.mkdir()
    mx.save_safetensors(str(src / "model.safetensors"), weights)
    (src / "config.json").write_text(json.dumps(config))
    return src


@pytest.mark.parametrize("tied", [False, True])
def test_logits_bit_identical(tmp_path, tied):
    src = _tiny_llama(tmp_path, tied)
    out = tmp_path / f"packed_{tied}"
    pack_model(src, out, log=lambda *_: None, version="1")  # the slow runtime reads format 1
    ref, _ = load_model(src)
    xpm, _ = load_packed_model(out)
    kinds = [type(m).__name__ for _, m in xpm.named_modules()]
    assert kinds.count("BinadeLinear") >= 6 and kinds.count("BinadeEmbedding") == 1
    tokens = mx.array([[1, 5, 77, 300, 42, 511, 0, 9]])
    a, b = ref(tokens), xpm(tokens)
    mx.eval(a, b)
    assert a.dtype == b.dtype and np.array_equal(np.array(a.astype(mx.float32)), np.array(b.astype(mx.float32)))


def test_dense_matches_original_weights(tmp_path):
    src = _tiny_llama(tmp_path, False)
    out = tmp_path / "packed"
    pack_model(src, out, log=lambda *_: None, version="1")  # the slow runtime reads format 1
    orig = {i.name: i for i in st.list_tensors(src)}
    arrays = mx.load(str(out / "model.safetensors"))
    for name, spec in packed_specs(out).items():
        m = BinadeEmbedding(spec) if "embed" in name else BinadeLinear(spec)
        m.update({"weight": {p: arrays[f"{name}.{p}"] for p in spec}})
        got = np.array(dense(m).view(mx.uint16))
        assert np.array_equal(got, st.read(orig[name])), name


@pytest.mark.parametrize("chunk", [4096, 1 << 24])
def test_chunked_decode_and_lookup(tmp_path, monkeypatch, chunk):
    """Tiny decode chunks (escapes and lookups across chunk boundaries) change nothing."""
    import binade.mlx.slow_linear as sl

    monkeypatch.setattr(sl, "CHUNK", chunk)
    src = _tiny_llama(tmp_path, True)
    out = tmp_path / "packed"
    pack_model(src, out, log=lambda *_: None, version="1")  # the slow runtime reads format 1
    orig = {i.name: i for i in st.list_tensors(src)}
    arrays = mx.load(str(out / "model.safetensors"))
    name = "model.embed_tokens.weight"
    spec = packed_specs(out)[name]
    m = BinadeEmbedding(spec)
    m.update({"weight": {p: arrays[f"{name}.{p}"] for p in spec}})
    ref = st.read(orig[name])
    assert np.array_equal(np.array(dense(m).view(mx.uint16)), ref)
    ids = mx.array([[511, 0, 17, 17, 300], [256, 1, 2, 510, 64]])
    got = np.array(m(ids).view(mx.uint16))
    assert np.array_equal(got, ref[np.array(ids)])
    if chunk == 4096:
        assert len(m._plan.chunks) > 1
