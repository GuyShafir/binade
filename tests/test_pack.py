import json
import subprocess
import sys

import mlx.core as mx
import numpy as np
import pytest
from safetensors import safe_open

from binade import st
from binade.pack import pack_model
from binade.unpack import PackedModel


def _model(tmp_path):
    rng = np.random.default_rng(0)
    bf = lambda a: mx.array(a.astype(np.float32)).astype(mx.bfloat16)
    emb = rng.standard_normal((512, 256)) * 0.02
    emb[3, :5] = [np.nan, np.inf, -np.inf, 0.0, -0.0]
    down = rng.standard_normal((256, 300)) * 0.02
    down[7] = 0.5  # a row with one exponent
    tensors = {
        "model.embed_tokens.weight": bf(emb),
        "model.layers.0.self_attn.q_proj.weight": bf(rng.standard_normal((300, 260)) * np.exp(rng.standard_normal((300, 1)))),
        "model.layers.0.experts.gate_up_proj": bf(rng.standard_normal((4, 256, 256)) * 0.01),
        "model.layers.0.mlp.down_proj.weight": bf(down),
        "model.layers.0.input_layernorm.weight": bf(rng.standard_normal(256)),
        "model.layers.0.linear_attn.A_log": mx.array(rng.standard_normal(32).astype(np.float32)),
        "lm_head.weight": bf(rng.standard_normal((512, 256)) * 0.05),
    }
    src = tmp_path / "src"
    src.mkdir()
    mx.save_safetensors(str(src / "model.safetensors"), tensors)
    (src / "config.json").write_text(json.dumps({"model_type": "test", "hidden_size": 256}))
    (src / "tokenizer.json").write_text("{}")
    return src


@pytest.mark.parametrize("version,gpu", [("1", True), ("2", True), ("2", False)])
def test_pack_roundtrip_bit_exact(tmp_path, version, gpu):
    src = _model(tmp_path)
    out = tmp_path / "packed"
    report = pack_model(src, out, chunk=5000, log=lambda *_: None, version=version)
    pm = PackedModel(out, gpu=gpu)
    orig = {i.name: i for i in st.list_tensors(src)}
    assert sorted(pm.names()) == sorted(orig)
    assert set(pm.packed) == {
        "model.embed_tokens.weight",
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.experts.gate_up_proj",
        "model.layers.0.mlp.down_proj.weight",
        "lm_head.weight",
    }
    for name, info in orig.items():
        assert np.array_equal(pm.tensor(name), st.read(info)), name
    assert report["packed_bits"] < 16 * report["packed_numel"]

    # the files are valid safetensors for both readers mlx_lm and HF use
    with safe_open(str(out / "model.safetensors"), framework="numpy") as f:
        keys = set(f.keys())
    assert f"model.layers.0.experts.gate_up_proj.{'idx' if version == '1' else 'bits'}" in keys
    loaded = mx.load(str(out / "model.safetensors"))
    assert loaded["model.layers.0.experts.gate_up_proj.raw"].shape == (4, 256, 256)
    config = json.loads((out / "config.json").read_text())
    assert config["quantization_config"]["quant_method"] == "binade" and config["quantization_config"]["version"] == version
    index = json.loads((out / "model.safetensors.index.json").read_text())
    assert set(index["weight_map"]) == keys
    assert (out / "tokenizer.json").exists() and (out / "binade_pack.json").exists()


@pytest.mark.parametrize("version", ["1", "2"])
def test_unpack_cli_restores_original_bytes(tmp_path, version):
    src = _model(tmp_path)
    out = tmp_path / "packed"
    pack_model(src, out, chunk=5000, log=lambda *_: None, version=version)
    back = tmp_path / "back"
    subprocess.run([sys.executable, "-m", "binade.unpack", str(out), "--out", str(back)], check=True)
    restored = {i.name: i for i in st.list_tensors(back)}
    for info in st.list_tensors(src):
        r = restored[info.name]
        assert (r.dtype, r.shape) == (info.dtype, info.shape)
        assert np.array_equal(st.read(r), st.read(info)), info.name
    assert "quantization_config" not in json.loads((back / "config.json").read_text())
