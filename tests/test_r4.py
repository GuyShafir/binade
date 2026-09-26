"""R4 runtime end to end: prompt pass (multi-row) and token-by-token decode (kernel) match BF16."""

import mlx.core as mx
import numpy as np
import pytest
from mlx_lm.generate import generate_step
from mlx_lm.utils import load_model

from binade.mlx.loader import load_packed_model
from binade.mlx.r4 import R4Embedding, R4Linear, convert
from binade.pack import pack_model
from test_runtime import _tiny_llama


@pytest.mark.parametrize("version", ["1", "2"])
@pytest.mark.parametrize("tied", [False, True])
def test_r4_runtime_matches_bf16(tmp_path, tied, version):
    src = _tiny_llama(tmp_path, tied)
    out = tmp_path / f"packed_{tied}"
    pack_model(src, out, log=lambda *_: None, version=version)
    ref, _ = load_model(src)
    model, _ = load_packed_model(out)
    assert convert(model) >= 6
    kinds = [type(m).__name__ for _, m in model.named_modules()]
    assert "R4Linear" in kinds and "R4Embedding" in kinds and "BinadeLinear" not in kinds

    prompt = mx.array([[1, 5, 77, 300, 42, 511, 0, 9]])
    a, b = ref(prompt), model(prompt)  # multi-row path
    mx.eval(a, b)
    assert np.array_equal(np.array(a.view(mx.uint16)), np.array(b.view(mx.uint16)))

    toks_ref = [int(t) for t, _ in generate_step(prompt[0], ref, max_tokens=12)]
    toks_r4 = [int(t) for t, _ in generate_step(prompt[0], model, max_tokens=12)]  # single-row steps use the kernel
    assert toks_ref == toks_r4


@pytest.mark.parametrize("version", ["1", "2"])
def test_r4_step_logits_bit_identical(tmp_path, version):
    """One decode step (single row through every projection and the lm_head) is bit-identical."""
    from mlx_lm.models.cache import make_prompt_cache

    src = _tiny_llama(tmp_path, True)
    out = tmp_path / "packed"
    pack_model(src, out, log=lambda *_: None, version=version)
    ref, _ = load_model(src)
    model, _ = load_packed_model(out)
    convert(model)
    prompt = mx.array([[3, 1, 4, 1, 5, 9, 2, 6]])
    ca, cb = make_prompt_cache(ref), make_prompt_cache(model)
    mx.eval(ref(prompt, cache=ca), model(prompt, cache=cb))
    step = mx.array([[271]])
    a, b = ref(step, cache=ca), model(step, cache=cb)
    mx.eval(a, b)
    assert np.array_equal(np.array(a.view(mx.uint16)), np.array(b.view(mx.uint16)))


@pytest.mark.parametrize("version", ["1", "2"])
def test_r4_decode_kernel_rebuilds_original(tmp_path, version):
    """The R4 decode kernel returns the original BF16 bits for every packed tensor."""
    from binade import st
    from binade.mlx.loader import packed_specs
    from binade.mlx.slow_linear import BinadeEmbedding, BinadeLinear
    from binade.mlx.r4 import R4Linear, transcode

    src = _tiny_llama(tmp_path, True)
    out = tmp_path / "packed"
    pack_model(src, out, log=lambda *_: None, version=version)
    orig = {i.name: i for i in st.list_tensors(src)}
    arrays = mx.load(str(out / "model.safetensors"))
    for name, spec in packed_specs(out).items():
        m = BinadeEmbedding(spec) if "embed" in name else BinadeLinear(spec)
        m.update({"weight": {p: arrays[f"{name}.{p}"] for p in spec}})
        got = np.array(R4Linear(transcode(m)).dense().view(mx.uint16))
        assert np.array_equal(got, st.read(orig[name])), name


def test_slow_runtime_refuses_rice(tmp_path):
    src = _tiny_llama(tmp_path, False)
    out = tmp_path / "packed"
    pack_model(src, out, log=lambda *_: None, version="2")
    model, _ = load_packed_model(out)
    with pytest.raises(RuntimeError, match="R4 runtime"):
        model(mx.array([[1, 2, 3]]))
