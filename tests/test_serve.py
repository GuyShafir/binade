"""binade serve: mlx_lm.server's model provider loads Binade packs onto the R4 runtime and stock
models as mlx-lm does."""

from argparse import Namespace

import mlx.core as mx
import numpy as np
from mlx_lm.generate import generate_step

from binade.mlx import serve
from binade.pack import pack_model
from test_runtime import _tiny_llama


def _with_tokenizer(src):
    """A word-level tokenizer over the tiny model's 512-token vocabulary, so mlx-lm's loaders find one."""
    from tokenizers import Tokenizer, models

    Tokenizer(models.WordLevel(vocab={f"t{i}": i for i in range(512)}, unk_token="t0")).save(str(src / "tokenizer.json"))
    (src / "tokenizer_config.json").write_text('{"eos_token": "t1", "model_max_length": 512}')
    return src


def _provider(model):
    import mlx_lm.server as server

    server.load = serve.load
    args = Namespace(model=str(model), adapter_path=None, draft_model=None, trust_remote_code=False, chat_template=None,
                     use_default_chat_template=False, pipeline=False)
    return server.ModelProvider(args)


def test_provider_loads_pack_and_stock(tmp_path):
    src = _with_tokenizer(_tiny_llama(tmp_path, True))
    out = tmp_path / "packed"
    pack_model(src, out, log=lambda *_: None)
    p = _provider(out)
    model, tok = p.load("default_model")
    kinds = [type(m).__name__ for _, m in model.named_modules()]
    assert "R4Linear" in kinds and "BinadeLinear" not in kinds
    ref, _ = p.load(str(src))  # switching to the stock model by path
    assert "R4Linear" not in [type(m).__name__ for _, m in ref.named_modules()]
    prompt = mx.array([1, 5, 77, 300, 42, 511, 0, 9])
    a = [int(t) for t, _ in generate_step(prompt, model, max_tokens=8)]
    b = [int(t) for t, _ in generate_step(prompt, ref, max_tokens=8)]
    assert a == b


def test_adapter_refused_on_pack(tmp_path):
    import pytest

    src = _tiny_llama(tmp_path, False)
    out = tmp_path / "packed"
    pack_model(src, out, log=lambda *_: None)
    with pytest.raises(ValueError, match="adapters"):
        serve.load(str(out), adapter_path=str(tmp_path))
