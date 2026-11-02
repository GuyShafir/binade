"""binade serve: mlx-lm's OpenAI-compatible HTTP server (mlx_lm.server) with Binade models on
the R4 runtime.

    binade serve --model Binade/gemma-4-12B-it [--port 8080] [any mlx_lm.server option]

A model path that is a Binade pack (a local directory or a Hub repo id, downloaded if needed)
loads through binade.mlx.r4.load; any other model loads as mlx-lm would, so one server can
switch between packed and stock models by name. Adapters and draft models are mlx-lm's:
they apply to stock models, and a Binade model given an adapter is refused (its modules are
not nn.Linear).
"""

from mlx_lm.utils import load as load_stock

from .. import st
from ..pack import pack_meta


def load(model_path, tokenizer_config=None, adapter_path=None, **kwargs):
    """mlx_lm.utils.load's signature: (model, tokenizer), Binade packs on the R4 runtime."""
    try:
        path = st.fetch(str(model_path))
    except FileNotFoundError:
        path = None
    if path is None or pack_meta(path) is None:
        return load_stock(model_path, tokenizer_config=tokenizer_config, adapter_path=adapter_path, **kwargs)
    if adapter_path is not None:
        raise ValueError(f"{model_path} is a Binade model; adapters are not supported on the R4 runtime")
    from .r4 import load as load_r4

    model, tokenizer, _ = load_r4(path, tokenizer_config=tokenizer_config)
    return model, tokenizer


def main():
    import mlx_lm.server as server

    server.load = load
    server.main()
