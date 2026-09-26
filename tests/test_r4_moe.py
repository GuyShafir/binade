"""R4 with partial last tiles (K % 128 != 0) and expert gathers, bit-identical to MLX."""

from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

from binade import bf16, rice
from binade.format import rank_table
from binade.mlx.cuda_kernels import available as cuda_available
from binade.mlx.r4 import R4Linear, transcode

metal_only = pytest.mark.skipif(cuda_available(), reason="the R4 GEMM mirrors MLX's Metal GEMM; on CUDA, multi-row inputs rebuild BF16")


def _weights(rng, shape, scale=0.02):
    w = rng.standard_normal(shape) * scale * np.exp(rng.standard_normal(shape[:-1] + (1,)))
    w = np.array(mx.array(w.astype(np.float32)).astype(mx.bfloat16).view(mx.uint16))
    w.reshape(-1)[:: 997] = 0x8000  # some -0
    w.reshape(-1)[5 :: 1999] = 0x3F80 | 0x4000  # a few large values (rare exponents)
    return w


def _r4(w: np.ndarray) -> R4Linear:
    """R4 module from uint16 BF16 bits [..., K] via a format 2 (Rice) holder."""
    K = w.shape[-1]
    raw, exps = bf16.split(w.reshape(-1, K))
    table, rank = rank_table(np.bincount(exps.ravel(), minlength=256))
    enc = rice.encode(exps, table, rank)
    holder = SimpleNamespace(weight={"raw": mx.array(raw), "table": mx.array(table), "meta": mx.array(enc.meta), "bits": mx.array(enc.bits), "roff": mx.array(enc.roff)})
    return R4Linear(transcode(holder))


def _bits(a: mx.array) -> np.ndarray:
    return np.array(a.astype(mx.bfloat16).view(mx.uint16))


@pytest.mark.parametrize("N,K", [(64, 704), (160, 2112), (300, 260), (48, 100), (64, 128), (40, 4)])
def test_partial_tile_matvec_and_dense(N, K):
    rng = np.random.default_rng(K)
    w = _weights(rng, (N, K))
    m = _r4(w)
    assert np.array_equal(_bits(m.dense()), w)
    x = mx.array(rng.standard_normal((1, K)).astype(np.float32)).astype(mx.bfloat16)
    W = mx.array(w).view(mx.bfloat16)
    ref, got = x @ W.T, m(x)
    mx.eval(ref, got)
    assert np.array_equal(_bits(got), _bits(ref))


@pytest.mark.parametrize("E,N,K,topk", [(8, 176, 704, 4), (8, 704, 176, 4), (6, 96, 384, 2), (16, 64, 520, 8)])
def test_gather_matches_gather_mm(E, N, K, topk):
    """SwitchLinear: gather_mm(x, W.swapaxes(-1, -2), rhs_indices) for one token, both with a
    shared input (gate/up) and with one input row per selected expert (down)."""
    rng = np.random.default_rng(N * K)
    w = _weights(rng, (E, N, K))
    m = _r4(w)
    W = mx.array(w).view(mx.bfloat16)
    idx = mx.array(rng.choice(E, size=topk, replace=False).astype(np.uint32)).reshape(1, 1, topk)
    for shared in (True, False):
        shape = (1, 1, 1, 1, K) if shared else (1, 1, topk, 1, K)
        x = mx.array(rng.standard_normal(shape).astype(np.float32)).astype(mx.bfloat16)
        ref = mx.gather_mm(x, W.swapaxes(-1, -2), rhs_indices=idx)  # [1, 1, topk, 1, N]
        xi = mx.zeros((topk,), dtype=mx.uint32) if shared else mx.arange(topk, dtype=mx.uint32)
        got = m._gather(x, xi, idx.reshape(-1), N, N)
        mx.eval(ref, got)
        assert np.array_equal(_bits(got), _bits(ref.reshape(topk, N))), shared


def test_gather_row_offset_selects_half_of_fused_rows():
    """gate_up_proj [E, 2n, K] holds gate rows 0..n-1 and up rows n..2n-1 of each expert."""
    rng = np.random.default_rng(7)
    E, n, K, topk = 8, 96, 704, 4
    w = _weights(rng, (E, 2 * n, K))
    m = _r4(w)
    W = mx.array(w).view(mx.bfloat16)
    idx = mx.array(np.array([3, 0, 7, 5], np.uint32))
    x = mx.array(rng.standard_normal((1, 1, 1, 1, K)).astype(np.float32)).astype(mx.bfloat16)
    for r0, half in ((0, W[:, :n]), (n, W[:, n:])):
        ref = mx.gather_mm(x, mx.contiguous(half).swapaxes(-1, -2), rhs_indices=idx.reshape(1, 1, topk))
        got = m._gather(x, mx.zeros((topk,), dtype=mx.uint32), idx, n, 2 * n, r0)
        mx.eval(ref, got)
        assert np.array_equal(_bits(got), _bits(ref.reshape(topk, n)))


def _tiny_gemma4_moe(tmp_path):
    """A 2-layer Gemma 4 MoE (26B layout scaled down). Expert down_proj (K = 320) and the
    dense MLP's down_proj (K = 336) end in a partial tile, like the 26B's K = 704 and 2112."""
    import json

    from mlx.utils import tree_flatten
    from mlx_lm.models import gemma4_text

    config = {
        "model_type": "gemma4_text",
        "hidden_size": 256,
        "num_hidden_layers": 2,
        "intermediate_size": 336,
        "num_attention_heads": 2,
        "head_dim": 64,
        "global_head_dim": 128,
        "num_key_value_heads": 2,
        "num_global_key_value_heads": 1,
        "attention_k_eq_v": True,
        "hidden_size_per_layer_input": 0,
        "num_kv_shared_layers": 0,
        "vocab_size": 512,
        "vocab_size_per_layer_input": 512,
        "enable_moe_block": True,
        "num_experts": 8,
        "top_k_experts": 2,
        "moe_intermediate_size": 320,
        "use_double_wide_mlp": False,
        "sliding_window": 16,
        "layer_types": ["sliding_attention", "full_attention"],
        "tie_word_embeddings": True,
    }
    mx.random.seed(1)
    model = gemma4_text.Model(gemma4_text.ModelArgs.from_dict(config))
    flat = dict(tree_flatten(model.parameters()))
    weights = {}
    for k, v in flat.items():
        v = (v * 8).astype(mx.bfloat16) if v.ndim >= 2 else v.astype(mx.bfloat16)
        if ".switch_glu.gate_proj.weight" in k:
            up = (flat[k.replace("gate_proj", "up_proj")] * 8).astype(mx.bfloat16)
            weights[k.replace(".switch_glu.gate_proj.weight", ".gate_up_proj")] = mx.concatenate([v, up], axis=1)
        elif ".switch_glu.down_proj.weight" in k:
            weights[k.replace(".switch_glu.down_proj.weight", ".down_proj")] = v
        elif ".switch_glu.up_proj.weight" not in k:
            weights[k] = v
    src = tmp_path / "src"
    src.mkdir()
    mx.save_safetensors(str(src / "model.safetensors"), weights)
    (src / "config.json").write_text(json.dumps(config))
    return src


@pytest.mark.parametrize("version", ["1", "2"])
def test_moe_model_r4_matches_bf16(tmp_path, version):
    from mlx_lm.generate import generate_step
    from mlx_lm.models.cache import make_prompt_cache
    from mlx_lm.utils import load_model

    from binade.mlx.loader import load_packed_model
    from binade.mlx.r4 import convert
    from binade.pack import pack_model

    src = _tiny_gemma4_moe(tmp_path)
    out = tmp_path / "packed"
    report = pack_model(src, out, log=lambda *_: None, version=version)
    assert any(n.endswith("experts.gate_up_proj") for n in report["tensors"])
    ref, _ = load_model(src)
    model, _ = load_packed_model(out)
    kinds = [type(m).__name__ for _, m in model.named_modules()]
    assert kinds.count("BinadeSwitchGLU") == 2
    convert(model)
    kinds = [type(m).__name__ for _, m in model.named_modules()]
    assert kinds.count("R4SwitchGLU") == 2 and "BinadeSwitchGLU" not in kinds

    for n in (5, 40):  # 10 expert slots: gather kernel; 80: sorted gather_mm on rebuilt experts
        prompt = mx.array(np.random.default_rng(n).integers(0, 512, size=(1, n)))
        ca, cb = make_prompt_cache(ref), make_prompt_cache(model)
        a, b = ref(prompt, cache=ca), model(prompt, cache=cb)
        mx.eval(a, b)
        assert np.array_equal(_bits(a), _bits(b)), n
        step = mx.array([[7]])
        a, b = ref(step, cache=ca), model(step, cache=cb)
        mx.eval(a, b)
        assert np.array_equal(_bits(a), _bits(b)), n
    p = mx.array([3, 1, 4, 1, 5, 9, 2, 6])
    assert [int(t) for t, _ in generate_step(p, ref, max_tokens=16)] == [int(t) for t, _ in generate_step(p, model, max_tokens=16)]


def test_streamed_reference_matches_resident(tmp_path):
    from mlx.utils import tree_flatten
    from mlx_lm.models.cache import make_prompt_cache
    from mlx_lm.utils import load_model

    from binade.mlx.loader import load_streamed_model

    src = _tiny_gemma4_moe(tmp_path)
    ref, _ = load_model(src)
    streamed, _, _ = load_streamed_model(src)
    assert type(streamed.layers[0]).__name__.startswith("Streamed")
    ca, cb = make_prompt_cache(ref), make_prompt_cache(streamed)
    for x in (mx.array([[5, 17, 300, 2, 99, 41]]), mx.array([[8]]), mx.array([[13]])):
        before = {k: id(v) for k, v in tree_flatten(streamed.layers[1].parameters())}
        a, b = ref(x, cache=ca), streamed(x, cache=cb)
        mx.eval(a, b)
        assert np.array_equal(_bits(a), _bits(b))
        after = {k: id(v) for k, v in tree_flatten(streamed.layers[1].parameters())}
        assert before.keys() == after.keys() and all(before[k] != after[k] for k in before)  # reloaded


@metal_only
@pytest.mark.parametrize("E,N,K,S", [(8, 176, 704, 67), (8, 704, 256, 64), (16, 96, 384, 200), (6, 64, 520, 9)])
def test_gather_mm_sorted_matches_mlx(E, N, K, S):
    """The prompt path: slots sorted by expert, gather_mm with sorted_indices on BF16 weights."""
    rng = np.random.default_rng(E * N + S)
    w = _weights(rng, (E, N, K))
    m = _r4(w)
    assert m._a["esc"].size > 1  # rare exponents take the escape path
    W = mx.array(w).view(mx.bfloat16)
    idx = mx.array(np.sort(rng.integers(0, E, size=S)).astype(np.uint32))
    x = mx.array(rng.standard_normal((S, 1, K)).astype(np.float32)).astype(mx.bfloat16)
    ref = mx.gather_mm(x, W.swapaxes(-1, -2), rhs_indices=idx, sorted_indices=True).reshape(S, N)
    got = m._gather_mm(x, idx, N, N)
    mx.eval(ref, got)
    assert np.array_equal(_bits(got), _bits(ref))
    # half of fused rows, as for gate_up_proj
    half = N // 2
    ref = mx.gather_mm(x, mx.contiguous(W[:, half:]).swapaxes(-1, -2), rhs_indices=idx, sorted_indices=True).reshape(S, N - half)
    got = m._gather_mm(x, idx, N - half, N, half)
    mx.eval(ref, got)
    assert np.array_equal(_bits(got), _bits(ref))


def _adversarial(R: int, K: int, seed: int = 0) -> np.ndarray:
    """uint16 BF16 rows [R, K] whose products with an all-ones input depend on summation order:
    +2^25 first, then ones that vanish against it (float's ulp at 2^25 is 4), then -2^25. In
    order the result is 0; any other grouping keeps some of the ones. Rows cycle through a
    per-lane pattern (MLX's gemv: lane 0 owns k = 128 j), a per-MMA-step pattern (steps of 8
    along K) and random values. Also exercises escapes (the 2^25 exponent is rare)."""
    big = 2.0**25
    w = np.zeros((R, K), np.float32)
    last = 128 * ((K - 1) // 128)
    for r in range(0, R, 3):
        w[r, 0], w[r, last] = big, -big
        w[r, 128:last:128] = 1.0
    for r in range(1, R, 3):
        w[r, 0], w[r, K - 8] = big, -big
        w[r, 8 : K - 8 : 8] = 1.0
    w[2::3] = np.random.default_rng(seed).standard_normal(w[2::3].shape) * 0.02
    return np.array(mx.array(w).astype(mx.bfloat16).view(mx.uint16))


@pytest.mark.parametrize("N,K", [(96, 704), (4096, 512), (512, 2048), (48, 2816)])
def test_order_sensitive_rows(N, K):
    """R4 on any number of rows against x @ W.T: matvec (1 row), BF16 rebuild (2..15 rows and
    split-K shapes) and the R4 GEMM (MLX's plain GEMM shapes), on order-sensitive data."""
    w = _adversarial(N, K)
    m = _r4(w)
    W = mx.array(w).view(mx.bfloat16)
    from binade.mlx.r4 import use_mma

    for M in (1, 2, 8, 15, 16, 40):
        x = mx.ones((1, M, K), dtype=mx.bfloat16)
        ref, got = x @ W.T, m._rows(x)
        mx.eval(ref, got)
        assert np.array_equal(_bits(got), _bits(ref)), M
        if use_mma(M, N, K):  # the fused GEMM the runtime uses when a rebuild would not fit
            fused = m._gather_mm(x, mx.zeros((M,), dtype=mx.uint32), N, N)
            mx.eval(fused)
            assert np.array_equal(_bits(fused).reshape(-1), _bits(ref).reshape(-1)), M
    exact = np.array(W.astype(mx.float32)).astype(np.float64).sum(axis=1)
    mlx = np.array((mx.ones((1, K), dtype=mx.bfloat16) @ W.T).astype(mx.float32))[0]
    assert np.any(mlx[0::3] != exact[0::3]) and np.any(mlx[1::3] != exact[1::3])  # rounding depends on order


@metal_only
@pytest.mark.parametrize("K", [704, 2816])
def test_order_sensitive_sorted_gather(K):
    E, N, S = 8, 48, 67
    w = _adversarial(E * N, K).reshape(E, N, K)
    m = _r4(w)
    W = mx.array(w).view(mx.bfloat16)
    rng = np.random.default_rng(K)
    idx = mx.array(np.sort(rng.integers(0, E, size=S)).astype(np.uint32))
    x = mx.ones((S, 1, K), dtype=mx.bfloat16)
    ref = mx.gather_mm(x, W.swapaxes(-1, -2), rhs_indices=idx, sorted_indices=True).reshape(S, N)
    got = m._gather_mm(x, idx, N, N)
    mx.eval(ref, got)
    assert np.array_equal(_bits(got), _bits(ref))


@pytest.mark.parametrize("K", [704, 2816, 520, 96])
def test_order_sensitive_gathers(K):
    """The decode-step expert gather (MLX's gemv_gather on Metal, gather_mv on CUDA)."""
    E, N, topk = 8, 48, 4
    w = _adversarial(E * N, K).reshape(E, N, K)
    m = _r4(w)
    W = mx.array(w).view(mx.bfloat16)
    sel = mx.array(np.array([5, 0, 7, 2], np.uint32))
    ref = mx.gather_mm(mx.ones((1, 1, 1, 1, K), dtype=mx.bfloat16), W.swapaxes(-1, -2), rhs_indices=sel.reshape(1, 1, topk))
    got = m._gather(mx.ones((1, K), dtype=mx.bfloat16), mx.zeros((topk,), dtype=mx.uint32), sel, N, N)
    mx.eval(ref, got)
    assert np.array_equal(_bits(got), _bits(ref.reshape(topk, N)))


@metal_only
def test_order_sensitive_data_catches_a_different_order():
    """K = 520 is not a multiple of MLX's K block (16), so MLX's gather GEMM adds the last,
    partial block first; the R4 GEMM (used only for aligned K) sums in order and differs."""
    E, N, K, S = 4, 48, 520, 16
    w = _adversarial(E * N, K).reshape(E, N, K)
    m = _r4(w)
    W = mx.array(w).view(mx.bfloat16)
    idx = mx.array(np.sort(np.arange(S) % E).astype(np.uint32))
    x = mx.ones((S, 1, K), dtype=mx.bfloat16)
    ref = mx.gather_mm(x, W.swapaxes(-1, -2), rhs_indices=idx, sorted_indices=True).reshape(S, N)
    got = m._gather_mm(x, idx, N, N)
    mx.eval(ref, got)
    assert not np.array_equal(_bits(got), _bits(ref))
