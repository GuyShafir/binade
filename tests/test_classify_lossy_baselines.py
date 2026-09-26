import mlx.core as mx
import numpy as np
import pytest

from binade import baselines, lossy
from binade.bf16 import round_mantissa
from binade.classify import classify

L = "model.language_model.layers.7."


@pytest.mark.parametrize(
    "name,shape,cls,sub",
    [
        (L + "self_attn.q_proj.weight", (4096, 3840), "linear", "attn"),
        (L + "self_attn.v_proj.weight", (2048, 3840), "linear", "attn"),
        (L + "mlp.down_proj.weight", (3840, 15360), "linear", "mlp"),
        (L + "experts.gate_up_proj", (128, 1408, 2816), "linear", "experts"),
        (L + "experts.down_proj", (128, 2816, 704), "linear", "experts"),
        ("model.layers.3.mlp.experts.12.gate_proj.weight", (768, 2048), "linear", "experts"),
        (L + "per_layer_input_gate.weight", (256, 2560), "linear", "ple_proj"),
        ("model.language_model.per_layer_model_projection.weight", (10752, 2560), "linear", "ple_proj"),
        (L + "router.proj.weight", (128, 2816), "other", "other"),
        (L + "input_layernorm.weight", (3840,), "other", "norm"),
        ("model.language_model.embed_tokens.weight", (262144, 3840), "embed", "embed_tokens"),
        ("lm_head.weight", (128256, 4096), "embed", "lm_head"),
        ("model.language_model.embed_tokens_per_layer.weight", (262144, 10752), "ple", "ple"),
        ("model.vision_tower.encoder.layers.0.self_attn.q_proj.linear.weight", (768, 768), "mm", "mm"),
        ("model.vision_embedder.patch_dense.weight", (3840, 6912), "mm", "mm"),
        ("model.embed_vision.embedding_projection.weight", (3840, 3840), "mm", "mm"),
        ("model.language_model.layers.3.linear_attn.in_proj_qkv.weight", (8192, 4096), "linear", "linattn"),
        ("model.language_model.layers.3.linear_attn.out_proj.weight", (4096, 4096), "linear", "linattn"),
        ("model.language_model.layers.3.linear_attn.in_proj_a.weight", (32, 4096), "other", "other"),
        ("model.visual.blocks.0.attn.qkv.weight", (3456, 1152), "mm", "mm"),
        ("mtp.layers.0.mlp.gate_proj.weight", (12288, 4096), "linear", "mlp"),
    ],
)
def test_classify(name, shape, cls, sub):
    k = classify(name, shape, "BF16")
    assert (k.cls, k.sub) == (cls, sub)


def test_classify_layer_and_dtype():
    k = classify(L + "mlp.up_proj.weight", (15360, 3840), "BF16")
    assert k.layer == 7 and k.block == "model.language_model.layers.7"
    k = classify("mtp.layers.0.self_attn.q_proj.weight", (8192, 4096), "BF16")
    assert k.layer is None and k.block == "mtp.layers.0"
    assert classify(L + "mlp.up_proj.weight", (15360, 3840), "F32").cls == "other"


def test_tile_errors_match_numpy():
    rng = np.random.default_rng(1)
    R, K, T, m = 4, 96, 32, 3
    f = (rng.standard_normal((R, K)) * 0.02).astype(np.float32)
    f[0, :T] = 0.0  # an all-zero tile -> NaN
    w16 = (f.view(np.uint32) >> 16).astype(np.uint16)
    d, a = lossy.elementwise(mx.array(w16), m)
    fro, mxr = (np.array(x) for x in lossy.tile_errors(d, a, T))

    def val(u):
        return (u.astype(np.uint32) << 16).view(np.float32).astype(np.float64)

    w, q = val(w16), val(round_mantissa(w16, m))
    for i in range(R * K // T):
        r, j = divmod(i, K // T)
        wt, qt = w[r, j * T : (j + 1) * T], q[r, j * T : (j + 1) * T]
        if not wt.any():
            assert np.isnan(fro[i]) and np.isnan(mxr[i])
            continue
        assert fro[i] == pytest.approx(np.linalg.norm(wt - qt) / np.linalg.norm(wt), rel=1e-5)
        nz = wt != 0
        assert mxr[i] == pytest.approx(np.max(np.abs(wt - qt)[nz] / np.abs(wt[nz])), rel=1e-5)
        assert mxr[i] <= 2.0 ** -(m + 1)  # RNE bound for normal numbers


def test_histogram_percentiles():
    x = mx.array(np.r_[np.full(900, 1e-3), np.full(100, 1e-2), np.nan].astype(np.float32))
    h = lossy.histogram(x)
    assert h.sum() == 1000
    p = lossy.percentiles(h)
    assert 1e-3 <= p["p50"] < 1e-3 * 1.02
    assert 1e-2 <= p["p99"] < 1e-2 * 1.02


def test_huffman():
    h = np.zeros(256, np.int64)
    h[[10, 11, 12, 13]] = [8, 4, 2, 2]  # dyadic: lengths 1, 2, 3, 3
    lengths = baselines.huffman_lengths(h)
    assert list(lengths[[10, 11, 12, 13]]) == [1, 2, 3, 3]
    assert baselines.huffman_bits(h) == 8 + 8 + 6 + 6
    assert baselines.entropy(h) == pytest.approx(1.75)
    rng = np.random.default_rng(0)
    g = np.bincount(np.clip(120 - rng.geometric(0.4, 100_000), 0, 255), minlength=256)
    assert np.sum(2.0 ** -baselines.huffman_lengths(g)[g > 0]) == pytest.approx(1.0)  # complete code
    H = baselines.entropy(g)
    assert H <= baselines.huffman_bits(g) / g.sum() < H + 1


def test_tensor_palette_bits():
    h = np.zeros(256, np.int64)
    h[[1, 2, 3, 4, 5]] = 10
    assert baselines.tensor_palette_bits(h) == 50 * 3 + 5 * 8


def test_window7_and_palette16():
    h = np.zeros(256, np.int64)
    h[100:107] = 10  # exactly one window
    h[120] = 3  # outliers
    assert baselines.window7_bits(h) == 3 * 73 + 16 * 3
    assert baselines.palette16_segment_bits(in_weights=128, numel=192, segments=3) == 4 * 128 + 8 * 64 + 3 + 128
