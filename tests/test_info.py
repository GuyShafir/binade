import math

import mlx.core as mx
import numpy as np
import pytest

from binade import st
from binade.info import entropy_mm, tensor_info


def test_entropy_mm_uniform():
    c = np.zeros((1, 256))
    c[0, :4] = 1000
    assert entropy_mm(c)[0] == pytest.approx(2 + 3 / (2 * 4000 * math.log(2)))


def _write(tmp_path, f):
    mx.save_safetensors(str(tmp_path / "model.safetensors"), {"layers.0.mlp.up_proj.weight": mx.array(f).astype(mx.bfloat16)})
    return st.list_tensors(tmp_path)[0]


def test_iid_has_no_information(tmp_path):
    f = np.random.default_rng(0).standard_normal((1024, 512)).astype(np.float32)
    r = tensor_info(_write(tmp_path, f))
    assert abs(r["H"] - r["H_given_col"]) < 0.01 and abs(r["H"] - r["H_given_row"]) < 0.01


def test_column_scale_is_detected(tmp_path):
    rng = np.random.default_rng(1)
    scale = 2.0 ** rng.integers(-6, 6, size=512)  # each column in its own binade range
    f = (rng.standard_normal((1024, 512)) * scale[None, :]).astype(np.float32)
    r = tensor_info(_write(tmp_path, f))
    assert r["H"] - r["H_given_col"] > 1.0
    assert abs(r["H"] - r["H_given_row"]) < 0.01
