import mlx.core as mx
import numpy as np
from safetensors import safe_open

from binade import st


def _write(tmp_path):
    rng = np.random.default_rng(0)
    tensors = {
        "a.weight": mx.array(rng.standard_normal((5, 7)).astype(np.float32)).astype(mx.bfloat16),
        "b.odd": mx.array(rng.integers(0, 255, (3,), dtype=np.uint8)),  # odd byte count shifts later offsets
        "c.weight": mx.array(rng.standard_normal((4, 3, 6)).astype(np.float32)).astype(mx.bfloat16),
        "d.norm": mx.array(rng.standard_normal((9,)).astype(np.float32)),
    }
    path = tmp_path / "model.safetensors"
    mx.save_safetensors(str(path), tensors)
    return path, tensors


def test_header_matches_safe_open(tmp_path):
    path, tensors = _write(tmp_path)
    infos = {t.name: t for t in st.list_tensors(tmp_path)}
    assert set(infos) == set(tensors)
    with safe_open(str(path), framework="numpy") as f:
        for k in f.keys():
            sl = f.get_slice(k)
            assert infos[k].dtype == sl.get_dtype()
            assert list(infos[k].shape) == list(sl.get_shape())


def test_read_matches_mlx_loader(tmp_path):
    _, tensors = _write(tmp_path)
    loaded = mx.load(str(tmp_path / "model.safetensors"))
    for info in st.list_tensors(tmp_path):
        ref = loaded[info.name]
        if ref.dtype == mx.bfloat16:
            ref = ref.view(mx.uint16)
        got = st.read(info)
        assert got.shape == tuple(ref.shape)
        assert np.array_equal(got, np.array(ref))


def test_read_row_ranges(tmp_path):
    _write(tmp_path)
    info = {t.name: t for t in st.list_tensors(tmp_path)}["c.weight"]
    full = st.read(info).reshape(info.rows, info.shape[-1])
    parts = [st.read(info, (r, min(info.rows, r + 5))) for r in range(0, info.rows, 5)]
    assert np.array_equal(np.concatenate(parts), full)


def test_provenance_parses_hf_cache_path():
    p = st.provenance("/x/hub/models--google--gemma-4-12B-it/snapshots/" + "a" * 40)
    assert p["repo_id"] == "google/gemma-4-12B-it" and p["revision"] == "a" * 40
