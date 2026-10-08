"""The fused R4 tensor-core GEMM on CUDA against MLX's BF16 matmul (cuBLASLt), bit for bit."""

import mlx.core as mx
import numpy as np
import pytest

from binade.mlx.cuda_kernels import available as cuda_available

pytestmark = pytest.mark.skipif(not cuda_available(), reason="CUDA backend only")

from test_r4_moe import _adversarial, _bits, _r4, _weights  # noqa: E402

# How often the fused path is taken is a speed policy (r4.CUDA_FUSED_MAX_ROWS); these tests run
# the kernels directly at every row count.


@pytest.mark.parametrize("N,K", [(512, 3840), (512, 2048), (2048, 3840), (3840, 2816), (1024, 704), (4096, 4096), (3840, 15360), (2816, 8192)])
@pytest.mark.parametrize("M", [2, 7, 16, 40, 128, 300, 512, 1024])
def test_fused_gemm_matches_cublas(N, K, M):
    """Order-sensitive and random data: the runtime's product is bit-identical to x @ W.T on any
    GPU, and on a GPU the fused GEMM mirrors (cublas_plan.mirrored) so is the fused R4 GEMM run
    with whatever split-K plan cuBLAS picks for (M, N, K)."""
    from binade.mlx import cublas_plan
    from binade.mlx import cuda_kernels as ck

    p = cublas_plan.plan(M, N, K)
    assert p is not None
    split = cublas_plan.supported(p)
    rng = np.random.default_rng(N + K + M)
    for w, x in ((_adversarial(N, K), mx.ones((M, K), dtype=mx.bfloat16)), (_weights(rng, (N, K)), mx.array(rng.standard_normal((M, K)).astype(np.float32)).astype(mx.bfloat16))):
        m = _r4(w)
        ref = x @ mx.array(w).view(mx.bfloat16).T
        got = m._rows(x)
        mx.eval(ref, got)
        assert np.array_equal(_bits(got), _bits(ref)), (p, M, N, K)
        sl = cublas_plan.slices(M, N, K)
        if split is not None and sl is not None and cublas_plan.mirrored():  # a GPU and algorithm the fused GEMM reproduces
            for tile in (None, "big128") + (("big256",) if sl == 1 else ()) + (("regs",) if K % 64 == 0 else ()):
                fused = ck.r4_gemm(m._a, x, N, K, cublas_plan.partitions(K, p[0]), split, tile, sl)
                mx.eval(fused)
                assert np.array_equal(_bits(fused), _bits(ref)), (p, M, N, K, tile)


@pytest.mark.parametrize("E,N,K,S", [(8, 48, 704, 67), (8, 96, 2816, 9), (16, 64, 704, 200), (4, 32, 2816, 300)])
def test_fused_gather_matches_gather_mm(E, N, K, S):
    """The MoE prompt path: slots sorted by expert, MLX's gather_mm with sorted indices (a CUTLASS
    grouped GEMM on CUDA) against the fused R4 gather GEMM, whole rows and the upper half of each
    expert's rows (gate_up's up half), on order-sensitive and random data."""
    from binade.mlx.cuda_kernels import r4_gather_mm

    rng = np.random.default_rng(E * N + S)
    idx = mx.array(np.sort(rng.integers(0, E, size=S)).astype(np.uint32))
    for w, x in ((_adversarial(E * N, K).reshape(E, N, K), mx.ones((S, 1, K), dtype=mx.bfloat16)), (_weights(rng, (E, N, K)), mx.array(rng.standard_normal((S, 1, K)).astype(np.float32)).astype(mx.bfloat16))):
        m = _r4(w)
        W = mx.array(w).view(mx.bfloat16)
        ref = mx.gather_mm(x, W.swapaxes(-1, -2), rhs_indices=idx, sorted_indices=True).reshape(S, N)
        got = r4_gather_mm(m._a, x, idx, N, N, 0, K)
        half = N // 2
        ref_h = mx.gather_mm(x, mx.contiguous(W[:, half:]).swapaxes(-1, -2), rhs_indices=idx, sorted_indices=True).reshape(S, N - half)
        got_h = r4_gather_mm(m._a, x, idx, N - half, N, half, K)
        mx.eval(ref, got, ref_h, got_h)
        assert np.array_equal(_bits(got), _bits(ref))
        assert np.array_equal(_bits(got_h), _bits(ref_h))
