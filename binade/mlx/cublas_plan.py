"""Which algorithm cuBLASLt runs for MLX's BF16 x @ W.T on CUDA, read through ctypes.

MLX 0.32.2's CUDA backend sends every multi-row BF16 matmul to cuBLASLt (matmul.cpp; gemv only
for one row). cuBLAS is closed, but on the GPUs we measured its BF16 GEMM sums each output as
tensor-core m16n8k16 steps along K, in order, from zero; with split-K into S partitions, each
partition is summed that way, rounded to BF16 (reduction scheme OUTPUT_TYPE), and the partials
are added in float32 in partition order; with reduction scheme COMPUTE_TYPE the partials stay
float32, and with INPLACE each partition adds its float32 sum to the BF16 output so far. One
algorithm (31, on the A100 and L4) sums each output in two accumulators over alternating
32-wide K slices instead (DESIGN.md section 9; the paper, paper/binade.pdf). A kernel that does
the same is bit-identical, so the runtime needs the algorithm, S and the reduction scheme, which
depend on the shape and the GPU.

`plan(M, N, K)` builds the same descriptors MLX does (gemms/cublas_gemm.cpp, cublas_utils.cpp:
A and B swapped for row-major output, TRANSA = T, TRANSB = N, compute and scale type float32,
host pointer mode, a 32 MiB workspace on compute capability 9 and up, else 4 MiB) and asks
cublasLtMatmulAlgoGetHeuristic for its first result, as MLX does; it returns the algorithm's
split-K count and reduction scheme (`config` returns every attribute, `slices` the K slicing of
a verified algorithm), or None if cuBLASLt cannot be loaded or queried.
"""

import ctypes
import glob
import os
import site
from functools import lru_cache

import mlx.core as mx

_COMPUTE_32F = 68
_R_32F, _R_16BF = 0, 14
_OP_N, _OP_T = 0, 1
_DESC_POINTER_MODE, _DESC_TRANSA, _DESC_TRANSB = 2, 3, 4
_PREF_MAX_WORKSPACE_BYTES = 1
_CFG_SPLITK_NUM, _CFG_REDUCTION_SCHEME = 2, 3
REDUCTION_NONE, REDUCTION_OUTPUT_TYPE = 0, 4


class _Algo(ctypes.Structure):
    _fields_ = [("data", ctypes.c_uint64 * 8)]


class _Heuristic(ctypes.Structure):
    _fields_ = [("algo", _Algo), ("workspaceSize", ctypes.c_size_t), ("state", ctypes.c_int), ("wavesCount", ctypes.c_float), ("reserved", ctypes.c_int * 4)]


@lru_cache(maxsize=None)
def _lib():
    """cuBLASLt from the CUDA wheels MLX links against (site-packages/nvidia/cublas/lib)."""
    roots = site.getsitepackages() + [site.getusersitepackages()]
    paths = sorted(p for d in roots for p in glob.glob(os.path.join(d, "nvidia", "cublas", "lib", "libcublasLt.so*")))
    if not paths:
        return None
    lib = ctypes.CDLL(paths[0])
    h = ctypes.c_void_p()
    if lib.cublasLtCreate(ctypes.byref(h)) != 0:
        return None
    ws = (32 if int(mx.device_info().get("compute_capability_major", 0)) >= 9 else 4) * 1024 * 1024
    pref = ctypes.c_void_p()
    v = ctypes.c_uint64(ws)
    if lib.cublasLtMatmulPreferenceCreate(ctypes.byref(pref)) or lib.cublasLtMatmulPreferenceSetAttribute(pref, _PREF_MAX_WORKSPACE_BYTES, ctypes.byref(v), ctypes.sizeof(v)):
        return None
    return lib, h, pref


def _layout(lib, rows: int, cols: int, ld: int):
    d = ctypes.c_void_p()
    if lib.cublasLtMatrixLayoutCreate(ctypes.byref(d), _R_16BF, ctypes.c_uint64(rows), ctypes.c_uint64(cols), ctypes.c_int64(ld)):
        raise RuntimeError("cublasLtMatrixLayoutCreate")
    return d


CONFIG = ("algo", "tile", "split_k", "reduction", "swizzle", "custom", "stages", "inner_shape", "cluster_shape")  # cublasLtMatmulAlgoConfigAttributes_t 0..8


@lru_cache(maxsize=4096)
def config(M: int, N: int, K: int) -> dict | None:
    """Every attribute of the algorithm cuBLASLt picks for MLX's BF16 x [M, K] @ W.T (CONFIG:
    algorithm and tile ids, split-K count, reduction scheme, ...); None if unavailable."""
    return _heuristic(M, N, K, tuple(range(len(CONFIG))))


@lru_cache(maxsize=4096)
def plan(M: int, N: int, K: int):
    """(split_k, reduction_scheme) cuBLASLt picks for MLX's BF16 x [M, K] @ W.T, W [N, K]; None
    if unavailable. Cached: the choice depends only on the shape, dtypes and GPU."""
    c = _heuristic(M, N, K, (_CFG_SPLITK_NUM, _CFG_REDUCTION_SCHEME))
    return None if c is None else (max(1, c["split_k"]), c["reduction"])


def _heuristic(M: int, N: int, K: int, attrs: tuple):
    ctx = _lib()
    if ctx is None:
        return None
    lib, h, pref = ctx
    desc = ctypes.c_void_p()
    if lib.cublasLtMatmulDescCreate(ctypes.byref(desc), _COMPUTE_32F, _R_32F):
        return None
    try:
        for attr, val in ((_DESC_TRANSA, _OP_T), (_DESC_TRANSB, _OP_N), (_DESC_POINTER_MODE, 0)):
            v = ctypes.c_int(val)
            if lib.cublasLtMatmulDescSetAttribute(desc, attr, ctypes.byref(v), ctypes.sizeof(v)):
                return None
        layouts = [_layout(lib, K, N, K), _layout(lib, K, M, K), _layout(lib, N, M, N)]
        res, n = _Heuristic(), ctypes.c_int(0)
        st = lib.cublasLtMatmulAlgoGetHeuristic(h, desc, layouts[0], layouts[1], layouts[2], layouts[2], pref, 1, ctypes.byref(res), ctypes.byref(n))
        for d in layouts:
            lib.cublasLtMatrixLayoutDestroy(d)
        if st or n.value < 1:
            return None
        out = {}
        for attr in attrs:
            # attributes are 16 or 32 bits wide and the buffer size must match: ask for it first
            v, need = ctypes.c_uint64(0), ctypes.c_size_t(0)
            if lib.cublasLtMatmulAlgoConfigGetAttribute(ctypes.byref(res.algo), attr, None, 0, ctypes.byref(need)) or need.value > 8:
                return None
            if lib.cublasLtMatmulAlgoConfigGetAttribute(ctypes.byref(res.algo), attr, ctypes.byref(v), need.value, ctypes.byref(need)):
                return None
            out[CONFIG[attr]] = int(v.value)
        return out
    finally:
        lib.cublasLtMatmulDescDestroy(desc)


SPLIT_ALIGN = 64  # partition size = round_up(ceil(K / S), SPLIT_ALIGN), measured on the RTX PRO 6000
REDUCTION_COMPUTE_TYPE = 2  # float32 partials (OUTPUT_TYPE: partials rounded to the output type, BF16)
REDUCTION_INPLACE = 1  # serial split-K in the output: each partition adds its float sum to the BF16 output so far


def partitions(K: int, S: int) -> list[int]:
    """K boundaries of cuBLAS's S split-K partitions: sizes of ceil(K / S) rounded up to a
    multiple of SPLIT_ALIGN; trailing partitions may be short or empty (an empty partition
    still adds a zero partial, as cuBLAS's does)."""
    if S == 1:
        return [0, K]
    size = -(-(-(-K // S)) // SPLIT_ALIGN) * SPLIT_ALIGN
    return [min(i * size, K) for i in range(S + 1)]


# Per compute capability, the cuBLASLt algorithms whose order the fused GEMM reproduces, as the
# number of 32-wide K slices summed apart and added at the end (1: in order; 2: sliced-K),
# verified by tests/test_cuda_gemm.py on that GPU: RTX PRO 6000, L4, A100.
ALGO_SLICES = {(12, 0): {21: 1}, (8, 9): {21: 1, 6: 1, 5: 1, 31: 2}, (8, 0): {21: 1, 6: 1, 31: 2}}


@lru_cache(maxsize=None)
def _algos() -> dict | None:
    """This GPU's entry of ALGO_SLICES; BINADE_CUDA_ALGOS="21:1,6:1" replaces it (to check a GPU)."""
    env = os.environ.get("BINADE_CUDA_ALGOS")
    if env:
        return {int(k): int(v) for k, v in (kv.split(":") for kv in env.split(","))}
    info = mx.device_info()
    return ALGO_SLICES.get((int(info.get("compute_capability_major", 0)), int(info.get("compute_capability_minor", 0))))


@lru_cache(maxsize=None)
def mirrored() -> bool:
    """Whether this GPU's cuBLAS kernels are ones the fused GEMM reproduces. The split-K rule and
    the in-order m16n8k16 steps were measured on compute capability 12.0; other GPUs can pick
    other kernels, so elsewhere the runtime rebuilds BF16 weights for cuBLAS itself (and on a
    listed GPU for any algorithm not listed). BINADE_CUDA_ALGOS sets the algorithms, to check a
    new GPU with tests/test_cuda_gemm.py."""
    return _algos() is not None


def slices(M: int, N: int, K: int) -> int | None:
    """K slices of cuBLAS's kernel for this product (1: in-order m16n8k16 steps; 2: two accumulators
    over alternating 32-wide K slices, added at the end), None for an algorithm not measured.
    Sliced-K is only known without split-K."""
    c, algos = config(M, N, K), _algos()
    if c is None or algos is None or c["algo"] not in algos:
        return None
    n = algos[c["algo"]]
    return n if n == 1 or c["split_k"] <= 1 else None


def supported(p) -> str | None:
    """How the fused kernel reproduces plan p: "none" (no split-K), "bf16" or "f32" partials,
    "inplace" (float partials added in order to the BF16 output so far); None for anything else
    (the runtime then rebuilds BF16 weights for cuBLAS)."""
    if p is None:
        return None
    S, scheme = p
    if S == 1 and scheme == REDUCTION_NONE:
        return "none"
    if S > 1 and scheme == REDUCTION_OUTPUT_TYPE:
        return "bf16"
    if S > 1 and scheme == REDUCTION_COMPUTE_TYPE:
        return "f32"
    if S > 1 and scheme == REDUCTION_INPLACE:
        return "inplace"
    return None
