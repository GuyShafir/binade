"""R4 runtime: Binade format 1 or 2 on disk, 4-bit rank codes in memory, the R4 kernel
for decoding.

`convert(model)` replaces every BinadeLinear / BinadeEmbedding (file arrays) with an
R4Linear / R4Embedding. The transcode runs on the GPU at load: ranks come from the
format 1 decode (exponents mapped through the inverse of the per-tensor table) or the
format 2 Rice decode kernel; codes are min(rank, 15) packed four to a uint16, and ranks
>= 15 go to an escape stream. The file's index arrays are released, so memory holds the
raw bytes plus about 4.06 bits per weight.

Single-row inputs (batch-1 decode) run the R4 kernel, whose outputs are bit-identical to
MLX's BF16 matvec. Multi-row inputs (prompt processing) rebuild the BF16 weight with the
R4 decode kernel and call MLX's matmul, bit-identical by construction.
"""

import re
from functools import lru_cache

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from ..format import TILE
from . import cublas_plan
from . import cuda_kernels as ck
from .kernel import _kernel_r4_gemm, _kernel_r4_pf
from .kernel import r4_decode as r4_decode_metal
from .moe import BinadeSwitchGLU, R4SwitchGLU, holders
from .rice_kernel import rice_ranks
from .slow_linear import CHUNK, BinadeEmbedding, BinadeLinear, _plan

PREFETCH = 2  # tiles loaded ahead in the matvec kernel
CUDA = ck.available()  # MLX's CUDA backend: cuda_kernels.py replaces the Metal kernels


def _u32(a: mx.array) -> mx.array:
    return a if a.dtype == mx.uint32 else a.astype(mx.uint32)


def r4_decode(*args, **kwargs) -> mx.array:
    return (ck.r4_decode if CUDA else r4_decode_metal)(*args, **kwargs)


@lru_cache(maxsize=None)
def _device() -> tuple[int, str]:
    arch = mx.device_info()["architecture"]  # e.g. applegpu_g16s: generation 16, class s
    m = re.search(r"g(\d+)", arch)
    return (int(m.group(1)) if m else 0), arch[-1]


def use_matvec(N: int, K: int) -> bool:
    """Whether MLX's batch-1 x @ W.T runs the gemv layout the R4 matvec mirrors (32 lanes per
    row, 4 weights per lane): not for K <= 64 (4 lanes per row) or K >= 16 N (K split
    across 8 SIMD groups), per gemv_axbpy in mlx/backend/metal/matmul.cpp. The expert
    gather (gather_mv) always uses it."""
    return 64 < K < 16 * N


CUDA_FUSED_MAX_ROWS = 128  # multi-row inputs up to this many rows take the fused R4 GEMM on CUDA when cuBLAS's plan allows (above, the rebuild's decode overlaps other layers' work and ties or wins end to end)
REBUILD_MARGIN = 1 << 30  # bytes kept free below the GPU working set when rebuilding a weight
FUSED_MAX_ROWS = 128  # up to this many rows the R4 GEMM beats rebuild + MLX GEMM (prefill_bench.md)


def gemm_tile(rows: int) -> tuple:
    """R4 GEMM tile (bm, bn, wm, wn) for about `rows` slots per expert: 64 x 64
    (four SIMD groups of 32 x 32) unless a 64-row block would be at least half empty. Tiling
    does not change any output's summation order, so this is a speed choice only (measured on
    Gemma 4 12B's shapes, scripts/prefill_bench.py)."""
    if rows <= 16:
        return (16, 32, 1, 2)
    if rows <= 32 or -(-rows // 64) * 64 - rows >= 32:
        return (32, 64, 2, 2)
    return (64, 64, 2, 2)


def memory_limit() -> int:
    """Bytes of GPU memory to plan within: Metal's recommended working set, or CUDA's total."""
    info = mx.device_info()
    return info.get("max_recommended_working_set_size") or info["total_memory"]


def rebuild_fits(nbytes: int) -> bool:
    """Whether a BF16 weight of nbytes can be built without crowding the GPU working set.
    Rebuilding and calling MLX's GEMM is exact by construction and, with room, faster than
    the fused R4 GEMM; near the limit, fresh allocations page (Gemma 4 26B)."""
    return mx.get_active_memory() + nbytes + REBUILD_MARGIN <= memory_limit()


def defer_rebuilds() -> bool:
    """Whether a prompt pass may leave its rebuilt BF16 weights in the lazy graph instead of
    evaluating each product as it is built. Evaluating bounds memory but synchronizes once per
    weight; deferring is safe when even every weight rebuilt at once (about 1.33 times the R4
    weights, 16 / 12.07 bits) fits beside what is resident. On the RTX PRO 6000 (96 GB) the
    first token of a 40-token 12B prompt comes 21 ms sooner (97 to 76 ms); on a 48 GB Mac
    holding the 12B it does not apply."""
    return 2.4 * mx.get_active_memory() <= memory_limit()


def use_mma(M: int, N: int, K: int) -> bool:
    """Whether MLX 0.32's x @ W.T for M rows runs its steel GEMM with K in aligned blocks of 16
    (so every output accumulates in simdgroup MMA steps of 8 along K, in order), which the R4
    gather GEMM reproduces. M <= 15 goes to gemv / gemv_wide instead, and the split-K GEMM
    sums K in partitions. Only M3 and M4 GPUs (generations 15, 16; no neural accelerators)
    are known to dispatch this way (mlx/backend/metal/matmul.cpp, Matmul::eval_gpu and
    steel_matmul_axpby)."""
    gen, devc = _device()
    if gen not in (15, 16) or K % 16 or M < 16:
        return False
    tm, tn, tk = -(-M // 16), -(-N // 16), K // 16
    split_k = tm * tn <= (2048 if devc in "sd" else 1024) and tk >= 8 and K >= max(M, N)
    return not split_k


def _rank_chunks(module, R: int, K: int):
    """[tiles, 128] uint8 ranks of consecutive whole rows, chunk by chunk."""
    w = module.weight
    if "bits" in w:  # format 2, Rice
        T = -(-K // TILE)
        rows = max(1, CHUNK // K)
        bits = w["bits"] if w["bits"].size else mx.zeros((1,), dtype=mx.uint32)
        ends = mx.concatenate([w["roff"][1:], mx.array([w["bits"].size], dtype=mx.uint32)])
        for r0 in range(0, R, rows):
            r1 = min(R, r0 + rows)
            rank, rend = rice_ranks(w["meta"][r0 * T : r1 * T], bits, w["roff"][r0:r1], K)
            if not mx.array_equal(rend, ends[r0:r1]).item():
                raise ValueError(f"Rice stream of rows {r0}..{r1} does not end where the next row starts")
            yield rank.reshape(-1, TILE)
        return
    table_np = np.array(w["table"])
    inv = np.full(256, 255, np.int64)
    for r in range(255, -1, -1):  # smallest rank wins for zero padding
        inv[table_np[r]] = r
    inv_mx = mx.array(inv.astype(np.uint8))
    for c in _plan(module).chunks:
        yield inv_mx[c.exponents(w)]


def transcode(module) -> dict:
    """R4 arrays for a format 1 or 2 module. Rows of K weights, K % 4 == 0 (raw bytes are read
    four at a time); a last tile with K % 128 weights is padded with code 0."""
    w = module.weight
    raw = w["raw"]
    K = raw.shape[-1]
    R = raw.size // K
    if K % 4:
        raise ValueError(f"R4 needs K % 4 == 0, got {K}")
    T = -(-K // TILE)
    table_np = np.array(w["table"])
    codes, flags, esc, per_row = [], [], [], []
    for rank in _rank_chunks(module, R, K):
        q = mx.minimum(rank, 15).astype(mx.uint16).reshape(-1, 32, 4)
        codes.append(q[..., 0] | (q[..., 1] << 4) | (q[..., 2] << 8) | (q[..., 3] << 12))
        hit = rank >= 15
        per_tile = hit.astype(mx.int32).sum(axis=1)
        mx.eval(codes[-1], per_tile)
        pt = np.array(per_tile)
        flags.append((pt > 0).astype(np.uint8))
        per_row.append(pt.reshape(-1, T).sum(axis=1))
        if pt.any():
            esc.append(table_np[np.array(rank)[np.array(hit)]])
    per_row = np.concatenate(per_row)
    return {
        "raw": raw.reshape(R, K),
        "table": w["table"],
        "codes": mx.concatenate(codes).reshape(-1),
        "flags": mx.array(np.concatenate(flags)),
        "esc": mx.array(np.concatenate(esc + [np.zeros(1, np.uint8)])),
        "row_esc": mx.array((np.cumsum(per_row) - per_row).astype(np.uint32)),
        "shape": tuple(raw.shape),
    }


class _R4:
    """Shared R4 storage, the kernel matvec and the chunked BF16 rebuild."""

    def _init_r4(self, a: dict):
        self._a = a
        self.N, self.K = a["raw"].shape
        self.T = -(-self.K // TILE)

    def _matvec(self, x: mx.array) -> mx.array:
        """x [..., K] holding one row -> [..., N]. Inputs and output keep the caller's shapes,
        so a call adds no reshape to the graph: decoding a MoE model on a fast GPU can be
        bound by the CPU's graph building, not the GPU."""
        a = self._a
        if CUDA:
            return ck.r4_matvec(a, x, self.N, self.K)
        bm = 8 if self.N >= 4096 else 4
        groups = -(-self.N // bm)
        (y,) = _kernel_r4_pf(PREFETCH, self.K % TILE != 0)(
            inputs=[x, a["raw"], a["codes"], a["flags"], a["esc"], a["table"], a["row_esc"]],
            template=[("N", self.N), ("K", self.K), ("T", self.T), ("BM", bm)],
            grid=(groups * bm * 32, 1, 1),
            threadgroup=(bm * 32, 1, 1),
            output_shapes=[(*x.shape[:-1], self.N)],
            output_dtypes=[mx.bfloat16],
        )
        return y

    def _gather(self, x: mx.array, xi: mx.array, ei: mx.array, n: int, rs: int, r0: int = 0, *, split: int | None = None, shape: tuple | None = None):
        """[G, n] outputs (or `shape`): pair g multiplies input row xi[g] of x [.., K] by storage
        rows ei[g] * rs + r0 .. + n (one expert's rows). Bit-identical to MLX's gemv_gather.
        With split, the pair [G, split], [G, n - split] (gate and up of fused expert rows)."""
        a = self._a
        if CUDA:
            return ck.r4_gather(a, x, xi, ei, n, rs, r0, self.K, split, shape)
        G = int(ei.size)
        bm = 8 if n >= 4096 else 4
        shapes = [(G, split), (G, n - split)] if split else [shape or (G, n)]
        out = _kernel_r4_pf(PREFETCH, self.K % TILE != 0, True, bool(split))(
            inputs=[x, a["raw"], a["codes"], a["flags"], a["esc"], a["table"], a["row_esc"], _u32(xi), _u32(ei)],
            template=[("N", n), ("K", self.K), ("T", self.T), ("BM", bm), ("RS", rs), ("R0", r0)] + ([("SPLIT", split)] if split else []),
            grid=(-(-n // bm) * bm * 32, G, 1),
            threadgroup=(bm * 32, 1, 1),
            output_shapes=shapes,
            output_dtypes=[mx.bfloat16] * len(shapes),
        )
        return (out[0], out[1]) if split else out[0]

    def _gather_mm(self, x: mx.array, idx: mx.array, n: int, rs: int, r0: int = 0, *, tile: tuple | None = None) -> mx.array:
        """[S, n] = rows of x [S, K] times storage rows idx[s] * rs + r0 .. + n, for slots sorted
        by expert. Bit-identical to MLX's gather_mm(x, W.swapaxes(-1, -2), idx, sorted_indices=True),
        and with all idx 0 to MLX's x @ W.T wherever use_mma holds. tile = (bm, bn, wm, wn),
        by default gemm_tile for the average slots per expert."""
        a = self._a
        S = int(idx.size)
        if self.K % 8:
            raise ValueError(f"the R4 GEMM needs K % 8 == 0, got {self.K}")
        bm, bn, wm, wn = tile or gemm_tile(-(-S // max(1, self.N // rs)))
        p = mx.array([S, n, self.K, self.T, rs, r0], dtype=mx.uint32)
        (y,) = _kernel_r4_gemm(bm, bn, wm, wn)(
            inputs=[x.reshape(S, self.K), idx.astype(mx.uint32), a["raw"], a["codes"], a["flags"], a["esc"], a["table"], a["row_esc"], p],
            grid=(-(-n // bn) * wm * wn * 32, -(-S // bm), 1),
            threadgroup=(wm * wn * 32, 1, 1),
            output_shapes=[(S, n)],
            output_dtypes=[mx.bfloat16],
        )
        return y

    def dense(self, n: int | None = None, rs: int | None = None, r0: int = 0) -> mx.array:
        """The BF16 weight, rebuilt from R4 by the decode kernel. With n, rs, r0: rows
        r0 .. r0 + n - 1 of every block of rs storage rows, as [N * n / rs, K]."""
        a = self._a
        if n is None:
            return r4_decode(a["raw"], a["codes"], a["flags"], a["esc"], a["table"], a["row_esc"], self.N, self.K).reshape(a["shape"])
        return r4_decode(a["raw"], a["codes"], a["flags"], a["esc"], a["table"], a["row_esc"], self.N // rs * n, self.K, n, rs, r0)


    def _cuda_fused(self, x: mx.array, M: int):
        """x @ W.T for M > 1 rows on CUDA with the fused R4 tensor-core GEMM, or None when the
        runtime must rebuild BF16 for cuBLAS instead: MLX sends these products to cuBLASLt, and
        the fused kernel reproduces cuBLAS's algorithm without split-K and with split-K over BF16
        or float32 partials or in place, summing K in order or in two slices (cublas_plan). The
        plan is cuBLASLt's own choice for this shape."""
        if M > CUDA_FUSED_MAX_ROWS or self.K % 32 or not cublas_plan.mirrored():
            return None
        p = cublas_plan.plan(M, self.N, self.K)
        split = cublas_plan.supported(p)
        sl = cublas_plan.slices(M, self.N, self.K)
        if split is None or sl is None or not ck.gemm_supported(M, self.N, self.K, sl):
            return None
        bounds = cublas_plan.partitions(self.K, p[0])
        return ck.r4_gemm(self._a, x.reshape(M, self.K), self.N, self.K, bounds, split, slices=sl)

    def _rows(self, x: mx.array) -> mx.array:
        """x [..., K] @ W.T for any number of rows, bit-identical to MLX on the BF16 weight.
        One row: the matvec kernel. Where MLX runs its plain GEMM (use_mma), the R4 GEMM for
        short inputs (up to FUSED_MAX_ROWS) or when a BF16 rebuild would crowd GPU memory.
        Otherwise the BF16 weight is rebuilt and MLX's own matmul runs (evaluated here unless
        defer_rebuilds, so rebuilt weights do not pile up in a lazy graph); its fixed decode
        cost pays off on long prompts."""
        lead = x.shape[:-1]
        M = x.size // self.K
        if M == 1 and (CUDA or use_matvec(self.N, self.K)):  # MLX's CUDA backend runs gemv for any K
            return self._matvec(x)
        if CUDA and M > 1:
            y = self._cuda_fused(x, M)
            if y is not None:
                return y.reshape(*lead, self.N)
        if not CUDA and use_mma(M, self.N, self.K) and (M <= FUSED_MAX_ROWS or not rebuild_fits(2 * self.N * self.K)):
            y = self._gather_mm(x.reshape(M, self.K), mx.zeros((M,), dtype=mx.uint32), self.N, self.N)
            return y.reshape(*lead, self.N)
        y = x @ self.dense().T
        if not defer_rebuilds():
            mx.eval(y)
        return y


class R4Linear(nn.Module, _R4):
    def __init__(self, a: dict):
        nn.Module.__init__(self)
        self._init_r4(a)

    def __call__(self, x: mx.array) -> mx.array:
        return self._rows(x)


class R4Embedding(nn.Module, _R4):
    def __init__(self, a: dict):
        nn.Module.__init__(self)
        self._init_r4(a)

    def __call__(self, ids: mx.array) -> mx.array:
        """Rows for token ids, rebuilt lazily with MLX ops. No host sync, so mlx_lm keeps
        building the next decode step while the GPU runs the current one."""
        a = self._a
        flat = ids.reshape(-1).astype(mx.uint32)
        n, W = flat.size, self.T * 32
        g = a["codes"][flat[:, None] * W + mx.arange(W, dtype=mx.uint32)[None, :]]
        c = ((g[..., None] >> mx.array([0, 4, 8, 12], dtype=mx.uint16)) & 15).astype(mx.uint8).reshape(n, W * 4)[:, : self.K]
        e = a["table"][c]
        hit = c == 15
        pos = a["row_esc"][flat][:, None].astype(mx.int64) + mx.cumsum(hit.astype(mx.int32), axis=1) - 1
        e = mx.where(hit, a["esc"][mx.maximum(pos, 0).astype(mx.uint32)], e).astype(mx.uint16)
        r = a["raw"][flat].astype(mx.uint16)
        out = (((r & 0x80) << 8) | (e << 7) | (r & 0x7F)).view(mx.bfloat16)
        return out.reshape(*ids.shape, self.K)

    def as_linear(self, x: mx.array) -> mx.array:
        return self._rows(x)


def _evaluated(a: dict) -> dict:
    mx.eval(a["raw"], a["codes"], a["flags"], a["esc"], a["row_esc"])
    return a


def convert(model: nn.Module, log=None) -> int:
    """Swap every v1 module for its R4 counterpart in place; returns how many were converted.
    Holds only paths between swaps, so each v1 module is freed as soon as it is replaced."""
    paths = [path for path, m in model.named_modules() if isinstance(m, (BinadeLinear, BinadeEmbedding, BinadeSwitchGLU))]

    def resolve(path):
        parent_path, _, leaf = path.rpartition(".")
        parent = model
        for part in parent_path.split(".") if parent_path else []:
            parent = parent[int(part)] if isinstance(parent, list) else getattr(parent, part)
        return parent, leaf

    for i, path in enumerate(paths, 1):
        parent, leaf = resolve(path)
        m = parent[int(leaf)] if isinstance(parent, list) else getattr(parent, leaf)
        if isinstance(m, BinadeSwitchGLU):
            gu, dn = (R4Linear(_evaluated(transcode(h))) for h in holders(m))
            new = R4SwitchGLU(gu, dn, m._activation)
            del gu, dn
        else:
            a = _evaluated(transcode(m))
            new = R4Embedding(a) if isinstance(m, BinadeEmbedding) else R4Linear(a)
            del a
        if isinstance(parent, list):
            parent[int(leaf)] = new
        else:
            setattr(parent, leaf, new)
        del m, new
        mx.clear_cache()
        if log and (i % 50 == 0 or i == len(paths)):
            log(f"R4 transcode {i}/{len(paths)}")
    return len(paths)


def load(path, log=None, cache_limit_gb: float = 2, wire: bool = True):
    """(model, tokenizer, dropped) for a Binade model, converted to the R4 runtime.

    Caps MLX's buffer cache (global) at cache_limit_gb: decoding needs only small buffers,
    and the few BF16 weights still rebuilt for multi-row inputs would otherwise stay cached
    next to the R4 weights. With wire (global, like mlx_lm's generate does while it runs),
    MLX wires memory up to the GPU's recommended working set before the weights are built,
    so macOS cannot page them out between load and use: on a 48 GB Mac holding the 26B
    (35.9 GiB), unwired weights made the first prompt pass 9 to 20 s instead of 0.6 s."""
    from .loader import load as load_v1

    if wire and not CUDA and mx.metal.is_available():
        mx.set_wired_limit(int(mx.device_info()["max_recommended_working_set_size"]))

    model, tokenizer, dropped = load_v1(path, cache_limit_gb=cache_limit_gb)
    convert(model, log)
    mx.clear_cache()
    return model, tokenizer, dropped
