"""R4 and Rice kernels for MLX's CUDA backend (mx.fast.cuda_kernel), the counterparts of
kernel.py and rice_kernel.py.

The matvec mirrors MLX 0.32.2's CUDA gemv (mlx/backend/cuda/gemms/gemv.cu, gemv_impl): one
warp per output row; thread t owns n consecutive weights of every block of 32 n along K,
accumulating float products in K order; then cooperative_groups::reduce over the warp,
called here too, so the reduction is the same code. n is 4 when K % 128 == 0, 2 when
K % 64 == 0, else 1, as MLX picks it (for aligned pointers). An R4 tile (128 weights) holds
whole blocks for every n, so a block's escapes are found with one warp prefix sum and a
running pointer per row. gemv and the expert gather (gather_mv) share gemv_impl, so one
kernel serves both. Products of two BF16 values are exact in float, so fused or separate
multiply-adds give the same sums.
"""

import mlx.core as mx

from ..format import TILE

HEADER = """
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
"""


def available() -> bool:
    """Whether MLX runs on its CUDA backend (then these kernels replace the Metal ones)."""
    return hasattr(mx, "cuda") and mx.cuda.is_available()


def n_per_thread(K: int) -> int:
    return 4 if K % 128 == 0 else 2 if K % 64 == 0 else 1


def _escapes(n: int) -> str:
    """Replace code-15 exponents in e[] from esc[], advancing epos (whole warp takes part)."""
    return f"""
      if (f) {{
        uint32_t cnt = 0;
        for (int j = 0; j < {n}; j++) {{ cnt += (in && c[j] == 15u) ? 1u : 0u; }}
        uint32_t incl = cnt;
        for (int o = 1; o < 32; o <<= 1) {{ uint32_t v = __shfl_up_sync(0xffffffffu, incl, o); if (lane >= o) incl += v; }}
        uint32_t q = epos + incl - cnt;
        for (int j = 0; j < {n}; j++) {{ if (in && c[j] == 15u) {{ e[j] = esc[q]; q++; }} }}
        epos += __shfl_sync(0xffffffffu, incl, 31);
      }}"""


def source_matvec(n: int, gather: bool) -> str:
    rowsel = (
        "uint32_t g = blockIdx.y; size_t row = size_t(ei[g]) * RS + R0 + size_t(r); const __nv_bfloat16* xv = x + size_t(xi[g]) * K; size_t yo = size_t(g) * N + r;"
        if gather
        else "size_t row = size_t(r); const __nv_bfloat16* xv = x; size_t yo = size_t(r);"
    )
    return f"""
    namespace cg = cooperative_groups;
    auto warp = cg::tiled_partition<32>(cg::this_thread_block());
    uint32_t lane = threadIdx.x % 32;
    int r = int(blockIdx.x) * 8 + int(threadIdx.x / 32);
    if (r >= N) {{ return; }}
    {rowsel}
    float sum = 0.0f;
    uint32_t epos = row_esc[row];
    for (int base = 0; base < K; base += {32 * n}) {{
      int col = base + {n} * int(lane);
      bool in = col < K;
      int t = base >> 7;
      int i = col & 127;
      uint32_t f = flags[row * T + t];
      uint32_t c[{n}]; uint32_t e[{n}]; uint32_t rw[{n}];
      uint32_t g4 = in ? uint32_t(codes[(row * T + t) * 32 + (i >> 2)]) : 0u;
      for (int j = 0; j < {n}; j++) {{
        c[j] = (g4 >> (4u * uint32_t((i & 3) + j))) & 15u;
        e[j] = tbl[c[j]];
        rw[j] = in ? uint32_t(raw[row * K + col + j]) : 0u;
      }}
      {_escapes(n)}
      if (in) {{
        for (int j = 0; j < {n}; j++) {{
          unsigned short u = (unsigned short)(((rw[j] & 0x80u) << 8) | (e[j] << 7) | (rw[j] & 0x7Fu));
          __nv_bfloat16 w = __ushort_as_bfloat16(u);
          sum += static_cast<float>(w) * static_cast<float>(xv[col + j]);
        }}
      }}
    }}
    sum = cg::reduce(warp, sum, cg::plus<float>());
    if (lane == 0) {{ y[yo] = static_cast<__nv_bfloat16>(sum); }}
"""


_MV = {}


def _matvec_kernel(n: int, gather: bool):
    key = (n, gather)
    if key not in _MV:
        inputs = ["x", "raw", "codes", "flags", "esc", "tbl", "row_esc"] + (["xi", "ei"] if gather else [])
        _MV[key] = mx.fast.cuda_kernel(
            name=f"binade_cuda_r4_mv_n{n}{'_g' if gather else ''}",
            input_names=inputs,
            output_names=["y"],
            source=source_matvec(n, gather),
            header=HEADER,
        )
    return _MV[key]


def r4_matvec(a: dict, x: mx.array, N: int, K: int) -> mx.array:
    T = -(-K // TILE)
    (y,) = _matvec_kernel(n_per_thread(K), False)(
        inputs=[x.reshape(-1), a["raw"], a["codes"], a["flags"], a["esc"], a["table"], a["row_esc"]],
        template=[("N", N), ("K", K), ("T", T)],
        grid=(-(-N // 8) * 256, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(N,)],
        output_dtypes=[mx.bfloat16],
    )
    return y


def r4_gather(a: dict, x: mx.array, xi: mx.array, ei: mx.array, n: int, rs: int, r0: int, K: int) -> mx.array:
    T = -(-K // TILE)
    G = int(ei.size)
    (y,) = _matvec_kernel(n_per_thread(K), True)(
        inputs=[x.reshape(-1), a["raw"], a["codes"], a["flags"], a["esc"], a["table"], a["row_esc"], xi.astype(mx.uint32), ei.astype(mx.uint32)],
        template=[("N", n), ("K", K), ("T", T), ("RS", rs), ("R0", r0)],
        grid=(-(-n // 8) * 256, G, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(G * n,)],
        output_dtypes=[mx.bfloat16],
    )
    return y.reshape(G, n)


SOURCE_DECODE = """
    uint32_t lane = threadIdx.x % 32;
    int orow = int(blockIdx.x) * 8 + int(threadIdx.x / 32);
    if (orow >= N) { return; }
    size_t row = size_t(orow / NB) * RS + R0 + size_t(orow % NB);
    uint32_t epos = row_esc[row];
    for (int t = 0; t < T; t++) {
      bool in = t * 128 + int(lane) * 4 < K;
      uint32_t f = flags[row * T + t];
      uint32_t g4 = uint32_t(codes[(row * T + t) * 32 + lane]);
      uint32_t c[4]; uint32_t e[4];
      for (int j = 0; j < 4; j++) { c[j] = (g4 >> (4u * uint32_t(j))) & 15u; e[j] = tbl[c[j]]; }
      if (f) {
        uint32_t cnt = 0;
        for (int j = 0; j < 4; j++) { cnt += c[j] == 15u ? 1u : 0u; }
        uint32_t incl = cnt;
        for (int o = 1; o < 32; o <<= 1) { uint32_t v = __shfl_up_sync(0xffffffffu, incl, o); if (lane >= o) incl += v; }
        uint32_t q = epos + incl - cnt;
        for (int j = 0; j < 4; j++) { if (c[j] == 15u) { e[j] = esc[q]; q++; } }
        epos += __shfl_sync(0xffffffffu, incl, 31);
      }
      if (in) {
        size_t k0 = row * K + size_t(t * 128) + size_t(lane) * 4;
        size_t o0 = size_t(orow) * K + size_t(t * 128) + size_t(lane) * 4;
        for (int j = 0; j < 4; j++) {
          uint32_t rj = raw[k0 + j];
          out[o0 + j] = (uint16_t)(((rj & 0x80u) << 8) | (e[j] << 7) | (rj & 0x7Fu));
        }
      }
    }
"""

_DECODE = []


def r4_decode(raw, codes, flags, esc, table, row_esc, N: int, K: int, nb: int | None = None, rs: int | None = None, r0: int = 0) -> mx.array:
    """BF16 weight [N, K] from R4 arrays; output row r is storage row (r // nb) * rs + r0 + r % nb."""
    if not _DECODE:
        _DECODE.append(
            mx.fast.cuda_kernel(
                name="binade_cuda_r4_decode",
                input_names=["raw", "codes", "flags", "esc", "tbl", "row_esc"],
                output_names=["out"],
                source=SOURCE_DECODE,
            )
        )
    (u,) = _DECODE[0](
        inputs=[raw, codes, flags, esc, table, row_esc],
        template=[("N", N), ("K", K), ("T", -(-K // TILE)), ("NB", nb or N), ("RS", rs or nb or N), ("R0", r0)],
        grid=(-(-N // 8) * 256, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(N * K,)],
        output_dtypes=[mx.uint16],
    )
    return u.view(mx.bfloat16).reshape(N, K)


SOURCE_RICE = """
    uint32_t lane = threadIdx.x % 32;
    int row = int(blockIdx.x) * 8 + int(threadIdx.x / 32);
    if (row >= R) { return; }
    unsigned long long cur = (unsigned long long)roff[row] * 32ull;
    for (int t = 0; t < TN; t++) {
      uint32_t m = meta[row * TN + t];
      uint32_t k = m & 3u;
      uint32_t qmax = m >> 2;
      int n = min(128, K - t * 128);
      uint32_t nv = uint32_t(max(0, min(4, n - int(lane) * 4)));
      uint32_t lo = 0u;
      uint32_t cnt0 = nv * k;
      if (cnt0 > 0u) {
        unsigned long long p = cur + (unsigned long long)(lane * 4u * k);
        uint32_t w = uint32_t(p >> 5), s = uint32_t(p & 31ull);
        lo = bits[w] >> s;
        if (s + cnt0 > 32u) { lo |= bits[w + 1] << (32u - s); }
        lo &= (1u << cnt0) - 1u;
      }
      cur += (unsigned long long)(uint32_t(n) * k);
      uint32_t q[4] = {0u, 0u, 0u, 0u};
      uint32_t alive = (1u << nv) - 1u;
      for (uint32_t j = 0; j < qmax; j++) {
        uint32_t c = __popc(alive);
        uint32_t incl = c;
        for (int o = 1; o < 32; o <<= 1) { uint32_t v = __shfl_up_sync(0xffffffffu, incl, o); if (lane >= o) incl += v; }
        uint32_t pre = incl - c;
        uint32_t tot = __shfl_sync(0xffffffffu, incl, 31);
        uint32_t b = 0u;
        if (c > 0u) {
          unsigned long long p = cur + pre;
          uint32_t w = uint32_t(p >> 5), s = uint32_t(p & 31ull);
          b = bits[w] >> s;
          if (s + c > 32u) { b |= bits[w + 1] << (32u - s); }
          b &= (1u << c) - 1u;
        }
        uint32_t na = 0u, bi = 0u;
        for (uint32_t i = 0; i < 4u; i++) {
          if ((alive >> i) & 1u) { uint32_t v = (b >> bi) & 1u; q[i] += v; na |= v << i; bi++; }
        }
        alive = na;
        cur += tot;
      }
      uint32_t mask = (1u << k) - 1u;
      uint32_t packed = 0u;
      for (uint32_t i = 0; i < 4u; i++) { packed |= ((q[i] << k) | ((lo >> (i * k)) & mask)) << (8u * i); }
      out[((size_t)row * TN + t) * 32 + lane] = packed;
    }
    if (lane == 0) { rend[row] = uint32_t((cur + 31ull) >> 5); }
"""

_RICE = []


def rice_ranks(meta: mx.array, bits: mx.array, roff: mx.array, K: int) -> tuple[mx.array, mx.array]:
    """Same contract as rice_kernel.rice_ranks, on CUDA."""
    if not _RICE:
        _RICE.append(
            mx.fast.cuda_kernel(
                name="binade_cuda_rice_ranks",
                input_names=["meta", "bits", "roff"],
                output_names=["out", "rend"],
                source=SOURCE_RICE,
            )
        )
    R = roff.size
    TN = -(-K // TILE)
    u, rend = _RICE[0](
        inputs=[meta, bits, roff],
        template=[("R", R), ("K", K), ("TN", TN)],
        grid=(-(-R // 8) * 256, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(R * TN * 32,), (R,)],
        output_dtypes=[mx.uint32, mx.uint32],
    )
    return u.view(mx.uint8).reshape(R, TN * TILE), rend
