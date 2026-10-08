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


def source_matvec(n: int, gather: bool, split: bool = False) -> str:
    """split (gather only): rows below SPLIT go to y [G, SPLIT], the rest to y2 [G, N - SPLIT]
    (the gate and up halves of fused expert rows, without slicing afterwards)."""
    rowsel = (
        "uint32_t g = blockIdx.y; size_t row = size_t(ei[g]) * RS + R0 + size_t(r); const __nv_bfloat16* xv = x + size_t(xi[g]) * K; size_t yo = size_t(g) * N + r;"
        if gather
        else "size_t row = size_t(r); const __nv_bfloat16* xv = x; size_t yo = size_t(r);"
    )
    store = (
        "if (lane == 0) { if (r < SPLIT) { y[size_t(g) * SPLIT + r] = static_cast<__nv_bfloat16>(sum); } else { y2[size_t(g) * (N - SPLIT) + (r - SPLIT)] = static_cast<__nv_bfloat16>(sum); } }"
        if split
        else "if (lane == 0) { y[yo] = static_cast<__nv_bfloat16>(sum); }"
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
    {store}
"""


_MV = {}


def _matvec_kernel(n: int, gather: bool, split: bool = False):
    key = (n, gather, split)
    if key not in _MV:
        inputs = ["x", "raw", "codes", "flags", "esc", "tbl", "row_esc"] + (["xi", "ei"] if gather else [])
        _MV[key] = mx.fast.cuda_kernel(
            name=f"binade_cuda_r4_mv_n{n}{'_g' if gather else ''}{'_s' if split else ''}",
            input_names=inputs,
            output_names=["y", "y2"] if split else ["y"],
            source=source_matvec(n, gather, split),
            header=HEADER,
        )
    return _MV[key]


def _u32(a: mx.array) -> mx.array:
    return a if a.dtype == mx.uint32 else a.astype(mx.uint32)


def r4_matvec(a: dict, x: mx.array, N: int, K: int) -> mx.array:
    """x [..., K] holding one row -> [..., N]. Inputs and output keep the caller's shapes, so
    the call adds no reshape to the graph (decoding on a fast GPU can be CPU-bound)."""
    T = -(-K // TILE)
    (y,) = _matvec_kernel(n_per_thread(K), False)(
        inputs=[x, a["raw"], a["codes"], a["flags"], a["esc"], a["table"], a["row_esc"]],
        template=[("N", N), ("K", K), ("T", T)],
        grid=(-(-N // 8) * 256, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(*x.shape[:-1], N)],
        output_dtypes=[mx.bfloat16],
    )
    return y


def r4_gather(a: dict, x: mx.array, xi: mx.array, ei: mx.array, n: int, rs: int, r0: int, K: int, split: int | None = None, shape: tuple | None = None):
    """[G, n] (or `shape`), or with split the pair [G, split], [G, n - split]."""
    T = -(-K // TILE)
    G = int(ei.size)
    shapes = [(G, split), (G, n - split)] if split else [shape or (G, n)]
    out = _matvec_kernel(n_per_thread(K), True, bool(split))(
        inputs=[x, a["raw"], a["codes"], a["flags"], a["esc"], a["table"], a["row_esc"], _u32(xi), _u32(ei)],
        template=[("N", n), ("K", K), ("T", T), ("RS", rs), ("R0", r0)] + ([("SPLIT", split)] if split else []),
        grid=(-(-n // 8) * 256, G, 1),
        threadgroup=(256, 1, 1),
        output_shapes=shapes,
        output_dtypes=[mx.bfloat16] * len(shapes),
    )
    return (out[0], out[1]) if split else out[0]


FLOAT_PARTIALS = ("f32", "inplace")  # split-K schemes whose partitions leave float32 sums


def source_r4_mma(bm: int, bn: int, wm: int, wn: int, split: str) -> str:
    """R4 GEMM on tensor cores, bit-identical to cuBLAS's BF16 GEMM as MLX calls it
    (binade/mlx/cublas_plan.py): every output is a sequence of mma.sync m16n8k16 steps along K,
    in order, from zero, over its split-K partition (blockIdx.z, bounds kr[]). split "none":
    the sum is rounded to BF16 into y; "bf16": the partition's sum is rounded to BF16 into
    part[S, M, N] (cuBLAS's OUTPUT_TYPE reduction scheme); "f32": kept in float32 (COMPUTE_TYPE);
    r4_gemm then adds the partials in partition order.

    A block of 4 warps (wm x wn) owns bm rows of x by bn weight rows and walks its partition in
    chunks of 32: it stages the chunk's inputs in shared memory while two threads per weight row
    decode 16 weights each from R4 (escape counts from the code nibbles, exchanged between the
    pair; one lookup turns a byte of codes into two exponents in place), then each warp runs the
    chunk's two m16n8k16 steps on its (bm / wm) x (bn / wn) tile. A partition starting inside a
    row finds its escape pointer by counting escapes in the flagged tiles before it. K % 32 == 0."""
    nthr = wm * wn * 32
    tm, tn = bm // wm // 16, bn // wn // 8
    assert nthr == 128 and bn * 2 == nthr and tm >= 1 and tn >= 1, (bm, bn, wm, wn)
    a_bytes = bm * 64 // nthr  # input bytes staged per thread per chunk
    assert a_bytes in (8, 16, 32), bm
    ld = 40  # padded row of a staged chunk, in BF16 elements
    store = "y[size_t(r) * N + c]" if split == "none" else "part[size_t(s) * M * N + size_t(r) * N + c]"
    conv = "" if split in FLOAT_PARTIALS else "__float2bfloat16_rn"
    a_stage = {
        8: "uint2 v = make_uint2(0u, 0u); int r = int(tid) >> 3, c = (int(tid) & 7) * 4; if (rb + r < M) { v = *reinterpret_cast<const uint2*>(x + size_t(rb + r) * K + k0 + c); } *reinterpret_cast<uint2*>(&As[r * LD + c]) = v;",
        16: "uint4 v = make_uint4(0u, 0u, 0u, 0u); int r = int(tid) >> 2, c = (int(tid) & 3) * 8; if (rb + r < M) { v = *reinterpret_cast<const uint4*>(x + size_t(rb + r) * K + k0 + c); } *reinterpret_cast<uint4*>(&As[r * LD + c]) = v;",
        32: "for (int h = 0; h < 2; h++) { uint4 v = make_uint4(0u, 0u, 0u, 0u); int i = int(tid) + h * 128; int r = i >> 2, c = (i & 3) * 8; if (rb + r < M) { v = *reinterpret_cast<const uint4*>(x + size_t(rb + r) * K + k0 + c); } *reinterpret_cast<uint4*>(&As[r * LD + c]) = v; }",
    }[a_bytes]
    return f"""
    constexpr int LD = {ld};
    __shared__ __align__(16) uint16_t As[{bm} * LD];
    __shared__ __align__(16) uint16_t Bs[{bn} * LD];
    __shared__ uint32_t lut2[256];
    uint32_t tid = threadIdx.x, lane = tid & 31u, warp = tid >> 5;
    for (int i = int(tid); i < 256; i += {nthr}) {{ lut2[i] = (uint32_t(tbl[i & 15]) << 7) | (uint32_t(tbl[i >> 4]) << 23); }}
    int s = int(blockIdx.z);
    int kb = kr[s], ke = kr[s + 1];
    int rb = int(blockIdx.y) * {bm}, cb = int(blockIdx.x) * {bn};
    // decoding role: weight row nl, weights 16 g .. 16 g + 15 of each chunk
    int nl = int(tid) >> 1, g = int(tid) & 1;
    int n = cb + nl;
    bool nok = n < N;
    size_t row = size_t(nok ? n : 0);
    // escape pointer at the partition start: escapes in [0, kb) of this row
    uint32_t cnt0 = 0;
    if (nok && kb > 0) {{
      int tk = kb >> 7;
      for (int t = g; t < tk; t += 2) {{
        if (flags[row * T + t]) {{
          const uint4* cw4 = reinterpret_cast<const uint4*>(codes + (row * T + t) * 32);
          for (int q = 0; q < 4; q++) {{ uint4 u = cw4[q]; uint32_t w[4] = {{u.x, u.y, u.z, u.w}};
            for (int z = 0; z < 4; z++) {{ cnt0 += __popc(w[z] & (w[z] >> 1) & (w[z] >> 2) & (w[z] >> 3) & 0x11111111u); }} }}
        }}
      }}
      int wpart = (kb & 127) >> 2;  // code words (4 weights each) of tile tk before kb
      if (g == 0 && wpart > 0 && flags[row * T + tk]) {{
        const uint16_t* cw = codes + (row * T + tk) * 32;
        for (int q = 0; q < wpart; q++) {{ uint32_t w = cw[q]; cnt0 += __popc(w & (w >> 1) & (w >> 2) & (w >> 3) & 0x1111u); }}
      }}
    }}
    cnt0 += __shfl_xor_sync(0xffffffffu, cnt0, 1);
    uint32_t epos = (nok ? row_esc[row] : 0u) + cnt0;
    // MMA role
    int qg = int(lane >> 2), qt = int(lane & 3u);
    int wr = int(warp) / {wn} * {bm // wm}, wc = int(warp) % {wn} * {bn // wn};
    float acc[{tm}][{tn}][4];
    #pragma unroll
    for (int i = 0; i < {tm}; i++) {{
      #pragma unroll
      for (int j = 0; j < {tn}; j++) {{ acc[i][j][0] = acc[i][j][1] = acc[i][j][2] = acc[i][j][3] = 0.f; }}
    }}
    for (int k0 = kb; k0 < ke; k0 += 32) {{
      __syncthreads();
      {{ {a_stage} }}
      {{
        int t = k0 >> 7, c = k0 & 127;
        uint2 cv = make_uint2(0u, 0u); uint4 rv = make_uint4(0u, 0u, 0u, 0u);
        if (nok) {{
          cv = *reinterpret_cast<const uint2*>(codes + (row * T + t) * 32 + ((c + 16 * g) >> 2));
          rv = *reinterpret_cast<const uint4*>(raw + row * K + k0 + 16 * g);
        }}
        uint32_t cw[2] = {{cv.x, cv.y}};
        uint32_t rw[4] = {{rv.x, rv.y, rv.z, rv.w}};
        uint32_t f[2];
        uint32_t cnt = 0;
        #pragma unroll
        for (int z = 0; z < 2; z++) {{ f[z] = cw[z] & (cw[z] >> 1) & (cw[z] >> 2) & (cw[z] >> 3) & 0x11111111u; cnt += __popc(f[z]); }}
        uint32_t other = __shfl_xor_sync(0xffffffffu, cnt, 1);
        uint32_t q = epos + (g ? other : 0u);
        epos += cnt + other;
        #pragma unroll
        for (int hh = 0; hh < 2; hh++) {{
          uint32_t pk[4];
          #pragma unroll
          for (int jj = 0; jj < 4; jj++) {{
            int j = hh * 4 + jj;
            uint32_t e2 = lut2[(cw[j >> 2] >> (8u * uint32_t(j & 3))) & 0xFFu];
            uint32_t r = (rw[j >> 1] >> (16u * uint32_t(j & 1))) & 0xFFFFu;
            pk[jj] = e2 | ((r & 0x80u) << 8) | (r & 0x7Fu) | ((r & 0x8000u) << 16) | ((r & 0x7F00u) << 8);
          }}
          if (f[hh]) {{
            #pragma unroll
            for (int jj = 0; jj < 8; jj++) {{
              if ((f[hh] >> (4u * uint32_t(jj))) & 1u) {{
                uint32_t sh = 7u + 16u * uint32_t(jj & 1);
                pk[jj >> 1] = (pk[jj >> 1] & ~(0xFFu << sh)) | (uint32_t(esc[q]) << sh);
                q++;
              }}
            }}
          }}
          *reinterpret_cast<uint4*>(&Bs[nl * LD + 16 * g + 8 * hh]) = nok ? make_uint4(pk[0], pk[1], pk[2], pk[3]) : make_uint4(0u, 0u, 0u, 0u);
        }}
      }}
      __syncthreads();
      #pragma unroll
      for (int kk = 0; kk < 32; kk += 16) {{
        uint32_t af[{tm}][4], bf[{tn}][2];
        #pragma unroll
        for (int i = 0; i < {tm}; i++) {{
          int r0 = wr + i * 16 + qg;
          af[i][0] = *reinterpret_cast<const uint32_t*>(&As[r0 * LD + kk + 2 * qt]);
          af[i][1] = *reinterpret_cast<const uint32_t*>(&As[(r0 + 8) * LD + kk + 2 * qt]);
          af[i][2] = *reinterpret_cast<const uint32_t*>(&As[r0 * LD + kk + 2 * qt + 8]);
          af[i][3] = *reinterpret_cast<const uint32_t*>(&As[(r0 + 8) * LD + kk + 2 * qt + 8]);
        }}
        #pragma unroll
        for (int j = 0; j < {tn}; j++) {{
          int n0 = wc + j * 8 + qg;
          bf[j][0] = *reinterpret_cast<const uint32_t*>(&Bs[n0 * LD + kk + 2 * qt]);
          bf[j][1] = *reinterpret_cast<const uint32_t*>(&Bs[n0 * LD + kk + 2 * qt + 8]);
        }}
        #pragma unroll
        for (int i = 0; i < {tm}; i++) {{
          #pragma unroll
          for (int j = 0; j < {tn}; j++) {{
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {{%0,%1,%2,%3}}, {{%4,%5,%6,%7}}, {{%8,%9}}, {{%0,%1,%2,%3}};\\n"
                         : "+f"(acc[i][j][0]), "+f"(acc[i][j][1]), "+f"(acc[i][j][2]), "+f"(acc[i][j][3])
                         : "r"(af[i][0]), "r"(af[i][1]), "r"(af[i][2]), "r"(af[i][3]), "r"(bf[j][0]), "r"(bf[j][1]));
          }}
        }}
      }}
    }}
    #pragma unroll
    for (int i = 0; i < {tm}; i++) {{
      #pragma unroll
      for (int j = 0; j < {tn}; j++) {{
        int c = cb + wc + j * 8 + 2 * qt;
        #pragma unroll
        for (int h = 0; h < 2; h++) {{
          int r = rb + wr + i * 16 + qg + 8 * h;
          if (r < M && c < N) {{ {store} = {conv}(acc[i][j][2 * h]); }}
          if (r < M && c + 1 < N) {{ {store.replace("+ c]", "+ c + 1]")} = {conv}(acc[i][j][2 * h + 1]); }}
        }}
      }}
    }}
"""


STEP_REGS = r'''
      #pragma unroll
      for (int j = @J0; j < @J1; j++) {
        uint32_t b0 = nok ? b0s[j] : 0u, b1 = nok ? b1s[j] : 0u;
        #pragma unroll
        for (int i = 0; i < @MT; i++) {
          const uint16_t* a = A + (16 * i + g) * LDA + 16 * j + 2 * t;
          uint32_t a0 = *reinterpret_cast<const uint32_t*>(a), a1 = *reinterpret_cast<const uint32_t*>(a + 8 * LDA);
          uint32_t a2 = *reinterpret_cast<const uint32_t*>(a + 8), a3 = *reinterpret_cast<const uint32_t*>(a + 8 * LDA + 8);
          asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                       : "+f"(@ACC[i][0]), "+f"(@ACC[i][1]), "+f"(@ACC[i][2]), "+f"(@ACC[i][3])
                       : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
        }
      }
'''


def source_r4_mma_regs(mt: int, split: str, wpb: int = 8, gather: bool = False, slices: int = 1) -> str:
    """The R4 tensor-core GEMM for short inputs (up to 16 mt rows), with the same per-output
    order as source_r4_mma (in-order mma.sync m16n8k16 steps per split-K partition, so the same
    cuBLAS mirror). A block of wpb warps owns 8 wpb weight rows, one warp per 8, for every input
    row (r4_gemm picks wpb so that a narrow matrix still spreads over the GPU's SMs).
    K goes in blocks of 64 weights: the block's input rows for the next 64 are loaded into
    registers and staged in a double-buffered shared tile (one barrier per block); each warp's
    B fragments are decoded straight into registers (the four threads of a row's quad hold one
    step's raw bytes and code words each and exchange them by shuffle; escapes are ordered by a
    prefix sum over the quad), with weight blocks loaded three ahead. Like the matvec kernel, every
    weight is read once. Partition bounds are multiples of 64 (cublas_plan); K % 64 == 0."""
    store = "y[size_t(rg) * N + c]" if split == "none" else "part[size_t(s) * M * N + size_t(rg) * N + c]"
    run_head = ("uint32_t e = idx[rb + seg]; int end = seg + 1; while (end < nrow && idx[rb + end] == e) { end++; }" if gather else "int end = nrow;")
    row_expr = "size_t(e) * RS + R0 + size_t(nok ? n : 0)" if gather else "size_t(nok ? n : 0)"
    conv = "" if split in FLOAT_PARTIALS else "__float2bfloat16_rn"
    rows = 16 * mt
    nthr = 32 * wpb
    xv = rows * 8  # uint4 (8 BF16) per 64-wide input block
    # with slices == 2 (cuBLAS's sliced-K kernels) steps 0, 1 of each 64-wide block go to acc and 2, 3 to acc2
    def step(j0: int, j1: int, acc: str) -> str:
        return STEP_REGS.replace("@J0", str(j0)).replace("@J1", str(j1)).replace("@ACC", acc).replace("@MT", str(mt))
    steps = step(0, 4, "acc") if slices == 1 else step(0, 2, "acc") + step(2, 4, "acc2")
    acc2 = f"float acc2[{mt}][4]; for (int i = 0; i < {mt}; i++) {{ acc2[i][0] = acc2[i][1] = acc2[i][2] = acc2[i][3] = 0.f; }}" if slices == 2 else ""
    v0, v1 = ("(acc[i][2 * h] + acc2[i][2 * h])", "(acc[i][2 * h + 1] + acc2[i][2 * h + 1])") if slices == 2 else ("acc[i][2 * h]", "acc[i][2 * h + 1]")
    per = -(-xv // nthr)  # per thread
    return f"""
    constexpr int LDA = 72;  // padded staged input row, BF16
    __shared__ __align__(16) uint16_t As[3][{rows} * LDA];  // 3-stage ring of 64-wide input blocks
    __shared__ uint8_t lut[16];
    if (threadIdx.x < 16) {{ lut[threadIdx.x] = tbl[threadIdx.x]; }}
    const uint32_t full = 0xffffffffu;
    uint32_t tid = threadIdx.x, lane = tid & 31u;
    int warp = int(tid >> 5);
    int g = int(lane >> 2), t = int(lane & 3u);
    int q0 = int(lane & ~3u);
    int n0 = (int(blockIdx.x) * {wpb} + warp) * 8;
    int n = n0 + g;
    bool nok = n < N;  // warps past N keep running for the barriers, with zero weights
    int s = int(blockIdx.z);
    int kb = kr[s], ke = kr[s + 1];
    int rb = int(blockIdx.y) * {rows};  // this block's input rows: rb .. rb + nrow - 1
    int nrow = min({rows}, M - rb);
    int sh = (t & 1) * 2;
    // input staging with cp.async: thread tid copies uint4 number tid + {nthr} v of a block (row (..) / 8,
    // column 8 ((..) % 8)) straight into the ring; rows past nrow are zero-filled; one commit group per block
    #define COPYX(k64, st) {{ _Pragma("unroll") for (int v = 0; v < {per}; v++) {{ int i = int(tid) + {nthr} * v; int r = i >> 3, c = (i & 7) * 8; \\
        if (i < {xv}) {{ bool ok = r < nrow && (k64) < ke; const __nv_bfloat16* gp = ok ? x + size_t(rb + r) * K + (k64) + c : x; \\
          uint32_t sa = static_cast<uint32_t>(__cvta_generic_to_shared(&As[st][r * LDA + c])); \\
          asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\\n" :: "r"(sa), "l"(gp), "r"(ok ? 16 : 0)); }} }} \\
        asm volatile("cp.async.commit_group;\\n" ::); }}
    #define LOADB(k64, R, C) {{ if (nok && (k64) < ke) {{ R = *reinterpret_cast<const uint4*>(rrow + (k64) + 16 * t); C = *reinterpret_cast<const uint2*>(crow + ((k64) >> 2) + 4 * t); }} \\
        else {{ R = make_uint4(0u, 0u, 0u, 0u); C = make_uint2(0u, 0u); }} }}
    // runs of input rows that share an expert (gathering: slots sorted by expert); one run otherwise
    for (int seg = 0; seg < nrow;) {{
    {run_head}
    size_t row = {row_expr};
    // escape pointer at the partition start: escapes in [0, kb) of this row, split over the quad
    uint32_t c0 = 0;
    if (nok && kb > 0) {{
      int tk = kb >> 7;
      for (int tt = t; tt < tk; tt += 4) {{
        if (flags[row * T + tt]) {{
          const uint4* cw4 = reinterpret_cast<const uint4*>(codes + (row * T + tt) * 32);
          for (int z = 0; z < 4; z++) {{ uint4 u = cw4[z]; uint32_t w[4] = {{u.x, u.y, u.z, u.w}};
            for (int y2 = 0; y2 < 4; y2++) {{ c0 += __popc(w[y2] & (w[y2] >> 1) & (w[y2] >> 2) & (w[y2] >> 3) & 0x11111111u); }} }}
        }}
      }}
      int wpart = (kb & 127) >> 2;
      if (wpart > 0 && flags[row * T + tk]) {{
        const uint16_t* cw = codes + (row * T + tk) * 32;
        for (int z = t; z < wpart; z += 4) {{ uint32_t w = cw[z]; c0 += __popc(w & (w >> 1) & (w >> 2) & (w >> 3) & 0x1111u); }}
      }}
    }}
    c0 += __shfl_xor_sync(full, c0, 1);
    c0 += __shfl_xor_sync(full, c0, 2);
    uint32_t epos = (nok ? row_esc[row] : 0u) + c0;
    const uint16_t* crow = codes + row * T * 32;  // the row's code words: word k / 4 holds weights k .. k + 3
    const uint8_t* rrow = raw + row * K;
    float acc[{mt}][4];
    #pragma unroll
    for (int i = 0; i < {mt}; i++) {{ acc[i][0] = acc[i][1] = acc[i][2] = acc[i][3] = 0.f; }}
    {acc2}
    uint4 r_cur, r_nxt, r_nx2; uint2 c_cur, c_nxt, c_nx2;
    LOADB(kb, r_cur, c_cur);
    LOADB(kb + 64, r_nxt, c_nxt);
    LOADB(kb + 128, r_nx2, c_nx2);
    COPYX(kb, 0);
    COPYX(kb + 64, 1);
    int buf = 0;
    for (int k64 = kb; k64 < ke; k64 += 64) {{
      asm volatile("cp.async.wait_group 1;\\n" ::);  // this block's copy is done (the next one may still be in flight)
      __syncthreads();  // ... for every thread; and every warp is done with the stage refilled below
      COPYX(k64 + 128, (buf + 2) % 3);
      uint4 br = r_cur; uint2 bc = c_cur;
      r_cur = r_nxt; c_cur = c_nxt;
      r_nxt = r_nx2; c_nxt = c_nx2;
      LOADB(k64 + 192, r_nx2, c_nx2);
      uint32_t cA = (bc.x & 0xFFFFu) | (bc.y << 16), cB = (bc.x >> 16) | (bc.y & 0xFFFF0000u);
      const uint16_t* A = As[buf];
      // gather the block's 4 steps with 12 shuffles: at shuffle k, threads 0, 1 of each quad read
      // step k from lane k (its A-half: code words 0, 2; raw words 0, 2) and threads 2, 3 read step
      // (k + 1) % 4 from lane (k + 1) % 4 (B-half: words 1, 3); each source lane sends the half
      // its readers need
      int qs = int(lane & 3u);
      uint32_t sc[4], sl[4], shh[4];
      #pragma unroll
      for (int k = 0; k < 4; k++) {{
        bool sendA = qs == k, sendB = qs == ((k + 1) & 3);
        uint32_t vc = sendA ? cA : (sendB ? cB : 0u);
        uint32_t vl = sendA ? br.x : (sendB ? br.y : 0u);
        uint32_t vh = sendA ? br.z : (sendB ? br.w : 0u);
        int src = q0 + (t < 2 ? k : ((k + 1) & 3));
        sc[k] = __shfl_sync(full, vc, src);
        sl[k] = __shfl_sync(full, vl, src);
        shh[k] = __shfl_sync(full, vh, src);
      }}
      uint32_t b0s[4], b1s[4], cw[4];
      uint32_t anyesc = 0;
      #pragma unroll
      for (int j = 0; j < 4; j++) {{
        // step j arrived at shuffle j (threads 0, 1) or at shuffle (j + 3) % 4 (threads 2, 3)
        uint32_t cc = t < 2 ? sc[j] : sc[(j + 3) & 3];
        uint32_t rlo = t < 2 ? sl[j] : sl[(j + 3) & 3];
        uint32_t rhi = t < 2 ? shh[j] : shh[(j + 3) & 3];
        cw[j] = cc;
        uint32_t clo = cc & 0xFFFFu, chi = cc >> 16;
        uint32_t c0 = (clo >> (4 * sh)) & 15u, c1 = (clo >> (4 * sh + 4)) & 15u, c2 = (chi >> (4 * sh)) & 15u, c3 = (chi >> (4 * sh + 4)) & 15u;
        anyesc |= (c0 == 15u) | (c1 == 15u) | (c2 == 15u) | (c3 == 15u);
        uint32_t r0b = (rlo >> (8 * sh)) & 255u, r1b = (rlo >> (8 * sh + 8)) & 255u, r2b = (rhi >> (8 * sh)) & 255u, r3b = (rhi >> (8 * sh + 8)) & 255u;
        uint32_t u0 = ((r0b & 0x80u) << 8) | (uint32_t(lut[c0]) << 7) | (r0b & 0x7Fu);
        uint32_t u1 = ((r1b & 0x80u) << 8) | (uint32_t(lut[c1]) << 7) | (r1b & 0x7Fu);
        uint32_t u2 = ((r2b & 0x80u) << 8) | (uint32_t(lut[c2]) << 7) | (r2b & 0x7Fu);
        uint32_t u3 = ((r3b & 0x80u) << 8) | (uint32_t(lut[c3]) << 7) | (r3b & 0x7Fu);
        b0s[j] = u0 | (u1 << 16); b1s[j] = u2 | (u3 << 16);
      }}
      if (__any_sync(full, anyesc)) {{
        // rare: some row has escapes in this block; patch steps in order (escape order = position order)
        #pragma unroll
        for (int j = 0; j < 4; j++) {{
          uint32_t clo = cw[j] & 0xFFFFu, chi = cw[j] >> 16;
          uint32_t c[4] = {{(clo >> (4 * sh)) & 15u, (clo >> (4 * sh + 4)) & 15u, (chi >> (4 * sh)) & 15u, (chi >> (4 * sh + 4)) & 15u}};
          uint32_t nlo = (c[0] == 15u ? 1u : 0u) + (c[1] == 15u ? 1u : 0u), nhi = (c[2] == 15u ? 1u : 0u) + (c[3] == 15u ? 1u : 0u);
          uint32_t ilo = nlo, ihi = nhi, v;
          v = __shfl_up_sync(full, ilo, 1, 4); if (t >= 1) {{ ilo += v; }}
          v = __shfl_up_sync(full, ilo, 2, 4); if (t >= 2) {{ ilo += v; }}
          v = __shfl_up_sync(full, ihi, 1, 4); if (t >= 1) {{ ihi += v; }}
          v = __shfl_up_sync(full, ihi, 2, 4); if (t >= 2) {{ ihi += v; }}
          uint32_t tlo = __shfl_sync(full, ilo, 3, 4), thi = __shfl_sync(full, ihi, 3, 4);
          uint32_t q = epos + ilo - nlo;
          if (c[0] == 15u) {{ b0s[j] = (b0s[j] & ~(0xFFu << 7)) | (uint32_t(esc[q]) << 7); q++; }}
          if (c[1] == 15u) {{ b0s[j] = (b0s[j] & ~(0xFFu << 23)) | (uint32_t(esc[q]) << 23); q++; }}
          q = epos + tlo + ihi - nhi;
          if (c[2] == 15u) {{ b1s[j] = (b1s[j] & ~(0xFFu << 7)) | (uint32_t(esc[q]) << 7); q++; }}
          if (c[3] == 15u) {{ b1s[j] = (b1s[j] & ~(0xFFu << 23)) | (uint32_t(esc[q]) << 23); q++; }}
          epos += tlo + thi;
        }}
      }}
{steps}
      buf = (buf + 1) % 3;
    }}
    asm volatile("cp.async.wait_group 0;\\n" ::);
    int c = n0 + 2 * t;
    #pragma unroll
    for (int i = 0; i < {mt}; i++) {{
      #pragma unroll
      for (int h = 0; h < 2; h++) {{
        int r = 16 * i + g + 8 * h;
        int rg = rb + r;
        if (r >= seg && r < end && c < N) {{ {store} = {conv}({v0}); }}
        if (r >= seg && r < end && c + 1 < N) {{ {store.replace("+ c]", "+ c + 1]")} = {conv}({v1}); }}
      }}
    }}
    seg = end;
    __syncthreads();  // the next run refills the ring
    }}
    #undef COPYX
    #undef LOADB
"""


BIG_STAGES = {128: 3, 256: 4}  # input stages per tile height (dynamic shared memory)


def big_smem(bm: int) -> int:
    return BIG_STAGES[bm] * bm * 64 + 2 * 128 * 64 + 1024


STEP_BIG = r'''
      #pragma unroll
      for (int kk = 0; kk < 2; kk++) {
        uint32_t af[4][4];
        #pragma unroll
        for (int i = 0; i < 4; i++) {
          asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n" : "=r"(af[i][0]), "=r"(af[i][1]), "=r"(af[i][2]), "=r"(af[i][3]) : "r"(A + (oA[i] ^ (32u * kk))));
        }
        #pragma unroll
        for (int jp = 0; jp < @NJ; jp++) {
          uint32_t b0, b1, b2, b3;
          asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n" : "=r"(b0), "=r"(b1), "=r"(b2), "=r"(b3) : "r"(B + (oB[jp] ^ (32u * kk))));
          #pragma unroll
          for (int i = 0; i < 4; i++) {
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                         : "+f"(@ACC[i][2 * jp][0]), "+f"(@ACC[i][2 * jp][1]), "+f"(@ACC[i][2 * jp][2]), "+f"(@ACC[i][2 * jp][3])
                         : "r"(af[i][0]), "r"(af[i][1]), "r"(af[i][2]), "r"(af[i][3]), "r"(b0), "r"(b1));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                         : "+f"(@ACC[i][2 * jp + 1][0]), "+f"(@ACC[i][2 * jp + 1][1]), "+f"(@ACC[i][2 * jp + 1][2]), "+f"(@ACC[i][2 * jp + 1][3])
                         : "r"(af[i][0]), "r"(af[i][1]), "r"(af[i][2]), "r"(af[i][3]), "r"(b2), "r"(b3));
          }
        }
      }
'''


def source_r4_mma_big(split: str, bm: int = 256, slices: int = 1) -> str:
    """The R4 tensor-core GEMM for long inputs, with the same per-output order as the other two
    (in-order mma.sync m16n8k16 steps per split-K partition). A block of 8 warps owns bm input
    rows (128 or 256) by 128 weight rows, each warp a 64 x (128 * 64 / bm) tile, like cuBLAS's
    256 x 128 kernel, and walks its partition in chunks of 32: inputs come with cp.async into a
    shared ring of BIG_STAGES[bm] stages, two threads per weight row decode 16 weights each into
    a double-buffered shared tile one chunk ahead (raw bytes and code words loaded two chunks
    ahead), and fragments are read with ldmatrix from rows swizzled in 16-byte pieces (no bank
    conflicts). Every address is computed once; one barrier per chunk. Launch rows of blocks
    walk the input tiles first, so blocks sharing a weight tile run together. K % 32 == 0,
    N % 2 == 0."""
    store = "y + size_t(r) * N + c" if split == "none" else "part + size_t(s) * M * N + size_t(r) * N + c"
    if split in FLOAT_PARTIALS:
        put = "*reinterpret_cast<float2*>({p}) = make_float2(acc[i][j][2 * h], acc[i][j][2 * h + 1]);"
    else:
        put = "*reinterpret_cast<__nv_bfloat162*>({p}) = __floats2bfloat162_rn(acc[i][j][2 * h], acc[i][j][2 * h + 1]);"
    put = put.format(p=store)
    wm = bm // 64
    wn = 8 // wm
    nt = 128 // wn // 8
    stages = BIG_STAGES[bm]
    a_bytes = bm * 64
    # with slices == 2 (cuBLAS's sliced-K kernels) chunks alternate between acc and acc2 (32-wide K slices)
    step = lambda acc: STEP_BIG.replace("@NJ", str(nt // 2)).replace("@ACC", acc)
    steps = step("acc") if slices == 1 else "      if (((k0 - kb) >> 5) & 1) {\n" + step("acc2") + "      } else {\n" + step("acc") + "      }\n"
    acc2 = f"float acc2[4][{nt}][4]; for (int i = 0; i < 4; i++) {{ for (int j = 0; j < {nt}; j++) {{ acc2[i][j][0] = acc2[i][j][1] = acc2[i][j][2] = acc2[i][j][3] = 0.f; }} }}" if slices == 2 else ""
    if slices == 2:
        put = put.replace("acc[i][j][2 * h]", "acc[i][j][2 * h] + acc2[i][j][2 * h]").replace("acc[i][j][2 * h + 1]", "acc[i][j][2 * h + 1] + acc2[i][j][2 * h + 1]")
    return f"""
    extern __shared__ __align__(128) uint8_t smem[];
    const uint32_t sA = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
    const uint32_t sB = sA + {stages * a_bytes};
    uint32_t* lut2 = reinterpret_cast<uint32_t*>(smem + {stages * a_bytes + 2 * 128 * 64});
    const uint32_t full = 0xffffffffu;
    uint32_t tid = threadIdx.x, lane = tid & 31u;
    int warp = int(tid >> 5);
    lut2[tid] = (uint32_t(tbl[tid & 15u]) << 7) | (uint32_t(tbl[tid >> 4]) << 23);
    __syncthreads();
    int s = int(blockIdx.z);
    int kb = kr[s], ke = kr[s + 1];
    int rb = int(blockIdx.x) * {bm}, cb = int(blockIdx.y) * 128;
    // byte offset of (row r, 16-byte piece c of 4) in a stage of 64-byte rows, swizzled
    #define SWB(r, c) (uint32_t(r) * 64u + 16u * uint32_t((c) ^ (((r) >> 1) & 3)))
    // decoding role: weight row nl, weights 16 gh .. 16 gh + 15 of each chunk
    int nl = int(tid) >> 1, gh = int(tid) & 1;
    int n = cb + nl;
    bool nok = n < N;
    size_t row = size_t(nok ? n : 0);
    uint32_t cnt0 = 0;
    if (nok && kb > 0) {{
      int tk = kb >> 7;
      for (int t = gh; t < tk; t += 2) {{
        if (flags[row * T + t]) {{
          const uint4* cw4 = reinterpret_cast<const uint4*>(codes + (row * T + t) * 32);
          for (int q = 0; q < 4; q++) {{ uint4 u = cw4[q]; uint32_t w[4] = {{u.x, u.y, u.z, u.w}};
            for (int z = 0; z < 4; z++) {{ cnt0 += __popc(w[z] & (w[z] >> 1) & (w[z] >> 2) & (w[z] >> 3) & 0x11111111u); }} }}
        }}
      }}
      int wpart = (kb & 127) >> 2;
      if (gh == 0 && wpart > 0 && flags[row * T + tk]) {{
        const uint16_t* cw = codes + (row * T + tk) * 32;
        for (int q = 0; q < wpart; q++) {{ uint32_t w = cw[q]; cnt0 += __popc(w & (w >> 1) & (w >> 2) & (w >> 3) & 0x1111u); }}
      }}
    }}
    cnt0 += __shfl_xor_sync(full, cnt0, 1);
    uint32_t epos = (nok ? row_esc[row] : 0u) + cnt0;
    const uint16_t* crow = codes + row * T * 32 + 4 * gh;  // word k / 4 holds the codes of weights k .. k + 3
    const uint8_t* rrow = raw + row * K + 16 * gh;
    const uint32_t dB = SWB(nl, 2 * gh);  // this thread's two decoded pieces: dB and dB ^ 16
    // input copies: rows (tid >> 2) + 64 v, piece tid & 3
    const int cr = int(tid) >> 2, cpc = int(tid) & 3;
    const uint32_t dA = SWB(cr, cpc);
    const __nv_bfloat16* xg = x + size_t(rb + cr) * K + 8 * cpc;
    #define COPYA(k0, st) {{ _Pragma("unroll") for (int v = 0; v < {wm}; v++) {{ \\
        bool ok = rb + cr + 64 * v < M && (k0) < ke; const __nv_bfloat16* gp = ok ? xg + size_t(64 * v) * K + (k0) : x; \\
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\\n" :: "r"(sA + (st) * {a_bytes}u + dA + 4096u * v), "l"(gp), "r"(ok ? 16 : 0)); }} \\
        asm volatile("cp.async.commit_group;\\n" ::); }}
    #define LOADB(k0, R, C) {{ if (nok && (k0) < ke) {{ R = *reinterpret_cast<const uint4*>(rrow + (k0)); C = *reinterpret_cast<const uint2*>(crow + ((k0) >> 2)); }} \\
        else {{ R = make_uint4(0u, 0u, 0u, 0u); C = make_uint2(0u, 0u); }} }}
    // weights 16 gh .. 16 gh + 15 of row nl into B stage st; escapes in position order (the pair's counts exchanged)
    #define DECODE(R, C, st) {{ uint32_t cw[2] = {{C.x, C.y}}; uint32_t rw[4] = {{R.x, R.y, R.z, R.w}}; uint32_t f[2]; uint32_t cnt = 0; \\
        _Pragma("unroll") for (int z = 0; z < 2; z++) {{ f[z] = cw[z] & (cw[z] >> 1) & (cw[z] >> 2) & (cw[z] >> 3) & 0x11111111u; cnt += __popc(f[z]); }} \\
        uint32_t other = __shfl_xor_sync(full, cnt, 1); uint32_t q = epos + (gh ? other : 0u); epos += cnt + other; \\
        _Pragma("unroll") for (int hh = 0; hh < 2; hh++) {{ uint32_t pk[4]; \\
          _Pragma("unroll") for (int jj = 0; jj < 4; jj++) {{ int j = hh * 4 + jj; uint32_t e2 = lut2[(cw[j >> 2] >> (8u * uint32_t(j & 3))) & 0xFFu]; \\
            uint32_t r2 = __byte_perm(rw[j >> 1], 0u, (j & 1) ? 0x7372u : 0x7170u); \\
            pk[jj] = e2 | ((r2 & 0x00800080u) << 8) | (r2 & 0x007F007Fu); }} \\
          if (f[hh]) {{ _Pragma("unroll") for (int jj = 0; jj < 8; jj++) {{ if ((f[hh] >> (4u * uint32_t(jj))) & 1u) {{ \\
            uint32_t sh = 7u + 16u * uint32_t(jj & 1); pk[jj >> 1] = (pk[jj >> 1] & ~(0xFFu << sh)) | (uint32_t(esc[q]) << sh); q++; }} }} }} \\
          uint4 v4 = nok ? make_uint4(pk[0], pk[1], pk[2], pk[3]) : make_uint4(0u, 0u, 0u, 0u); \\
          asm volatile("st.shared.v4.u32 [%0], {{%1, %2, %3, %4}};\\n" :: "r"(sB + (st) * 8192u + (dB ^ (16u * hh))), "r"(v4.x), "r"(v4.y), "r"(v4.z), "r"(v4.w)); }} }}
    // MMA role: warp tile 64 x {128 // wn} at (wr, wc); ldmatrix offsets for the chunk's first step (the second: ^ 32)
    int wr = (warp / {wn}) * 64, wc = (warp % {wn}) * {128 // wn};
    int qg = int(lane >> 2), qt = int(lane & 3u);
    uint32_t oA[4], oB[{nt // 2}];
    #pragma unroll
    for (int i = 0; i < 4; i++) {{ int r = wr + 16 * i + (int(lane) & 7) + ((int(lane) >> 3) & 1) * 8; oA[i] = SWB(r, int(lane) >> 4); }}
    #pragma unroll
    for (int jp = 0; jp < {nt // 2}; jp++) {{ int r = wc + 16 * jp + ((int(lane) >> 4) & 1) * 8 + (int(lane) & 7); oB[jp] = SWB(r, (int(lane) >> 3) & 1); }}
    float acc[4][{nt}][4];
    #pragma unroll
    for (int i = 0; i < 4; i++) {{
      #pragma unroll
      for (int j = 0; j < {nt}; j++) {{ acc[i][j][0] = acc[i][j][1] = acc[i][j][2] = acc[i][j][3] = 0.f; }}
    }}
    {acc2}
    uint4 rc, rn; uint2 cc, cn;
    LOADB(kb, rc, cc);
    LOADB(kb + 32, rn, cn);
    #pragma unroll
    for (int st = 0; st < {stages - 1}; st++) {{ COPYA(kb + 32 * st, st); }}
    DECODE(rc, cc, 0);
    rc = rn; cc = cn;
    LOADB(kb + 64, rn, cn);
    int a = 0, b = 0;
    for (int k0 = kb; k0 < ke; k0 += 32) {{
      asm volatile("cp.async.wait_group {stages - 2};\\n" ::);
      __syncthreads();  // chunk k0's inputs and weights are in; everyone is done with the stages refilled below
      COPYA(k0 + {32 * (stages - 1)}, (a + {stages - 1}) % {stages});
      if (k0 + 32 < ke) {{ DECODE(rc, cc, b ^ 1); }}
      rc = rn; cc = cn;
      LOADB(k0 + 96, rn, cn);
      const uint32_t A = sA + uint32_t(a) * {a_bytes}u, B = sB + uint32_t(b) * 8192u;
{steps}
      a = (a + 1) % {stages}; b ^= 1;
    }}
    asm volatile("cp.async.wait_group 0;\\n" ::);
    #pragma unroll
    for (int i = 0; i < 4; i++) {{
      #pragma unroll
      for (int j = 0; j < {nt}; j++) {{
        int c = cb + wc + 8 * j + 2 * qt;
        #pragma unroll
        for (int h = 0; h < 2; h++) {{
          int r = rb + wr + 16 * i + qg + 8 * h;
          if (r < M && c < N) {{ {put} }}
        }}
      }}
    }}
    #undef SWB
    #undef COPYA
    #undef LOADB
    #undef DECODE
"""


SOURCE_SPLIT_REDUCE = """
    // cuBLAS's split-K reduction: partials (BF16 or float32) added in float32 in partition order
    size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= size_t(M) * N) { return; }
    float a = 0.f;
    for (int s = 0; s < S; s++) { a += static_cast<float>(part[size_t(s) * M * N + i]); }
    y[i] = __float2bfloat16_rn(a);
"""

SOURCE_SPLIT_INPLACE = """
    // cuBLAS's in-place (serial) split-K: partition s adds its float32 sum to the BF16 output
    // left by partitions 0 .. s - 1, rounding to BF16 each time
    size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= size_t(M) * N) { return; }
    __nv_bfloat16 o = __float2bfloat16_rn(part[i]);
    for (int s = 1; s < S; s++) { o = __float2bfloat16_rn(static_cast<float>(o) + part[size_t(s) * M * N + i]); }
    y[i] = o;
"""

_MMA = {}
_REDUCE = {}
MMA_TILES = ((16, (16, 64, 1, 4)), (None, (64, 64, 2, 2)))  # (max rows, (bm, bn, wm, wn))


def mma_tile(M: int) -> tuple:
    return next(t for r, t in MMA_TILES if r is None or M <= r)


REGS_MAX_ROWS = 64  # up to this many rows, the register-decoding kernel (all rows in one launch row)
REGS_MIN_BLOCKS = 376  # fewer warps per block until there are this many blocks (2 per SM of an RTX PRO 6000)
_BOUNDS = {}


def _bounds(b: tuple) -> mx.array:
    """Split-K bounds as a cached, evaluated device array: a fresh array per call would copy from
    the host each time and can stall the CPU/GPU pipeline."""
    if b not in _BOUNDS:
        _BOUNDS[b] = mx.array(b, dtype=mx.int32)
        mx.eval(_BOUNDS[b])
    return _BOUNDS[b]


def gemm_supported(M: int, N: int, K: int, slices: int = 1) -> bool:
    """Whether r4_gemm's default kernel for this shape can sum in `slices` K slices (the 64 x 64
    shared-memory kernel sums in order only)."""
    return K % 32 == 0 and (slices == 1 or (M <= REGS_MAX_ROWS and K % 64 == 0) or N % 2 == 0)


def r4_gemm(a: dict, x: mx.array, N: int, K: int, bounds: list, split: str = "none", tile: str | tuple | None = None, slices: int = 1) -> mx.array:
    """x [M, K] @ W.T [M, N] with W in R4, bit-identical to MLX's cuBLAS matmul when bounds and
    split are that matmul's split-K partitions and partial type (cublas_plan). K % 32 == 0.
    The kernel: up to REGS_MAX_ROWS rows (K % 64 == 0) the register kernel, else the 128 x 128
    one (N even), else the 64 x 64 shared-memory one; tile ("regs", "big128", "big256" or
    (bm, bn, wm, wn)) forces one, for tests and benchmarks. slices = 2 mirrors cuBLAS's sliced-K
    kernels (cublas_plan.slices), without split-K."""
    M = x.shape[0]
    S = len(bounds) - 1
    split = "none" if S == 1 else split
    T = -(-K // TILE)
    kr = _bounds(tuple(bounds))
    out_shape, out_dtype = ((S, M, N) if S > 1 else (M, N)), (mx.float32 if split in FLOAT_PARTIALS else mx.bfloat16)
    if (tile == "regs" or (tile is None and M <= REGS_MAX_ROWS)) and K % 64 == 0:
        mt = min(4, -(-M // 16))
        gy = -(-M // (16 * mt))  # launch rows of 16 mt inputs (each re-decodes the weights)
        wpb = next(w for w in (8, 4, 2) if w == 2 or -(-N // (8 * w)) * gy * S >= REGS_MIN_BLOCKS)
        key = ("regs", mt, split, wpb, slices)
        if key not in _MMA:
            _MMA[key] = mx.fast.cuda_kernel(
                name=f"binade_cuda_r4_mma_regs{mt}_{split}_w{wpb}{'_s%d' % slices if slices > 1 else ''}",
                input_names=["x", "raw", "codes", "flags", "esc", "tbl", "row_esc", "kr"],
                output_names=["y" if split == "none" else "part"],
                source=source_r4_mma_regs(mt, split, wpb, slices=slices),
            )
        (out,) = _MMA[key](
            inputs=[x, a["raw"], a["codes"], a["flags"], a["esc"], a["table"], a["row_esc"], kr],
            template=[("M", M), ("N", N), ("K", K), ("T", T)],
            grid=(-(-N // (8 * wpb)) * 32 * wpb, gy, S),
            threadgroup=(32 * wpb, 1, 1),
            output_shapes=[out_shape],
            output_dtypes=[out_dtype],
        )
        return out if S == 1 else _split_reduce(out, M, N, S, split)
    if tile is None and N % 2 == 0:
        tile = "big128"  # 128 x 128 tiles beat 256 x 128 at every length measured (two blocks per SM)
    if tile in ("big128", "big256"):
        bm = 128 if tile == "big128" else 256
        key = ("big", bm, split, slices)
        if key not in _MMA:
            _MMA[key] = mx.fast.cuda_kernel(
                name=f"binade_cuda_r4_mma_big{bm}_{split}{'_s%d' % slices if slices > 1 else ''}",
                input_names=["x", "raw", "codes", "flags", "esc", "tbl", "row_esc", "kr"],
                output_names=["y" if split == "none" else "part"],
                source=source_r4_mma_big(split, bm, slices),
                shared_memory=big_smem(bm),
            )
        (out,) = _MMA[key](
            inputs=[x, a["raw"], a["codes"], a["flags"], a["esc"], a["table"], a["row_esc"], kr],
            template=[("M", M), ("N", N), ("K", K), ("T", T)],
            grid=(-(-M // bm) * 256, -(-N // 128), S),
            threadgroup=(256, 1, 1),
            output_shapes=[out_shape],
            output_dtypes=[out_dtype],
        )
        return out if S == 1 else _split_reduce(out, M, N, S, split)
    if slices != 1:
        raise ValueError("the 64 x 64 kernel sums K in order only")
    bm, bn, wm, wn = tile or mma_tile(M)
    key = (bm, bn, wm, wn, split)
    if key not in _MMA:
        _MMA[key] = mx.fast.cuda_kernel(
            name=f"binade_cuda_r4_mma_{bm}x{bn}_{wm}x{wn}_{split}",
            input_names=["x", "raw", "codes", "flags", "esc", "tbl", "row_esc", "kr"],
            output_names=["y" if split == "none" else "part"],
            source=source_r4_mma(bm, bn, wm, wn, split),
        )
    (out,) = _MMA[key](
        inputs=[x, a["raw"], a["codes"], a["flags"], a["esc"], a["table"], a["row_esc"], kr],
        template=[("M", M), ("N", N), ("K", K), ("T", T)],
        grid=(-(-N // bn) * 128, -(-M // bm), S),
        threadgroup=(128, 1, 1),
        output_shapes=[out_shape],
        output_dtypes=[out_dtype],
    )
    return out if S == 1 else _split_reduce(out, M, N, S, split)


def r4_gather_mm(a: dict, x: mx.array, idx: mx.array, n: int, rs: int, r0: int, K: int) -> mx.array:
    """[S, n] = rows of x [S, K] times storage rows idx[s] * rs + r0 .. + n, slots sorted by expert:
    bit-identical to MLX's CUDA gather_mm(x, W.swapaxes(-1, -2), idx, sorted_indices=True), which
    runs a CUTLASS grouped GEMM (per expert, m16n8k16 steps along K in order, no split-K) for
    one-row slots. K % 64 == 0."""
    S = int(idx.size)
    per_expert = S / max(1, a["raw"].shape[0] // rs)
    mt = 1 if per_expert <= 16 else 2 if per_expert <= 32 else 4
    wpb = next(w for w in (8, 4, 2) if w == 2 or -(-n // (8 * w)) * -(-S // (16 * mt)) >= REGS_MIN_BLOCKS)
    key = ("gather", mt, wpb)
    if key not in _MMA:
        _MMA[key] = mx.fast.cuda_kernel(
            name=f"binade_cuda_r4_mma_gather{mt}_w{wpb}",
            input_names=["x", "raw", "codes", "flags", "esc", "tbl", "row_esc", "kr", "idx"],
            output_names=["y"],
            source=source_r4_mma_regs(mt, "none", wpb, gather=True),
        )
    (y,) = _MMA[key](
        inputs=[x.reshape(S, K), a["raw"], a["codes"], a["flags"], a["esc"], a["table"], a["row_esc"], _bounds((0, K)), _u32(idx)],
        template=[("M", S), ("N", n), ("K", K), ("T", -(-K // TILE)), ("RS", rs), ("R0", r0)],
        grid=(-(-n // (8 * wpb)) * 32 * wpb, -(-S // (16 * mt)), 1),
        threadgroup=(32 * wpb, 1, 1),
        output_shapes=[(S, n)],
        output_dtypes=[mx.bfloat16],
    )
    return y


def _split_reduce(out: mx.array, M: int, N: int, S: int, split: str) -> mx.array:
    if split not in _REDUCE:
        _REDUCE[split] = mx.fast.cuda_kernel(name=f"binade_cuda_split_reduce_{split}", input_names=["part"], output_names=["y"], source=SOURCE_SPLIT_INPLACE if split == "inplace" else SOURCE_SPLIT_REDUCE)
    (y,) = _REDUCE[split](
        inputs=[out],
        template=[("M", M), ("N", N), ("S", S)],
        grid=(-(-(M * N) // 256) * 256, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(M, N)],
        output_dtypes=[mx.bfloat16],
    )
    return y


SOURCE_DECODE = """
    // One warp per output row, lane l owning weights 4l..4l+3 of each tile: one 32-bit load of
    // raw bytes and one 8-byte store of four BF16 values per lane and tile (rows of K weights,
    // K % 4 == 0, keep both aligned); exponent table in shared memory.
    __shared__ uint8_t lut[16];
    if (threadIdx.x < 16) { lut[threadIdx.x] = tbl[threadIdx.x]; }
    __syncthreads();
    uint32_t lane = threadIdx.x % 32;
    int orow = int(blockIdx.x) * 8 + int(threadIdx.x / 32);
    if (orow >= N) { return; }
    size_t row = size_t(orow / NB) * RS + R0 + size_t(orow % NB);
    uint32_t epos = row_esc[row];
    const uint32_t* rw = reinterpret_cast<const uint32_t*>(raw + row * K);
    uint2* o = reinterpret_cast<uint2*>(out + size_t(orow) * K);
    for (int t = 0; t < T; t++) {
      bool in = t * 128 + int(lane) * 4 < K;
      uint32_t f = flags[row * T + t];
      uint32_t g4 = uint32_t(codes[(row * T + t) * 32 + lane]);
      uint32_t r = in ? rw[t * 32 + lane] : 0u;
      uint32_t c[4]; uint32_t e[4];
      for (int j = 0; j < 4; j++) { c[j] = (g4 >> (4u * uint32_t(j))) & 15u; e[j] = lut[c[j]]; }
      if (f) {
        uint32_t cnt = 0;
        for (int j = 0; j < 4; j++) { cnt += c[j] == 15u ? 1u : 0u; }
        uint32_t incl = cnt;
        for (int s = 1; s < 32; s <<= 1) { uint32_t v = __shfl_up_sync(0xffffffffu, incl, s); if (lane >= s) incl += v; }
        uint32_t q = epos + incl - cnt;
        for (int j = 0; j < 4; j++) { if (c[j] == 15u) { e[j] = esc[q]; q++; } }
        epos += __shfl_sync(0xffffffffu, incl, 31);
      }
      if (in) {
        uint32_t u[4];
        for (int j = 0; j < 4; j++) {
          uint32_t rj = (r >> (8u * uint32_t(j))) & 0xFFu;
          u[j] = ((rj & 0x80u) << 8) | (e[j] << 7) | (rj & 0x7Fu);
        }
        o[t * 32 + lane] = make_uint2(u[0] | (u[1] << 16), u[2] | (u[3] << 16));
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
        output_shapes=[(N, K)],
        output_dtypes=[mx.bfloat16],
    )
    return u


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
