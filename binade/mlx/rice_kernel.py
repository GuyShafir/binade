"""Binade v2 (Rice) exponent streams -> frequency ranks on the GPU (binade/rice.py layout).

One SIMD group per row walks the row's tiles in order; lane i owns weights 4i..4i+3 of
every tile, the R4 layout. A tile's low plane is read at fixed positions; each unary
plane j is read at the prefix count of weights still alive (simd_prefix_exclusive_sum),
at most 4 bits per lane. Used by the load-time transcode to R4 and by unpack.
"""

import mlx.core as mx

from ..format import TILE


def _read(dst: str, pos: str, cnt: str) -> list[str]:
    """dst = cnt (<= 12) bits of `bits` starting at bit pos. Inputs are indexed directly:
    small ones land in Metal's constant address space, so no pointer is passed around."""
    return [
        f"{dst} = 0u;",
        f"if ({cnt} > 0u) {{",
        f"  uint w_ = uint({pos} >> 5); uint s_ = uint({pos} & 31ul);",
        f"  {dst} = bits[w_] >> s_;",
        f"  if (s_ + {cnt} > 32u) {{ {dst} |= bits[w_ + 1] << (32u - s_); }}",
        f"  {dst} &= (1u << {cnt}) - 1u;",
        "}",
    ]


def source() -> str:
    lines = [
        "uint lane = thread_index_in_simdgroup;",
        "int row = int(threadgroup_position_in_grid.x) * BM + int(simdgroup_index_in_threadgroup);",
        "if (row >= R) { return; }",
        "ulong cur = ulong(roff[row]) * 32ul;",
        "for (int t = 0; t < TN; t++) {",
        "  uint m = meta[row * TN + t];",
        "  uint k = m & 3u;",
        "  uint qmax = m >> 2;",
        "  int n = min(128, K - t * 128);",
        "  uint nv = uint(clamp(n - int(lane) * 4, 0, 4));",
        "  uint lo;",
        *("  " + l for l in _read("lo", "(cur + ulong(lane * 4u * k))", "(nv * k)")),
        "  cur += ulong(uint(n) * k);",
        "  uint q[4] = {0u, 0u, 0u, 0u};",
        "  uint alive = (1u << nv) - 1u;",
        "  for (uint j = 0; j < qmax; j++) {",
        "    uint c = popcount(alive);",
        "    uint pre = simd_prefix_exclusive_sum(c);",
        "    uint tot = simd_sum(c);",
        "    uint b;",
        *("    " + l for l in _read("b", "(cur + ulong(pre))", "c")),
        "    uint na = 0u; uint bi = 0u;",
        "    for (uint i = 0; i < 4u; i++) {",
        "      if ((alive >> i) & 1u) { uint v = (b >> bi) & 1u; q[i] += v; na |= v << i; bi++; }",
        "    }",
        "    alive = na;",
        "    cur += ulong(tot);",
        "  }",
        "  uint mask = (1u << k) - 1u;",
        "  uint packed = 0u;",
        "  for (uint i = 0; i < 4u; i++) { packed |= ((q[i] << k) | ((lo >> (i * k)) & mask)) << (8u * i); }",
        "  out[(size_t(row) * TN + t) * 32 + lane] = packed;",
        "}",
        "if (lane == 0) { rend[row] = uint((cur + 31ul) >> 5); }",
    ]
    return "\n".join("    " + l for l in lines)


_KERNEL = []


def rice_ranks(meta: mx.array, bits: mx.array, roff: mx.array, K: int, bm: int = 8) -> tuple[mx.array, mx.array]:
    """(uint8 ranks [R, ceil(K/128) * 128], uint32 end word [R]) for the rows of roff
    (absolute word offsets into bits) whose tiles meta holds. Padding weights in a partial
    last tile come out as 0. A well-formed stream ends each row where the next begins."""
    from .cuda_kernels import available, rice_ranks as rice_ranks_cuda

    if available():
        return rice_ranks_cuda(meta, bits, roff, K)
    if not _KERNEL:
        _KERNEL.append(
            mx.fast.metal_kernel(
                name="binade_rice_ranks",
                input_names=["meta", "bits", "roff"],
                output_names=["out", "rend"],
                source=source(),
            )
        )
    R = roff.size
    TN = -(-K // TILE)
    u, rend = _KERNEL[0](
        inputs=[meta, bits, roff],
        template=[("R", R), ("K", K), ("TN", TN), ("BM", bm)],
        grid=(-(-R // bm) * bm * 32, 1, 1),
        threadgroup=(bm * 32, 1, 1),
        output_shapes=[(R * TN * 32,), (R,)],
        output_dtypes=[mx.uint32, mx.uint32],
    )
    return u.view(mx.uint8).reshape(R, TN * TILE), rend
