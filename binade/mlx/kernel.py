"""Binade v1 matrix-vector kernel for batch-1 decode (DESIGN.md 9), built with mx.fast.metal_kernel.

It mirrors MLX's own batch-1 BF16 kernel. For y = x W^T with one input row, MLX 0.32.2
dispatches `gemv` with bm 4 or 8, sm 1, sn 32, tm 4, tn 4 (mlx/backend/metal/matmul.cpp):
lane i owns weights 4i..4i+3 of every 128-weight block along K, accumulates `bf16 * float`
products in K order, then reduces with simd_shuffle_down. A Binade tile is exactly one such
block, so the kernel decodes the tile in registers and feeds the same accumulation order:
outputs match MLX bit for bit. (MLX switches to K split across SIMD groups when K >= 16 N;
no Gemma 4 projection does.) Rows per SIMD group and SIMD groups per threadgroup do not
change any row's summation order, so they are free performance knobs.

Per row-tile the width and mode come from one meta byte, uniform across the SIMD group;
escapes cost one simd_prefix_exclusive_sum per escape-mode tile. Row starts in the index
and escape streams are derived once at load (8 bytes per row).

Variants (all bit-identical):
  tm        rows per SIMD group
  prologue  tile word offsets computed up front with SIMD prefix sums into threadgroup
            memory, instead of a running sum inside the tile loop
  fast3     b = 3 path with per-lane word and shift constants (99.6% of 12B tiles)
  regtab    fast3 reads the first 8 table entries (all a b = 3 tile can index) from
            registers instead of threadgroup memory
  tgmeta    meta bytes staged in threadgroup memory by the prologue
  unroll    unroll factor for the tile loop
  xf32      x passed as float32 (converted once per call) instead of bf16 per tile
  swar      rebuild BF16 values two at a time in 32-bit words; escape hits counted
            with a 4-bit mask and popcount
  specesc   in escape-mode tiles every lane loads esc[cursor + lane] before decoding (one
            coalesced 32-byte load, independent of the codes); hit lanes then take their
            byte with simd_shuffle, and only in-tile escape positions >= 32 load directly.
            Removes the decode-dependent load from each tile's critical path.
  ebase     per-tile escape starts from a load-time escape-count array (1 byte per tile,
            counted in nbytes), prefix-summed in the prologue: no escape cursor carried
            from tile to tile (needs prologue)
  tgesc     the prologue copies the SIMD group's escape bytes (first WIN of them) into
            threadgroup memory with coalesced loads; escape reads hit threadgroup memory,
            positions past the window read device memory (needs prologue)
  batch     (tm 1, prologue, fast3) decode `batch` consecutive tiles first, issuing all
            their index, raw and escape loads, then run their multiply-adds in K order:
            one load latency covers the batch instead of each tile waiting on its own
  ablate    timing only, wrong outputs: "noesc" skips escapes, "nolut" skips the table,
            "noidx" skips the index stream (constant exponent)
"""

import mlx.core as mx
import numpy as np

from ..format import ESCAPE, TILE, unpack_codes

DEFAULT = {"tm": 4, "prologue": False, "fast3": False, "regtab": False, "tgmeta": False, "unroll": 1, "xf32": False, "swar": False, "specesc": False, "ebase": False, "tgesc": False, "batch": 1, "ablate": ""}
WIN = 512  # escape bytes staged per SIMD group by tgesc
OPTIONS = tuple(DEFAULT)


def source(tm: int, prologue: bool, fast3: bool, regtab: bool = False, tgmeta: bool = False, unroll: int = 1, xf32: bool = False, swar: bool = False, specesc: bool = False, ebase: bool = False, tgesc: bool = False, batch: int = 1, ablate: str = "") -> str:
    if batch > 1:
        if not (tm == 1 and prologue and fast3) or any((regtab, tgmeta, xf32, swar, specesc, ebase, tgesc, ablate)) or unroll > 1:
            raise ValueError("batch is implemented for tm=1 with prologue and fast3 only")
        return _batched(batch)
    if tgesc and (not prologue or specesc):
        raise ValueError("tgesc needs prologue and replaces specesc")
    if ebase and not prologue:
        raise ValueError("ebase needs prologue")
    if specesc and swar:
        raise ValueError("specesc replaces the escape block; use it without swar")
    if (regtab and not fast3) or (tgmeta and not prologue):
        raise ValueError("regtab needs fast3, tgmeta needs prologue")
    lines = [
        "uint lane = thread_index_in_simdgroup;",
        "uint sg = simdgroup_index_in_threadgroup;",
        "uint tg = threadgroup_position_in_grid.x;",
        "uint tid = thread_position_in_threadgroup.x;",
        "threadgroup uchar lut[256];",
        "for (uint i = tid; i < 256; i += BM * 32) { lut[i] = tbl[i]; }",
        "threadgroup_barrier(mem_flags::mem_threadgroup);",
        f"int row0 = (int(tg) * BM + int(sg)) * {tm};",
        "if (row0 >= N) { return; }",
        f"row0 = row0 + {tm} <= N ? row0 : N - {tm};  // same inward shift as MLX",
        f"float result[{tm}];",
        f"uint wpos[{tm}];",
        f"uint epos[{tm}];",
        f"for (int tm = 0; tm < {tm}; tm++) {{ result[tm] = 0.0f; wpos[tm] = row_word[row0 + tm]; epos[tm] = row_esc[row0 + tm]; }}",
    ]
    if prologue:
        lines += [
            f"threadgroup uint offs[BM * {tm} * T];",
            f"threadgroup uint* my = offs + sg * ({tm} * T);",
            *([f"threadgroup uchar metas[BM * {tm} * T];", f"threadgroup uchar* mym = metas + sg * ({tm} * T);"] if tgmeta else []),
            *([f"threadgroup uint eoffs[BM * {tm} * T];", f"threadgroup uint* mye = eoffs + sg * ({tm} * T);"] if ebase else []),
            f"for (int tm = 0; tm < {tm}; tm++) {{",
            "    uint carry = wpos[tm];",
            *(["    uint ecarry = epos[tm];"] if ebase else []),
            "    for (int c = 0; c < T; c += 32) {",
            "        int t = c + int(lane);",
            "        uint mt = t < T ? uint(meta[size_t(row0 + tm) * T + t]) : 0u;",
            "        uint w4 = 4u * (mt & 15u);",
            *(["        if (t < T) { mym[tm * T + t] = uchar(mt); }"] if tgmeta else []),
            "        uint before = simd_prefix_exclusive_sum(w4);",
            "        if (t < T) { my[tm * T + t] = carry + before; }",
            "        carry += simd_sum(w4);",
            *(["        uint ec = t < T ? uint(ecnt[size_t(row0 + tm) * T + t]) : 0u;",
               "        uint ebefore = simd_prefix_exclusive_sum(ec);",
               "        if (t < T) { mye[tm * T + t] = ecarry + ebefore; }",
               "        ecarry += simd_sum(ec);"] if ebase else []),
            "    }",
            "}",
            *([f"threadgroup uchar wins[BM * {WIN}];",
               f"threadgroup uchar* win = wins + sg * {WIN};",
               "uint wbase = row_esc[row0];",
               f"for (uint i = lane; i < {WIN}u; i += 32u) {{ win[i] = esc[wbase + i]; }}"] if tgesc else []),
            "simdgroup_barrier(mem_flags::mem_threadgroup);",
        ]
    if fast3:
        lines += ["uint w3 = (lane * 12u) >> 5;", "uint sh3 = (lane * 12u) & 31u;"]
    if regtab:
        lines += ["uint tab0 = uint(lut[0]) | (uint(lut[1]) << 8) | (uint(lut[2]) << 16) | (uint(lut[3]) << 24);",
                  "uint tab1 = uint(lut[4]) | (uint(lut[5]) << 8) | (uint(lut[6]) << 16) | (uint(lut[7]) << 24);"]
    base = "my[tm * T + t]" if prologue else "wpos[tm]"
    general = [
        "    uint s = lane * 4u * b;",
        f"    uint w = {base} + (s >> 5);",
        "    uint sh = s & 31u;",
        "    uint lo = idx[w];",
        "    uint bits = sh == 0u ? lo : (lo >> sh) | (idx[w + 1] << (32u - sh));",
        "    uint mask = (1u << b) - 1u;",
        "    for (int j = 0; j < 4; j++) { c[j] = (bits >> (uint(j) * b)) & mask; }",
    ]
    three = [
        f"    uint w = {base} + w3;",
        "    uint lo = idx[w];",
        "    uint bits = sh3 == 0u ? lo : (lo >> sh3) | (idx[w + 1] << (32u - sh3));",
        "    for (int j = 0; j < 4; j++) { c[j] = (bits >> (uint(j) * 3u)) & 7u; }",
        *(["    for (int j = 0; j < 4; j++) { e[j] = ((c[j] < 4u ? tab0 : tab1) >> (8u * (c[j] & 3u))) & 0xFFu; }"] if regtab else []),
    ]
    general_tail = ["    for (int j = 0; j < 4; j++) { e[j] = lut[c[j]]; }"] if regtab else []
    zero_tail = ["    for (int j = 0; j < 4; j++) { e[j] = lut[0]; }"] if regtab else []
    if fast3:
        decode = ["if (b == 3u) {"] + three + ["} else if (b > 0u) {"] + general + general_tail + ["} else {"] + zero_tail + ["}"]
    else:
        decode = ["if (b > 0u) {"] + general + ["}"]
    lines += [
        *([f"#pragma unroll({unroll})"] if unroll > 1 else []),
        "for (int t = 0; t < T; t++) {",
        "  int k0 = t * 128 + int(lane) * 4;",
        *(["  float4 v = *(reinterpret_cast<const device float4*>(x + k0));"] if xf32 else
          ["  float v[4];", "  for (int j = 0; j < 4; j++) { v[j] = static_cast<float>(x[k0 + j]); }"]),
        f"  for (int tm = 0; tm < {tm}; tm++) {{",
        "    size_t row = size_t(row0 + tm);",
        "    uint m = " + ("mym[tm * T + t];" if tgmeta else "meta[row * T + t];"),
        "    uint b = " + ("0u;" if ablate == "noidx" else "m & 15u;"),
        "    uint c[4] = {0u, 0u, 0u, 0u};",
        "    uint e[4];",
        *(["    uint e0 = mye[tm * T + t];"] if ebase else ["    uint e0 = epos[tm];"]),
        *(["    uint spec = (m & 16u) ? uint(esc[e0 + lane]) : 0u;  // before decoding: independent of the codes"] if specesc else []),
        *["    " + l for l in decode],
        *([] if regtab else ["    for (int j = 0; j < 4; j++) { e[j] = lut[c[j]]; }"]),
        *(["    for (int j = 0; j < 4; j++) { c[j] = 0u; }"] if ablate == "noidx" else []),
        *(["    for (int j = 0; j < 4; j++) { e[j] = c[j] + 118u; }"] if ablate == "nolut" else []),
        "    if (" + ("false" if ablate in ("noesc", "noidx") else "m & 16u") + ") {",
        "      uint code = (1u << b) - 1u;",
        *(["      uint hm = (c[0] == code ? 1u : 0u) | (c[1] == code ? 2u : 0u) | (c[2] == code ? 4u : 0u) | (c[3] == code ? 8u : 0u);",
           "      uint cnt = popcount(hm);",
           "      uint p = epos[tm] + simd_prefix_exclusive_sum(cnt);",
           "      if (hm & 1u) { e[0] = esc[p]; p++; }",
           "      if (hm & 2u) { e[1] = esc[p]; p++; }",
           "      if (hm & 4u) { e[2] = esc[p]; p++; }",
           "      if (hm & 8u) { e[3] = esc[p]; }"] if swar else
          ["      uint cnt = 0;",
           "      for (int j = 0; j < 4; j++) { cnt += c[j] == code ? 1u : 0u; }",
           "      uint k = simd_prefix_exclusive_sum(cnt);",
           "      bool far = false;",
           "      for (int j = 0; j < 4; j++) {",
           "        bool hit = c[j] == code;",
           "        if (simd_any(hit)) {  // uniform: every lane runs the shuffle",
           "          uint got = simd_shuffle(spec, ushort(k & 31u));",
           "          if (hit) { e[j] = got; far = far || k >= 32u; }",
           "        }",
           "        k += hit ? 1u : 0u;",
           "      }",
           "      if (simd_any(far)) {  // rare: more than 32 escapes in this tile",
           "        uint q = simd_prefix_exclusive_sum(cnt);",
           "        for (int j = 0; j < 4; j++) {",
           "          if (c[j] == code) { if (q >= 32u) { e[j] = esc[e0 + q]; } q++; }",
           "        }",
           "      }"] if specesc else
          ["      uint cnt = 0;",
           "      for (int j = 0; j < 4; j++) { cnt += c[j] == code ? 1u : 0u; }",
           "      uint p = e0 + " + ("0u;" if ablate == "noprefix" else "simd_prefix_exclusive_sum(cnt);"),
           *(["      for (int j = 0; j < 4; j++) {",
              "        if (c[j] == code) {",
              f"          uint o = p - wbase;",
              f"          if (o < {WIN}u) {{ e[j] = win[o]; }} else {{ e[j] = esc[p]; }}",
              "          p++;",
              "        }",
              "      }"] if tgesc else
             ["      for (int j = 0; j < 4; j++) { if (c[j] == code) { e[j] = " + ("117u" if ablate == "noload" else "esc[p]") + "; p++; } }"])]),
        *([] if ablate == "nosum" or ebase else ["      epos[tm] += simd_sum(cnt);"]),
        "    }",
        "    uint r = *(reinterpret_cast<const device uint*>(raw + row * K + size_t(k0)));",
        *(["    uint t01 = (r & 0xFFu) | ((r & 0xFF00u) << 8);",
           "    uint t23 = ((r >> 16) & 0xFFu) | ((r >> 8) & 0xFF0000u);",
           "    bfloat2 w01 = as_type<bfloat2>(((t01 & 0x00800080u) << 8) | (t01 & 0x007F007Fu) | ((e[0] | (e[1] << 16)) << 7));",
           "    bfloat2 w23 = as_type<bfloat2>(((t23 & 0x00800080u) << 8) | (t23 & 0x007F007Fu) | ((e[2] | (e[3] << 16)) << 7));",
           "    result[tm] += w01.x * v[0];",
           "    result[tm] += w01.y * v[1];",
           "    result[tm] += w23.x * v[2];",
           "    result[tm] += w23.y * v[3];"] if swar else
          ["    for (int j = 0; j < 4; j++) {",
           "      uint rj = (r >> (8u * uint(j))) & 0xFFu;",
           "      bfloat16_t wv = as_type<bfloat16_t>(ushort(((rj & 0x80u) << 8) | (e[j] << 7) | (rj & 0x7Fu)));",
           "      result[tm] += wv * v[j];",
           "    }"]),
        *([] if prologue else ["    wpos[tm] += 4u * b;"]),
        "  }",
        "}",
        f"for (int tm = 0; tm < {tm}; tm++) {{",
        "  for (ushort sn = 16; sn >= 1; sn >>= 1) { result[tm] += simd_shuffle_down(result[tm], sn); }",
        "}",
        f"if (lane == 0) {{ for (int tm = 0; tm < {tm}; tm++) {{ y[row0 + tm] = static_cast<bfloat16_t>(result[tm]); }} }}",
    ]
    return "\n".join("    " + l for l in lines)


def _batched(B: int) -> str:
    """tm = 1 kernel that decodes B tiles (loads in flight together) before their multiply-adds."""
    lines = [
        "uint lane = thread_index_in_simdgroup;",
        "uint sg = simdgroup_index_in_threadgroup;",
        "uint tg = threadgroup_position_in_grid.x;",
        "uint tid = thread_position_in_threadgroup.x;",
        "threadgroup uchar lut[256];",
        "for (uint i = tid; i < 256; i += BM * 32) { lut[i] = tbl[i]; }",
        "threadgroup_barrier(mem_flags::mem_threadgroup);",
        "int row0 = int(tg) * BM + int(sg);",
        "if (row0 >= N) { return; }",
        "size_t row = size_t(row0);",
        "threadgroup uint offs[BM * T];",
        "threadgroup uint* my = offs + sg * T;",
        "{",
        "  uint carry = row_word[row0];",
        "  for (int c = 0; c < T; c += 32) {",
        "    int t = c + int(lane);",
        "    uint w4 = t < T ? 4u * (uint(meta[row * T + t]) & 15u) : 0u;",
        "    uint before = simd_prefix_exclusive_sum(w4);",
        "    if (t < T) { my[t] = carry + before; }",
        "    carry += simd_sum(w4);",
        "  }",
        "}",
        "simdgroup_barrier(mem_flags::mem_threadgroup);",
        "uint epos = row_esc[row0];",
        "uint w3 = (lane * 12u) >> 5;",
        "uint sh3 = (lane * 12u) & 31u;",
        "float result = 0.0f;",
        f"for (int t0 = 0; t0 < T; t0 += {B}) {{",
        f"  uint e[{B}][4];",
        f"  uint r[{B}];",
        f"  for (int u = 0; u < {B}; u++) {{",
        "    int t = t0 + u;",
        "    if (t >= T) { break; }",
        "    uint m = meta[row * T + t];",
        "    uint b = m & 15u;",
        "    uint c[4] = {0u, 0u, 0u, 0u};",
        "    if (b == 3u) {",
        "      uint w = my[t] + w3;",
        "      uint lo = idx[w];",
        "      uint bits = sh3 == 0u ? lo : (lo >> sh3) | (idx[w + 1] << (32u - sh3));",
        "      for (int j = 0; j < 4; j++) { c[j] = (bits >> (uint(j) * 3u)) & 7u; }",
        "    } else if (b > 0u) {",
        "      uint s = lane * 4u * b;",
        "      uint w = my[t] + (s >> 5);",
        "      uint sh = s & 31u;",
        "      uint lo = idx[w];",
        "      uint bits = sh == 0u ? lo : (lo >> sh) | (idx[w + 1] << (32u - sh));",
        "      uint mask = (1u << b) - 1u;",
        "      for (int j = 0; j < 4; j++) { c[j] = (bits >> (uint(j) * b)) & mask; }",
        "    }",
        "    for (int j = 0; j < 4; j++) { e[u][j] = lut[c[j]]; }",
        "    if (m & 16u) {",
        "      uint code = (1u << b) - 1u;",
        "      uint cnt = 0;",
        "      for (int j = 0; j < 4; j++) { cnt += c[j] == code ? 1u : 0u; }",
        "      uint p = epos + simd_prefix_exclusive_sum(cnt);",
        "      for (int j = 0; j < 4; j++) { if (c[j] == code) { e[u][j] = esc[p]; p++; } }",
        "      epos += simd_sum(cnt);",
        "    }",
        "    r[u] = *(reinterpret_cast<const device uint*>(raw + row * K + size_t(t * 128 + int(lane) * 4)));",
        "  }",
        f"  for (int u = 0; u < {B}; u++) {{",
        "    int t = t0 + u;",
        "    if (t >= T) { break; }",
        "    int k0 = t * 128 + int(lane) * 4;",
        "    for (int j = 0; j < 4; j++) {",
        "      uint rj = (r[u] >> (8u * uint(j))) & 0xFFu;",
        "      bfloat16_t wv = as_type<bfloat16_t>(ushort(((rj & 0x80u) << 8) | (e[u][j] << 7) | (rj & 0x7Fu)));",
        "      result += wv * static_cast<float>(x[k0 + j]);",
        "    }",
        "  }",
        "}",
        "for (ushort sn = 16; sn >= 1; sn >>= 1) { result += simd_shuffle_down(result, sn); }",
        "if (lane == 0) { y[row0] = static_cast<bfloat16_t>(result); }",
    ]
    return "\n".join("    " + l for l in lines)


_KERNELS = {}


def _kernel(opts: tuple, math_mode: str = "safe"):
    key = (opts, math_mode)
    if key not in _KERNELS:
        o = dict(zip(OPTIONS, opts))
        tag = "_".join(f"{k}{v if isinstance(v, str) else int(v)}" for k, v in o.items())
        _KERNELS[key] = mx.fast.metal_kernel(
            name=f"binade_gemv_{tag}_{math_mode}",
            input_names=["x", "raw", "idx", "esc", "meta", "tbl", "row_word", "row_esc", "ecnt"],
            output_names=["y"],
            source=source(**o),
            compile_options={"math_mode": math_mode},
        )
    return _KERNELS[key]


def escape_counts(meta: np.ndarray, idx: np.ndarray, T: int = TILE) -> np.ndarray:
    """Escapes per tile (int64 [ntiles]), from the codes of escape-mode tiles."""
    b = (meta & 0x0F).astype(np.int64)
    start = 4 * (np.cumsum(b) - b)
    counts = np.zeros(b.size, np.int64)
    sel_all = np.flatnonzero((meta & ESCAPE) != 0)
    for w in np.unique(b[sel_all]):
        sel = sel_all[b[sel_all] == w]
        codes = unpack_codes(idx[start[sel][:, None] + np.arange(4 * w)], int(w), T)
        counts[sel] = (codes == (1 << int(w)) - 1).sum(axis=1)
    return counts


class BinadeMatrix:
    """A Binade v1 weight [N, K] ready for the kernel: packed arrays plus per-row stream starts."""

    def __init__(self, raw, table, meta, idx, esc):
        raw_np = np.asarray(raw)
        self.N, self.K = raw_np.shape[0], int(np.prod(raw_np.shape[1:]))
        if self.K % TILE or self.N < 4:
            raise ValueError(f"kernel needs K % {TILE} == 0 and N >= 4, got {self.N}x{self.K}")
        self.T = self.K // TILE
        meta_np, idx_np = np.asarray(meta), np.asarray(idx)
        b = (meta_np & 0x0F).astype(np.int64).reshape(self.N, self.T)
        words = 4 * b.sum(axis=1)
        per_tile = escape_counts(meta_np, idx_np)
        if per_tile.max(initial=0) > 255:
            raise ValueError("escape count per tile exceeds a byte")
        self.ecnt = mx.array(per_tile.astype(np.uint8))  # derived at load; read only by the ebase variant
        escs = per_tile.reshape(self.N, self.T).sum(axis=1)
        self.row_word = mx.array((np.cumsum(words) - words).astype(np.uint32))
        self.row_esc = mx.array((np.cumsum(escs) - escs).astype(np.uint32))
        self.raw = mx.array(raw_np.reshape(self.N, self.K))
        self.idx = mx.array(np.concatenate([idx_np, np.zeros(2, np.uint32)]))  # each lane may read 2 words
        esc_np = np.asarray(esc)
        self.esc = mx.array(np.concatenate([esc_np, np.zeros(WIN, np.uint8)]))  # staged/speculative reads may run past the end
        self.meta = mx.array(meta_np)
        self.table = mx.array(np.asarray(table))

    def nbytes(self, ebase: bool = False) -> int:
        """Bytes a matvec streams from the weight (x and y excluded)."""
        arrays = (self.raw, self.idx, self.esc, self.meta, self.table, self.row_word, self.row_esc) + ((self.ecnt,) if ebase else ())
        return sum(a.nbytes for a in arrays)

    def matvec(self, x: mx.array, bm: int | None = None, math_mode: str = "safe", **opts) -> mx.array:
        """y = x W^T for a single input row x [K] (or [1, K]), bf16 out like MLX. opts: see OPTIONS."""
        o = {**DEFAULT, **opts}
        tm = o["tm"]
        x = x.astype(mx.float32) if o["xf32"] else x
        bm = bm or (8 if self.N >= 4096 else 4)
        if self.N < tm:
            raise ValueError(f"N={self.N} < rows per SIMD group {tm}")
        groups = -(-self.N // (bm * tm))
        (y,) = _kernel(tuple(o[k] for k in OPTIONS), math_mode)(
            inputs=[x.reshape(-1), self.raw, self.idx, self.esc, self.meta, self.table, self.row_word, self.row_esc, self.ecnt],
            template=[("N", self.N), ("K", self.K), ("T", self.T), ("BM", bm)],
            grid=(groups * bm * 32, 1, 1),
            threadgroup=(bm * 32, 1, 1),
            output_shapes=[(self.N,)],
            output_dtypes=[mx.bfloat16],
        )
        return y


# ---------------------------------------------------------------------------
# R4: a runtime layout transcoded from v1 at load time (the file format is unchanged).
#
# Every weight stores min(rank, 15) in 4 bits; code 15 escapes to the exponent byte in
# the escape stream, so only ranks >= 15 escape (0.024% of Gemma 4 12B weights; 2.9% of
# tiles hold one). Lane i's four codes are one uint16, so there are no per-tile widths,
# no word straddling and no offsets to derive, and the escape path runs only in flagged
# tiles. Costs about 12.07 bits per weight in memory against v1's 11.28 on disk.
# Because every address but an escape's is known in advance, the runtime kernel
# (source_r4_pf) loads tile t + 2 while working on tile t: in a dependent chain of real
# 12B layers that takes R4 from 1.14x to 1.25x BF16.


def source_r4(tm: int, shuffle: bool = False) -> str:
    """shuffle: codes index only table entries 0..15, so each lane keeps entry (lane & 15)
    in a register and lookups are simd_shuffle; no threadgroup memory and no barrier."""
    lines = [
        "uint lane = thread_index_in_simdgroup;",
        "uint sg = simdgroup_index_in_threadgroup;",
        "uint tg = threadgroup_position_in_grid.x;",
        "uint tid = thread_position_in_threadgroup.x;",
        *(["uint tab = uint(tbl[lane & 15u]);"] if shuffle else [
            "threadgroup uchar lut[256];",
            "for (uint i = tid; i < 256; i += BM * 32) { lut[i] = tbl[i]; }",
            "threadgroup_barrier(mem_flags::mem_threadgroup);"]),
        f"int row0 = (int(tg) * BM + int(sg)) * {tm};",
        "if (row0 >= N) { return; }",
        f"row0 = row0 + {tm} <= N ? row0 : N - {tm};",
        f"float result[{tm}];",
        f"uint epos[{tm}];",
        f"for (int tm = 0; tm < {tm}; tm++) {{ result[tm] = 0.0f; epos[tm] = row_esc[row0 + tm]; }}",
        "for (int t = 0; t < T; t++) {",
        "  int k0 = t * 128 + int(lane) * 4;",
        "  float v[4];",
        "  for (int j = 0; j < 4; j++) { v[j] = static_cast<float>(x[k0 + j]); }",
        f"  for (int tm = 0; tm < {tm}; tm++) {{",
        "    size_t row = size_t(row0 + tm);",
        "    uint g = codes[(row * T + t) * 32 + lane];",
        "    uint c[4];",
        "    uint e[4];",
        "    for (int j = 0; j < 4; j++) { c[j] = (g >> (4u * uint(j))) & 15u; e[j] = "
        + ("simd_shuffle(tab, ushort(c[j]));" if shuffle else "lut[c[j]];") + " }",
        "    if (flags[row * T + t]) {",
        "      uint cnt = 0;",
        "      for (int j = 0; j < 4; j++) { cnt += c[j] == 15u ? 1u : 0u; }",
        "      uint p = epos[tm] + simd_prefix_exclusive_sum(cnt);",
        "      for (int j = 0; j < 4; j++) { if (c[j] == 15u) { e[j] = esc[p]; p++; } }",
        "      epos[tm] += simd_sum(cnt);",
        "    }",
        "    uint r = *(reinterpret_cast<const device uint*>(raw + row * K + size_t(k0)));",
        "    for (int j = 0; j < 4; j++) {",
        "      uint rj = (r >> (8u * uint(j))) & 0xFFu;",
        "      bfloat16_t wv = as_type<bfloat16_t>(ushort(((rj & 0x80u) << 8) | (e[j] << 7) | (rj & 0x7Fu)));",
        "      result[tm] += wv * v[j];",
        "    }",
        "  }",
        "}",
        f"for (int tm = 0; tm < {tm}; tm++) {{",
        "  for (ushort sn = 16; sn >= 1; sn >>= 1) { result[tm] += simd_shuffle_down(result[tm], sn); }",
        "}",
        f"if (lane == 0) {{ for (int tm = 0; tm < {tm}; tm++) {{ y[row0 + tm] = static_cast<bfloat16_t>(result[tm]); }} }}",
    ]
    return "\n".join("    " + l for l in lines)


def source_r4_pf(dist: int, partial: bool = False, gather: bool = False) -> str:
    """tm = 1 R4 matvec that loads tile t + dist's codes, raw bytes and flag while working
    on tile t (every address but the escape's is known in advance).

    partial: K % 128 != 0. The last tile holds K % 128 weights; like MLX's gemv tail
    (load_safe), lanes past K still add 0 * 0 to their sums, which can turn a -0 into +0.
    gather: G matvecs in one launch, one per grid row g: output row r of pair g uses
    storage row ei[g] * RS + R0 + r and input vector xi[g] (MLX's gemv_gather)."""
    D = dist

    def raw_at(tt: str) -> str:
        load = f"*(reinterpret_cast<const device uint*>(raw + row * K + size_t({tt} * 128) + size_t(lane) * 4))"
        return f"(({tt}) * 128 + int(lane) * 4 < K ? {load} : 0u)" if partial else load

    lines = [
        "uint lane = thread_index_in_simdgroup;",
        "uint sg = simdgroup_index_in_threadgroup;",
        "uint tg = threadgroup_position_in_grid.x;",
        "uint tid = thread_position_in_threadgroup.x;",
        "threadgroup uchar lut[256];",
        "for (uint i = tid; i < 256; i += BM * 32) { lut[i] = tbl[i]; }",
        "threadgroup_barrier(mem_flags::mem_threadgroup);",
        "int row0 = int(tg) * BM + int(sg);",
        "if (row0 >= N) { return; }",
    ]
    if gather:
        lines += [
            "uint g = threadgroup_position_in_grid.y;",
            "size_t row = size_t(ei[g]) * RS + R0 + size_t(row0);",
            "size_t xb = size_t(xi[g]) * K;",
            "size_t yo = size_t(g) * N + size_t(row0);",
        ]
    else:
        lines += ["size_t row = size_t(row0);", "size_t xb = 0;", "size_t yo = size_t(row0);"]
    lines += [
        "float result = 0.0f;",
        "uint epos = row_esc[row];",
        f"uint gq[{D}]; uint rq[{D}]; uint fq[{D}];",
        f"for (int d = 0; d < {D}; d++) {{ int t = d < T ? d : T - 1; gq[d] = codes[(row * T + t) * 32 + lane]; rq[d] = {raw_at('t')}; fq[d] = flags[row * T + t]; }}",
        "for (int t = 0; t < T; t++) {",
        "  uint g4 = gq[0]; uint r = rq[0]; uint f = fq[0];",
        f"  for (int d = 0; d + 1 < {D}; d++) {{ gq[d] = gq[d + 1]; rq[d] = rq[d + 1]; fq[d] = fq[d + 1]; }}",
        f"  int tn = t + {D} < T ? t + {D} : T - 1;",
        f"  gq[{D - 1}] = codes[(row * T + tn) * 32 + lane]; rq[{D - 1}] = {raw_at('tn')}; fq[{D - 1}] = flags[row * T + tn];",
        "  int k0 = t * 128 + int(lane) * 4;",
        "  float v[4];",
        (
            "  for (int j = 0; j < 4; j++) { v[j] = k0 + j < K ? static_cast<float>(x[xb + k0 + j]) : 0.0f; }"
            if partial
            else "  for (int j = 0; j < 4; j++) { v[j] = static_cast<float>(x[xb + k0 + j]); }"
        ),
        "  uint c[4]; uint e[4];",
        "  for (int j = 0; j < 4; j++) { c[j] = (g4 >> (4u * uint(j))) & 15u; e[j] = lut[c[j]]; }",
        "  if (f) {",
        "    uint cnt = 0;",
        "    for (int j = 0; j < 4; j++) { cnt += c[j] == 15u ? 1u : 0u; }",
        "    uint p = epos + simd_prefix_exclusive_sum(cnt);",
        "    for (int j = 0; j < 4; j++) { if (c[j] == 15u) { e[j] = esc[p]; p++; } }",
        "    epos += simd_sum(cnt);",
        "  }",
        "  for (int j = 0; j < 4; j++) {",
        "    uint rj = (r >> (8u * uint(j))) & 0xFFu;",
        "    bfloat16_t wv = as_type<bfloat16_t>(ushort(((rj & 0x80u) << 8) | (e[j] << 7) | (rj & 0x7Fu)));",
        *(["    wv = k0 + j < K ? wv : bfloat16_t(0.0f);"] if partial else []),
        "    result += wv * v[j];",
        "  }",
        "}",
        "for (ushort sn = 16; sn >= 1; sn >>= 1) { result += simd_shuffle_down(result, sn); }",
        "if (lane == 0) { y[yo] = static_cast<bfloat16_t>(result); }",
    ]
    return "\n".join("    " + l for l in lines)


_R4PF = {}


def _kernel_r4_pf(dist: int, partial: bool = False, gather: bool = False):
    key = (dist, partial, gather)
    if key not in _R4PF:
        inputs = ["x", "raw", "codes", "flags", "esc", "tbl", "row_esc"] + (["xi", "ei"] if gather else [])
        _R4PF[key] = mx.fast.metal_kernel(
            name=f"binade_gemv_r4_pf{dist}{'_p' if partial else ''}{'_g' if gather else ''}",
            input_names=inputs,
            output_names=["y"],
            source=source_r4_pf(dist, partial, gather),
        )
    return _R4PF[key]


_R4 = {}


def _kernel_r4(tm: int, shuffle: bool = False):
    if (tm, shuffle) not in _R4:
        _R4[tm, shuffle] = mx.fast.metal_kernel(
            name=f"binade_gemv_r4_tm{tm}_s{int(shuffle)}",
            input_names=["x", "raw", "codes", "flags", "esc", "tbl", "row_esc"],
            output_names=["y"],
            source=source_r4(tm, shuffle),
        )
    return _R4[tm, shuffle]


class R4Matrix:
    """Runtime 4-bit layout of a v1 weight [N, K] (K % 128 == 0), transcoded at load."""

    def __init__(self, raw, table, meta, idx, esc):
        from ..format import decode_tiles

        raw_np = np.asarray(raw)
        self.N, self.K = raw_np.shape[0], int(np.prod(raw_np.shape[1:]))
        if self.K % TILE or self.N < 4:
            raise ValueError(f"R4 needs K % {TILE} == 0 and N >= 4, got {self.N}x{self.K}")
        self.T = self.K // TILE
        table_np = np.asarray(table)
        e, _ = decode_tiles(np.asarray(meta), np.asarray(idx), np.asarray(esc), table_np)  # [nt, 128] exponents
        inv = np.full(256, 255, np.int64)
        present = np.flatnonzero(np.bincount(e.ravel(), minlength=256))
        inv[table_np[: present.size]] = np.arange(present.size)  # rank of each present exponent
        rank = inv[e]
        codes = np.minimum(rank, 15).astype(np.uint16).reshape(-1, 32, 4)
        packed = codes[..., 0] | (codes[..., 1] << 4) | (codes[..., 2] << 8) | (codes[..., 3] << 12)
        hit = rank >= 15
        per_tile = hit.sum(axis=1)
        per_row = per_tile.reshape(self.N, self.T).sum(axis=1)
        self.raw = mx.array(raw_np.reshape(self.N, self.K))
        self.codes = mx.array(packed.reshape(-1).astype(np.uint16))
        self.flags = mx.array((per_tile > 0).astype(np.uint8))
        self.esc = mx.array(np.concatenate([e[hit].astype(np.uint8), np.zeros(1, np.uint8)]))
        self.table = mx.array(table_np)
        self.row_esc = mx.array((np.cumsum(per_row) - per_row).astype(np.uint32))

    def nbytes(self) -> int:
        return sum(a.nbytes for a in (self.raw, self.codes, self.flags, self.esc, self.table, self.row_esc))

    def matvec(self, x: mx.array, tm: int = 1, bm: int | None = None, shuffle: bool = False, pf: int = 0) -> mx.array:
        bm = bm or (8 if self.N >= 4096 else 4)
        groups = -(-self.N // (bm * tm))
        kernel = _kernel_r4_pf(pf) if pf else _kernel_r4(tm, shuffle)
        if pf and tm != 1:
            raise ValueError("prefetch variant is tm = 1")
        (y,) = kernel(
            inputs=[x.reshape(-1), self.raw, self.codes, self.flags, self.esc, self.table, self.row_esc],
            template=[("N", self.N), ("K", self.K), ("T", self.T), ("BM", bm)],
            grid=(groups * bm * 32, 1, 1),
            threadgroup=(bm * 32, 1, 1),
            output_shapes=[(self.N,)],
            output_dtypes=[mx.bfloat16],
        )
        return y


def source_r4_decode() -> str:
    """R4 -> BF16 rows: the matvec kernel's decode with a coalesced store instead of the multiply-adds.
    Output row r comes from storage row (r / NB) * RS + R0 + r % NB (identity: NB = RS = N,
    R0 = 0), so one half of each expert's fused gate/up rows decodes into its own array."""
    lines = [
        "uint lane = thread_index_in_simdgroup;",
        "uint sg = simdgroup_index_in_threadgroup;",
        "uint tg = threadgroup_position_in_grid.x;",
        "uint tid = thread_position_in_threadgroup.x;",
        "threadgroup uchar lut[256];",
        "for (uint i = tid; i < 256; i += BM * 32) { lut[i] = tbl[i]; }",
        "threadgroup_barrier(mem_flags::mem_threadgroup);",
        "int orow = int(tg) * BM + int(sg);",
        "if (orow >= N) { return; }",
        "size_t row = size_t(orow / NB) * RS + R0 + size_t(orow % NB);",
        "uint epos = row_esc[row];",
        "for (int t = 0; t < T; t++) {",
        "  bool in = t * 128 + int(lane) * 4 < K;  // false only in a partial last tile",
        "  size_t base = row * T + t;",
        "  uint g = codes[base * 32 + lane];",
        "  uint c[4];",
        "  uint e[4];",
        "  for (int j = 0; j < 4; j++) { c[j] = (g >> (4u * uint(j))) & 15u; e[j] = lut[c[j]]; }",
        "  if (flags[base]) {",
        "    uint cnt = 0;",
        "    for (int j = 0; j < 4; j++) { cnt += c[j] == 15u ? 1u : 0u; }",
        "    uint p = epos + simd_prefix_exclusive_sum(cnt);",
        "    for (int j = 0; j < 4; j++) { if (c[j] == 15u) { e[j] = esc[p]; p++; } }",
        "    epos += simd_sum(cnt);",
        "  }",
        "  size_t k0 = row * K + size_t(t * 128) + size_t(lane) * 4;",
        "  size_t o0 = size_t(orow) * K + size_t(t * 128) + size_t(lane) * 4;",
        "  uint r = in ? *(reinterpret_cast<const device uint*>(raw + k0)) : 0u;",
        "  uint u[4];",
        "  for (int j = 0; j < 4; j++) {",
        "    uint rj = (r >> (8u * uint(j))) & 0xFFu;",
        "    u[j] = ((rj & 0x80u) << 8) | (e[j] << 7) | (rj & 0x7Fu);",
        "  }",
        "  if (in) { *(reinterpret_cast<device uint2*>(out + o0)) = uint2(u[0] | (u[1] << 16), u[2] | (u[3] << 16)); }",
        "}",
    ]
    return "\n".join("    " + l for l in lines)


_R4_DECODE = []


def r4_decode(raw, codes, flags, esc, table, row_esc, N: int, K: int, nb: int | None = None, rs: int | None = None, r0: int = 0) -> mx.array:
    """BF16 weight [N, K] (uint16 bit patterns viewed as bfloat16) from R4 arrays; output row r
    is storage row (r // nb) * rs + r0 + r % nb (default: the first N rows)."""
    if not _R4_DECODE:
        _R4_DECODE.append(
            mx.fast.metal_kernel(
                name="binade_r4_decode",
                input_names=["raw", "codes", "flags", "esc", "tbl", "row_esc"],
                output_names=["out"],
                source=source_r4_decode(),
            )
        )
    bm = 8
    (u,) = _R4_DECODE[0](
        inputs=[raw, codes, flags, esc, table, row_esc],
        template=[("N", N), ("K", K), ("T", -(-K // TILE)), ("BM", bm), ("NB", nb or N), ("RS", rs or nb or N), ("R0", r0)],
        grid=(-(-N // bm) * bm * 32, 1, 1),
        threadgroup=(bm * 32, 1, 1),
        output_shapes=[(N * K,)],
        output_dtypes=[mx.uint16],
    )
    return u.view(mx.bfloat16).reshape(N, K)


def source_r4_gemm(bm: int, bn: int) -> str:
    """R4 GEMM over slots sorted by expert, bit-identical to MLX's GEMM (steel) and its sorted
    gather_mm (gather_mm_rhs). MLX accumulates each output element in float
    simdgroup_multiply_accumulate steps of 8 along K, in order, whatever its block sizes, when
    K is a multiple of its K block and no split-K applies; this kernel runs the same steps
    (fragment layout from steel's BaseMMAFrag::get_coord) on weights decoded from R4.
    A threadgroup of 4 SIMD groups (2 x 2) owns bm slots x bn outputs and walks
    K one R4 tile (128 weights) at a time, in order: it stages the slots' inputs in
    threadgroup memory, each SIMD group decodes bn / 4 weight rows (a running escape pointer
    per row, as in the matvec kernel), then every SIMD group runs the tile's MMA steps of 8.
    Slots are sorted by expert; each run of equal experts in the block redoes the K walk
    with that expert's rows, like MLX's gather_mm_rhs. Runtime p = [S, N, K, T, RS, R0]."""
    assert bm % 16 == 0 and bn % 16 == 0
    tm, tn, rows_per_sg = bm // 16, bn // 16, bn // 4
    ld = 128 + 8  # padded row of a staged tile, in bf16 elements
    return f"""
    const int S = int(p[0]), N = int(p[1]), K = int(p[2]), T = int(p[3]);
    const size_t RS = size_t(p[4]), R0 = size_t(p[5]);
    threadgroup ushort As[{bm * ld}];
    threadgroup ushort Bs[{bn * ld}];
    uint lane = thread_index_in_simdgroup;
    uint sg = simdgroup_index_in_threadgroup;
    uint tid = thread_position_in_threadgroup.x;
    int rb = int(threadgroup_position_in_grid.y) * {bm};
    int cb = int(threadgroup_position_in_grid.x) * {bn};
    if (rb >= S) {{ return; }}
    short qid = lane / 4;
    short fm = (qid & 4) + ((lane / 2) % 4);
    short fn = (qid & 2) * 2 + (lane % 2) * 2;
    int sm = int(sg / 2) * {bm // 2};
    int sn = int(sg % 2) * {bn // 2};
    int rows = min({bm}, S - rb);
    int seg = 0;
    while (seg < rows) {{
      uint e = idx[rb + seg];
      int end = seg + 1;
      while (end < rows && idx[rb + end] == e) {{ end++; }}
      simdgroup_float8x8 C[{tm}][{tn}];
      for (int i = 0; i < {tm}; i++) {{ for (int j = 0; j < {tn}; j++) {{ C[i][j] = simdgroup_float8x8(0); }} }}
      uint epos[{rows_per_sg}];
      for (int r = 0; r < {rows_per_sg}; r++) {{
        int n = cb + int(sg) * {rows_per_sg} + r;
        epos[r] = n < N ? row_esc[size_t(e) * RS + R0 + size_t(n)] : 0u;
      }}
      for (int t = 0; t < T; t++) {{
        int k0 = t * 128;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // stage the block's inputs: {bm} rows x 128, four bf16 per load
        for (int i = int(tid); i < {bm * 32}; i += 128) {{
          int r = i / 32, c = (i % 32) * 4;
          uint2 v = uint2(0u, 0u);
          if (rb + r < S && k0 + c < K) {{ v = *(reinterpret_cast<const device uint2*>(x + size_t(rb + r) * K + k0 + c)); }}
          *(reinterpret_cast<threadgroup uint2*>(As + r * {ld} + c)) = v;
        }}
        // decode this SIMD group's weight rows of tile t
        for (int r = 0; r < {rows_per_sg}; r++) {{
          int nl = int(sg) * {rows_per_sg} + r;
          int n = cb + nl;
          uint u[4] = {{0u, 0u, 0u, 0u}};
          if (n < N) {{
            size_t row = size_t(e) * RS + R0 + size_t(n);
            bool in = k0 + int(lane) * 4 < K;
            uint g = codes[(row * T + t) * 32 + lane];
            uint rw = in ? *(reinterpret_cast<const device uint*>(raw + row * K + k0 + lane * 4)) : 0u;
            uint c4[4]; uint ex[4];
            for (int j = 0; j < 4; j++) {{ c4[j] = (g >> (4u * uint(j))) & 15u; ex[j] = tbl[c4[j]]; }}
            if (flags[row * T + t]) {{
              uint cnt = 0;
              for (int j = 0; j < 4; j++) {{ cnt += c4[j] == 15u ? 1u : 0u; }}
              uint q = epos[r] + simd_prefix_exclusive_sum(cnt);
              for (int j = 0; j < 4; j++) {{ if (c4[j] == 15u) {{ ex[j] = esc[q]; q++; }} }}
              epos[r] += simd_sum(cnt);
            }}
            for (int j = 0; j < 4; j++) {{
              uint rj = (rw >> (8u * uint(j))) & 0xFFu;
              u[j] = in ? (((rj & 0x80u) << 8) | (ex[j] << 7) | (rj & 0x7Fu)) : 0u;
            }}
          }}
          *(reinterpret_cast<threadgroup uint2*>(Bs + nl * {ld} + lane * 4)) = uint2(u[0] | (u[1] << 16), u[2] | (u[3] << 16));
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        int steps = min(128, K - k0) / 8;
        for (int s = 0; s < steps; s++) {{
          int kk = s * 8;
          simdgroup_float8x8 A[{tm}];
          simdgroup_float8x8 B[{tn}];
          for (int i = 0; i < {tm}; i++) {{
            int ar = sm + i * 8 + fm;
            A[i].thread_elements()[0] = static_cast<float>(as_type<bfloat16_t>(As[ar * {ld} + kk + fn]));
            A[i].thread_elements()[1] = static_cast<float>(as_type<bfloat16_t>(As[ar * {ld} + kk + fn + 1]));
          }}
          for (int j = 0; j < {tn}; j++) {{
            int bc = sn + j * 8 + fn;
            B[j].thread_elements()[0] = static_cast<float>(as_type<bfloat16_t>(Bs[bc * {ld} + kk + fm]));
            B[j].thread_elements()[1] = static_cast<float>(as_type<bfloat16_t>(Bs[(bc + 1) * {ld} + kk + fm]));
          }}
          for (int i = 0; i < {tm}; i++) {{ for (int j = 0; j < {tn}; j++) {{ simdgroup_multiply_accumulate(C[i][j], A[i], B[j], C[i][j]); }} }}
        }}
      }}
      for (int i = 0; i < {tm}; i++) {{
        int orow = sm + i * 8 + fm;
        if (orow >= seg && orow < end) {{
          for (int j = 0; j < {tn}; j++) {{
            int n0 = cb + sn + j * 8 + fn;
            if (n0 < N) {{ y[size_t(rb + orow) * N + n0] = static_cast<bfloat16_t>(C[i][j].thread_elements()[0]); }}
            if (n0 + 1 < N) {{ y[size_t(rb + orow) * N + n0 + 1] = static_cast<bfloat16_t>(C[i][j].thread_elements()[1]); }}
          }}
        }}
      }}
      seg = end;
    }}
"""


_R4GEMM = {}


def _kernel_r4_gemm(bm: int, bn: int):
    if (bm, bn) not in _R4GEMM:
        _R4GEMM[bm, bn] = mx.fast.metal_kernel(
            name=f"binade_r4_gemm_{bm}x{bn}",
            input_names=["x", "idx", "raw", "codes", "flags", "esc", "tbl", "row_esc", "p"],
            output_names=["y"],
            source=source_r4_gemm(bm, bn),
        )
    return _R4GEMM[bm, bn]

