"""Packed mixture-of-experts weights (Gemma 4 layout) for the R4 runtime.

Gemma 4 checkpoints store each layer's experts as experts.gate_up_proj [E, 2n, D] (gate
rows 0..n-1, up rows n..2n-1 of every expert) and experts.down_proj [E, D, n]; mlx-lm's
sanitize splits gate_up_proj into the gate_proj and up_proj of a SwitchGLU. A packed
model keeps both tensors whole: BinadeSwitchGLU holds their file arrays in place of the
SwitchGLU (keys experts.switch_glu.gate_up_proj.* and .down_proj.*), and
R4SwitchGLU runs them.

R4SwitchGLU mirrors SwitchGLU.__call__. Without sorting (fewer than 64 expert slots, so
every decode step) mlx-lm's gather_mm multiplies one input row by one expert per slot,
MLX's gemv_gather; the R4 gather kernel reproduces it bit for bit, gate and up rows in
one launch. With sorting (prompt processing) gather_mm is a GEMM over slots sorted by
expert (gather_mm_rhs); the R4 gather GEMM reproduces it bit for bit, reading R4 directly,
so no BF16 copy of the experts is ever built. K must be a multiple of 64 there (MLX's
tile loop runs a K remainder first otherwise); other K rebuild BF16 experts instead.
"""

from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn

from .slow_linear import _placeholders

ROLES = {"gate_up_proj": "gate_up_proj", "down_proj": "down_proj"}


class BinadeSwitchGLU(nn.Module):
    """File arrays of one layer's experts; runs only after conversion to R4SwitchGLU."""

    def __init__(self, gate_up_spec: dict, down_spec: dict, activation):
        super().__init__()
        self.gate_up_proj = _placeholders(gate_up_spec)
        self.down_proj = _placeholders(down_spec)
        self._activation = activation

    def __call__(self, x, indices):
        raise RuntimeError("packed experts run on the R4 runtime (binade.mlx.r4.load)")


class R4SwitchGLU(nn.Module):
    def __init__(self, gate_up, down, activation):
        """gate_up, down: R4 modules (binade.mlx.r4._R4) over [E * 2n, D] and [E * D, n] rows."""
        super().__init__()
        self._gu = gate_up
        self._down = down
        self._activation = activation
        self.E = gate_up._a["shape"][0]
        self.n = gate_up._a["shape"][1] // 2
        self.D = down._a["shape"][1]

    def _dense(self):
        """BF16 gate, up [E, n, D] and down [E, D, n]: the arrays sanitize makes, each decoded
        straight into place (no fused [E, 2n, D] copy to split)."""
        n, E = self.n, self.E
        gate = self._gu.dense(n, 2 * n, 0).reshape(E, n, self.D)
        up = self._gu.dense(n, 2 * n, n).reshape(E, n, self.D)
        return gate, up, self._down.dense()

    def __call__(self, x: mx.array, indices: mx.array) -> mx.array:
        from mlx_lm.models.switch_layers import _gather_sort, _scatter_unsort

        do_sort = indices.size >= 64
        if not do_sort:
            k = indices.shape[-1]
            G = indices.size
            ei = indices.reshape(-1).astype(mx.uint32)
            xi = mx.arange(G, dtype=mx.uint32) // k
            h = self._gu._gather(x.reshape(-1, self.D), xi, ei, 2 * self.n, 2 * self.n)  # [G, 2n]
            h = self._activation(h[:, self.n :], h[:, : self.n])
            y = self._down._gather(h, mx.arange(G, dtype=mx.uint32), ei, self.D, self.D)
            return y.reshape(*indices.shape, self.D)
        x = mx.expand_dims(x, (-2, -3))
        x, idx, inv_order = _gather_sort(x, indices)
        from .cuda_kernels import available as cuda

        if not cuda() and self._gu.K % 64 == 0 and self._down.K % 64 == 0:  # the R4 GEMM mirrors MLX's Metal GEMM
            S = idx.size
            h = self._gu._gather_mm(x, idx, 2 * self.n, 2 * self.n)  # [S, 2n]: gate, then up
            h = self._activation(h[:, self.n :], h[:, : self.n])
            y = self._down._gather_mm(h, idx, self.D, self.D).reshape(S, 1, self.D)
            return _scatter_unsort(y, inv_order, indices.shape).squeeze(-2)
        gate, up, down = self._dense()
        x_up = mx.gather_mm(x, up.swapaxes(-1, -2), rhs_indices=idx, sorted_indices=True)
        x_gate = mx.gather_mm(x, gate.swapaxes(-1, -2), rhs_indices=idx, sorted_indices=True)
        x = mx.gather_mm(self._activation(x_up, x_gate), down.swapaxes(-1, -2), rhs_indices=idx, sorted_indices=True)
        x = _scatter_unsort(x, inv_order, indices.shape)
        y = x.squeeze(-2)
        mx.eval(y)
        return y


def holders(m: BinadeSwitchGLU):
    """(gate_up, down) objects shaped like packed linear modules, for r4.transcode."""
    return SimpleNamespace(weight=m.gate_up_proj, _plan=None), SimpleNamespace(weight=m.down_proj, _plan=None)
