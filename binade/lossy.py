"""Lossy dial, report only (DESIGN.md 11).

For m kept mantissa bits (round half to even, exponent exact), per tile:
  rel_fro  ||W - Q||_F / ||W||_F
  max_rel  max_i |w_i - q_i| / |w_i| over nonzero finite w_i
Per-tile values are streamed into log-spaced histograms so whole-model percentiles
need no per-tile storage.
"""

import mlx.core as mx
import numpy as np

from .bf16 import round_mantissa
from .tile import tiles

MANTISSA_BITS = (2, 3, 4, 5, 6, 7)
BINS_PER_DECADE = 200
LOG10_MIN = -8  # bin 0 collects everything below 1e-8, including exact zeros
NBINS = -LOG10_MIN * BINS_PER_DECADE + 1
METRICS = ("rel_fro", "max_rel")


def elementwise(w16: mx.array, m: int) -> tuple[mx.array, mx.array]:
    """(|w - q|, |w|) as float32 for finite weights, zeros elsewhere. w16: uint16 BF16 bits."""
    finite = ((w16 >> 7) & 0xFF) != 0xFF
    w = w16.view(mx.bfloat16).astype(mx.float32)
    q = round_mantissa(w16, m).view(mx.bfloat16).astype(mx.float32)
    d = mx.where(finite, mx.abs(w - q), 0.0)
    a = mx.where(finite, mx.abs(w), 0.0)
    return d, a


def tile_errors(d: mx.array, a: mx.array, T: int) -> tuple[mx.array, mx.array]:
    """Per-tile (rel_fro, max_rel) from [R, K] error and magnitude arrays. NaN for all-zero tiles."""
    out_fro, out_max = [], []
    for (dt, at) in zip(tiles(d, T), tiles(a, T)):
        if dt is None:
            continue
        num = (dt * dt).sum(axis=1)
        den = (at * at).sum(axis=1)
        out_fro.append(mx.where(den > 0, mx.sqrt(num / den), mx.nan))
        rel = mx.where(at > 0, dt / mx.where(at > 0, at, 1.0), 0.0)
        out_max.append(mx.where(at.max(axis=1) > 0, rel.max(axis=1), mx.nan))
    return mx.concatenate(out_fro), mx.concatenate(out_max)


def histogram(x: mx.array) -> np.ndarray:
    """Log-spaced histogram of non-NaN values (int64 [NBINS])."""
    x = np.asarray(x)
    x = x[~np.isnan(x)]
    with np.errstate(divide="ignore"):
        b = np.floor((np.log10(x) - LOG10_MIN) * BINS_PER_DECADE)
    b = np.clip(np.nan_to_num(b, neginf=0), 0, NBINS - 1).astype(np.int64)
    return np.bincount(b, minlength=NBINS)


def bin_upper(i: int) -> float:
    """Upper edge of bin i (a conservative value for a percentile landing in it)."""
    return 0.0 if i == 0 else 10 ** (LOG10_MIN + (i + 1) / BINS_PER_DECADE)


def percentiles(h: np.ndarray, qs=(50, 90, 99, 99.9)) -> dict:
    """Percentiles from a histogram, reported as bin upper edges."""
    total = int(h.sum())
    if total == 0:
        return {}
    cdf = np.cumsum(h)
    return {f"p{q:g}": bin_upper(int(np.searchsorted(cdf, q / 100 * total))) for q in qs}
