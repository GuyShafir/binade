"""Information the row and column index carry about the exponent (see scripts/mutual_info.py)."""

import math

import numpy as np

from . import st
from .bf16 import exponent

CHUNK = 1 << 24


def entropy_mm(counts: np.ndarray) -> np.ndarray:
    """Miller-Madow entropy (bits) of each row of a [G, 256] count matrix."""
    counts = np.atleast_2d(counts).astype(np.float64)
    n = counts.sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = counts / n[:, None]
        h = -np.nansum(np.where(counts > 0, p * np.log2(p), 0.0), axis=1)
    m = (counts > 0).sum(axis=1)
    return h + (m - 1) / (2 * np.maximum(n, 1) * math.log(2))


def tensor_info(info: st.TensorInfo) -> dict:
    R, K = info.rows, info.shape[-1]
    step = max(1, CHUNK // K)
    col = np.zeros((K, 256), np.int64)
    h_row_sum = 0.0
    for r0 in range(0, R, step):
        r1 = min(R, r0 + step)
        e = exponent(st.read(info, (r0, r1))).astype(np.int64)
        col += np.bincount((np.arange(K)[None, :] * 256 + e).ravel(), minlength=K * 256).reshape(K, 256)
        rows = np.bincount((np.arange(r1 - r0)[:, None] * 256 + e).ravel(), minlength=(r1 - r0) * 256).reshape(-1, 256)
        h_row_sum += float((entropy_mm(rows) * K).sum())
    n = R * K
    h = float(entropy_mm(col.sum(axis=0))[0])
    h_col = float((entropy_mm(col) * R).sum() / n)
    h_row = h_row_sum / n
    return {"numel": n, "rows": R, "cols": K, "H": h, "H_given_row": h_row, "H_given_col": h_col}
