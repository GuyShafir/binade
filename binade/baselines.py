"""Reference exponent codes to compare against (DESIGN.md 4). All return bits per exponent."""

import heapq

import numpy as np


def entropy(hist: np.ndarray) -> float:
    """Shannon entropy (bits) of a histogram."""
    p = hist[hist > 0] / hist.sum()
    return float(-(p * np.log2(p)).sum())


def huffman_lengths(hist: np.ndarray) -> np.ndarray:
    """Huffman code length per symbol (0 for absent symbols; 1 if only one symbol)."""
    lengths = np.zeros(len(hist), np.int64)
    syms = [int(s) for s in np.flatnonzero(hist)]
    if len(syms) == 1:
        lengths[syms[0]] = 1
        return lengths
    heap = [(int(hist[s]), s, [s]) for s in syms]
    heapq.heapify(heap)
    while len(heap) > 1:
        c1, t1, s1 = heapq.heappop(heap)
        c2, t2, s2 = heapq.heappop(heap)
        for s in s1 + s2:
            lengths[s] += 1
        heapq.heappush(heap, (c1 + c2, min(t1, t2), s1 + s2))
    return lengths


def huffman_bits(hist: np.ndarray, lengths: np.ndarray | None = None) -> int:
    """Total bits to code `hist` with a static Huffman code (its own unless `lengths` given)."""
    if lengths is None:
        lengths = huffman_lengths(hist)
    return int((hist.astype(np.int64) * lengths).sum())


def tensor_palette_bits(hist: np.ndarray) -> int:
    """One palette per tensor, b = ceil(log2(distinct)); palette entries at 8 bits each."""
    d = int((hist > 0).sum())
    b = int(np.ceil(np.log2(d))) if d > 1 else 0
    return int(hist.sum()) * b + 8 * d


def palette16_segment_bits(in_weights: int, numel: int, segments: int) -> int:
    """Unweight-class: per-tensor palette of the 16 most frequent exponents, 4-bit index.
    A 64-weight row segment holding any other exponent keeps its 8-bit exponents.
    Charges the 16-byte palette and one flag bit per segment."""
    return 4 * in_weights + 8 * (numel - in_weights) + segments + 16 * 8


def window7_bits(hist: np.ndarray) -> int:
    """ZipServ-class: 7 contiguous exponents per tensor at a fixed 3-bit code (code 000 is
    the outlier escape); an outlier is also stored as a full 16-bit BF16 value."""
    covered = max(int(hist[b : b + 7].sum()) for b in range(0, 250))
    n = int(hist.sum())
    return 3 * n + 16 * (n - covered)
