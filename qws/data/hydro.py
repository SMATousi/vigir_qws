"""Minimal DEM hydrology: depression filling, D8 flow directions, flow accumulation.

Implemented in-house with numba (decision G5): the available ``pysheds`` release is
incompatible with NumPy 2.x. The algorithms are standard:

* **Priority-flood with epsilon** (Barnes, Lehman & Mulla 2014): cells are processed from
  the lowest border inwards with a min-heap. Each unvisited neighbour is raised to at least
  its parent's elevation + epsilon, so every cell drains to the border.
* **D8**: each cell drains to the neighbour with the steepest positive descent
  (drop / distance, diagonals at sqrt(2)).
* **Accumulation**: cells are visited in decreasing filled elevation, and each passes its
  accumulated area to its receiver. Area is counted in cells (1 m^2 on the 1 m DEM),
  including the cell itself.
"""

from __future__ import annotations

import heapq

import numpy as np

try:
    from numba import njit
except ImportError:  # pragma: no cover - numba is part of the 'gully' extra
    def njit(*a, **k):
        return (lambda f: f) if not a or not callable(a[0]) else a[0]

_DR = np.array([-1, -1, -1, 0, 0, 1, 1, 1])
_DC = np.array([-1, 0, 1, -1, 1, -1, 0, 1])
_DIST = np.array([np.sqrt(2), 1, np.sqrt(2), 1, 1, np.sqrt(2), 1, np.sqrt(2)])


@njit(cache=True)
def _priority_flood(z, eps):
    H, W = z.shape
    out = z.copy()
    seen = np.zeros((H, W), dtype=np.bool_)
    heap = [(0.0, 0, 0)]
    heap.pop()
    for r in range(H):
        for c in range(W):
            if r == 0 or c == 0 or r == H - 1 or c == W - 1:
                heapq.heappush(heap, (out[r, c], r, c))
                seen[r, c] = True
    while len(heap) > 0:
        e, r, c = heapq.heappop(heap)
        for k in range(8):
            rr = r + _DR[k]
            cc = c + _DC[k]
            if rr < 0 or cc < 0 or rr >= H or cc >= W or seen[rr, cc]:
                continue
            seen[rr, cc] = True
            if out[rr, cc] <= e:
                out[rr, cc] = e + eps
            heapq.heappush(heap, (out[rr, cc], rr, cc))
    return out


@njit(cache=True)
def _d8(z):
    H, W = z.shape
    rec = -np.ones((H, W), dtype=np.int64)  # flat index of receiver, -1 = outlet
    for r in range(H):
        for c in range(W):
            best, bk = 0.0, -1
            for k in range(8):
                rr = r + _DR[k]
                cc = c + _DC[k]
                if rr < 0 or cc < 0 or rr >= H or cc >= W:
                    continue
                s = (z[r, c] - z[rr, cc]) / _DIST[k]
                if s > best:
                    best, bk = s, k
            if bk >= 0:
                rec[r, c] = (r + _DR[bk]) * W + (c + _DC[bk])
    return rec


@njit(cache=True)
def _accumulate(order, rec_flat, n):
    acc = np.ones(n)
    for i in order:
        j = rec_flat[i]
        if j >= 0:
            acc[j] += acc[i]
    return acc


def fill_depressions(z: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    return _priority_flood(np.ascontiguousarray(z, dtype=np.float64), eps)


def d8_receivers(z: np.ndarray) -> np.ndarray:
    return _d8(np.ascontiguousarray(z, dtype=np.float64))


def flow_accumulation(z: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    """Upslope contributing area (in cells, including the cell) after depression filling."""
    filled = fill_depressions(z, eps)
    rec = d8_receivers(filled).reshape(-1)
    order = np.argsort(-filled.reshape(-1), kind="stable")
    return _accumulate(order, rec, filled.size).reshape(z.shape)
