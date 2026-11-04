"""Spatiotemporal site graph with orientation-aware couplings (decisions G2, G3, G6; doc 11 §5).

Sites are all pixels of all years in year-major order: ``site = t * H * W + r * W + c``.
Edges:

* **spatial** (8-neighbour, within a year), weight
  w = contrast_ik * align_ik * dist_ik with
  - contrast_ik = exp(-|rgb_i - rgb_k|^2 / (2 s^2)) (s = RMS edge difference of that image);
  - align_ik = mean over both endpoints of (1 - coh) + coh * cos^2(angle(edge) - line_dir),
    with line_dir and coherence coh from the luminance structure tensor. In coherent linear
    structures, couplings ALONG the line stay strong and couplings ACROSS it vanish, so the
    smoothness prior connects thin gullies instead of eroding them. ``orientation=False``
    gives isotropic weights (align = 1) for comparison;
  - dist_ik = 1 for axial and 1/sqrt(2) for diagonal neighbours.
* **temporal**: the same pixel in consecutive years, weight ``tau``. Evidence that a gully is
  visible in one year raises its probability in neighbouring years without forcing it. EGs
  are ephemeral (decision G1).

All weights are >= 0, so with symmetric LFs the posterior is ferromagnetic and MAP is exact by
graph cut.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter

from ..graph import Sites
from .lfs import luminance

_OFFSETS = [(0, 1), (1, 0), (1, 1), (1, -1)]  # (dr, dc): right, down, down-right, down-left


def orientation_field(img: np.ndarray, sigma: float = 1.0, rho: float = 2.5) -> tuple[np.ndarray, np.ndarray]:
    """Line direction (radians, in (col, row) coordinates) and coherence in [0, 1] per pixel."""
    lum = gaussian_filter(luminance(img), sigma)
    gy, gx = np.gradient(lum)  # gy along rows, gx along columns
    jxx, jyy, jxy = gaussian_filter(gx * gx, rho), gaussian_filter(gy * gy, rho), gaussian_filter(gx * gy, rho)
    grad_dir = 0.5 * np.arctan2(2 * jxy, jxx - jyy)
    tr = jxx + jyy
    coh = np.where(tr > 1e-12, ((np.sqrt((jxx - jyy) ** 2 + 4 * jxy**2)) / (tr + 1e-12)) ** 2, 0.0)
    return grad_dir + np.pi / 2, np.clip(coh, 0.0, 1.0)


_STRUCTURE_CACHE: dict[tuple[int, int, int], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}


def _structure(T: int, H: int, W: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Edge list, label map and centroids are identical for every location of a given shape:
    built once and shared (read-only) to keep memory flat across hundreds of locations."""
    key = (T, H, W)
    if key not in _STRUCTURE_CACHE:
        idx = np.arange(T * H * W).reshape(T, H, W)
        e = []
        for t in range(T):
            for dr, dc in _OFFSETS:
                r0, r1 = max(0, -dr), H - max(0, dr)
                c0, c1 = max(0, -dc), W - max(0, dc)
                e.append(np.stack([idx[t, r0:r1, c0:c1].reshape(-1), idx[t, r0 + dr:r1 + dr, c0 + dc:c1 + dc].reshape(-1)], 1))
        for t in range(T - 1):
            e.append(np.stack([idx[t].reshape(-1), idx[t + 1].reshape(-1)], 1))
        edges = np.sort(np.concatenate(e), axis=1).astype(np.int32)
        rr, cc = np.indices((T * H, W))
        cent = np.stack([rr.reshape(-1), cc.reshape(-1)], 1).astype(np.float32)
        for a in (edges, cent, idx):
            a.setflags(write=False)
        _STRUCTURE_CACHE[key] = (edges, idx.reshape(T * H, W), cent)
    return _STRUCTURE_CACHE[key]


def spatiotemporal_sites(images: np.ndarray, tau: float = 0.5, orientation: bool = True,
                         sigma: float | None = None) -> Sites:
    """Sites object for a (T, H, W, 3) stack. ``Sites.labels`` stacks the years vertically
    as a (T*H, W) map, so ``to_pixels`` returns the T masks stacked vertically."""
    T, H, W = images.shape[:3]
    w_list = []
    for t in range(T):
        img = images[t]
        line_dir, coh = orientation_field(img) if orientation else (None, None)
        diffs = []
        pairs = []
        for dr, dc in _OFFSETS:
            r0, r1 = max(0, -dr), H - max(0, dr)
            c0, c1 = max(0, -dc), W - max(0, dc)
            a = (slice(r0, r1), slice(c0, c1))
            b = (slice(r0 + dr, r1 + dr), slice(c0 + dc, c1 + dc))
            d2 = np.sum((img[a] - img[b]) ** 2, axis=-1)
            diffs.append(d2.reshape(-1))
            pairs.append((a, b, dr, dc))
        s2 = np.mean(np.concatenate(diffs)) or 1.0
        s2 = s2 if sigma is None else sigma**2
        for (a, b, dr, dc), d2 in zip(pairs, diffs):
            w = np.exp(-d2 / (2 * s2))
            if orientation:
                ang = np.arctan2(dr, dc)
                al_a = (1 - coh[a]) + coh[a] * np.cos(ang - line_dir[a]) ** 2
                al_b = (1 - coh[b]) + coh[b] * np.cos(ang - line_dir[b]) ** 2
                w = w * (0.5 * (al_a + al_b)).reshape(-1)
            if dr and dc:
                w = w / np.sqrt(2)
            w_list.append(w)
    for t in range(T - 1):
        w_list.append(np.full(H * W, tau))
    edges, labels, cent = _structure(T, H, W)
    assert edges.shape[0] == sum(len(w) for w in w_list)
    return Sites(labels, edges, np.concatenate(w_list).astype(np.float32), images.reshape(-1, images.shape[-1]), cent)
