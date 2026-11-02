"""Site construction: pixel grids and superpixel region-adjacency graphs (doc 01 §2, doc 04 §5.1)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Sites:
    """Sites of one image and their adjacency graph.

    Attributes:
        labels: ``(H, W)`` int map from pixel to site id in ``0..n-1``.
        edges: ``(m, 2)`` unordered site pairs with ``i < k``.
        weights: Edge affinities ``w_ik`` in ``(0, 1]``.
        features: Mean image value per site, ``(n, C)``.
        centroids: Mean pixel coordinate (row, col) per site, ``(n, 2)``.
    """

    labels: np.ndarray
    edges: np.ndarray
    weights: np.ndarray
    features: np.ndarray
    centroids: np.ndarray

    @property
    def n(self) -> int:
        return int(self.features.shape[0])

    @property
    def m(self) -> int:
        return int(self.edges.shape[0])

    @property
    def shape(self) -> tuple[int, int]:
        return tuple(self.labels.shape)  # type: ignore[return-value]

    def to_pixels(self, values: np.ndarray) -> np.ndarray:
        """Broadcast per-site values to an ``(H, W)`` map."""
        return np.asarray(values)[self.labels]

    def from_pixels(self, pixel_values: np.ndarray) -> np.ndarray:
        """Average an ``(H, W)`` map over each site."""
        flat = self.labels.reshape(-1)
        sums = np.bincount(flat, weights=np.asarray(pixel_values, dtype=float).reshape(-1), minlength=self.n)
        counts = np.bincount(flat, minlength=self.n)
        return sums / np.maximum(counts, 1)


def _as_channels(image: np.ndarray) -> np.ndarray:
    img = np.asarray(image, dtype=float)
    return img[..., None] if img.ndim == 2 else img


def _site_stats(img: np.ndarray, labels: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    flat = labels.reshape(-1)
    counts = np.maximum(np.bincount(flat, minlength=n), 1)
    feats = np.stack([np.bincount(flat, weights=img[..., c].reshape(-1), minlength=n) / counts
                      for c in range(img.shape[-1])], axis=1)
    rr, cc = np.indices(labels.shape)
    cent = np.stack([np.bincount(flat, weights=rr.reshape(-1), minlength=n) / counts,
                     np.bincount(flat, weights=cc.reshape(-1), minlength=n) / counts], axis=1)
    return feats, cent


def contrast_weights(features: np.ndarray, edges: np.ndarray, sigma: float | None = None) -> np.ndarray:
    """w_ik = exp(-||f_i - f_k||^2 / (2 sigma^2)); sigma defaults to the RMS edge difference.

    ``sigma = inf`` gives uniform weights.
    """
    if edges.size == 0:
        return np.zeros(0)
    d2 = np.sum((features[edges[:, 0]] - features[edges[:, 1]]) ** 2, axis=1)
    if sigma is None:
        sigma = float(np.sqrt(np.mean(d2))) if np.mean(d2) > 0 else 1.0
    if np.isinf(sigma):
        return np.ones_like(d2)
    return np.exp(-d2 / (2.0 * sigma**2))


def grid_sites(image: np.ndarray, connectivity: int = 4, sigma: float | None = None) -> Sites:
    """One site per pixel, 4- or 8-neighbour grid."""
    img = _as_channels(image)
    H, W = img.shape[:2]
    labels = np.arange(H * W).reshape(H, W)
    pairs = [(labels[:, :-1], labels[:, 1:]), (labels[:-1, :], labels[1:, :])]
    if connectivity == 8:
        pairs += [(labels[:-1, :-1], labels[1:, 1:]), (labels[:-1, 1:], labels[1:, :-1])]
    elif connectivity != 4:
        raise ValueError("connectivity must be 4 or 8")
    edges = np.concatenate([np.stack([a.reshape(-1), b.reshape(-1)], axis=1) for a, b in pairs])
    edges = np.sort(edges, axis=1)
    feats = img.reshape(H * W, -1)
    rr, cc = np.indices((H, W))
    cent = np.stack([rr.reshape(-1), cc.reshape(-1)], axis=1).astype(float)
    return Sites(labels, edges, contrast_weights(feats, edges, sigma), feats, cent)


def sites_from_labels(image: np.ndarray, labels: np.ndarray, sigma: float | None = None) -> Sites:
    """Region-adjacency graph of an arbitrary label map (4-connectivity between pixels)."""
    img = _as_channels(image)
    _, labels = np.unique(np.asarray(labels), return_inverse=True)
    labels = labels.reshape(img.shape[:2])
    n = int(labels.max()) + 1
    a = np.concatenate([labels[:, :-1].reshape(-1), labels[:-1, :].reshape(-1)])
    b = np.concatenate([labels[:, 1:].reshape(-1), labels[1:, :].reshape(-1)])
    diff = a != b
    lo, hi = np.minimum(a[diff], b[diff]), np.maximum(a[diff], b[diff])
    keys = np.unique(lo.astype(np.int64) * n + hi)
    edges = np.stack([keys // n, keys % n], axis=1)
    feats, cent = _site_stats(img, labels, n)
    return Sites(labels, edges, contrast_weights(feats, edges, sigma), feats, cent)


def superpixel_sites(image: np.ndarray, n_segments: int = 500, compactness: float = 10.0,
                     sigma: float | None = None) -> Sites:
    """SLIC superpixels (scikit-image) and their region-adjacency graph."""
    from skimage.segmentation import slic

    img = np.asarray(image, dtype=float)
    channel_axis = -1 if img.ndim == 3 else None
    labels = slic(img, n_segments=n_segments, compactness=compactness, channel_axis=channel_axis,
                  start_label=0)
    return sites_from_labels(img, labels, sigma)
