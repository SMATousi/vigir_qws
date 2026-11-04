"""Generic example labeling functions. Domain-specific LFs are out of scope (doc 00 §4)."""

from __future__ import annotations

import numpy as np

from ..graph import Sites
from .base import PairwiseLF, PixelLF


def _intensity(image: np.ndarray, channel: int | None) -> np.ndarray:
    img = np.asarray(image, dtype=float)
    if img.ndim == 2:
        return img
    return img[..., channel] if channel is not None else img.mean(axis=-1)


class ThresholdLF(PixelLF):
    """+1 above ``high``, -1 below ``low``, abstain in between (or the reverse if ``invert``)."""

    def __init__(self, low: float, high: float, channel: int | None = None, invert: bool = False,
                 name: str | None = None, min_coverage: float = 0.5):
        self.low, self.high, self.channel, self.invert = low, high, channel, invert

        def fn(image: np.ndarray) -> np.ndarray:
            x = _intensity(image, self.channel)
            v = np.where(x > self.high, 1, np.where(x < self.low, -1, 0))
            return -v if self.invert else v

        super().__init__(fn, name or f"threshold[{low},{high}]", min_coverage)


class BoxLF(PixelLF):
    """Weak box prior: +1 (or abstain) inside the box, -1 beyond ``margin`` pixels outside it."""

    def __init__(self, box: tuple[int, int, int, int], margin: int = 0, inside_vote: int = 0,
                 name: str = "box", min_coverage: float = 0.5):
        r0, c0, r1, c1 = box

        def fn(image: np.ndarray) -> np.ndarray:
            H, W = np.asarray(image).shape[:2]
            v = -np.ones((H, W), dtype=np.int8)
            v[max(r0 - margin, 0):r1 + margin, max(c0 - margin, 0):c1 + margin] = 0
            v[r0:r1, c0:c1] = inside_vote
            return v

        super().__init__(fn, name, min_coverage)


class ScribbleLF(PixelLF):
    """User scribbles: a fixed ``(H, W)`` map in {-1, 0, +1}."""

    def __init__(self, scribbles: np.ndarray, name: str = "scribble", min_coverage: float = 0.01):
        s = np.asarray(scribbles, dtype=np.int8)
        super().__init__(lambda image: s, name, min_coverage)


class EdgeContrastLF(PairwiseLF):
    """Pairwise LF on site features: -1 ("different") if the feature distance exceeds
    ``differ``, +1 ("same") if it is below ``same``, abstain otherwise."""

    def __init__(self, same: float, differ: float, name: str = "edge_contrast"):
        if same > differ:
            raise ValueError("need same <= differ")
        self.same, self.differ, self.name = same, differ, name

    def votes(self, image: np.ndarray, sites: Sites) -> np.ndarray:
        e = sites.edges
        d = np.linalg.norm(sites.features[e[:, 0]] - sites.features[e[:, 1]], axis=1)
        return np.where(d > self.differ, -1, np.where(d < self.same, 1, 0)).astype(np.int8)
