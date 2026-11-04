"""Weak labeling functions for ephemeral-gully (EG) segmentation (spec doc 11 §4).

Every LF votes per pixel and per year: ``votes(loc, terrain) -> (6, H, W)`` int8 in {-1, 0, +1}
(+1 EG, -1 not EG, 0 abstain). None of them uses any label. They encode:

* **Terrain** (USGS 3DEP 1 m DEM; decision G4): gullies form along concentrated-flow paths.
  - ``FlowPathLF``: a large upslope area votes +1; a tiny one votes -1.
  - ``ConvergenceLF``: valley-shaped (convergent) terrain votes +1; ridges vote -1.
* **Imagery** (NAIP RGB, per year):
  - ``ThinLineLF``: thin curvilinear dark or bright structures (multi-scale Sato tubeness)
    vote +1; featureless pixels vote -1.
  - ``StraightLineLF``: long straight segments (roads, field edges, ditches, crop rows) vote
    -1. EGs are sinuous.
  - ``NonCropLF``: bright low-saturation pixels (roads, buildings) and dark, highly textured
    pixels (trees) vote -1.
  - ``PersistentVegetationLF``: pixels that stay green through bare-soil years (field-edge
    strips, riparian corridors, grassed waterways) vote -1.
  - ``PermanentLineLF``: thin lines present at the same place in at least 5 of 6 years
    (permanent features, not *ephemeral* gullies) vote -1.
* **Physics consistency**: ``LineOnFlowLF``. A thin line on a flow path votes +1, which is
  high precision. A thin line far from any flow path votes -1 (tillage marks, crop rows).

Terrain is a single DEM snapshot, so terrain LFs vote identically in all years. Whether a
gully was *visible* in a given year is decided by the imagery LFs and the temporal coupling
(doc 11 §5).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from scipy.ndimage import binary_dilation, distance_transform_edt, gaussian_filter, uniform_filter

from ..data.dem3dep import Terrain
from ..data.gully import GullyLocation


class GullyLF(ABC):
    name = "gully_lf"
    needs_terrain = False

    @abstractmethod
    def votes(self, loc: GullyLocation, terrain: Terrain | None) -> np.ndarray:
        """(years, H, W) int8 votes."""

    def _abstain(self, loc: GullyLocation) -> np.ndarray:
        return np.zeros(loc.images.shape[:3], dtype=np.int8)


def luminance(img: np.ndarray) -> np.ndarray:
    return img[..., :3] @ np.array([0.299, 0.587, 0.114])


def tubeness(img: np.ndarray, sigmas=(1.0, 1.5, 2.0)) -> np.ndarray:
    """Max of dark- and bright-ridge Sato tubeness on luminance, normalized per image to [0, 1]."""
    from skimage.filters import sato

    lum = luminance(img)
    r = np.maximum(sato(lum, sigmas=sigmas, black_ridges=True), sato(lum, sigmas=sigmas, black_ridges=False))
    return r / (r.max() + 1e-12)


def tubeness_stack(loc: GullyLocation) -> np.ndarray:
    """Per-year tubeness, computed once per location and cached in ``loc.meta``."""
    if "tubeness" not in loc.meta:
        loc.meta["tubeness"] = np.stack([tubeness(img) for img in loc.images])
    return loc.meta["tubeness"]


def _repeat_years(mask_votes: np.ndarray, years: int) -> np.ndarray:
    return np.repeat(mask_votes[None].astype(np.int8), years, axis=0)


class FlowPathLF(GullyLF):
    """+1 where upslope area >= 10**pos (default 1,000 m^2), dilated by ``dilate`` px (gullies are wider
    than the 1-px D8 path); -1 where it is <= 10**neg (default 100 m^2) within ``clear`` px."""

    name, needs_terrain = "flow_path", True

    def __init__(self, pos: float = 3.0, neg: float = 2.0, dilate: int = 1, clear: int = 3):
        self.pos, self.neg, self.dilate, self.clear = pos, neg, dilate, clear

    def votes(self, loc, terrain):
        if terrain is None:
            return self._abstain(loc)
        la = np.nan_to_num(terrain.log_acc, nan=0.0)
        on = binary_dilation(la >= self.pos, iterations=self.dilate) if self.dilate else la >= self.pos
        near_any = binary_dilation(la > self.neg, iterations=self.clear)
        v = np.where(on, 1, np.where(~near_any, -1, 0))
        v[np.isnan(terrain.log_acc)] = 0
        return _repeat_years(v, loc.images.shape[0])


class ConvergenceLF(GullyLF):
    """+1 in the top ``q_pos`` quantile of terrain convergence (valleys), -1 below the
    ``q_neg`` quantile (ridges and planar divergent slopes)."""

    name, needs_terrain = "convergence", True

    def __init__(self, q_pos: float = 0.92, q_neg: float = 0.5):
        self.q_pos, self.q_neg = q_pos, q_neg

    def votes(self, loc, terrain):
        if terrain is None or np.isnan(terrain.convergence).all():
            return self._abstain(loc)
        c = terrain.convergence
        hi, lo = np.nanquantile(c, self.q_pos), np.nanquantile(c, self.q_neg)
        v = np.where(c >= hi, 1, np.where(c <= lo, -1, 0))
        v[np.isnan(c)] = 0
        return _repeat_years(v, loc.images.shape[0])


class ThinLineLF(GullyLF):
    """Per year: +1 for the top ``q_pos`` fraction of tubeness, -1 below the ``q_neg`` quantile."""

    name = "thin_line"

    def __init__(self, q_pos: float = 0.97, q_neg: float = 0.6):
        self.q_pos, self.q_neg = q_pos, q_neg

    def votes(self, loc, terrain):
        out = np.zeros(loc.images.shape[:3], dtype=np.int8)
        for t, r in enumerate(tubeness_stack(loc)):
            hi, lo = np.quantile(r, self.q_pos), np.quantile(r, self.q_neg)
            out[t] = np.where(r >= hi, 1, np.where(r <= lo, -1, 0))
        return out


def straight_line_mask(img: np.ndarray, min_length: int = 40, gap: int = 4, width: int = 2, seed: int = 0
                       ) -> np.ndarray:
    """Pixels within ``width`` px of long straight segments (probabilistic Hough on Canny edges)."""
    from skimage.draw import line as draw_line
    from skimage.feature import canny
    from skimage.transform import probabilistic_hough_line

    edges = canny(luminance(img), sigma=1.2)
    segs = probabilistic_hough_line(edges, threshold=8, line_length=min_length, line_gap=gap, rng=seed)
    m = np.zeros(img.shape[:2], dtype=bool)
    for (c0, r0), (c1, r1) in segs:
        rr, cc = draw_line(r0, c0, r1, c1)
        m[rr, cc] = True
    return binary_dilation(m, iterations=width) if m.any() else m


class StraightLineLF(GullyLF):
    """-1 on and around long straight segments (roads, field edges, ditches, crop rows)."""

    name = "straight_line"

    def __init__(self, min_length: int = 40, width: int = 2):
        self.min_length, self.width = min_length, width

    def votes(self, loc, terrain):
        out = np.zeros(loc.images.shape[:3], dtype=np.int8)
        for t, img in enumerate(loc.images):
            out[t] = np.where(straight_line_mask(img, self.min_length, width=self.width, seed=t), -1, 0)
        return out


class NonCropLF(GullyLF):
    """-1 for roads/buildings (bright, unsaturated) and trees (dark green, highly textured)."""

    name = "non_crop"

    def __init__(self, bright: float = 0.62, sat: float = 0.12, dark: float = 0.22, texture: float = 0.045):
        self.bright, self.sat, self.dark, self.texture = bright, sat, dark, texture

    def votes(self, loc, terrain):
        from skimage.color import rgb2hsv

        out = np.zeros(loc.images.shape[:3], dtype=np.int8)
        for t, img in enumerate(loc.images):
            hsv = rgb2hsv(img)
            lum = luminance(img)
            local_std = np.sqrt(np.maximum(uniform_filter(lum**2, 5) - uniform_filter(lum, 5) ** 2, 0))
            road = (hsv[..., 2] > self.bright) & (hsv[..., 1] < self.sat)
            tree = (lum < self.dark) & (local_std > self.texture)
            out[t] = np.where(binary_dilation(road | tree, iterations=1), -1, 0)
        return out


def excess_green(img: np.ndarray) -> np.ndarray:
    """ExG = 2g - r - b on chromaticity-normalized RGB (robust to illumination)."""
    tot = img[..., :3].sum(-1) + 1e-6
    r, g, b = (img[..., k] / tot for k in range(3))
    return 2 * g - r - b


class PersistentVegetationLF(GullyLF):
    """-1 for pixels that are vegetated in nearly every year while the field as a whole is not.

    Permanent vegetation along field edges, riparian corridors and grassed waterways often
    follows flow paths. Unlike an ephemeral gully in a cropped field, it stays green through
    bare-soil years. A pixel votes -1 in all years if its excess green is above the per-year
    tile median + ``margin`` in at least ``min_years`` years, and if at least one year has a
    low tile median (bare field), so that the persistence is informative.
    """

    name = "persistent_vegetation"

    def __init__(self, margin: float = 0.03, min_years: int = 5, bare_median: float = 0.05, dilate: int = 1):
        self.margin, self.min_years, self.bare_median, self.dilate = margin, min_years, bare_median, dilate

    def votes(self, loc, terrain):
        exg = np.stack([gaussian_filter(excess_green(img), 1.0) for img in loc.images])
        med = np.median(exg.reshape(exg.shape[0], -1), axis=1)
        if med.min() > self.bare_median:  # always-green tile: persistence carries no information
            return self._abstain(loc)
        persistent = ((exg > (med[:, None, None] + self.margin)).sum(axis=0) >= self.min_years)
        if self.dilate:
            persistent = binary_dilation(persistent, iterations=self.dilate)
        return _repeat_years(np.where(persistent, -1, 0), loc.images.shape[0])


class PermanentLineLF(GullyLF):
    """-1 for linear features present at the same place in (almost) every year.

    EGs are *ephemeral*: tillage erases them, and they reappear after storms, so a given
    gully is visible in only some of the 6 acquisitions. A thin line visible at the same
    place (within ``dilate`` px) in at least ``min_years`` years is a permanent feature
    (field boundary, ditch, road edge, tree line), not an EG, and votes -1 in all years.
    """

    name = "permanent_line"

    def __init__(self, q: float = 0.95, min_years: int = 5, dilate: int = 1):
        self.q, self.min_years, self.dilate = q, min_years, dilate

    def votes(self, loc, terrain):
        tb = tubeness_stack(loc)
        lines = np.stack([binary_dilation(r >= np.quantile(r, self.q), iterations=self.dilate) for r in tb])
        permanent = lines.sum(axis=0) >= self.min_years
        return _repeat_years(np.where(permanent, -1, 0), loc.images.shape[0])


class LineOnFlowLF(GullyLF):
    """+1 where a thin line (tubeness top ``q``) lies within ``near`` px of a flow path
    (upslope area >= 10**pos); -1 where a thin line lies farther than ``far`` px from any
    path with area >= 10**pos_far."""

    name, needs_terrain = "line_on_flow", True

    def __init__(self, q: float = 0.95, pos: float = 2.7, near: float = 3.0, pos_far: float = 2.3, far: float = 8.0):
        self.q, self.pos, self.near, self.pos_far, self.far = q, pos, near, pos_far, far

    def votes(self, loc, terrain):
        if terrain is None:
            return self._abstain(loc)
        la = np.nan_to_num(terrain.log_acc, nan=0.0)
        d_flow = distance_transform_edt(la < self.pos)
        d_any = distance_transform_edt(la < self.pos_far)
        out = np.zeros(loc.images.shape[:3], dtype=np.int8)
        for t, tb in enumerate(tubeness_stack(loc)):
            line = tb >= np.quantile(tb, self.q)
            out[t] = np.where(line & (d_flow <= self.near), 1, np.where(line & (d_any > self.far), -1, 0))
        out[:, np.isnan(terrain.log_acc)] = 0
        return out


def default_gully_lfs(use_terrain: bool = True) -> list[GullyLF]:
    lfs: list[GullyLF] = [ThinLineLF(), StraightLineLF(), NonCropLF(), PersistentVegetationLF(), PermanentLineLF()]
    if use_terrain:
        lfs = [FlowPathLF(), ConvergenceLF(), LineOnFlowLF()] + lfs
    return lfs


def apply_gully_lfs(loc: GullyLocation, terrain: Terrain | None, lfs: list[GullyLF]) -> np.ndarray:
    """(years * H * W, m) int8 vote matrix in year-major site order (doc 11 §5)."""
    return np.stack([lf.votes(loc, terrain).reshape(-1) for lf in lfs], axis=1).astype(np.int8)
