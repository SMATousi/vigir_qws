"""Generic labeling functions for natural object-centric images (Oxford-IIIT Pet, R5).

None of them uses ground truth. They encode weak, generic priors:

* ``BorderBackgroundLF``: pixels near the image border are background.
* ``CenterForegroundLF``: pixels in a central ellipse are foreground.
* ``ColorModelLF``: colour histograms fitted on the border band (background) and the
  centre (foreground); votes by log-likelihood ratio.
* ``SaliencyLF``: spectral-residual saliency (Hou & Zhang, 2007); high is foreground.
* ``ColorContrastLF``: pairwise; strong colour differences mean "different", tiny
  ones "same" (thresholds relative to the image's own edge-difference distribution).
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter, uniform_filter

from ..graph import Sites
from .base import PairwiseLF, PixelLF


def _rgb(image: np.ndarray) -> np.ndarray:
    img = np.asarray(image, dtype=float)
    return np.stack([img] * 3, axis=-1) if img.ndim == 2 else img[..., :3]


def border_band(shape: tuple[int, int], frac: float) -> np.ndarray:
    H, W = shape
    w = max(1, int(round(frac * min(H, W))))
    band = np.zeros((H, W), dtype=bool)
    band[:w], band[-w:], band[:, :w], band[:, -w:] = True, True, True, True
    return band


def center_ellipse(shape: tuple[int, int], frac: float) -> np.ndarray:
    H, W = shape
    rr, cc = np.indices((H, W))
    return ((rr - (H - 1) / 2) / (frac * H / 2)) ** 2 + ((cc - (W - 1) / 2) / (frac * W / 2)) ** 2 <= 1.0


class BorderBackgroundLF(PixelLF):
    def __init__(self, frac: float = 0.06, name: str = "border_bg", min_coverage: float = 0.5):
        super().__init__(lambda img: np.where(border_band(np.asarray(img).shape[:2], frac), -1, 0), name,
                         min_coverage)


class CenterForegroundLF(PixelLF):
    def __init__(self, frac: float = 0.35, name: str = "center_fg", min_coverage: float = 0.5):
        super().__init__(lambda img: np.where(center_ellipse(np.asarray(img).shape[:2], frac), 1, 0), name,
                         min_coverage)


class ColorModelLF(PixelLF):
    """Histogram colour model: background from the border band, foreground from the
    central ellipse; vote +1 / -1 where the log-likelihood ratio exceeds +-``margin``."""

    def __init__(self, bins: int = 8, border_frac: float = 0.06, center_frac: float = 0.35, margin: float = 1.0,
                 name: str = "color_model", min_coverage: float = 0.5):
        self.bins, self.border_frac, self.center_frac, self.margin = bins, border_frac, center_frac, margin
        super().__init__(self._votes, name, min_coverage)

    def _votes(self, image: np.ndarray) -> np.ndarray:
        img = _rgb(image)
        q = np.clip((img * self.bins).astype(int), 0, self.bins - 1)
        idx = (q[..., 0] * self.bins + q[..., 1]) * self.bins + q[..., 2]
        nb = self.bins**3

        def hist(sel: np.ndarray) -> np.ndarray:
            h = np.bincount(idx[sel], minlength=nb).astype(float) + 1.0  # Laplace smoothing
            return h / h.sum()

        bg = hist(border_band(img.shape[:2], self.border_frac))
        fg = hist(center_ellipse(img.shape[:2], self.center_frac))
        llr = np.log(fg[idx]) - np.log(bg[idx])
        return np.where(llr > self.margin, 1, np.where(llr < -self.margin, -1, 0))


def spectral_residual_saliency(image: np.ndarray, size: int = 64, sigma: float = 2.5) -> np.ndarray:
    """Spectral-residual saliency map in [0, 1], computed at ``size`` and resized back."""
    from skimage.transform import resize

    gray = _rgb(image).mean(axis=-1)
    small = resize(gray, (size, size), anti_aliasing=True)
    F = np.fft.fft2(small)
    log_amp = np.log(np.abs(F) + 1e-9)
    residual = log_amp - uniform_filter(log_amp, size=3, mode="nearest")
    sal = np.abs(np.fft.ifft2(np.exp(residual + 1j * np.angle(F)))) ** 2
    sal = gaussian_filter(sal, sigma)
    sal = resize(sal, gray.shape, anti_aliasing=False)
    return (sal - sal.min()) / (np.ptp(sal) + 1e-12)


class SaliencyLF(PixelLF):
    """+1 for the most salient ``high`` fraction of pixels, -1 for the least salient ``low`` fraction."""

    def __init__(self, high: float = 0.2, low: float = 0.4, name: str = "saliency", min_coverage: float = 0.5):
        def fn(img: np.ndarray) -> np.ndarray:
            s = spectral_residual_saliency(img)
            hi, lo = np.quantile(s, 1 - high), np.quantile(s, low)
            return np.where(s >= hi, 1, np.where(s <= lo, -1, 0))

        super().__init__(fn, name, min_coverage)


class ColorContrastLF(PairwiseLF):
    """Pairwise LF on site mean colours, with per-image relative thresholds: edges whose
    colour distance is above the ``differ_q`` quantile vote "different", and edges below
    the ``same_q`` quantile vote "same"."""

    def __init__(self, same_q: float = 0.3, differ_q: float = 0.9, name: str = "color_contrast"):
        if not 0 <= same_q < differ_q <= 1:
            raise ValueError("need 0 <= same_q < differ_q <= 1")
        self.same_q, self.differ_q, self.name = same_q, differ_q, name

    def votes(self, image: np.ndarray, sites: Sites) -> np.ndarray:
        e = sites.edges
        if e.size == 0:
            return np.zeros(0, dtype=np.int8)
        d = np.linalg.norm(sites.features[e[:, 0]] - sites.features[e[:, 1]], axis=1)
        lo, hi = np.quantile(d, self.same_q), np.quantile(d, self.differ_q)
        return np.where(d > hi, -1, np.where(d < lo, 1, 0)).astype(np.int8)


def default_natural_lfs() -> tuple[list, list]:
    """The generic LF set used for Oxford-IIIT Pet (spec doc 09 §6)."""
    unary = [BorderBackgroundLF(), CenterForegroundLF(), ColorModelLF(), SaliencyLF()]
    return unary, [ColorContrastLF()]
