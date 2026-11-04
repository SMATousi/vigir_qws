"""Generator of a small dataset in the exact Oxford-IIIT Pet file layout.

Used to exercise the real-image pipeline (JPEG decoding, trimaps with a boundary band,
split lists, resizing, superpixels, generic LFs, ignore-masked metrics) while the real
dataset's hosts are not reachable from this environment (spec doc 09 §6). Images are
clearly synthetic: a textured, roughly centred blob ("pet") on a textured background
with overlapping colour distributions. They are NOT a substitute for real Pet results.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion, gaussian_filter


def _texture(shape: tuple[int, int], rng: np.random.Generator, scale: float) -> np.ndarray:
    t = gaussian_filter(rng.standard_normal(shape), scale)
    return (t - t.mean()) / (t.std() + 1e-9)


def make_petlike(shape: tuple[int, int], rng: np.random.Generator, color_sep: float = 0.35,
                 boundary: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """One image (H, W, 3, uint8) and its trimap (H, W, uint8: 1 pet, 2 background, 3 boundary)."""
    H, W = shape
    rr, cc = np.indices(shape)
    cy, cx = H / 2 + rng.normal(0, H / 10), W / 2 + rng.normal(0, W / 10)
    ry, rx = rng.uniform(0.2, 0.38) * H, rng.uniform(0.2, 0.38) * W
    ang = np.arctan2(rr - cy, cc - cx)
    wobble = 1 + 0.18 * np.sin(3 * ang + rng.uniform(0, 2 * np.pi)) + 0.1 * np.sin(5 * ang)
    fg = ((rr - cy) / ry) ** 2 + ((cc - cx) / rx) ** 2 <= wobble**2
    bg_col = rng.uniform(0.2, 0.8, 3)
    direction = rng.normal(size=3)
    fg_col = np.clip(bg_col + color_sep * direction / np.linalg.norm(direction), 0.05, 0.95)
    img = np.where(fg[..., None], fg_col, bg_col)
    img = img + 0.12 * np.stack([_texture(shape, rng, 1.5) for _ in range(3)], -1)
    img = img + 0.10 * _texture(shape, rng, 6.0)[..., None]  # illumination
    img = np.clip(img, 0, 1)
    trimap = np.where(fg, 1, 2).astype(np.uint8)
    band = binary_dilation(fg, iterations=boundary) & ~binary_erosion(fg, iterations=boundary)
    trimap[band] = 3
    return (img * 255).round().astype(np.uint8), trimap


def write_petlike_dataset(root: str | Path, n_images: int = 20, shape: tuple[int, int] = (96, 128),
                          seed: int = 0, test_fraction: float = 0.5, **kw) -> Path:
    """Write images/, annotations/trimaps/ and annotations/{trainval,test}.txt under ``root``."""
    from skimage.io import imsave

    root = Path(root)
    (root / "images").mkdir(parents=True, exist_ok=True)
    (root / "annotations" / "trimaps").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    names = [f"petlike_{k:04d}" for k in range(n_images)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # "low contrast image" warnings for trimaps
        for name in names:
            img, tri = make_petlike(shape, rng, **kw)
            imsave(root / "images" / f"{name}.jpg", img, quality=92)
            imsave(root / "annotations" / "trimaps" / f"{name}.png", tri)
    n_test = int(round(test_fraction * n_images))
    lines = [f"{n} 1 1 1" for n in names]
    (root / "annotations" / "trainval.txt").write_text("\n".join(lines[n_test:]) + "\n")
    (root / "annotations" / "test.txt").write_text("\n".join(lines[:n_test]) + "\n")
    return root
