"""Oxford-IIIT Pet loader (decision R5; roadmap 5.1).

Expected layout (as distributed by the VGG group):

    root/images/<name>.jpg
    root/annotations/trimaps/<name>.png      1 = pet, 2 = background, 3 = boundary/unclassified
    root/annotations/{trainval,test}.txt     optional split lists ("<name> <class> <species> <breed>")

Trimaps are binarized to foreground (+1) / background (-1). Boundary pixels (3) become
0 = ignore and are excluded from every metric (spec doc 09 §6).
"""

from __future__ import annotations

import tarfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np

PET_URLS = {
    "images": "https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz",
    "annotations": "https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz",
}


@dataclass
class PetSample:
    name: str
    image: np.ndarray   # (H, W, 3) float in [0, 1]
    mask: np.ndarray    # (H, W) int8: +1 pet, -1 background, 0 ignore (trimap boundary)


def trimap_to_mask(trimap: np.ndarray) -> np.ndarray:
    t = np.asarray(trimap)
    return np.where(t == 1, 1, np.where(t == 2, -1, 0)).astype(np.int8)


def list_names(root: str | Path, split: str | None = None) -> list[str]:
    root = Path(root)
    if split is not None:
        lst = root / "annotations" / f"{split}.txt"
        if lst.exists():
            return [ln.split()[0] for ln in lst.read_text().splitlines() if ln.strip() and not ln.startswith("#")]
    return sorted(p.stem for p in (root / "annotations" / "trimaps").glob("*.png") if not p.name.startswith("._"))


def _resize(arr: np.ndarray, max_side: int | None, order: int) -> np.ndarray:
    if max_side is None or max(arr.shape[:2]) <= max_side:
        return arr
    from skimage.transform import resize

    scale = max_side / max(arr.shape[:2])
    shape = (max(1, round(arr.shape[0] * scale)), max(1, round(arr.shape[1] * scale)))
    out = resize(arr, shape + arr.shape[2:], order=order, preserve_range=True, anti_aliasing=order > 0)
    return out.astype(arr.dtype) if order == 0 else out


def load_pet(root: str | Path, split: str | None = "trainval", limit: int | None = None, max_side: int | None = 128,
             seed: int | None = 0) -> list[PetSample]:
    """Load (a random subset of) the dataset, downscaled so the longer side is ``max_side``.

    Trimaps are resized with nearest-neighbour interpolation, so labels are never blended.
    """
    from skimage.io import imread

    root = Path(root)
    names = list_names(root, split)
    if limit is not None and limit < len(names):
        rng = np.random.default_rng(seed)
        names = sorted(rng.choice(names, size=limit, replace=False).tolist())
    out = []
    for name in names:
        img_path = root / "images" / f"{name}.jpg"
        tri_path = root / "annotations" / "trimaps" / f"{name}.png"
        if not img_path.exists() or not tri_path.exists():
            continue
        img = imread(img_path)
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)
        img = img[..., :3].astype(float) / 255.0
        tri = imread(tri_path)
        if tri.ndim == 3:
            tri = tri[..., 0]
        out.append(PetSample(name, _resize(img, max_side, 1), trimap_to_mask(_resize(tri, max_side, 0))))
    return out


def download_pet(root: str | Path) -> Path:  # pragma: no cover - needs network access to robots.ox.ac.uk
    """Download and extract images and annotations into ``root`` (about 800 MB)."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    for key, url in PET_URLS.items():
        archive = root / f"{key}.tar.gz"
        if not archive.exists():
            urllib.request.urlretrieve(url, archive)
        with tarfile.open(archive) as tar:
            tar.extractall(root, filter="data")
    return root
