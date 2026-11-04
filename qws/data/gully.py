"""ViGIR Ephemeral Gully (EG) dataset (WACV 2025) — loader and spatially blocked split.

Dataset (decision G1, spec doc 11): 621 tile locations x 6 NAIP acquisitions
(``GD_Images/rgb_{year}_tile_{tile}.tif``), each 128 x 128 RGB, ~1 m/pixel GeoTIFF (WGS84),
co-registered across years. Labels (``GT_L6_Tol.json``) are *location-level*: '1' means at
least one EG is visible at the location in some year, '0' none. There is no pixel ground truth.

The dataset is copyright ViGIR and is never committed to the repository (decision G7):
``download_gully`` fetches it into ``data/`` (git-ignored).
"""

from __future__ import annotations

import json
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

GULLY_URL = "https://vigir.missouri.edu/Research/GullyDetection/GD_Dataset_ViGIR_WACV_2025.zip"
N_YEARS = 6


@dataclass
class GullyLocation:
    tile: int
    images: np.ndarray          # (6, H, W, 3) float in [0, 1], years in acquisition order
    label: int                  # 1 = at least one EG at this location, 0 = none
    bounds: tuple[float, float, float, float]  # (west, south, east, north) in degrees (WGS84)
    meta: dict = field(default_factory=dict)

    @property
    def shape(self) -> tuple[int, int]:
        return tuple(self.images.shape[1:3])  # type: ignore[return-value]


def download_gully(root: str | Path = "data/gully") -> Path:  # pragma: no cover - network
    """Download and extract the dataset into ``root`` (about 125 MB)."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    archive = root / "GD_Dataset_ViGIR_WACV_2025.zip"
    if not archive.exists():
        urllib.request.urlretrieve(GULLY_URL, archive)
    if not (root / "GD_Images").exists():
        with zipfile.ZipFile(archive) as z:
            z.extractall(root)
    return root


def read_labels(root: str | Path, label_file: str = "GT_L6_Tol.json") -> dict[int, int]:
    data = json.loads((Path(root) / label_file).read_text())
    return {int(k): int(v["label"]) for k, v in data.items()}


def tile_bounds(path: str | Path) -> tuple[float, float, float, float]:
    """(west, south, east, north) from the GeoTIFF tie point and pixel scale."""
    import tifffile

    with tifffile.TiffFile(path) as t:
        tags = t.pages[0].tags
        tp, sc = tags["ModelTiepointTag"].value, tags["ModelPixelScaleTag"].value
        h, w = t.pages[0].shape[:2]
    west, north = tp[3], tp[4]
    return west, north - h * sc[1], west + w * sc[0], north


def load_location(root: str | Path, tile: int, label: int | None = None) -> GullyLocation:
    import tifffile

    root = Path(root)
    paths = [root / "GD_Images" / f"rgb_{t}_tile_{tile}.tif" for t in range(N_YEARS)]
    images = np.stack([tifffile.imread(p)[..., :3] for p in paths]).astype(np.float32) / 255.0
    return GullyLocation(tile, images, -1 if label is None else label, tile_bounds(paths[0]))


def load_gully(root: str | Path = "data/gully", tiles: list[int] | None = None) -> list[GullyLocation]:
    labels = read_labels(root)
    tiles = sorted(labels) if tiles is None else tiles
    return [load_location(root, t, labels.get(t)) for t in tiles]


# ------------------------------------------------------------------------------ spatial split
def spatial_clusters(bounds: dict[int, tuple[float, float, float, float]], cell_deg: float = 0.02) -> dict[int, int]:
    """Group tiles into geographic cells of ``cell_deg`` degrees (~2 km); tiles in one cell
    share a cluster id. Neighbouring tiles are strongly correlated (shared fields, drainage,
    imagery), so they must not be split across dev and test."""
    keys = {t: (int(np.floor((b[0] + b[2]) / 2 / cell_deg)), int(np.floor((b[1] + b[3]) / 2 / cell_deg)))
            for t, b in bounds.items()}
    ids = {k: i for i, k in enumerate(sorted(set(keys.values())))}
    return {t: ids[k] for t, k in keys.items()}


def spatial_split(labels: dict[int, int], clusters: dict[int, int], test_fraction: float = 0.5,
                  seed: int = 0, trials: int = 5000) -> tuple[list[int], list[int]]:
    """Stratified group split: whole clusters go to dev or test.

    Clusters are lumpy (up to ~30 tiles), so a greedy pass balances poorly. Instead,
    ``trials`` random cluster assignments are drawn (seeded) and the one minimizing
    |test size - target| / n + |test positive rate - global positive rate| is kept.
    """
    rng = np.random.default_rng(seed)
    cids = sorted(set(clusters.values()))
    size = np.array([sum(1 for t in clusters if clusters[t] == c) for c in cids], dtype=float)
    pos = np.array([sum(labels[t] for t in clusters if clusters[t] == c) for c in cids], dtype=float)
    n, rate = size.sum(), pos.sum() / size.sum()
    best, best_cost = None, np.inf
    for _ in range(trials):
        test = rng.random(len(cids)) < test_fraction
        st = size[test].sum()
        if st == 0 or st == n:
            continue
        cost = abs(st - test_fraction * n) / n + abs(pos[test].sum() / st - rate) + abs(pos[~test].sum() / (n - st) - rate)
        if cost < best_cost:
            best, best_cost = test, cost
    test_ids = {c for c, t in zip(cids, best) if t}
    dev = sorted(t for t, c in clusters.items() if c not in test_ids)
    test = sorted(t for t, c in clusters.items() if c in test_ids)
    return dev, test


def make_split(root: str | Path = "data/gully", seed: int = 0, cell_deg: float = 0.02) -> dict:
    """Compute (and cache in ``root/split.json``) the spatially blocked dev/test split."""
    root = Path(root)
    cache = root / f"split_seed{seed}_cell{cell_deg}.json"
    if cache.exists():
        return json.loads(cache.read_text())
    labels = read_labels(root)
    bounds = {t: tile_bounds(root / "GD_Images" / f"rgb_0_tile_{t}.tif") for t in labels}
    clusters = spatial_clusters(bounds, cell_deg)
    dev, test = spatial_split(labels, clusters, seed=seed)
    out = {"seed": seed, "cell_deg": cell_deg, "dev": dev, "test": test,
           "n_clusters": len(set(clusters.values())),
           "dev_pos": sum(labels[t] for t in dev), "test_pos": sum(labels[t] for t in test)}
    cache.write_text(json.dumps(out))
    return out
