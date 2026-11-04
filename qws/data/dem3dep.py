"""USGS 3DEP 1 m LiDAR DEM access and terrain features for the gully tiles (decision G4).

Ephemeral gullies form where overland flow concentrates, so terrain is the strongest
physical prior. 1 m DEMs are read straight from the public USGS bucket
(``prd-tnm.s3.amazonaws.com``), with HTTP range requests on the cloud GeoTIFFs; only the
window around each tile is transferred.

Pipeline per tile (spec doc 11 §3):

1. Find the 10 km x 10 km 1 m DEM files (UTM, NAD83) that overlap the tile's bounds plus a
   ``margin`` (upslope area outside the tile matters). Files are named
   ``USGS_1M_{zone}_x{E // 10 km}y{ceil(N / 10 km)}_{project}.tif``. When several projects
   cover a file cell, the most recent acquisition year wins.
2. Mosaic the window at 1 m (``rasterio.merge``), fill depressions (priority-flood with an
   epsilon gradient, so flats drain), then compute D8 flow directions and flow accumulation
   (``qws.data.hydro``).
3. Derive slope, log flow accumulation, the topographic wetness index
   TWI = ln(a / tan(slope)), with a the specific catchment area, and a convergence index
   (negative Laplacian of the smoothed DEM).
4. Reproject every product onto the tile's own WGS84 pixel grid (bilinear), so terrain and
   imagery pixels align, and cache it as ``.npz``.
"""

from __future__ import annotations

import json
import math
import re
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np

BUCKET = "https://prd-tnm.s3.amazonaws.com"
PREFIX_1M = "StagedProducts/Elevation/1m/Projects/"
_KEY_RE = re.compile(r"USGS_1M_(\d+)_x(\d+)y(\d+)_(.+)\.tif$")


def _list(prefix: str, delimiter: str | None = None) -> tuple[list[str], list[str]]:
    """List keys and common prefixes under ``prefix`` (handles pagination)."""
    keys, prefixes, token = [], [], None
    while True:
        q = f"{BUCKET}/?list-type=2&prefix={prefix}" + (f"&delimiter={delimiter}" if delimiter else "")
        if token:
            q += "&continuation-token=" + urllib.request.quote(token, safe="")
        xml = urllib.request.urlopen(q, timeout=60).read().decode()
        keys += re.findall(r"<Key>([^<]*)</Key>", xml)
        prefixes += re.findall(r"<Prefix>([^<]*)</Prefix>", xml)[1:]  # first is the query prefix
        m = re.search(r"<NextContinuationToken>([^<]*)</NextContinuationToken>", xml)
        if not m:
            return keys, prefixes
        token = m.group(1)


def _year(project: str) -> int:
    years = [int(y) for y in re.findall(r"(?<!\d)(20\d\d|19\d\d)(?!\d)", project)]
    letter = re.search(r"_([A-D])(\d\d)$", project)  # e.g. ..._D20 (delivery year 2020)
    if letter:
        years.append(2000 + int(letter.group(2)))
    return max(years) if years else 0


def build_index(states: tuple[str, ...] = ("IA", "MO", "IL"), cache: str | Path | None = None) -> dict[str, list[str]]:
    """Map 'zone_x_y' -> list of 1 m DEM keys (most recent project first) for the given states."""
    if cache is not None and Path(cache).exists():
        return json.loads(Path(cache).read_text())
    index: dict[str, list[str]] = {}
    for st in states:
        _, projects = _list(f"{PREFIX_1M}{st}_", "/")
        for proj in projects:
            keys, _ = _list(f"{proj}TIFF/")
            for k in keys:
                m = _KEY_RE.search(k)
                if m:
                    index.setdefault(f"{m.group(1)}_{m.group(2)}_{m.group(3)}", []).append(k)
    for cell in index:
        index[cell].sort(key=lambda k: _year(_KEY_RE.search(k).group(4)), reverse=True)
    if cache is not None:
        Path(cache).parent.mkdir(parents=True, exist_ok=True)
        Path(cache).write_text(json.dumps(index))
    return index


@dataclass
class Terrain:
    dem: np.ndarray        # elevation (m) on the tile grid
    slope: np.ndarray      # degrees
    log_acc: np.ndarray    # log10 of upslope contributing area (m^2)
    twi: np.ndarray        # topographic wetness index
    convergence: np.ndarray  # -Laplacian of smoothed DEM (positive = valley / convergent)
    source: str            # DEM keys used

    def save(self, path: Path) -> None:
        np.savez_compressed(path, dem=self.dem, slope=self.slope, log_acc=self.log_acc, twi=self.twi,
                            convergence=self.convergence, source=np.array(self.source))

    @classmethod
    def load(cls, path: Path) -> "Terrain":
        z = np.load(path)
        return cls(z["dem"], z["slope"], z["log_acc"], z["twi"], z["convergence"], str(z["source"]))


def _utm_epsg(lon: float) -> tuple[int, int]:
    zone = int((lon + 180) // 6) + 1
    return zone, 26900 + zone  # NAD83 / UTM zone N (the 3DEP 1 m CRS)


def fetch_window(index: dict[str, list[str]], bounds: tuple[float, float, float, float], margin: float = 300.0):
    """Mosaicked 1 m DEM (UTM) covering the WGS84 ``bounds`` plus ``margin`` metres.

    Returns (dem, transform, crs, keys) or None if 3DEP has no 1 m coverage.
    """
    import rasterio
    from pyproj import Transformer
    from rasterio.merge import merge

    west, south, east, north = bounds
    zone, epsg = _utm_epsg((west + east) / 2)
    tr = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    xs, ys = tr.transform([west, east, west, east], [south, south, north, north])
    E0, E1, N0, N1 = min(xs) - margin, max(xs) + margin, min(ys) - margin, max(ys) + margin
    keys = []
    for x in range(int(E0 // 10000), int(E1 // 10000) + 1):
        for y in range(int(math.ceil(N0 / 10000)), int(math.ceil(N1 / 10000)) + 1):
            cands = index.get(f"{zone}_{x}_{y}")
            if cands:
                keys.append(cands[0])
    if not keys:
        return None
    srcs = [rasterio.open(f"/vsicurl/{BUCKET}/{k}") for k in keys]
    try:
        mosaic, transform = merge(srcs, bounds=(E0, N0, E1, N1), res=1.0, nodata=np.nan)
        crs = srcs[0].crs
    finally:
        for s in srcs:
            s.close()
    return mosaic[0].astype(np.float64), transform, crs, keys


def terrain_products(dem: np.ndarray, transform, crs) -> dict[str, np.ndarray]:
    """Slope, flow accumulation, TWI and convergence on the (UTM, 1 m) DEM window."""
    from scipy.ndimage import gaussian_filter, laplace

    from .hydro import flow_accumulation

    valid = np.isfinite(dem)
    fill = np.where(valid, dem, np.nanmin(dem) if valid.any() else 0.0)
    acc = flow_accumulation(fill)  # upslope cells (1 m^2 each)
    gy, gx = np.gradient(gaussian_filter(fill, 1.0))
    slope = np.degrees(np.arctan(np.hypot(gx, gy)))
    twi = np.log((acc + 1.0) / (np.tan(np.radians(slope)) + 1e-3))
    conv = -laplace(gaussian_filter(fill, 3.0))
    out = {"dem": fill, "slope": slope, "log_acc": np.log10(acc + 1.0), "twi": twi, "convergence": conv}
    for k in out:
        out[k] = np.where(valid, out[k], np.nan)
    return out


def terrain_for_bounds(index: dict[str, list[str]], bounds: tuple[float, float, float, float],
                       shape: tuple[int, int], margin: float = 300.0) -> Terrain | None:
    """Terrain products resampled onto the tile's WGS84 grid of ``shape`` pixels."""
    from rasterio.transform import from_bounds
    from rasterio.warp import Resampling, reproject

    win = fetch_window(index, bounds, margin)
    if win is None:
        return None
    dem, transform, crs, keys = win
    prods = terrain_products(dem, transform, crs)
    dst_t = from_bounds(*bounds, shape[1], shape[0])
    out = {}
    for name, arr in prods.items():
        dst = np.full(shape, np.nan, dtype=np.float64)
        reproject(arr, dst, src_transform=transform, src_crs=crs, dst_transform=dst_t, dst_crs="EPSG:4326",
                  resampling=Resampling.bilinear, src_nodata=np.nan, dst_nodata=np.nan)
        out[name] = dst
    return Terrain(**out, source=";".join(keys))


def cached_terrain(index: dict[str, list[str]], tile: int, bounds, shape, cache_dir: str | Path,
                   margin: float = 300.0) -> Terrain | None:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{tile}.npz"
    miss = cache_dir / f"{tile}.none"
    if path.exists():
        return Terrain.load(path)
    if miss.exists():
        return None
    t = terrain_for_bounds(index, bounds, shape, margin)
    if t is None:
        miss.touch()
        return None
    t.save(path)
    return t
