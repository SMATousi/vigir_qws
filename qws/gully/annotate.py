"""Pixel annotations for EG segmentation: formats, export for annotators, metrics (decision G9).

The dataset has no pixel ground truth. Pixel metrics become available once a small set of
locations is annotated. Two formats are accepted, in a directory ``ann_dir``:

* ``<tile>_y<year>.png``: uint8 mask per tile and year, 0 = background, 1 = EG,
  255 = not annotated (ignored). Years not provided count as not annotated.
* ``<tile>.geojson``: a FeatureCollection in WGS84 (EPSG:4326), typically digitized in QGIS
  on the GeoTIFFs. Each feature needs a ``year`` property (0-5). LineStrings are buffered by
  ``width_m`` metres (gullies are 1–4 m wide). Polygons are used as drawn. Every year of a
  location with a GeoJSON file counts as fully annotated (unmarked pixels are background),
  unless the property ``"annotated_years"`` on the collection lists the reviewed years.

Metrics: strict pixel IoU / precision / recall / F1 over annotated pixels, and
tolerance-based precision / recall / F1 (a prediction counts as correct within ``tol`` px of
an annotated EG pixel, and vice versa). The tolerant variant is the standard for thin
structures, because hand-drawn lines are rarely pixel-exact.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_edt

IGNORE = -1


def load_png_annotations(ann_dir: str | Path, tile: int, shape: tuple[int, int, int]) -> np.ndarray | None:
    from skimage.io import imread

    out = np.full(shape, IGNORE, dtype=np.int8)
    found = False
    for t in range(shape[0]):
        p = Path(ann_dir) / f"{tile}_y{t}.png"
        if p.exists():
            m = imread(p)
            m = m[..., 0] if m.ndim == 3 else m
            out[t] = np.where(m == 255, IGNORE, (m > 0).astype(np.int8))
            found = True
    return out if found else None


def load_geojson_annotations(ann_dir: str | Path, tile: int, bounds: tuple[float, float, float, float],
                             shape: tuple[int, int, int], width_m: float = 3.0) -> np.ndarray | None:
    """Rasterize GeoJSON lines (buffered) and polygons onto the tile grid, per year."""
    from rasterio.features import rasterize
    from rasterio.transform import from_bounds

    p = Path(ann_dir) / f"{tile}.geojson"
    if not p.exists():
        return None
    fc = json.loads(p.read_text())
    T, H, W = shape
    years = fc.get("properties", {}).get("annotated_years", list(range(T)))
    out = np.full(shape, IGNORE, dtype=np.int8)
    for t in years:
        out[t] = 0
    transform = from_bounds(*bounds, W, H)
    # metres -> degrees (local, isotropic enough at 128 m scale)
    lat = 0.5 * (bounds[1] + bounds[3])
    deg_per_m = 1.0 / (111_320.0 * np.cos(np.radians(lat)))
    for feat in fc.get("features", []):
        t = int(feat.get("properties", {}).get("year", -1))
        if not 0 <= t < T:
            continue
        geom = feat["geometry"]
        if geom["type"] in ("LineString", "MultiLineString"):
            from shapely.geometry import shape as to_shape

            geom = to_shape(geom).buffer(0.5 * width_m * deg_per_m).__geo_interface__
        burned = rasterize([(geom, 1)], out_shape=(H, W), transform=transform, fill=0, all_touched=True)
        out[t] = np.where(burned > 0, 1, out[t] if out[t].min() >= 0 else 0)
    return out


def load_annotations(ann_dir: str | Path, tile: int, bounds, shape) -> np.ndarray | None:
    a = load_geojson_annotations(ann_dir, tile, bounds, shape)
    return a if a is not None else load_png_annotations(ann_dir, tile, shape)


def pixel_metrics(pred: np.ndarray, ann: np.ndarray, tol: float = 2.0) -> dict[str, float]:
    """Strict and tolerance-based metrics over annotated pixels of all years."""
    valid = ann != IGNORE
    p = pred.astype(bool) & valid
    g = (ann == 1) & valid
    tp, fp, fn = np.sum(p & g), np.sum(p & ~g), np.sum(~p & g & valid)
    out = {"iou": tp / max(tp + fp + fn, 1), "precision": tp / max(tp + fp, 1), "recall": tp / max(tp + fn, 1)}
    out["f1"] = 2 * tp / max(2 * tp + fp + fn, 1)
    tp_p = tp_g = n_p = n_g = 0
    for t in range(pred.shape[0]):
        if not valid[t].any():
            continue
        d_to_g = distance_transform_edt(~g[t]) if g[t].any() else np.full(g[t].shape, np.inf)
        d_to_p = distance_transform_edt(~p[t]) if p[t].any() else np.full(p[t].shape, np.inf)
        tp_p += np.sum(p[t] & (d_to_g <= tol))
        n_p += p[t].sum()
        tp_g += np.sum(g[t] & (d_to_p <= tol))
        n_g += g[t].sum()
    prec, rec = tp_p / max(n_p, 1), tp_g / max(n_g, 1)
    out.update(tol_precision=prec, tol_recall=rec, tol_f1=2 * prec * rec / max(prec + rec, 1e-12), tol_px=tol,
               annotated_pixels=int(valid.sum()), eg_pixels=int(g.sum()))
    return {k: float(v) for k, v in out.items()}


def export_for_annotation(items, out_dir: str | Path, predictions=None, scale: int = 4) -> Path:
    """Write upscaled PNGs (and optional prediction overlays) plus instructions for annotators."""
    from skimage.io import imsave

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    preds = {p.tile: p for p in (predictions or [])}
    for it in items:
        for t, img in enumerate(it.loc.images):
            big = np.kron(img, np.ones((scale, scale, 1)))
            imsave(out / f"{it.tile}_y{t}_image.png", (big * 255).astype(np.uint8), check_contrast=False)
            if it.tile in preds and preds[it.tile].marginal is not None:
                over = img.copy()
                over[preds[it.tile].marginal[t] > 0.5] = [1, 0, 0]
                imsave(out / f"{it.tile}_y{t}_prediction.png", (np.kron(over, np.ones((scale, scale, 1))) * 255
                                                                ).astype(np.uint8), check_contrast=False)
    (out / "README.txt").write_text(
        "Ephemeral-gully pixel annotation\\n\\n"
        "Preferred: digitize each visible EG as a line in QGIS on the original GeoTIFF\\n"
        "(GD_Images/rgb_<year>_tile_<tile>.tif) and save one GeoJSON per tile, <tile>.geojson,\\n"
        "in EPSG:4326, with an integer property 'year' (0-5) on every feature.\\n"
        "Optional collection property 'annotated_years': [...] lists the reviewed years.\\n\\n"
        f"Alternative: paint a mask <tile>_y<year>.png at the original 128x128 size (these PNGs\\n"
        f"are upscaled x{scale} for viewing): 0 = background, 1 = EG, 255 = not annotated.\\n")
    return out
