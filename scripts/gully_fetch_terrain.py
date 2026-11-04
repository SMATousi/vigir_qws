"""Fetch and cache USGS 3DEP 1 m terrain products for every gully tile (spec doc 11 §3).

Usage:  python scripts/gully_fetch_terrain.py [--root data/gully] [--margin 300]
Needs network access to prd-tnm.s3.amazonaws.com. Cached in <root>/terrain/<tile>.npz.
"""

import argparse
import os
import time

from qws.data.dem3dep import build_index, cached_terrain
from qws.data.gully import read_labels, tile_bounds

os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
if os.path.exists("/root/.ccr/ca-bundle.crt"):
    os.environ.setdefault("CURL_CA_BUNDLE", "/root/.ccr/ca-bundle.crt")

ap = argparse.ArgumentParser()
ap.add_argument("--root", default="data/gully")
ap.add_argument("--margin", type=float, default=300.0)
args = ap.parse_args()
index = build_index(cache=f"{args.root}/dem_index_1m.json")
tiles = sorted(read_labels(args.root))
t0, missing = time.time(), []
for k, tile in enumerate(tiles):
    b = tile_bounds(f"{args.root}/GD_Images/rgb_0_tile_{tile}.tif")
    for attempt in range(3):
        try:
            t = cached_terrain(index, tile, b, (128, 128), f"{args.root}/terrain", args.margin)
            break
        except Exception as exc:  # transient network errors: retry with backoff
            print(f"tile {tile} attempt {attempt + 1} failed: {exc}", flush=True)
            time.sleep(2 ** (attempt + 1))
    else:
        t = None
    if t is None:
        missing.append(tile)
    if k % 50 == 0:
        print(f"{k + 1}/{len(tiles)} tiles, {time.time() - t0:.0f}s, missing {len(missing)}", flush=True)
print(f"done: {len(tiles) - len(missing)} with terrain, missing {missing}", flush=True)
