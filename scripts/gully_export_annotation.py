"""Export a set of locations for pixel annotation (decision G9; spec doc 11 §8).

Usage:  python scripts/gully_export_annotation.py [--root data/gully] [--n 40] [--out data/gully/annotation_todo]
        [--predict]

Selects ``--n`` positive *test* locations (seeded random sample; negatives need no annotation
because they are known to be EG-free) and writes, per location and year, an upscaled image
(and, with ``--predict``, the current model's prediction overlay) plus annotation
instructions. Annotations go back into ``<root>/annotations`` as ``<tile>.geojson`` (QGIS
lines on the GeoTIFFs) or ``<tile>_y<year>.png`` masks. The output stays under data/, which is
never committed.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from qws.data.gully import make_split, read_labels
from qws.gully.annotate import export_for_annotation
from qws.gully.pipeline import GullyConfig, infer, prepare, warm_start

ap = argparse.ArgumentParser()
ap.add_argument("--root", default="data/gully")
ap.add_argument("--n", type=int, default=40)
ap.add_argument("--out", default="data/gully/annotation_todo")
ap.add_argument("--predict", action="store_true")
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()

labels = read_labels(args.root)
split = make_split(args.root)
pos_test = [t for t in split["test"] if labels[t] == 1]
tiles = sorted(np.random.default_rng(args.seed).choice(pos_test, size=min(args.n, len(pos_test)), replace=False).tolist())
cfg = GullyConfig(root=args.root)
items = prepare(tiles, cfg)
preds = None
if args.predict:
    dev = prepare(split["dev"], cfg)
    warm = warm_start(dev, cfg)
    params = warm.copy()
    params.beta = cfg.beta
    preds = infer(items, params, warm, cfg)
out = export_for_annotation(items, args.out, preds)
(Path(out) / "tiles.json").write_text(json.dumps(tiles))
print(f"exported {len(tiles)} locations to {out}")
