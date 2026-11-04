"""End-task evaluation: label training images weakly, train a U-Net, test on held-out images.

Compared training signals (all evaluated on the same held-out test images):

* ``structured_soft``: soft masks from the structured label model (this work);
* ``independent_soft``: soft masks from the independent (beta = 0) label model;
* ``majority_hard``: per-site majority vote over the LFs;
* ``ground_truth``: the true masks (upper bound; never available in weak supervision).

For reference, the label model is also applied directly (transductively) to the test images.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..eval import iou
from ..experiment import DEFAULT_CONFIG, build_dataset, fit_and_label, merge
from ..learn import infer_map
from ..model import independent_posterior, warm_start
from .unet import UNetConfig, predict_unet, train_unet


def run_downstream(config: dict | None = None, unet: UNetConfig | None = None) -> dict[str, Any]:
    cfg = merge(DEFAULT_CONFIG, config or {})
    unet = unet or UNetConfig(seed=int(cfg["seed"]))
    kind = cfg["dataset"]["kind"]
    ignore = kind in ("pet", "petlike")
    if kind in ("pet", "petlike"):
        train_items, _ = build_dataset(merge(cfg, {"dataset": {"split": "trainval"}}))
        test_cfg = merge(cfg, {"dataset": {"split": "test", "kind": "pet" if kind == "petlike" else kind}})
        test_items, _ = build_dataset(test_cfg)
    else:
        train_items, _ = build_dataset(cfg)
        test_items, _ = build_dataset(merge(cfg, {"seed": int(cfg["seed"]) + 1000}))

    fl = fit_and_label(cfg, train_items)
    data = fl["data"]
    indep = warm_start(data, beta=0.0)
    signals = {
        "structured_soft": [im.sites.to_pixels(mu) for mu, im in zip(fl["marginals"], train_items)],
        "independent_soft": [im.sites.to_pixels(0.5 * (1 + independent_posterior(indep, lfo).mean))
                             for (lfo, _), im in zip(data, train_items)],
        "majority_hard": [im.sites.to_pixels((lfo.L.sum(axis=1) > 0).astype(float))
                          for (lfo, _), im in zip(data, train_items)],
        "ground_truth": [(im.mask > 0).astype(float) for im in train_items],
    }
    valid = [im.mask != 0 for im in train_items] if ignore else None
    images = [np.atleast_3d(im.image) if im.image.ndim == 3 else np.repeat(im.image[..., None], 3, -1)
              for im in train_items]
    test_images = [im.image if im.image.ndim == 3 else np.repeat(im.image[..., None], 3, -1) for im in test_items]

    def test_iou(preds):
        return float(np.mean([iou(np.where(p > 0.5, 1, -1), im.mask, ignore) for p, im in zip(preds, test_items)]))

    def train_iou(soft):
        return float(np.mean([iou(np.where(s > 0.5, 1, -1), im.mask, ignore) for s, im in zip(soft, train_items)]))

    out: dict[str, Any] = {"n_train": len(train_items), "n_test": len(test_items), "unet": unet.__dict__,
                           "label_model": fl["result"].params.to_dict(), "signals": {}}
    for name, soft in signals.items():
        # ground truth is the oracle; it uses the true masks only where they are defined
        model, hist = train_unet(images, soft, unet, valid if name == "ground_truth" else None)
        out["signals"][name] = {"train_label_iou": train_iou(soft), "test_iou": test_iou(predict_unet(model, test_images, unet.size)),
                                "final_loss": hist[-1]}
    # label model applied directly to the test images (transductive reference)
    test_data = [(im.lfo, im.sites) for im in test_items]
    from ..experiment import _solver

    direct = infer_map(fl["result"].params, test_data, _solver(cfg["map_solver"], int(cfg["seed"])))
    out["label_model_direct_test_iou"] = float(np.mean([iou(im.sites.to_pixels(d), im.mask, ignore)
                                                        for d, im in zip(direct, test_items)]))
    return out
