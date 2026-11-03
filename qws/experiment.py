"""End-to-end experiment driver used by the CLI (doc 05 §5): a declarative config in,
a JSON-serializable report out."""

from __future__ import annotations

import copy
import time
from typing import Any

import numpy as np

from .eval import SyntheticImage, ece, iou, synthetic_dataset
from .learn import CalibratedSampler, TrainConfig, Trainer, infer_map, infer_marginals
from .model import LabelModelParams, independent_posterior, warm_start
from .solvers import get_solver

DEFAULT_CONFIG: dict[str, Any] = {
    "seed": 0,
    "dataset": {
        "kind": "synthetic",
        "n_images": 8,
        "shape": [16, 16],
        "sites": "grid",
        "n_segments": 100,
        "smoothness": 3.0,
        "true_params": {
            "theta": [1.0, 0.6, 0.3, 0.8],
            "alpha": [0.5, 0.0, -0.5, 0.2],
            "thetaP": [1.0],
            "alphaP": [0.0],
        },
    },
    "model": {"beta": 0.5},
    "solver": {"name": "gibbs", "num_sweeps": 100},
    "map_solver": {"name": "sa", "num_sweeps": 300},
    "calibrate": False,
    "train": {"epochs": 20, "lr": 0.05, "num_samples": 100, "polish_sweeps": 0},
    "output": None,
}


def merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        out[k] = merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def _solver(spec: dict, seed: int):
    spec = dict(spec)
    name = spec.pop("name")
    if name in ("exact", "gibbs", "sa", "qaoa_sim", "qaoa_qiskit") and "seed" not in spec:
        spec["seed"] = seed
    return get_solver(name, **spec)


def _true_params(tp: dict) -> LabelModelParams:
    return LabelModelParams(np.array(tp["alpha"]), np.array(tp["theta"]), np.array(tp.get("alphaP", [])),
                            np.array(tp.get("thetaP", [])), b=tp.get("b", 0.0), beta=tp.get("beta", 0.0),
                            omega=np.array(tp.get("omega", [])), zeta=np.array(tp.get("zeta", [])))


def build_dataset(cfg: dict) -> tuple[list[SyntheticImage], LabelModelParams | None]:
    """Images with sites, LF votes and ground truth for every dataset kind.

    * ``synthetic``: votes from the model's own conditional law (known true parameters).
    * ``pet``: Oxford-IIIT Pet from ``dataset.root`` (decision R5) with the generic LFs.
    * ``petlike``: a generated dataset in the exact Pet file layout (written to
      ``dataset.root``), run through the same real-image pipeline as ``pet``.
    """
    ds = cfg["dataset"]
    seed = int(cfg["seed"])
    if ds["kind"] == "synthetic":
        true = _true_params(ds["true_params"])
        return synthetic_dataset(true, ds["n_images"], tuple(ds["shape"]), ds["sites"], ds["n_segments"],
                                 ds["smoothness"], seed=seed, mask_source=ds.get("mask_source", "blobs")), true
    if ds["kind"] in ("pet", "petlike"):
        from .data import load_pet
        from .data.petlike import write_petlike_dataset
        from .graph import superpixel_sites
        from .lf import apply_lfs, default_natural_lfs

        root = ds.get("root") or "data/oxford-iiit-pet"
        if ds["kind"] == "petlike":
            write_petlike_dataset(root, n_images=ds.get("generate", 2 * ds["n_images"]), seed=seed)
        samples = load_pet(root, ds.get("split", "trainval"), ds["n_images"], ds.get("max_side", 128), seed)
        if not samples:
            raise FileNotFoundError(f"no Oxford-IIIT Pet images under {root!r} (see qws.data.download_pet)")
        unary, pairwise = default_natural_lfs()
        items = []
        for smp in samples:
            sites = superpixel_sites(smp.image, n_segments=ds.get("n_segments", 200),
                                     compactness=ds.get("compactness", 10.0))
            y = np.where(sites.from_pixels(smp.mask) > 0, 1, -1).astype(np.int8)  # ignore pixels count as 0
            items.append(SyntheticImage(smp.image, smp.mask, sites, y, apply_lfs(smp.image, sites, unary, pairwise)))
        return items, None
    raise ValueError(f"unknown dataset kind {ds['kind']!r}")


def fit_and_label(cfg: dict, items: list[SyntheticImage]) -> dict[str, Any]:
    """Warm start, train with the configured sampler, then infer hard and soft masks."""
    seed = int(cfg["seed"])
    data = [(im.lfo, im.sites) for im in items]
    sampler = _solver(cfg["solver"], seed)
    if cfg["calibrate"]:
        sampler = CalibratedSampler(sampler)
    tc = TrainConfig(**{**cfg["train"], "beta": cfg["model"]["beta"], "seed": seed})
    t0 = time.perf_counter()
    init = warm_start(data, beta=tc.beta)
    result = Trainer(sampler, tc).fit(data, init)
    train_time = time.perf_counter() - t0
    masks = infer_map(result.params, data, _solver(cfg["map_solver"], seed))
    marg = infer_marginals(result.params, data, _solver(cfg["solver"], seed), num_samples=200, seed=seed)
    return {"result": result, "masks": masks, "marginals": marg, "train_time_s": train_time, "data": data}


def run_experiment(config: dict | None = None) -> dict[str, Any]:
    cfg = merge(DEFAULT_CONFIG, config or {})
    items, true = build_dataset(cfg)
    fl = fit_and_label(cfg, items)
    result, data = fl["result"], fl["data"]
    ignore = cfg["dataset"]["kind"] in ("pet", "petlike")  # trimap boundary pixels are excluded

    def mean_iou(preds):
        return float(np.mean([iou(im.sites.to_pixels(p), im.mask, ignore) for p, im in zip(preds, items)]))

    majority = [np.where(lfo.L.sum(axis=1) > 0, 1, -1) for lfo, _ in data]
    indep_params = warm_start(data, beta=0.0)
    independent = [np.where(independent_posterior(indep_params, lfo).mean > 0, 1, -1) for lfo, _ in data]
    lp = result.params
    report = {
        "config": cfg,
        "train_time_s": fl["train_time_s"],
        "epochs_run": len(result.history),
        "true_params": true.to_dict() if true is not None else None,
        "learned_params": lp.to_dict(),
        "theta_error": float(np.abs(lp.theta - true.theta).max()) if true is not None else None,
        "thetaP_error": float(np.abs(lp.thetaP - true.thetaP).max()) if true is not None and true.q else None,
        "iou": {"site_oracle": mean_iou([im.y for im in items]), "majority_vote": mean_iou(majority),
                "independent_model": mean_iou(independent), "structured_map": mean_iou(fl["masks"])},
        "ece_structured": float(np.mean([ece(im.sites.to_pixels(mu), im.mask, ignore_zero=ignore)
                                         for mu, im in zip(fl["marginals"], items)])),
        "lf_accuracy": _lf_accuracy(items),
        "history": result.history,
    }
    if cfg.get("output"):
        from .io import save_json

        save_json(cfg["output"], report)
    return report


def _lf_accuracy(items: list[SyntheticImage]) -> dict[str, list[float]]:
    """Empirical accuracy and coverage of each unary LF against the site ground truth
    (diagnostic only; never used for training)."""
    L = np.concatenate([im.lfo.L for im in items])
    y = np.concatenate([im.y for im in items])
    voted = L != 0
    acc = [float(np.mean(L[voted[:, j], j] == y[voted[:, j]])) if voted[:, j].any() else float("nan")
           for j in range(L.shape[1])]
    return {"names": list(items[0].lfo.unary_names), "accuracy": acc, "coverage": voted.mean(axis=0).tolist()}
