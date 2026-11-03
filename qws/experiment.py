"""End-to-end experiment driver used by the CLI (doc 05 §5): a declarative config in,
a JSON-serializable report out."""

from __future__ import annotations

import copy
import time
from typing import Any

import numpy as np

from .eval import ece, iou, synthetic_dataset
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


def run_experiment(config: dict | None = None) -> dict[str, Any]:
    cfg = merge(DEFAULT_CONFIG, config or {})
    seed = int(cfg["seed"])
    ds = cfg["dataset"]
    if ds["kind"] != "synthetic":
        raise NotImplementedError("only synthetic datasets are wired up; real datasets await open question Q1")
    tp = ds["true_params"]
    true = LabelModelParams(np.array(tp["alpha"]), np.array(tp["theta"]), np.array(tp.get("alphaP", [])),
                            np.array(tp.get("thetaP", [])))
    images = synthetic_dataset(true, ds["n_images"], tuple(ds["shape"]), ds["sites"], ds["n_segments"],
                               ds["smoothness"], seed=seed)
    data = [(im.lfo, im.sites) for im in images]

    sampler = _solver(cfg["solver"], seed)
    if cfg["calibrate"]:
        sampler = CalibratedSampler(sampler)
    tc = TrainConfig(**{**cfg["train"], "beta": cfg["model"]["beta"], "seed": seed})
    t0 = time.perf_counter()
    init = warm_start(data, beta=tc.beta)
    result = Trainer(sampler, tc).fit(data, init)
    train_time = time.perf_counter() - t0

    map_solver = _solver(cfg["map_solver"], seed)
    masks = infer_map(result.params, data, map_solver)
    marg = infer_marginals(result.params, data, _solver(cfg["solver"], seed), num_samples=200, seed=seed)

    def mean_iou(preds):
        return float(np.mean([iou(im.sites.to_pixels(p), im.mask) for p, im in zip(preds, images)]))

    majority = [np.where(lfo.L.sum(axis=1) > 0, 1, -1) for lfo, _ in data]
    indep_params = warm_start(data, beta=0.0)
    independent = [np.where(independent_posterior(indep_params, lfo).mean > 0, 1, -1) for lfo, _ in data]
    report = {
        "config": cfg,
        "train_time_s": train_time,
        "epochs_run": len(result.history),
        "true_params": true.to_dict(),
        "learned_params": result.params.to_dict(),
        "theta_error": float(np.abs(result.params.theta - true.theta).max()),
        "thetaP_error": float(np.abs(result.params.thetaP - true.thetaP).max()) if true.q else None,
        "iou": {"site_oracle": mean_iou([im.y for im in images]), "majority_vote": mean_iou(majority), "independent_model": mean_iou(independent),
                "structured_map": mean_iou(masks)},
        "ece_structured": float(np.mean([ece(im.sites.to_pixels(mu), im.mask) for mu, im in zip(marg, images)])),
        "history": result.history,
    }
    if cfg.get("output"):
        from .io import save_json

        save_json(cfg["output"], report)
    return report
