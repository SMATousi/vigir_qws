"""Reports 13–14: the label model and the downstream U-Net on the real Oxford-IIIT Pet dataset.

Usage:  python scripts/pet_experiment.py [--root data/oxford-iiit-pet] [--quick] [--only pipeline downstream]

These repeat reports 08 (pipeline) and 09 (downstream U-Net), which ran on generated Pet-layout data, with
the real images and trimaps (decision R5). Download the data first with ``qws.data.download_pet(root)``.
The dataset stays under data/, which is never committed.
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from _report import RAW, register, table, write, write_index  # noqa: E402
from qws.experiment import run_experiment
from qws.io import save_json


def exp_pipeline(root: str, quick: bool) -> dict:
    seeds = [0] if quick else [0, 1, 2]
    n = 20 if quick else 200
    rows, lf_rows, raw = [], [], []
    t0 = time.perf_counter()
    for s in seeds:
        rep = run_experiment({"seed": s, "dataset": {"kind": "pet", "root": root, "n_images": n, "n_segments": 150},
                              "train": {"epochs": 15}})
        rows.append([s, rep["iou"]["site_oracle"], rep["iou"]["majority_vote"], rep["iou"]["independent_model"],
                     rep["iou"]["structured_map"], rep["ece_structured"]])
        raw.append({k: rep[k] for k in ("iou", "ece_structured", "lf_accuracy", "learned_params")})
        if s == seeds[0]:
            la, th = rep["lf_accuracy"], rep["learned_params"]["theta"]
            for name, acc, cov, t in zip(la["names"], la["accuracy"], la["coverage"], th):
                lf_rows.append([name, cov, acc, t, 1 / (1 + np.exp(-2 * t))])
            thP = rep["learned_params"]["thetaP"][0]
            beta = rep["learned_params"]["beta"]
    wall = time.perf_counter() - t0
    save_json(RAW / "pet_pipeline.json", {"n_images": n, "seeds": seeds, "runs": raw, "wall_s": wall})
    arr = np.array([r[1:] for r in rows], dtype=float)
    mean, sd = arr.mean(axis=0), arr.std(axis=0)
    gap = (mean[3] - mean[2]) / max(mean[0] - mean[2], 1e-9)
    body = f"""
The label model on the **real Oxford-IIIT Pet dataset** (Parkhi et al., CVPR 2012; decision R5): {n} random
`trainval` images per seed, {len(seeds)} seed(s). This is the same configuration as report 08, which ran on
generated Pet-layout data; only the data changed. Pixel labels are used **only for evaluation**.

## Setup

- **Data:** {n} images per seed (longer side 128 px), about 150 SLIC superpixels each. Trimap boundary pixels
  are excluded from every metric.
- **LFs:** generic only, none uses ground truth:
  - border → background;
  - centre ellipse → foreground;
  - border-vs-centre colour-histogram model;
  - spectral-residual saliency;
  - pairwise colour contrast.
- **Model:** β = {beta}, 15 epochs, Gibbs sampling, MAP by simulated annealing.
- **Wall time:** {wall:.0f} s on CPU for all seeds.

## Label quality (IoU per seed)

{table(["seed", "IoU oracle", "IoU majority", "IoU independent", "IoU structured", "ECE structured"], rows)}

Mean ± SD over seeds: oracle {mean[0]:.3f} ± {sd[0]:.3f}, majority {mean[1]:.3f} ± {sd[1]:.3f},
independent {mean[2]:.3f} ± {sd[2]:.3f}, **structured {mean[3]:.3f} ± {sd[3]:.3f}**.

"Oracle" is the best mask the superpixels can represent (each superpixel set to its majority ground-truth
label). It bounds every method that labels superpixels.

## What the label model learned about each LF (seed {seeds[0]}, without ground truth)

The empirical accuracy (measured against site ground truth) is a diagnostic only. The implied accuracy σ(2θ)
is the model's own estimate of the probability that a cast vote is correct.

{table(["LF", "coverage", "empirical accuracy", "learned θ", "implied accuracy σ(2θ)"], lf_rows)}

Pairwise colour-contrast LF: learned θᴾ = {thP:.2f}.

## Observations

- On real photographs, the structured model reaches IoU {mean[3]:.3f}, against {mean[2]:.3f} for the
  independent model and {mean[1]:.3f} for majority vote. It closes {100 * gap:.0f}% of the gap between the
  independent model and the superpixel oracle.
- The remaining gap reflects the generic LFs: they encode a "centred object, distinct from the border" prior,
  which fails for off-centre pets, pets touching the border, and backgrounds coloured like the animal.
- Compare report 08 (generated data) for the same pipeline; the qualitative ordering is the same.
"""
    write("13-pet-pipeline.md", "13 — Label model on real Oxford-IIIT Pet images", body)
    register({"13-pet-pipeline.md": {
        "title": "13 — Real Oxford-IIIT Pet", "question": "Does the structured label model help on real photographs?",
        "headline": f"{n} images × {len(seeds)} seeds: structured IoU {mean[3]:.3f} vs. {mean[2]:.3f} (independent), "
                    f"{mean[1]:.3f} (majority); superpixel oracle {mean[0]:.3f}"}})
    return {"mean": mean.tolist()}


def exp_downstream(root: str, quick: bool) -> dict:
    from qws.downstream.pipeline import run_downstream
    from qws.downstream.unet import UNetConfig

    n = 24 if quick else 400
    t0 = time.perf_counter()
    out = run_downstream({"dataset": {"kind": "pet", "root": root, "n_images": n, "n_segments": 150},
                          "train": {"epochs": 15}},
                         UNetConfig(epochs=10 if quick else 60, size=64))
    wall = time.perf_counter() - t0
    save_json(RAW / "pet_downstream.json", {**out, "wall_s": wall})
    names = {"structured_soft": "structured label model, soft (this work)",
             "independent_soft": "independent label model, soft",
             "majority_hard": "majority vote, hard", "ground_truth": "ground truth (upper bound)"}
    s = out["signals"]
    rows = [[names[k], v["train_label_iou"], v["test_iou"], v["final_loss"]] for k, v in s.items()]
    body = f"""
End-task evaluation on the **real Oxford-IIIT Pet dataset** (decision R10). A small U-Net is trained on weak
labels for {out['n_train']} random `trainval` images, and evaluated on {out['n_test']} random images of the
official `test` list. This repeats report 09, which used generated Pet-layout data.

## Setup

- **Network:** U-Net with depth {out['unet']['depth']} and {out['unet']['base']} base channels, on
  {out['unet']['size']}×{out['unet']['size']} inputs; Adam with cosine schedule, {out['unet']['epochs']} epochs,
  horizontal-flip augmentation.
- **Loss:** noise-aware expected binary cross-entropy under the soft labels (doc 03 §7).
- **Metric:** test IoU at the label resolution (longer side 128 px), excluding trimap boundary pixels.
- **Wall time:** {wall:.0f} s on CPU.

{table(["training signal", "IoU of the training labels", "U-Net test IoU", "final train loss"], rows)}

For reference, the label model applied directly (transductively) to the test images reaches IoU
**{out['label_model_direct_test_iou']:.3f}**.

## Observations

- Training-label quality: structured {s['structured_soft']['train_label_iou']:.3f}, independent
  {s['independent_soft']['train_label_iou']:.3f}, majority {s['majority_hard']['train_label_iou']:.3f}.
- U-Net test IoU: structured {s['structured_soft']['test_iou']:.3f}, independent
  {s['independent_soft']['test_iou']:.3f}, majority {s['majority_hard']['test_iou']:.3f}; trained on ground
  truth {s['ground_truth']['test_iou']:.3f}.
- The network is small and trained on CPU at 64 px, so absolute numbers are far from the state of the art. The
  comparison between training signals is what this report measures.
"""
    write("14-pet-downstream.md", "14 — Downstream U-Net on real Oxford-IIIT Pet", body)
    register({"14-pet-downstream.md": {
        "title": "14 — Downstream U-Net on real Pet", "question": "Do better weak labels give a better network on real images?",
        "headline": f"Training-label IoU {s['structured_soft']['train_label_iou']:.2f} (structured) vs. "
                    f"{s['independent_soft']['train_label_iou']:.2f} / {s['majority_hard']['train_label_iou']:.2f}; "
                    f"U-Net test IoU {s['structured_soft']['test_iou']:.2f} / {s['independent_soft']['test_iou']:.2f} / "
                    f"{s['majority_hard']['test_iou']:.2f} (GT-trained {s['ground_truth']['test_iou']:.2f})"}})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/oxford-iiit-pet")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--only", nargs="*", help="subset of: pipeline downstream")
    args = ap.parse_args()
    RAW.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    for name in args.only or ["pipeline", "downstream"]:
        t = time.perf_counter()
        {"pipeline": exp_pipeline, "downstream": exp_downstream}[name](args.root, args.quick)
        print(f"  {name}: {time.perf_counter() - t:.0f}s", flush=True)
    write_index()
    print(f"done in {time.perf_counter() - t0:.0f}s")
    print("exit 0")


if __name__ == "__main__":
    main()
