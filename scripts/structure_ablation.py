"""Report 16: does learning *with* spatial structure beat learning independently and smoothing afterwards?

Usage:  python scripts/structure_ablation.py [--pet-root data/oxford-iiit-pet] [--quick]

Four variants on the same images and LF votes (spec doc 06, question Q1; paper Section "Learning with
structure"):

* independent      learn at beta = 0, infer at beta = 0 (Snorkel-style label model, per site)
* learn-then-smooth  learn at beta = 0 (independent model), then MAP with smoothness beta (unary LFs only)
* joint (unary)    learn the unary LF parameters *with* smoothness beta, MAP with beta (unary LFs only)
* joint (full)     as joint, plus the pairwise LF in learning and inference (the paper's model)

independent -> learn-then-smooth isolates smoothing at inference; learn-then-smooth -> joint (unary) isolates
joint learning (the question); joint (unary) -> joint (full) isolates the pairwise LF. Each is run for several
beta values, so no variant is favoured by a beta tuned for it. Ground truth is used only for evaluation.
MAP is exact (graph cut) for the unary-only variants, whose posteriors are ferromagnetic, and simulated annealing
for the full model, whose pairwise LF makes it frustrated.
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from _report import RAW, register, table, write, write_index  # noqa: E402
from qws.eval import iou
from qws.experiment import DEFAULT_CONFIG, build_dataset, merge
from qws.io import save_json
from qws.learn import TrainConfig, Trainer, infer_map
from qws.lf import LFOutputs
from qws.model import independent_posterior, warm_start
from qws.solvers import GibbsSampler, GraphCutSolver, SimulatedAnnealing

VARIANTS = ["independent", "learn-then-smooth", "joint (unary)", "joint (full)"]


def sigma2(t):
    return 1.0 / (1.0 + np.exp(-2.0 * np.asarray(t, dtype=float)))


def run_one(items, beta: float, seed: int, epochs: int, samples: int, ignore: bool, true_theta=None) -> dict:
    full = [(im.lfo, im.sites) for im in items]
    unary = [(LFOutputs(lfo.L, np.zeros((s.m, 0), np.int8), lfo.unary_names), s) for lfo, s in full]

    def score(masks):
        return float(np.mean([iou(im.sites.to_pixels(m), im.mask, ignore) for m, im in zip(masks, items)]))

    gc, sa = GraphCutSolver(), SimulatedAnnealing(num_sweeps=300, seed=seed)
    out = {}
    indep = warm_start(unary, beta=0.0)
    out["independent"] = (score([np.where(independent_posterior(indep, lfo).mean > 0, 1, -1) for lfo, _ in unary]),
                          indep.theta.copy())
    smooth = indep.copy()
    smooth.beta = beta
    out["learn-then-smooth"] = (score(infer_map(smooth, unary, gc)), indep.theta.copy())
    cfg = TrainConfig(epochs=epochs, num_samples=samples, beta=beta, seed=seed)
    ju = Trainer(GibbsSampler(num_sweeps=100, seed=seed), cfg).fit(unary, warm_start(unary, beta=beta)).params
    out["joint (unary)"] = (score(infer_map(ju, unary, gc)), ju.theta.copy())
    jf = Trainer(GibbsSampler(num_sweeps=100, seed=seed), cfg).fit(full, warm_start(full, beta=beta)).params
    out["joint (full)"] = (score(infer_map(jf, full, sa)), jf.theta.copy())
    res = {"iou": {k: v[0] for k, v in out.items()}, "theta": {k: v[1].tolist() for k, v in out.items()}}
    if true_theta is not None:
        res["acc_err"] = {k: float(np.abs(sigma2(v[1]) - sigma2(true_theta)).mean()) for k, v in out.items()}
    return res


def experiment(name: str, cfg_ds: dict, seeds, betas, epochs, samples, ignore: bool) -> dict:
    runs = []
    for s in seeds:
        items, true = build_dataset(merge(DEFAULT_CONFIG, {"seed": s, "dataset": cfg_ds}))
        for b in betas:
            t = time.perf_counter()
            r = run_one(items, b, s, epochs, samples, ignore, None if true is None else true.theta)
            runs.append({"seed": s, "beta": b, **r})
            print(name, s, b, {k: round(v, 3) for k, v in r["iou"].items()}, f"{time.perf_counter() - t:.0f}s",
                  flush=True)
    rows = []
    for b in betas:
        rr = [r for r in runs if r["beta"] == b]
        row = [f"{b:g}"]
        for v in VARIANTS:
            x = [r["iou"][v] for r in rr]
            row.append(f"{np.mean(x):.3f} ± {np.std(x):.3f}")
        gain = [r["iou"]["joint (unary)"] - r["iou"]["learn-then-smooth"] for r in rr]
        row.append(f"{np.mean(gain):+.3f} ({sum(g > 0 for g in gain)}/{len(gain)})")
        rows.append(row)
    acc_rows = []
    if "acc_err" in runs[0]:
        for b in betas:
            rr = [r for r in runs if r["beta"] == b]
            acc_rows.append([f"{b:g}"] + [float(np.mean([r["acc_err"][v] for r in rr])) for v in VARIANTS])
    return {"runs": runs, "rows": rows, "acc_rows": acc_rows, "seeds": list(seeds), "betas": list(betas)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pet-root", default="data/oxford-iiit-pet")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    t0 = time.perf_counter()
    q = args.quick
    betas = [0.5] if q else [0.25, 0.5, 1.0]
    syn = experiment("synthetic", {"kind": "synthetic"}, [0] if q else [0, 1, 2, 3, 4], betas,
                     5 if q else 20, 50 if q else 100, False)
    weak = experiment("synthetic-weak", {"kind": "synthetic", "true_params": {
        "theta": [0.5, 0.3, 0.15, 0.4], "alpha": [0.5, 0.0, -0.5, 0.2], "thetaP": [0.5], "alphaP": [0.0]}},
        [0] if q else [0, 1, 2, 3, 4], betas, 5 if q else 20, 50 if q else 100, False)
    pet = experiment("pet", {"kind": "pet", "root": args.pet_root, "n_images": 20 if q else 200, "n_segments": 150},
                     [0] if q else [0, 1, 2], betas, 5 if q else 15, 50 if q else 100, True)
    save_json(RAW / "structure_ablation.json", {"synthetic": syn, "synthetic_weak": weak, "pet": pet,
                                                "wall_s": time.perf_counter() - t0})
    hdr = ["β", *VARIANTS, "joint − smooth (wins)"]
    acc_hdr = ["β", *[f"{v}" for v in VARIANTS]]
    body = f"""
Does learning the LF parameters *with* the spatial prior beat the simpler pipeline of learning them
independently (as Snorkel does) and smoothing the masks afterwards? Four variants share the same images, LF
votes and smoothness β:

| variant | learning | inference |
|---|---|---|
| independent | β = 0 | β = 0 |
| learn-then-smooth | β = 0 | MAP with β, unary LFs |
| joint (unary) | with β | MAP with β, unary LFs |
| joint (full) | with β and the pairwise LF | MAP with β and the pairwise LF |

independent → learn-then-smooth is the effect of smoothing at inference. **learn-then-smooth → joint (unary)
is the effect of joint learning**, the question of this report. joint (unary) → joint (full) is the effect of the
pairwise LF. Each β is reported, so no variant benefits from a β tuned for it. Ground truth is used only for
evaluation. MAP is exact (graph cut) for the unary-only variants and simulated annealing for the full model.
The last column is the mean IoU gain of joint (unary) over learn-then-smooth, and how many runs it wins.

## Synthetic, report 01 setting (votes from the model's law; θ* = [1.0, 0.6, 0.3, 0.8], θᴾ* = 1.0)

16×16 grids, 8 images, seeds {syn['seeds']}.

{table(hdr, syn['rows'])}

Mean absolute error of the implied accuracies σ(2θ) of the unary LFs:

{table(acc_hdr, syn['acc_rows'])}

## Synthetic, weak LFs (θ* = [0.5, 0.3, 0.15, 0.4], θᴾ* = 0.5)

Same setting with weaker LFs, so that the posterior is uncertain. Seeds {weak['seeds']}.

{table(hdr, weak['rows'])}

Mean absolute error of the implied accuracies σ(2θ):

{table(acc_hdr, weak['acc_rows'])}

## Real images: Oxford-IIIT Pet

{20 if q else 200} random `trainval` images per seed, about 150 SLIC superpixels each, the generic LFs of report 13;
trimap boundary pixels excluded. Seeds {pet['seeds']}.

{table(hdr, pet['rows'])}

## Observations

OBSERVATIONS_PLACEHOLDER
"""
    write("16-structure-ablation.md", "16 — Joint structured learning vs. learn-then-smooth", body)
    p05 = {r[0]: r for r in pet["rows"]}["0.5"]
    register({"16-structure-ablation.md": {
        "title": "16 — Joint learning vs. learn-then-smooth",
        "question": "Does learning with the spatial prior beat learning independently and smoothing afterwards?",
        "headline": f"Pet, β = 0.5: independent {p05[1]}, learn-then-smooth {p05[2]}, joint (unary) {p05[3]}, "
                    f"joint (full) {p05[4]}"}})
    write_index()
    print(f"done in {time.perf_counter() - t0:.0f}s")
    print("exit 0")


if __name__ == "__main__":
    main()
