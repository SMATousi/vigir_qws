"""Ephemeral-gully weak segmentation experiment -> results/12-gully.md (spec doc 11).

Usage:  python scripts/gully_experiment.py [--root data/gully] [--quick]
Needs the dataset (qws.data.gully.download_gully) and cached terrain (scripts/gully_fetch_terrain.py).

Protocol (test labels are used only once, for the final evaluation):
1. Spatially blocked split into dev and test (whole ~2 km clusters; stratified).
2. Model selection on dev only, by 3-fold cross-validation grouped by geographic cluster:
   the LF set (all LFs vs. core LFs without the permanence/vegetation LFs), beta and tau
   are chosen by the mean validation AUC of the largest-MAP-component score. Stage 1
   (exact independent MIL label model) is trained on the training folds.
3. Final stage 1 on all of dev with the selected configuration.
4. Test locations: MAP + marginals; location-level AUC / best F1 for every score, overall
   and per region (Iowa vs. Missouri/Illinois).
5. Ablations on test (MAP-based), each changing one component of the selected model.
"""

from __future__ import annotations

import argparse
import copy
import time
from dataclasses import replace

import numpy as np

from _report import RAW, OUT, register, table, write, write_index  # noqa: E402
from qws.data.gully import make_split
from qws.gully.graph import spatiotemporal_sites
from qws.gully.lfs import (
    ConvergenceLF, FlowPathLF, LineOnFlowLF, NonCropLF, StraightLineLF, ThinLineLF, default_gully_lfs,
)
from qws.gully.pipeline import GullyConfig, evaluate_locations, infer, prepare, warm_start
from qws.io import save_json
from qws.model import posterior_ising
from qws.solvers import GraphCutSolver


def set_tau(items, tau: float) -> None:
    """Replace the temporal edge weights (the last (T-1)*H*W edges) in place."""
    for it in items:
        T, H, W = it.loc.images.shape[:3]
        w = it.sites.weights.copy()
        w[-(T - 1) * H * W:] = tau
        it.sites = replace(it.sites, weights=w)


def map_scores(items, params) -> list:
    """MAP-only predictions (fast) for tuning and ablations."""
    from qws.gully.pipeline import Prediction, _stack, location_scores

    gc = GraphCutSolver()
    out = []
    for it in items:
        y = gc.minimize([posterior_ising(params, it.lfo, it.sites)])[0].best()[0]
        pred = Prediction(it.tile, _stack(it, y > 0), None, _stack(it, y > 0), _stack(it, y > 0))
        pred.scores = {k: v for k, v in location_scores(pred, it).items() if k.startswith("structured")}
        out.append(pred)
    return out


def auc_of(preds, items, key="structured_largest_cc"):
    ev = evaluate_locations(preds, items)
    return ev[key]["auc"], ev.get("structured_frac", {}).get("auc", float("nan"))


def figure(test_items, preds, path, n_pos=3, n_neg=2):
    """Small example figure: image / marginal overlay for selected test locations (cited dataset)."""
    from skimage.io import imsave

    byscore = sorted(zip(preds, test_items), key=lambda pi: -pi[0].scores["structured_largest_cc"])
    pos = [pi for pi in byscore if pi[1].loc.label == 1][:n_pos]
    neg = [pi for pi in byscore if pi[1].loc.label == 0][:n_neg]
    rows = []
    for pred, it in pos + neg:
        top, bot = [], []
        for t in range(6):
            img = it.loc.images[t]
            over = img.copy()
            over[pred.marginal[t] > 0.5] = [1, 0, 0]
            top.append(np.pad(img, ((1, 1), (1, 1), (0, 0)), constant_values=1))
            bot.append(np.pad(over, ((1, 1), (1, 1), (0, 0)), constant_values=1))
        rows += [np.concatenate(top, 1), np.concatenate(bot, 1)]
    path.parent.mkdir(parents=True, exist_ok=True)
    imsave(path, (np.concatenate(rows, 0) * 255).astype(np.uint8), check_contrast=False)
    return [(it.tile, it.loc.label) for _, it in pos + neg]


ALL_LFS = [lf.name for lf in default_gully_lfs(True)]
CORE = [n for n in ALL_LFS if n not in ("persistent_vegetation", "permanent_line")]
TERRAIN_LFS = {"flow_path", "convergence", "line_on_flow"}


def subset(items, names):
    """Items restricted to a subset of LF columns (votes are cached once for all LFs)."""
    from qws.gully.pipeline import GullyItem
    from qws.lf import LFOutputs

    cols = [items[0].lfo.unary_names.index(n) for n in names]
    return [GullyItem(it.loc, it.terrain, it.sites, LFOutputs(it.lfo.L[:, cols], it.lfo.P, tuple(names)))
            for it in items]


def nonempty_fraction(preds):
    return float(np.mean([p.map_mask.any() for p in preds]))


def main() -> None:
    from qws.data.gully import spatial_clusters

    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/gully")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    t_all = time.perf_counter()
    split = make_split(args.root, seed=0)
    dev_t, test_t = split["dev"], split["test"]
    if args.quick:
        dev_t, test_t = dev_t[::3], test_t[::3]
    cfg = GullyConfig(root=args.root, warm_iters=150 if args.quick else 400)
    t0 = time.perf_counter()
    dev_all, test_all = prepare(dev_t, cfg), prepare(test_t, cfg)
    t_prep = time.perf_counter() - t0
    variants = {"all LFs": ALL_LFS, "core LFs (no permanence/vegetation)": CORE}

    # ---- grouped 3-fold CV on dev for (LF set, beta, tau)
    clusters = spatial_clusters({it.tile: it.loc.bounds for it in dev_all})
    cids = np.random.default_rng(0).permutation(sorted(set(clusters.values())))
    fold_of = {c: k % 3 for k, c in enumerate(cids)}
    folds = [[i for i, it in enumerate(dev_all) if fold_of[clusters[it.tile]] == f] for f in range(3)]
    cv = []
    t0 = time.perf_counter()
    for vname, names in variants.items():
        dv = subset(dev_all, names)
        for f in range(3):
            tr = [dv[i] for i in range(len(dv)) if i not in set(folds[f])]
            va = subset([dev_all[i] for i in folds[f]], names)
            w = warm_start(tr, cfg)
            for tau in (0.25, 1.0):
                set_tau(va, tau)
                for beta in (0.0, 0.5, 1.0, 2.0):
                    p = w.copy()
                    p.beta = beta
                    a_cc, a_frac = auc_of(map_scores(va, p), va)
                    cv.append({"variant": vname, "fold": f, "beta": beta, "tau": tau, "auc_cc": a_cc, "auc_frac": a_frac})
    t_cv = time.perf_counter() - t0
    keys = sorted({(r["variant"], r["beta"], r["tau"]) for r in cv})
    cv_mean = {k: float(np.nanmean([r["auc_cc"] for r in cv if (r["variant"], r["beta"], r["tau"]) == k])) for k in keys}
    best_variant, best_beta, best_tau = max(cv_mean, key=cv_mean.get)
    cfg = replace(cfg, beta=best_beta, tau=best_tau)
    names = variants[best_variant]

    # ---- final stage 1 on all of dev, test inference
    dev, test = subset(dev_all, names), subset(test_all, names)
    set_tau(dev, cfg.tau)
    set_tau(test, cfg.tau)
    t0 = time.perf_counter()
    warm = warm_start(dev, cfg)
    t_stage1 = time.perf_counter() - t0
    params = warm.copy()
    params.beta = cfg.beta
    # MIL diagnostics: expected EG pixels per positive dev location under stage 1, and P(no EG pixel)
    exp_pos, log_none = [], []
    for it in dev:
        if it.loc.label != 1:
            continue
        h = warm.b + it.lfo.L.astype(float) @ warm.theta + (it.lfo.L != 0).astype(float) @ warm.omega
        exp_pos.append(float(np.sum(0.5 * (1 + np.tanh(h)))))
        log_none.append(float(np.sum(-np.logaddexp(0.0, 2.0 * h))))
    mil_diag = {"median_expected_eg_pixels": float(np.median(exp_pos)), "max_log10_p_none": float(np.max(log_none) / np.log(10))}
    fold_sd = float(np.mean([np.std([r["auc_cc"] for r in cv if (r["variant"], r["beta"], r["tau"]) == k]) for k in keys]))
    t0 = time.perf_counter()
    preds = infer(test, params, warm, cfg)
    t_infer = time.perf_counter() - t0
    ev = evaluate_locations(preds, test)
    south = np.array([it.loc.bounds[1] < 41 for it in test])
    region = {}
    for rname, mask in (("Iowa", ~south), ("Missouri/Illinois", south)):
        sel = [k for k in range(len(test)) if mask[k]]
        evr = evaluate_locations([preds[k] for k in sel], [test[k] for k in sel])
        region[rname] = {"n": len(sel), "pos": int(sum(test[k].loc.label for k in sel)),
                         **{k: evr[k]["auc"] for k in ("structured_largest_cc", "independent_frac", "majority_frac",
                                                       "lf_line_on_flow_pos_frac", "terrain_max_log_acc") if k in evr}}
    fig_tiles = figure(test, preds, OUT / "figures" / "gully_examples.png")

    # ---- ablations on test (one change each, MAP-based)
    abl = {}

    def run(label, items, prm):
        ps = map_scores(items, prm)
        a = auc_of(ps, items)
        abl[label] = (a[0], a[1], nonempty_fraction(ps))

    run(f"selected model ({best_variant}, β={cfg.beta}, τ={cfg.tau})", test, params)
    other = [v for v in variants if v != best_variant][0]
    d2, t2 = subset(dev_all, variants[other]), subset(test_all, variants[other])
    set_tau(d2, cfg.tau)
    set_tau(t2, cfg.tau)
    w2 = warm_start(d2, cfg)
    w2.beta = cfg.beta
    run(f"other LF set: {other}", t2, w2)
    img_names = [n for n in names if n not in TERRAIN_LFS]
    d2, t2 = subset(dev_all, img_names), subset(test_all, img_names)
    set_tau(d2, cfg.tau)
    set_tau(t2, cfg.tau)
    w2 = warm_start(d2, cfg)
    w2.beta = cfg.beta
    run("imagery-only LFs (no terrain)", t2, w2)
    del d2, t2
    iso = [copy.copy(it) for it in test]
    for it in iso:
        it.sites = spatiotemporal_sites(it.loc.images, cfg.tau, orientation=False)
    run("isotropic couplings (no orientation)", iso, params)
    del iso
    set_tau(test, 0.0)
    run("no temporal coupling (τ = 0)", test, params)
    set_tau(test, cfg.tau)
    p0 = warm.copy()
    p0.beta = 0.0
    run("independent model (β = 0)", test, p0)
    for label, kw in (("symmetric LFs (no class-conditional ω, ζ)", {"class_conditional": False}),
                      ("no MIL constraint", {"mil": False})):
        w2 = warm_start(dev, replace(cfg, **kw))
        w2.beta = cfg.beta
        run(label, test, w2)

    raw = {"split": {k: (v if not isinstance(v, list) else len(v)) for k, v in split.items()}, "config": vars(cfg),
           "selected": {"variant": best_variant, "beta": best_beta, "tau": best_tau}, "lf_names": names,
           "stage1_params": warm.to_dict(), "cv": cv, "cv_mean": {str(k): v for k, v in cv_mean.items()},
           "test_eval": ev, "test_by_region": region, "mil_diagnostics": mil_diag, "cv_fold_sd": fold_sd,
           "ablations": {k: {"auc_largest_cc": v[0], "auc_frac": v[1], "nonempty_map_fraction": v[2]} for k, v in abl.items()},
           "figure_tiles": fig_tiles, "timing_s": {"prepare": t_prep, "cv": t_cv, "stage1": t_stage1,
                                                   "test_inference": t_infer, "total": time.perf_counter() - t_all}}
    save_json(RAW / "gully.json", raw)

    lf_rows = [[n, warm.alpha[j], warm.theta[j], warm.omega[j], warm.zeta[j],
                float(np.mean([np.mean(it.lfo.L[:, j] != 0) for it in dev]))] for j, n in enumerate(names)]
    order = sorted(ev.items(), key=lambda kv: -kv[1]["auc"])
    ev_rows = [[k, v["auc"], v["best_f1"], v["n"]] for k, v in order]
    cv_rows = [[v, b_, t_, cv_mean[(v, b_, t_)]] for (v, b_, t_) in sorted(cv_mean, key=cv_mean.get, reverse=True)]
    abl_rows = [[k, v[0], v[1], v[2]] for k, v in abl.items()]
    reg_rows = [[r, d["n"], d["pos"], d.get("structured_largest_cc", float("nan")), d.get("independent_frac", float("nan")),
                 d.get("majority_frac", float("nan")), d.get("lf_line_on_flow_pos_frac", float("nan")),
                 d.get("terrain_max_log_acc", float("nan"))] for r, d in region.items()]
    pos_rate = np.mean([it.loc.label for it in test])
    s_cc = ev["structured_largest_cc"]
    body = f"""
**First results of the ephemeral-gully (EG) branch** ([spec/11](../spec/11-gully-detection.md)), on the ViGIR EG
dataset (Tousi et al., WACV 2025; © ViGIR, used with permission and cited). The dataset has only
**location-level** labels: an EG is present at this location in some year, or not. There is no pixel ground
truth yet, so segmentation is evaluated indirectly, by how well each location's predicted masks detect the
labelled locations. Pixel metrics will follow once annotations exist (decision G9).

## Setup

- **Data:** 621 locations × 6 NAIP acquisitions (128 × 128 RGB, about 1 m/px), plus USGS 3DEP 1 m LiDAR terrain
  for every location.
- **Split:** spatially blocked over {split['n_clusters']} geographic clusters of about 2 km. Dev has
  {len(dev_t)} locations ({sum(it.loc.label for it in dev)} positive); test has {len(test_t)}
  ({int(round(pos_rate * len(test_t)))} positive). No cluster is shared between them.
- **Sites and couplings:** all pixels of all 6 years (98,304 per location); orientation-aware 8-neighbour
  spatial couplings plus temporal couplings.
- **Stage 1** (exact): an independent class-conditional label model. Dev negatives are clamped to background,
  and dev positives carry the exact "at least one EG pixel" constraint. Training on all dev took
  {t_stage1:.0f} s.
- **Stage 2:** the structured posterior. MAP by graph cut (exact; the posterior is ferromagnetic); marginals
  by Gibbs sampling started at the MAP state. Test inference took {t_infer:.0f} s.
- **Model selection on dev only:** 3-fold cross-validation grouped by geographic cluster ({t_cv:.0f} s),
  choosing the LF set, β and τ. Selected: **{best_variant}, β = {best_beta}, τ = {best_tau}**.

## Location-level detection on test ({len(test_t)} locations; used only for this evaluation)

{table(["score", "AUC", "best F1", "n"], ev_rows)}

Best F1 is optimistic, because its threshold is chosen on test; read it next to AUC. A constant "all positive"
predictor has F1 = {2 * pos_rate / (1 + pos_rate):.3f}.

## By region (test AUC)

{table(["region", "n", "positives", "structured largest component", "independent", "majority vote", "LF line_on_flow", "terrain alone"], reg_rows)}

## Model selection: dev cross-validation (mean validation AUC of the largest MAP component)

{table(["LF set", "β", "τ", "CV AUC"], cv_rows)}

## Ablations on test (MAP-based; one change at a time from the selected model)

{table(["variant", "AUC (largest component)", "AUC (EG fraction)", "locations with non-empty MAP"], abl_rows)}

An AUC of exactly 0.5 with a non-empty fraction of 0 means the MAP mask is empty at every location: the evidence
never overcomes the learned prior that EG pixels are rare. That variant cannot segment at all.

## What stage 1 learned about each LF (all of dev, without pixel labels)

The conditional accuracy of a cast vote is σ(2θ). ω > 0 means the LF votes more often on EG pixels; ζ is the
vote bias.

{table(["LF", "α", "θ", "ω", "ζ", "coverage"], lf_rows)}

Learned class bias b = {warm.b:.2f}. This is consistent with EG pixels being rare.

## Examples (test locations; top: image per year, bottom: P(EG) > 0.5 in red)

![examples](figures/gully_examples.png)

Locations shown, as (tile, label): {fig_tiles}. These are the three highest-scoring positive and the two
highest-scoring negative test locations.

## Notes

- **Model selection noise.** The mean fold-to-fold standard deviation of the CV AUC is {fold_sd:.3f}. Configurations
  whose CV AUCs differ by less than that are not reliably distinguished. In particular, the effect of β (spatial
  smoothing) on *location detection* is not resolved by this data. Its effect on mask shape (continuity across
  gaps and years) can only be measured against pixel annotations.
- **One dev fold is below chance.** Every configuration scores an AUC between 0.26 and 0.32 on one of the three dev folds.
  In that geographic group, the LF evidence ranks negative locations above positive ones. This is the main source
  of the fold-to-fold spread, and the clearest sign that the LFs do not transfer equally across landscapes.
- **The multiple-instance constraint is vacuous at this scale.** Under stage 1, a positive dev location is expected
  to contain a median of {mil_diag['median_expected_eg_pixels']:.0f} EG pixels, and P(no EG pixel at all) ≤
  10^{mil_diag['max_log10_p_none']:.0f}. The constraint "at least one EG pixel among 98,304" therefore barely
  changes the posterior; compare the "no MIL constraint" ablation. An informative location-level constraint would
  need to be stronger, for example "at least one connected component of length ≥ k" or a calibrated count
  likelihood (open question Q20).
- The regions differ markedly: training-free single-LF AUCs are much lower on dev than on test. Cross-validation
  numbers on dev are therefore lower than the test numbers, and they should not be compared directly.
- **What the examples show** (visual inspection, not a metric).
  - The top positives are channels along the DEM flow path. Some of them persist in all six years and look like
    grassed waterways rather than bare ephemeral gullies, which location labels cannot tell apart.
  - The highest-scoring negative (tile 493) has one blank (all-black, no-data) acquisition, and the model marks
    pixels in it. No-data years should be masked out before LFs vote.
- **Pixel-level quality is not measured yet.** Pixel IoU and ±2 px tolerant precision and recall are computed
  automatically once annotations exist (`qws.gully.annotate`; `scripts/gully_export_annotation.py`).
"""
    write("12-gully.md", "12 — Ephemeral-gully weak segmentation (ViGIR EG dataset)", body)
    register({"12-gully.md": {
        "title": "12 — Ephemeral gullies", "question": "Can weak LFs + location labels segment EGs and detect EG locations?",
        "headline": f"Test location AUC {s_cc['auc']:.3f} (structured largest component) vs. "
                    f"{ev['independent_frac']['auc']:.3f} (independent) / {ev['majority_frac']['auc']:.3f} (majority) / "
                    f"{ev.get('terrain_max_log_acc', {}).get('auc', float('nan')):.3f} (terrain alone); model chosen by "
                    f"grouped CV on dev; no pixel GT yet"}})
    write_index()
    print(f"done in {time.perf_counter() - t_all:.0f}s")


if __name__ == "__main__":
    main()
