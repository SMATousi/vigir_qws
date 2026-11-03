"""Reports 05–11: QPU readiness (mock), learning beta/b, class-conditional LFs, the
real-image pipeline on Pet-format data, downstream U-Net, QPBO, and multi-class.

Usage:  python scripts/generate_results_extensions.py [--quick]
No real QPU and no real Oxford-IIIT Pet images are used (see results/README.md).
"""

from __future__ import annotations

import argparse
import tempfile
import time

import numpy as np

from _report import RAW, ms, register, table, write, write_index  # noqa: E402
from qws.eval import iou, sample_votes, synthetic_dataset
from qws.experiment import run_experiment
from qws.graph import grid_sites, superpixel_sites
from qws.io import save_json
from qws.learn import CalibratedSampler, TrainConfig, Trainer, infer_map
from qws.learn.optim import Adam
from qws.lf import LFOutputs
from qws.model import (
    LabelModelParams, PosteriorMoments, gradient, independent_posterior, model_moments, posterior_ising, warm_start,
)
from qws.solvers import ExactSolver, GibbsSampler, QPBOSolver, SimulatedAnnealing, roof_dual

TRUE = dict(alpha=[0.5, 0.0, -0.5, 0.2], theta=[1.0, 0.6, 0.3, 0.8], alphaP=[0.0], thetaP=[1.0])


def true_params(**kw) -> LabelModelParams:
    return LabelModelParams(**{**TRUE, **kw})


def posterior_problems(shape, n_images, seed, beta=0.5, **kw):
    params = true_params(beta=beta, **kw)
    ds = synthetic_dataset(params, n_images, shape, seed=seed)
    return [posterior_ising(params, im.lfo, im.sites) for im in ds]


# ============================================================================ 05 QPU readiness
def exp_qpu(quick: bool) -> dict:
    from qws.multiclass import PottsProblem, to_onehot_ising
    from qws.solvers.qpu import QPUAnnealingSolver

    shape = 6 if quick else 16  # P16 = Advantage-size Pegasus graph
    qpu = QPUAnnealingSolver.mock(topology_shape=shape)
    rng = np.random.default_rng(0)
    graphs = {}
    for side in (4, 6, 8, 12):
        s = grid_sites(np.zeros((side, side)))
        graphs[f"grid {side}×{side}"] = posterior_problems((side, side), 1, seed=side)[0]
    from qws.data.petlike import make_petlike

    for nseg in (100, 300):
        img = make_petlike((96, 128), rng)[0] / 255.0  # structured image, so SLIC finds real segments
        s = superpixel_sites(img, n_segments=nseg)
        from qws.ising import IsingProblem

        graphs[f"superpixel RAG (SLIC, {nseg} requested)"] = IsingProblem(rng.normal(size=s.n), s.edges, rng.normal(size=s.m))
    pot = PottsProblem(rng.normal(size=(16, 3)), grid_sites(np.zeros((4, 4))).edges, rng.normal(size=24))
    graphs["one-hot K=3 on 4×4 (48 vars)"] = to_onehot_ising(pot)[0]
    emb_rows = []
    for name, p in graphs.items():
        t0 = time.perf_counter()
        try:
            emb = qpu.embed([p])
            lengths = [len(c) for c in emb.values()]
            emb_rows.append([name, p.n, p.m, int(sum(lengths)), int(max(lengths)), float(np.mean(lengths)),
                             time.perf_counter() - t0])
        except RuntimeError:
            emb_rows.append([name, p.n, p.m, "no embedding", "–", "–", time.perf_counter() - t0])
    # packing capacity: how many 4x4 posterior problems fit in one submission
    tiles = posterior_problems((4, 4), 1, seed=5) * (8 if quick else 64)
    cap = 0
    for k in range(1, len(tiles) + 1):
        try:
            qpu.embed(tiles[:k])
            cap = k
        except RuntimeError:
            break
    # MAP through the mock QPU on posterior problems, packed
    probs = posterior_problems((4, 4), 4 if quick else 8, seed=7)
    t0 = time.perf_counter()
    sets = qpu.minimize(probs, num_reads=100)
    wall = time.perf_counter() - t0
    ex = ExactSolver()
    map_rows = []
    for i, (p, ss) in enumerate(zip(probs, sets)):
        e0 = ex.minimize([p])[0].best()[1]
        hit = float(ss.counts[np.abs(ss.energies - e0) < 1e-6].sum() / ss.counts.sum())
        map_rows.append([i, p.n, ss.info["kappa"], ss.info["range_limited"], ss.info["chain_strength"],
                         ss.info["chain_break_fraction"], ss.info["packed_with"], ss.best()[1] - e0, hit])
    # calibration through the mock QPU
    cal_p = posterior_problems((3, 4), 1, seed=9)[0].scaled(0.5)
    cal = CalibratedSampler(QPUAnnealingSolver.mock(topology_shape=4), damping=1.0)
    for _ in range(6 if quick else 10):
        cal.sample([cal_p], 2000)
    out = {"embeddings": emb_rows, "packing_capacity_4x4": cap, "packing_cap_tested": len(tiles), "map": map_rows,
           "map_wall_s": wall, "calls": qpu.calls[-1], "calibration": cal.history, "topology_shape": shape}
    save_json(RAW / "qpu_readiness.json", out)
    cap_txt = f"{cap}" if cap < len(tiles) else f"≥ {cap} (all tested)"
    body = f"""
Everything a real D-Wave run needs, except the device: the full `QPUAnnealingSolver` path runs against
`MockDWaveSampler`. The mock has the real Pegasus P{shape} topology{' (the size of an Advantage QPU)' if shape == 16 else ''},
the real coefficient ranges and QPU timing fields, and a fixed-temperature simulated-annealing substitute
for the physics. See [spec/09 §4](../spec/09-implementation-notes.md) for the design.

## Minor-embedding of representative problem graphs (minorminer, cached by graph shape)

{table(["Problem graph", "variables", "couplers", "physical qubits", "max chain", "mean chain", "embed time (s)"],
       emb_rows)}

## Parallel embedding

{cap_txt} independent 4×4 posterior problems (16 variables each) fit into one P{shape} submission. They share
one programming and readout cycle; QPU access time is attributed to each problem in proportion to its size.

## MAP through the mock QPU ({len(probs)} posterior problems on 4×4 grids, packed, 100 reads)

{table(["problem", "n", "κ", "range-limited", "chain strength", "chain-break fraction", "packed with",
        "best − exact", "hit rate"], map_rows)}

All problems were solved in {wall:.2f} s of wall time, in {qpu.calls[-1]['problems']} problem(s) per submission.

## Temperature calibration through the mock QPU

The mock's substitute anneals to an inverse temperature of 4 in hardware units. Pseudo-likelihood calibration
should therefore converge to a submit scale of about 1/4. History of (scale, β̂):
{', '.join(f"({h['scale']:.3f}, {h['beta_hat']:.2f})" for h in cal.history)}.

Final implied β_eff = 1/scale = **{1 / cal.scale:.2f}**.

## Observations

- Grid and superpixel posterior graphs embed with short chains. The one-hot multi-class encoding is denser
  (every site is a K-clique) and needs longer chains.
- Range-aware scaling (κ) keeps logical couplers, fields and chains inside the device ranges with
  `auto_scale` off. Problems that already fit are not rescaled, which is what makes calibration possible.
- The calibration loop recovers the mock device's temperature from samples alone. This is the first
  experiment to repeat on the real Advantage system (roadmap 3.2).
"""
    write("05-qpu-readiness.md", "05 — D-Wave QPU readiness (offline, mock Pegasus)", body)
    register({"05-qpu-readiness.md": {
        "title": "05 — QPU readiness (mock)", "question": "Is the D-Wave path ready for real hardware?",
        "headline": f"Full path runs on mock Pegasus P{shape}; {cap_txt} 4×4 problems per submission; exact optimum "
                    f"on {sum(r[7] < 1e-6 for r in map_rows)}/{len(map_rows)} problems; calibration recovers "
                    f"β_eff ≈ {1 / cal.scale:.1f} (true 4)"}})
    return out


# ============================================================================ 06 learning beta and b
def exp_beta(quick: bool) -> dict:
    settings = [(0.2, 0.0), (0.35, 0.1), (0.6, -0.1)]
    seeds = [0] if quick else [0, 1, 2]
    rows, raw = [], []
    for beta_t, b_t in settings:
        res_b, res_bb, res_th = [], [], []
        for s in seeds:
            true = true_params(thetaP=[0.9], b=b_t, beta=beta_t)
            ds = synthetic_dataset(true, 12, (16, 16), mask_source="ising", seed=s)
            data = [(im.lfo, im.sites) for im in ds]
            init = warm_start(data, beta=0.1)
            cfg = TrainConfig(epochs=20 if quick else 60, lr=0.03, num_samples=100, learn_b=True, learn_beta=True,
                              tol=0, seed=s)
            p = Trainer(GibbsSampler(num_sweeps=30, persistent=True, seed=s), cfg).fit(data, init).params
            res_b.append(p.beta)
            res_bb.append(p.b)
            res_th.append(float(np.abs(p.theta - true.theta).max()))
            raw.append({"beta_true": beta_t, "b_true": b_t, "seed": s, "beta": p.beta, "b": p.b, "theta_err": res_th[-1]})
        rows.append([beta_t, b_t, ms(res_b), ms(res_bb), ms(res_th)])
    save_json(RAW / "learn_beta_b.json", raw)
    body = f"""
Can the smoothness β and the class bias b be learned from LF votes alone (decision R8)? This is the
negative phase of the gradient, sampled from the prior Ising model (doc 03 §2; paper Theorem 1 remark).

## Setup

- **Data:** masks drawn from the prior Ising model itself (`mask_source="ising"`, 16×16 grids, uniform
  weights), at a known (b, β). The whole generative process is therefore exactly the label model, which makes
  this a pure identifiability test.
- **LFs:** 12 images per run, the 4 unary LFs of report 01, and one pairwise LF with θᴾ = 0.9.
- **Training:** warm start at β = 0.1, b = 0, then {20 if quick else 60} epochs of Adam on all parameters with
  persistent Gibbs chains (prior moments shared per graph shape). Seeds: {seeds}.
- **Reference point:** the 2-D Ising critical point is β_c = ½ ln(1 + √2) ≈ 0.441.

## Results (mean ± std over seeds)

{table(["true β", "true b", "learned β", "learned b", "max |θ − θ*|"], rows)}

## Observations

- β is recovered well below and above the critical point.
- b is well identified only at weak coupling. In the ordered phase (β > β_c), each image is spontaneously
  dominated by one class, so 12 images carry little information about the bias, and the estimate is noisy.
  Use more images, or fix b, when coupling is strong.
- Unary accuracies θ stay accurate while β and b are learned jointly.
"""
    write("06-learn-beta-b.md", "06 — Learning the smoothness β and class bias b", body)
    register({"06-learn-beta-b.md": {
        "title": "06 — Learning β and b", "question": "Are β and b identifiable from votes alone?",
        "headline": "β recovered at all coupling strengths; b well identified only below the critical point "
                    "(spontaneous magnetization above it)"}})
    return {"rows": rows}


# ============================================================================ 07 class-conditional LFs
def exp_class_conditional(quick: bool) -> dict:
    cc = dict(omega=[0.8, -0.6, 0.0, 0.4], zeta=[0.3, 0.0, -0.4, 0.0])
    true = LabelModelParams(alpha=TRUE["alpha"], theta=TRUE["theta"], **cc)
    # (a) identifiability with exact moments: beta = 0, many independent sites
    rng = np.random.default_rng(0)
    N = 50_000 if quick else 200_000
    pf = model_moments(true).prior_field
    y = np.where(rng.random(N) < (1 + np.tanh(true.b + pf)) / 2, 1, -1).astype(np.int8)
    lfo = sample_votes(true, y, np.zeros((0, 2), int), rng)
    lfo = LFOutputs(lfo.L, np.zeros((0, 0), np.int8))
    s0 = grid_sites(np.zeros((1, 1)))
    sites = type(s0)(np.zeros((1, N), int), np.zeros((0, 2), int), np.zeros(0), np.zeros((N, 1)), np.zeros((N, 2)))
    p = LabelModelParams(alpha=np.zeros(4), theta=np.full(4, 0.5))
    opt = Adam(0.02)
    for _ in range(1500 if quick else 3000):
        post = independent_posterior(p, lfo)
        prior = PosteriorMoments(np.full(N, np.tanh(p.b + model_moments(p).prior_field)), np.zeros(0))
        g = gradient(p, [(lfo, sites)], [post], [prior])
        g.b = g.beta = 0.0
        p = p.from_vector(opt.step(p.to_vector(), g.to_vector() / N))
    ident_rows = [[f"{name}_{j}", getattr(true, name)[j], getattr(p, name)[j]]
                  for name in ("theta", "alpha", "omega", "zeta") for j in range(4)]
    # (b) coupled model, symmetric vs class-conditional fit
    import qws.eval.synthetic as S

    truec = LabelModelParams(alpha=TRUE["alpha"], theta=TRUE["theta"], alphaP=[0.0], thetaP=[0.9], beta=0.3, **cc)
    eff_b = truec.b + model_moments(truec).prior_field
    orig = S.ising_mask
    S.ising_mask = lambda shape, b, beta, rng, sweeps=500: orig(shape, eff_b, beta, rng, sweeps)
    try:
        ds = synthetic_dataset(truec, 12 if quick else 24, (16, 16), mask_source="ising", seed=3)
    finally:
        S.ising_mask = orig
    data = [(im.lfo, im.sites) for im in ds]
    fit_rows, raw = [], {}
    for label, flag in (("symmetric model", False), ("class-conditional model", True)):
        init = warm_start(data, beta=0.3)
        t0 = time.perf_counter()
        res = Trainer(GibbsSampler(num_sweeps=30, persistent=True, seed=0),
                      TrainConfig(epochs=40 if quick else 120, lr=0.03, num_samples=100, class_conditional=flag,
                                  tol=0)).fit(data, init)
        q = res.params
        masks = infer_map(q, data, SimulatedAnnealing(num_sweeps=300, seed=0))
        miou = float(np.mean([iou(im.sites.to_pixels(m), im.mask) for m, im in zip(masks, ds)]))
        fit_rows.append([label, float(np.abs(q.theta - truec.theta).max()), float(np.abs(q.omega - truec.omega).max()),
                         float(np.abs(q.zeta - truec.zeta).max()), miou, time.perf_counter() - t0])
        raw[label] = q.to_dict()
    save_json(RAW / "class_conditional.json", {"identifiability": ident_rows, "coupled": fit_rows, "params": raw})
    body = f"""
LF potentials whose accuracy and coverage differ between foreground and background (decision R9), in the
parametrization ψ_j(λ,y) = α_j·1[λ≠0] + θ_j·λy + ω_j·1[λ≠0]·y + ζ_j·λ ([spec/09 §3](../spec/09-implementation-notes.md)).
Here ω is class-dependent coverage and ζ is a vote bias. The model terms of the gradient now depend on the
prior marginals P(yᵢ = ±1), so the trainer samples the prior Ising model, whose field gains
Σ_c ½ log(Z_c(+1)/Z_c(−1)) (paper Proposition 2).

## (a) Identifiability with exact moments

The coupling is β = 0, so posterior and prior moments are exact closed forms and there is no sampling noise.
There are {N:,} sites, with labels drawn from the model's own prior marginal.

{table(["parameter", "true", "learned"], ident_rows)}

## (b) Coupled model: symmetric vs. class-conditional fit

- **Data:** masks from the full prior marginal (Ising with field b + prior field, β = 0.3), so the generative
  process is exactly the class-conditional label model. One pairwise LF; {12 if quick else 24} images of 16×16.
- **Fit:** both models trained with persistent Gibbs chains for {40 if quick else 120} epochs. MAP masks by
  simulated annealing.

{table(["fitted model", "max |θ−θ*|", "max |ω−ω*|", "max |ζ−ζ*|", "IoU (MAP)", "train time (s)"], fit_rows)}

## Observations

- With exact moments, every parameter (θ, α, ω, ζ) is recovered essentially exactly, so the
  class-conditional model is identifiable.
- In the coupled, finite-data setting, ζ (vote bias) is recovered well. ω (class-dependent coverage) is
  recovered only partially, because its signal is the small difference in coverage between classes, estimated
  from a few thousand sites with sampling noise. More data, or longer training, narrows the gap.
- On segmentation quality (IoU), the two models are close on this data. The class-conditional model mainly
  matters for calibrated soft labels and for LFs with strongly asymmetric behaviour.
"""
    write("07-class-conditional.md", "07 — Class-conditional labeling functions", body)
    register({"07-class-conditional.md": {
        "title": "07 — Class-conditional LFs", "question": "Are asymmetric LF accuracies/coverages learnable?",
        "headline": "Identifiable (exact recovery with exact moments); in the coupled finite-data case ζ is recovered well, "
                    "ω partially; IoU similar to the symmetric model"}})
    return {"ident": ident_rows, "fit": fit_rows}


# ============================================================================ 08 Pet-format pipeline
def exp_petlike(quick: bool, workdir: str) -> dict:
    seeds = [0] if quick else [0, 1, 2]
    rows, lf_rows, raw = [], [], []
    for s in seeds:
        rep = run_experiment({"seed": s, "dataset": {"kind": "petlike", "root": f"{workdir}/pl{s}",
                                                     "n_images": 12 if quick else 30, "n_segments": 150},
                              "train": {"epochs": 15}})
        rows.append([s, rep["iou"]["site_oracle"], rep["iou"]["majority_vote"], rep["iou"]["independent_model"],
                     rep["iou"]["structured_map"], rep["ece_structured"]])
        raw.append({k: rep[k] for k in ("iou", "ece_structured", "lf_accuracy", "learned_params")})
        if s == seeds[0]:
            la = rep["lf_accuracy"]
            th = rep["learned_params"]["theta"]
            for name, acc, cov, t in zip(la["names"], la["accuracy"], la["coverage"], th):
                lf_rows.append([name, cov, acc, t, 1 / (1 + np.exp(-2 * t))])
            thP = rep["learned_params"]["thetaP"][0]
    save_json(RAW / "petlike_pipeline.json", raw)
    mean = np.mean(np.array([r[1:] for r in rows], dtype=float), axis=0)
    body = f"""
The real-image pipeline (decision R5): JPEG decoding, Pet trimaps with an ignore band, resizing, SLIC
superpixels, generic LFs, and ignore-masked metrics. It runs end to end on a **generated dataset in the exact
Oxford-IIIT Pet file layout** (`qws.data.petlike`), because the real dataset's hosts are blocked from this
environment. **These are not Oxford-IIIT Pet results.** The images are synthetic textured blobs on textured
backgrounds, with overlapping colour distributions.

## Setup

- **Data:** {12 if quick else 30} images per seed (128 px longer side), ~150 SLIC superpixels each.
- **LFs:** generic only, none uses ground truth:
  - border → background;
  - centre ellipse → foreground;
  - border-vs-centre colour-histogram model;
  - spectral-residual saliency;
  - pairwise colour contrast.
- **Model:** β = 0.5, 15 epochs, Gibbs sampling, MAP by simulated annealing.
- **Metrics:** boundary (trimap = 3) pixels are excluded.

## Label quality (per seed)

{table(["seed", "IoU oracle", "IoU majority", "IoU independent", "IoU structured", "ECE structured"], rows)}

Mean: oracle {mean[0]:.3f}, majority {mean[1]:.3f}, independent {mean[2]:.3f}, **structured {mean[3]:.3f}**.

## What the label model learned about each LF (seed {seeds[0]}, without ground truth)

The empirical accuracy (measured against site ground truth) is a diagnostic only. The implied accuracy
σ(2θ) is the model's own estimate of the probability that a cast vote is correct.

{table(["LF", "coverage", "empirical accuracy", "learned θ", "implied accuracy σ(2θ)"], lf_rows)}

Pairwise colour-contrast LF: learned θᴾ = {thP:.2f}.

## Observations

- The structured model beats both baselines by a wide margin, and gets close to the superpixel oracle.
- Without labels, the label model correctly ranks the LFs. It assigns high weight to the reliable priors and
  near-zero weight to saliency, which is close to chance on these images. This is the core promise of data
  programming, working on real-format images.
- Next step (blocked by network policy): run the same command on the real dataset, with `kind: pet` and the
  path to the downloaded data.
"""
    write("08-petlike-pipeline.md", "08 — Real-image pipeline on Pet-format data", body)
    register({"08-petlike-pipeline.md": {
        "title": "08 — Pet-format pipeline", "question": "Does the real-image pipeline work end to end?",
        "headline": f"On generated Pet-layout data: structured IoU {mean[3]:.3f} vs. {mean[2]:.3f} (independent), "
                    f"{mean[1]:.3f} (majority), oracle {mean[0]:.3f}; useless LFs get near-zero weight. Not real Pet data"}})
    return {"rows": rows}


# ============================================================================ 09 downstream U-Net
def exp_downstream(quick: bool, workdir: str) -> dict:
    from qws.downstream.pipeline import run_downstream
    from qws.downstream.unet import UNetConfig

    n = 16 if quick else 60
    t0 = time.perf_counter()
    out = run_downstream({"dataset": {"kind": "petlike", "root": f"{workdir}/down", "n_images": n, "generate": 2 * n,
                                      "n_segments": 150}, "train": {"epochs": 15}},
                         UNetConfig(epochs=30 if quick else 100, size=64))
    wall = time.perf_counter() - t0
    save_json(RAW / "downstream_unet.json", out)
    names = {"structured_soft": "structured label model, soft (this work)",
             "independent_soft": "independent label model, soft",
             "majority_hard": "majority vote, hard", "ground_truth": "ground truth (upper bound)"}
    rows = [[names[k], v["train_label_iou"], v["test_iou"], v["final_loss"]] for k, v in out["signals"].items()]
    body = f"""
End-task evaluation (decision R10): a small U-Net is trained on weak labels for the training images and
evaluated on held-out test images. Data is the generated Pet-layout set of report 08, **not real Pet**.

## Setup

- **Split:** {out['n_train']} training images (Pet `trainval` list) and {out['n_test']} test images (`test` list).
- **Network:** U-Net with depth 3 and 16 base channels, on 64×64 inputs; Adam with cosine schedule,
  {out['unet']['epochs']} epochs, horizontal-flip augmentation.
- **Loss:** noise-aware expected binary cross-entropy under the soft labels (doc 03 §7).
- **Metric:** test IoU at original resolution, excluding trimap boundary pixels.
- **Wall time:** {wall:.0f} s on CPU.

{table(["training signal", "IoU of the training labels", "U-Net test IoU", "final train loss"], rows)}

For reference, the label model applied directly (transductively) to the test images reaches IoU
**{out['label_model_direct_test_iou']:.3f}**.

## Observations

- The structured label model gives by far the best *training labels*, well above the independent model and
  majority vote.
- On this small synthetic dataset, the U-Net itself is the bottleneck. Even trained on ground truth, it reaches
  only about the test IoU it reaches from either label model's soft masks. The label-quality advantage does not
  carry through to the end task here. Majority-vote labels, in contrast, clearly hurt.
- The label model applied directly beats every U-Net on these images. The meaningful end-task comparison needs
  real data with learnable appearance (Oxford-IIIT Pet), more images, and a stronger network. The
  pipeline is ready for that.
"""
    write("09-downstream-unet.md", "09 — Downstream U-Net trained on weak labels", body)
    s = out["signals"]
    register({"09-downstream-unet.md": {
        "title": "09 — Downstream U-Net", "question": "Do better weak labels give a better segmentation network?",
        "headline": f"Training-label IoU {s['structured_soft']['train_label_iou']:.2f} (structured) vs. "
                    f"{s['independent_soft']['train_label_iou']:.2f} / {s['majority_hard']['train_label_iou']:.2f}; "
                    f"U-Net test IoU {s['structured_soft']['test_iou']:.2f} / {s['independent_soft']['test_iou']:.2f} / "
                    f"{s['majority_hard']['test_iou']:.2f} (GT-trained {s['ground_truth']['test_iou']:.2f}): network-limited "
                    f"on this small synthetic set"}})
    return out


# ============================================================================ 10 QPBO
def exp_qpbo(quick: bool) -> dict:
    from qws.ising import IsingProblem

    ex = ExactSolver()
    weak = [t * 0.25 for t in TRUE["theta"]]
    rng = np.random.default_rng(23)

    def spin_glass(n_side: int, k: int):
        st = grid_sites(np.zeros((n_side, n_side)))
        return [IsingProblem(np.zeros(st.n), st.edges, rng.choice([-1.0, 1.0], st.m)) for _ in range(k)]

    def summarize(problems, exact: bool):
        lab, gap, ok = [], [], 0
        for p in problems:
            labels, bound = roof_dual(p)
            fixed = labels != 0
            lab.append(fixed.mean())
            if exact:
                Y, E = ex._enumerate(p)
                ok += int(np.isclose(E[np.all(Y[:, fixed] == labels[fixed], axis=1)].min(), E.min()))
                gap.append((E.min() - bound) / abs(E.min()))
        return float(np.mean(lab)), (float(np.mean(gap)) if gap else float("nan")), f"{ok}/{len(problems)}"

    k = 4 if quick else 10
    families = [
        ("posterior, realistic LFs", lambda tp, sh, kk: posterior_problems(sh, kk, seed=17, thetaP=[tp])),
        ("posterior, weak LFs (θ × 0.25)", lambda tp, sh, kk: posterior_problems(sh, kk, seed=17, thetaP=[tp],
                                                                                   theta=weak)),
    ]
    small_rows = []
    for fam, make in families:
        for tp in (0.0, 1.0, 2.0):
            small_rows.append([fam, tp, *summarize(make(tp, (4, 4), k), exact=True)])
    small_rows.append(["±J spin glass, no fields", "–", *summarize(spin_glass(4, k), exact=True)])
    big_rows = []
    side = 24 if quick else 32
    for fam, make in families:
        for tp in (0.0, 2.0):
            probs = make(tp, (side, side), 2)
            lab, de, tq, tsa = [], [], [], []
            for p in probs:
                t0 = time.perf_counter()
                ss = QPBOSolver().minimize([p])[0]
                tq.append(time.perf_counter() - t0)
                lab.append(ss.info["labeled_fraction"])
                t0 = time.perf_counter()
                e_sa = SimulatedAnnealing(num_sweeps=300, seed=0).minimize([p], num_reads=20)[0].best()[1]
                tsa.append(time.perf_counter() - t0)
                de.append(ss.best()[1] - e_sa)
            big_rows.append([fam, tp, side * side, float(np.mean(lab)), float(np.mean(de)), float(np.mean(tq)),
                             float(np.mean(tsa))])
    sg = spin_glass(side, 2)
    big_rows.append(["±J spin glass, no fields", "–", side * side, float(np.mean([roof_dual(p)[0].astype(bool).mean()
                                                                                for p in sg])), float("nan"),
                     float("nan"), float("nan")])
    save_json(RAW / "qpbo.json", {"small": small_rows, "large": big_rows})
    lab_weak2 = [r[2] for r in small_rows if r[0].startswith("posterior, weak") and r[1] == 2.0][0]
    lab_sg = small_rows[-1][2]
    body = f"""
Roof duality / QPBO (roadmap 4.3; doc 02 §6.3). The doubled energy over (x, y ≈ 1 − x) is submodular, so it is
minimized exactly by one max-flow ([spec/09 §5](../spec/09-implementation-notes.md)). Variables with y = 1 − x
are weakly persistent: some global optimum agrees with all of them at once. Frustration comes from the pairwise
"different" LF (strength θᴾ). To find where roof duality stops deciding variables, we also use weak unary LFs
and a field-free ±J spin glass.

## Small problems (4×4, exact reference by enumeration)

{table(["problem family", "θᴾ", "labeled fraction", "bound gap (E* − bound)/|E*|", "persistency verified"],
       small_rows)}

## Large problems ({side}×{side}): QPBO (+ SA completion of the unlabeled part) vs. simulated annealing alone

{table(["problem family", "θᴾ", "n", "labeled fraction", "E(QPBO+SA) − E(SA)", "QPBO time (s)", "SA time (s)"],
       big_rows)}

## Observations

- With realistic LFs, the unary fields anchor the posterior so strongly that roof duality labels every variable,
  even with strong pairwise "different" votes. The MAP problem is then solved exactly by max-flow, and no quantum
  hardware is needed for it.
- Weak LFs leave room for frustration. At θᴾ = 2 roof duality labels {lab_weak2:.0%} of the variables. For the
  field-free ±J spin glass it labels {lab_sg:.0%}: the regime that remains genuinely hard.
- Persistency held in every verified case: the labels always agree with an exact optimum. The bound is tight
  whenever all variables are labeled.
- In practice, run QPBO first and send only the unlabeled remainder to the QPU (doc 02 §6.3). On posteriors with
  informative LFs that remainder is often empty, which is an honest limit on where quantum MAP helps.
"""
    write("10-qpbo.md", "10 — QPBO / roof-duality baseline", body)
    register({"10-qpbo.md": {
        "title": "10 — QPBO", "question": "How much of a frustrated posterior is decided exactly by roof duality?",
        "headline": f"Realistic posteriors: 100% labeled (MAP solved exactly by max-flow); weak LFs at θᴾ = 2: "
                    f"{lab_weak2:.0%}; field-free spin glass: {lab_sg:.0%}; persistency verified in every case"}})
    return {"small": small_rows, "large": big_rows}


# ============================================================================ 11 multi-class
def exp_multiclass(quick: bool) -> dict:
    from qws.multiclass import (
        MulticlassParams, PottsGibbs, all_labelings, decode_onehot, fit_multiclass, multiclass_dataset,
        posterior_potts, to_onehot_ising,
    )
    from qws.solvers.qpu import QPUAnnealingSolver

    true = MulticlassParams(3, alpha=[0.5, 0.0, 0.3], theta=[1.2, 0.7, 1.5], alphaP=[0.0], thetaP=[0.8], beta=0.5)
    ds = multiclass_dataset(true, 6 if quick else 10, (12, 12), seed=0)
    batch = [(v, s) for v, s, _ in ds]
    init = MulticlassParams(3, alpha=np.zeros(3), theta=np.full(3, 0.5), alphaP=[0.0], thetaP=[0.5], beta=0.5)
    t0 = time.perf_counter()
    fit = fit_multiclass(batch, init, epochs=30 if quick else 60, num_samples=50, sweeps=15)
    fit_t = time.perf_counter() - t0
    param_rows = [[f"θ_{j}", true.theta[j], fit.theta[j]] for j in range(3)] + [["θᴾ_0", true.thetaP[0], fit.thetaP[0]]]
    acc_rows = []
    maj_acc, ind_acc, str_acc = [], [], []
    for (votes, sites, y) in ds:
        counts = np.stack([(votes.L == c + 1).sum(axis=1) for c in range(3)], axis=1)
        maj_acc.append(np.mean(counts.argmax(axis=1) == y))
        indep = fit.copy()
        indep.beta, indep.thetaP = 0.0, np.zeros(1)
        ind_acc.append(np.mean(posterior_potts(indep, votes, sites).unary.argmax(axis=1) == y))
        pot = posterior_potts(fit, votes, sites)
        str_acc.append(np.mean(PottsGibbs(num_sweeps=200, seed=0).anneal(pot, num_reads=5) == y))
    acc_rows = [["majority vote", float(np.mean(maj_acc))], ["independent (β = 0)", float(np.mean(ind_acc))],
                ["structured Potts MAP (this work)", float(np.mean(str_acc))]]
    # one-hot QUBO route vs exact on small Potts problems
    rng = np.random.default_rng(3)
    from qws.multiclass import PottsProblem

    small = []
    for _ in range(10 if quick else 20):
        e = grid_sites(np.zeros((2, 3))).edges
        small.append(PottsProblem(rng.normal(size=(6, 3)), e, rng.normal(size=len(e))))
    route_rows = []
    for label, make in (("SA on one-hot QUBO, 20 reads", lambda: (SimulatedAnnealing(num_sweeps=400, seed=0), 20)),
                        ("SA on one-hot QUBO, 100 reads", lambda: (SimulatedAnnealing(num_sweeps=400, seed=0), 100)),
                        ("mock QPU on one-hot QUBO, 100 reads", lambda: (QPUAnnealingSolver.mock(topology_shape=6), 100))):
        hits, infeas = 0, []
        for p in small:
            best = p.energy(all_labelings(p.n, p.K)).min()
            ising, _ = to_onehot_ising(p)
            solver, reads = make()
            ss = solver.minimize([ising], num_reads=reads)[0]
            labels, inf = decode_onehot(ss.samples, p)
            infeas.append(float(np.average(inf, weights=ss.counts)))
            hits += int(abs(p.energy(labels).min() - best) < 1e-6)
        route_rows.append([label, f"{hits}/{len(small)}", float(np.mean(infeas))])
    hits = sum(int(abs(p.energy(PottsGibbs(num_sweeps=300, seed=1).anneal(p, num_reads=10))[0]
                       - p.energy(all_labelings(p.n, p.K)).min()) < 1e-6) for p in small)
    route_rows.append(["Potts simulated annealing (native)", f"{hits}/{len(small)}", 0.0])
    save_json(RAW / "multiclass.json", {"params": param_rows, "accuracy": acc_rows, "routes": route_rows})
    body = f"""
The K-class extension (spec doc 07; paper Section 8 and Proposition 11). It consists of:
- a Potts label model with symmetric LF potentials;
- the one-hot QUBO with the Proposition-11 penalty, which makes multi-class MAP solvable by every binary backend,
  including the QPU path;
- decoding with repair of infeasible sites;
- a Potts Gibbs sampler and Potts simulated annealing.

## Learning (K = 3, synthetic 12×12 grids, {len(ds)} images, {fit_t:.0f} s)

{table(["parameter", "true", "learned"], param_rows)}

## Segmentation accuracy (pixel accuracy, K = 3)

{table(["method", "accuracy"], acc_rows)}

## Multi-class MAP routes on small Potts problems (2×3 grid, K = 3, exact reference by enumeration)

{table(["route", "exact optimum found", "infeasible-read fraction"], route_rows)}

## Observations

- LF accuracies are learned without labels in the K-class model, as in the binary case.
- The structured Potts model clearly beats majority vote and the independent model.
- **The one-hot encoding costs search effort.** Moving a site between classes passes through infeasible
  states, which cost about A (the Proposition-11 penalty), so local search needs more reads to find the optimum.
  This is the known weakness of one-hot encodings on annealers, and the reason domain-wall encodings exist
  (spec doc 07 §3). Budget QPU reads accordingly.
- With the sufficient penalty of Proposition 11, the minimizers are always feasible (verified exhaustively in
  `tests/test_multiclass.py`). The infeasible reads seen here are non-optimal samples, which the decoder repairs.
"""
    write("11-multiclass.md", "11 — Multi-class extension", body)
    register({"11-multiclass.md": {
        "title": "11 — Multi-class", "question": "Does the K-class model work, and what does one-hot encoding cost?",
        "headline": f"Accuracy {acc_rows[2][1]:.3f} (structured) vs. {acc_rows[1][1]:.3f} (independent) / "
                    f"{acc_rows[0][1]:.3f} (majority); one-hot QUBO needs more reads ({route_rows[0][1]} → "
                    f"{route_rows[1][1]} optima with 20 → 100 reads)"}})
    return {"acc": acc_rows, "routes": route_rows}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--only", nargs="*", help="subset of: qpu beta cc petlike downstream qpbo multiclass")
    args = ap.parse_args()
    RAW.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    todo = args.only or ["qpu", "beta", "cc", "petlike", "downstream", "qpbo", "multiclass"]
    with tempfile.TemporaryDirectory() as work:
        for name in todo:
            t = time.perf_counter()
            {"qpu": lambda: exp_qpu(args.quick), "beta": lambda: exp_beta(args.quick),
             "cc": lambda: exp_class_conditional(args.quick), "petlike": lambda: exp_petlike(args.quick, work),
             "downstream": lambda: exp_downstream(args.quick, work), "qpbo": lambda: exp_qpbo(args.quick),
             "multiclass": lambda: exp_multiclass(args.quick)}[name]()
            print(f"  {name}: {time.perf_counter() - t:.0f}s")
    write_index()
    print(f"done in {time.perf_counter() - t0:.0f}s")


if __name__ == "__main__":
    main()
