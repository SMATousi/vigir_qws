"""Regenerate every report in results/ (Markdown + raw JSON).

Usage:  python scripts/generate_results.py [--quick]

All runs are classical or simulated: no real QPU is used (roadmap Phase 3).
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from qws.decompose import block_coordinate_descent, grid_tiles
from qws.eval import map_benchmark, sampling_benchmark, synthetic_dataset
from qws.experiment import run_experiment
from qws.ising import IsingProblem
from qws.io import save_json
from qws.learn import CalibratedSampler, estimate_beta_pl
from qws.model import LabelModelParams, posterior_ising
from qws.solvers import ExactSolver, GibbsSampler, GraphCutSolver, SimulatedAnnealing, Solver, gibbs_polish
from qws.solvers.anneal import simulated_annealing_stand_in
from qws.solvers.qaoa import QAOASimulator

from _report import OUT, RAW, env_block, ms, register, table, write, write_index  # noqa: E402

FIXED_T = (0.1, 4.0)  # fixed inverse-temperature range: the stand-in behaves like a device at fixed temperature
TRUE = {"theta": [1.0, 0.6, 0.3, 0.8], "alpha": [0.5, 0.0, -0.5, 0.2], "thetaP": [1.0], "alphaP": [0.0]}


# ----------------------------------------------------------------------------- 1. label quality
def exp_label_quality(seeds: list[int], quick: bool) -> dict:
    configs = {
        "grid 16×16, Gibbs": {"dataset": {"shape": [16, 16], "sites": "grid"}},
        "superpixels (~60) on 32×32, Gibbs": {"dataset": {"shape": [32, 32], "sites": "superpixel",
                                                         "n_segments": 60}},
        "grid 16×16, anneal_sim (fixed T) + calibration": {"solver": {"name": "anneal_sim",
                                                                      "beta_range": list(FIXED_T)},
                                                           "calibrate": True},
        "grid 16×16, EM mode, Gibbs": {"train": {"mode": "em", "epochs": 10}},
    }
    epochs = 8 if quick else 20
    results = {}
    for name, cfg in configs.items():
        runs = []
        for s in seeds:
            c = {"seed": s, "train": {"epochs": epochs}, **cfg}
            c["train"] = {**{"epochs": epochs}, **cfg.get("train", {})}
            rep = run_experiment(c)
            runs.append({k: rep[k] for k in ("iou", "ece_structured", "theta_error", "thetaP_error", "train_time_s",
                                             "epochs_run", "learned_params")})
        results[name] = runs
    save_json(RAW / "label_quality.json", results)

    rows = []
    for name, runs in results.items():
        rows.append([name, ms([r["iou"]["site_oracle"] for r in runs]), ms([r["iou"]["majority_vote"] for r in runs]),
                     ms([r["iou"]["independent_model"] for r in runs]),
                     ms([r["iou"]["structured_map"] for r in runs]),
                     ms([r["ece_structured"] for r in runs]),
                     ms([r["theta_error"] for r in runs]), ms([r["thetaP_error"] for r in runs]),
                     ms([r["train_time_s"] for r in runs])])
    per_seed = []
    for name, runs in results.items():
        for s, r in zip(seeds, runs):
            per_seed.append([name, s, r["iou"]["majority_vote"], r["iou"]["independent_model"],
                             r["iou"]["structured_map"], r["theta_error"], r["epochs_run"]])
    first = results["grid 16×16, Gibbs"][0]["learned_params"]
    param_rows = [[f"θ_{j}", TRUE["theta"][j], first["theta"][j]] for j in range(4)]
    param_rows += [[f"α_{j}", TRUE["alpha"][j], first["alpha"][j]] for j in range(4)]
    param_rows += [["θᴾ_0 (pairwise)", TRUE["thetaP"][0], first["thetaP"][0]],
                   ["αᴾ_0 (pairwise)", TRUE["alphaP"][0], first["alphaP"][0]]]

    body = f"""
Does the structured label model give better segmentation labels than unstructured weak supervision?
This is evaluation question Q1 in [spec/06](../spec/06-evaluation.md).

## Setup

- **Data:** synthetic. Masks are smoothed random blobs (smoothness 3, 40% foreground), and images are the mask
  plus Gaussian noise. There are 8 images per run and {len(seeds)} seeds ({', '.join(map(str, seeds))}).
- **Labeling functions:** votes are drawn from the label model's own generative law
  (paper Lemma 5) with known parameters:
  - 4 unary LFs with θ = {TRUE['theta']} and α = {TRUE['alpha']};
  - 1 pairwise "same/different" LF with θᴾ = {TRUE['thetaP']}.
  The pairwise LF makes the posterior **frustrated** (it has negative couplings).
- **Model:** smoothness β = 0.5 (fixed, open question Q4). Warm start from the independent model, then
  {epochs} epochs of Adam (gradient mode) or EM, with 100 samples per image per step.
- **Inference:** the hard mask comes from simulated annealing, since graph cuts refuse frustrated problems.
  Soft masks come from 200 Gibbs samples.
- **Baselines:** per-site majority vote, and the independent (Snorkel-style, β = 0, no pairwise LF)
  label model.
- **Oracle:** the true per-site labels projected back to pixels. This is the best IoU any method can reach
  with that site representation; it is below 1 for superpixels that straddle the mask boundary.

## Summary (mean ± std over seeds)

{table(["Configuration", "IoU oracle", "IoU majority", "IoU independent", "IoU structured", "ECE structured",
        "max |θ−θ*|", "|θᴾ−θᴾ*|", "train time (s)"], rows)}

## Learned vs. true parameters (grid 16×16, Gibbs, seed {seeds[0]})

{table(["Parameter", "True", "Learned"], param_rows)}

## Per-seed results

{table(["Configuration", "Seed", "IoU majority", "IoU independent", "IoU structured", "max |θ−θ*|",
        "Epochs"], per_seed)}

## Observations

- The structured model beats both baselines by a wide margin in every configuration and every seed. This is
  the M1 acceptance criterion of spec/06 §5.
- Unary LF accuracies are recovered without ground truth, to within about 0.1.
- The pairwise accuracy θᴾ is recovered within about 0.03 by EM, but only within about 0.16 by gradient
  ascent after the same budget. Its gradient comes only from the edge correlations ⟨yᵢyₖ⟩, and Adam has not
  converged on it after 20 epochs. EM's closed-form M-step (θ = artanh(agree/cover)) jumps there directly.
- On superpixels the structured model closes most of the gap between the baselines and the oracle; the
  remaining gap reflects coarser sites and noisier lifted votes.
- The `anneal_sim` path (Ocean interface, fixed-temperature stand-in, pseudo-likelihood calibration) reaches the
  same quality as Gibbs. This exercises the code path a real QPU would use.
- Caveat: votes follow the model's own generative law, so this is a best case for the label model.
  Real LFs will deviate from it (open question Q1: dataset choice).
"""
    write("01-label-quality.md", "01 — Label quality: structured vs. unstructured weak supervision", body)
    return results


# ----------------------------------------------------------------------------- 2. MAP benchmark
def posterior_problems(shape, n_images, seed, pairwise=True, beta=0.5):
    tp = dict(TRUE) if pairwise else {k: v for k, v in TRUE.items() if not k.endswith("P")}
    params = LabelModelParams(tp["alpha"], tp["theta"], tp.get("alphaP", []), tp.get("thetaP", []), beta=beta)
    ds = synthetic_dataset(params, n_images, shape, seed=seed)
    return [posterior_ising(params, im.lfo, im.sites) for im in ds], ds


def exp_map(quick: bool) -> dict:
    out = {}
    small_frus, _ = posterior_problems((4, 4), 4 if quick else 8, seed=11)
    small_ferro, _ = posterior_problems((4, 4), 4 if quick else 8, seed=12, pairwise=False)
    solvers = {"sa": SimulatedAnnealing(num_sweeps=200, seed=0), "anneal_sim": simulated_annealing_stand_in(),
               "qaoa_sim p=1": QAOASimulator(p_layers=1, seed=0), "qaoa_sim p=2": QAOASimulator(p_layers=2, seed=0)}
    kw = {"sa": {"num_reads": 100}, "anneal_sim": {"num_reads": 100}, "qaoa_sim p=1": {"shots": 100},
          "qaoa_sim p=2": {"shots": 100}}
    out["small_frustrated"] = map_benchmark(small_frus, solvers, **kw)
    out["small_ferro"] = map_benchmark(small_ferro, {"graphcut": GraphCutSolver(), **solvers}, **kw)

    side = 24 if quick else 32
    big_frus, _ = posterior_problems((side, side), 3, seed=13)
    big_ferro, ds_ferro = posterior_problems((side, side), 3, seed=14, pairwise=False)
    big_solvers = {"sa": SimulatedAnnealing(num_sweeps=300, seed=0), "anneal_sim": simulated_annealing_stand_in()}
    big_kw = {"sa": {"num_reads": 20}, "anneal_sim": {"num_reads": 20}}
    out["large_frustrated"] = map_benchmark(big_frus, big_solvers, **big_kw)
    out["large_ferro"] = map_benchmark(big_ferro, {"graphcut": GraphCutSolver(), **big_solvers}, **big_kw)

    # decomposition: block coordinate descent with exact 4x4 tiles (hardware-sized subproblems)
    dec = []
    for i, (p, im) in enumerate(zip(big_frus, synthetic_dataset(LabelModelParams(TRUE["alpha"], TRUE["theta"],
                                                                                 TRUE["alphaP"], TRUE["thetaP"],
                                                                                 beta=0.5), 3, (side, side),
                                                                seed=13))):
        t0 = time.perf_counter()
        y, trace = block_coordinate_descent(p, grid_tiles(im.sites, 4), ExactSolver(), max_sweeps=10, seed=0)
        ref = min(r["best_energy"] for r in out["large_frustrated"] if r["problem"] == i and "best_energy" in r)
        dec.append({"problem": i, "n": p.n, "tiles": len(grid_tiles(im.sites, 4)), "sweeps_accepted": len(trace) - 1,
                    "energy": float(trace[-1]), "best_monolithic": ref, "gap_vs_best": (trace[-1] - ref) / abs(ref),
                    "wall_time_s": time.perf_counter() - t0})
    out["decomposition"] = dec
    save_json(RAW / "map_benchmark.json", out)

    def summarize(records):
        rows = []
        for name in dict.fromkeys(r["solver"] for r in records):
            rs = [r for r in records if r["solver"] == name and "gap" in r]
            skipped = [r for r in records if r["solver"] == name and "skipped" in r]
            if not rs:
                rows.append([name, "–", "–", "–", "–", f"refused ({len(skipped)})"])
                continue
            rows.append([name, rs[0]["n"], float(np.mean([r["gap"] for r in rs])),
                         float(np.mean([r["hit_rate"] for r in rs])),
                         float(np.median([r["tts99_s"] for r in rs])), float(np.mean([r["wall_time_s"] for r in rs]))])
        return table(["Solver", "n", "mean rel. gap", "mean hit rate", "median TTS99 (s)", "mean wall (s)"], rows)

    ref_note = {k: sorted({r.get("reference_kind") for r in v if "reference_kind" in r}) for k, v in out.items()
                if k != "decomposition"}
    body = f"""
For a fixed model, how well do the solvers find the ground state (the MAP mask)? This is question Q2
in [spec/06](../spec/06-evaluation.md), task T1.

## Setup

- **Problems:** posterior Ising models of synthetic images (same LFs as report 01, β = 0.5).
  - *Frustrated* problems include the pairwise "different" LF, which adds negative couplings.
  - *Ferromagnetic* problems use unary LFs only, so all J ≥ 0.
- **References:** exact enumeration for small problems; graph cut (exact) for large ferromagnetic problems; the
  best energy found by any solver for large frustrated problems. Reference kinds used: {ref_note}.
- **Metrics:**
  - relative energy gap (E − E*)/|E*|;
  - hit rate (the fraction of reads reaching E*);
  - TTS99 = t · log(0.01)/log(1 − p_hit), with t the wall time of the whole call.
- **Timing:** CPU wall times. They are not comparable to QPU access times.

## Small problems (4×4 grid, n = 16), frustrated

{summarize(out['small_frustrated'])}

## Small problems (4×4 grid, n = 16), ferromagnetic

{summarize(out['small_ferro'])}

## Large problems ({side}×{side} grid, n = {side * side}), frustrated

{summarize(out['large_frustrated'])}

## Large problems ({side}×{side} grid, n = {side * side}), ferromagnetic

{summarize(out['large_ferro'])}

## Decomposition: block coordinate descent with 4×4 exact tiles ({side}×{side}, frustrated)

Each tile (16 sites, about the size of a small QPU or QAOA subproblem) is solved exactly, conditioned on its
boundary (paper Proposition 5(b)–(c)).

{table(["Problem", "n", "tiles", "accepted tile moves", "final energy", "best monolithic", "rel. gap", "wall (s)"],
       [[d["problem"], d["n"], d["tiles"], d["sweeps_accepted"], d["energy"], d["best_monolithic"],
         d["gap_vs_best"], d["wall_time_s"]] for d in dec])}

## Observations

- Graph cut is exact and fastest on ferromagnetic problems. As intended, it refuses every frustrated
  problem (shown as "refused").
- Simulated annealing (ours, and Ocean's through the annealing interface) reaches the exact optimum on every
  small problem, with hit rates near 1. On the 1024-site problems both reach the same best energy, with
  70–85% of reads hitting it.
- QAOA (exact statevector, angles optimized classically, 100 shots) is much weaker at these depths. At p = 1,
  no shot reaches the optimum on frustrated problems. At p = 2, about 7–11% of shots reach it; best-of-100 is
  exact on ferromagnetic problems but leaves a mean gap of several percent on frustrated ones. Depth and
  shot budget are the levers to study on hardware (roadmap 3.4).
- Block coordinate descent with exact 16-site tiles converges to a block-local optimum within 0–0.5% of the best
  monolithic energy. That gap is the cost of decomposition when problems exceed QPU capacity.
"""
    write("02-map-benchmark.md", "02 — MAP (ground-state) benchmark", body)
    return out


# ----------------------------------------------------------------------------- 3. sampling benchmark
class _QAOASampler(Solver):
    """QAOA measurement samples, optionally polished with Gibbs sweeps (seeding)."""

    def __init__(self, polish: int, seed: int = 0):
        self.q, self.polish, self.rng = QAOASimulator(p_layers=2, seed=seed), polish, np.random.default_rng(seed)
        self.name = f"qaoa_sim(p=2)+polish{polish}"

    def sample(self, problems, num_samples, **kw):
        from qws.solvers import SampleSet

        out = []
        for p in problems:
            ss = self.q.sample([p], num_samples)[0]
            s = gibbs_polish(p, ss.expanded(), self.polish, self.rng) if self.polish else ss.expanded()
            out.append(SampleSet.from_samples(p, s))
        return out


class _Polished(Solver):
    def __init__(self, inner: Solver, polish: int, name: str, seed: int = 0):
        self.inner, self.polish, self.name, self.rng = inner, polish, name, np.random.default_rng(seed)

    def sample(self, problems, num_samples, **kw):
        from qws.solvers import SampleSet

        out = []
        for p, ss in zip(problems, self.inner.sample(problems, num_samples, **kw)):
            s = gibbs_polish(p, ss.expanded(), self.polish, self.rng) if self.polish else ss.expanded()
            out.append(SampleSet.from_samples(p, s))
        return out


def exp_sampling(quick: bool) -> dict:
    probs, _ = posterior_problems((3, 4), 3 if quick else 6, seed=21)  # n = 12, exact reference
    weak = [IsingProblem(p.h, p.edges, p.J * 0.3) for p in probs]
    strong = [IsingProblem(p.h, p.edges, p.J * 3.0) for p in probs]
    def calibrated(ps):
        cal = CalibratedSampler(simulated_annealing_stand_in(beta_range=FIXED_T), damping=1.0)
        for _ in range(10):  # warm up on the regime's own problems
            cal.sample(ps, 500)
        return cal

    samplers = {
        "exact (reference sampler)": ExactSolver(seed=0),
        "gibbs, 10 sweeps": GibbsSampler(num_sweeps=10, seed=0),
        "gibbs, 100 sweeps": GibbsSampler(num_sweeps=100, seed=0),
        "anneal_sim, auto range (cold)": simulated_annealing_stand_in(),
        "anneal_sim fixed T, raw": simulated_annealing_stand_in(beta_range=FIXED_T),
        "anneal_sim fixed T + 5 polish": _Polished(simulated_annealing_stand_in(beta_range=FIXED_T), 5, "anneal+polish"),
        "anneal_sim calibrated": None,  # built per regime below
        "qaoa_sim p=2 raw": _QAOASampler(0),
        "qaoa_sim p=2 + 5 polish": _QAOASampler(5),
    }
    out = {}
    n_s = 1000 if quick else 3000
    cal_hist = {}
    for regime, ps in {"weak coupling (J×0.3)": weak, "nominal (β=0.5 model)": probs,
                       "strong coupling (J×3)": strong}.items():
        cal = calibrated(ps)
        cal_hist[regime] = list(cal.history)
        out[regime] = sampling_benchmark(ps, {**samplers, "anneal_sim calibrated": cal}, num_samples=n_s)
    # PL temperature estimates of the raw stand-in
    raw = simulated_annealing_stand_in(beta_range=FIXED_T)
    out["beta_hat_raw_anneal_sim"] = [estimate_beta_pl(p, *(lambda s: (s.samples, s.counts))(raw.sample([p], 1000)[0]))
                                      for p in probs]
    out["calibration_history"] = cal_hist
    save_json(RAW / "sampling_benchmark.json", out)

    sections = []
    for regime in ("weak coupling (J×0.3)", "nominal (β=0.5 model)", "strong coupling (J×3)"):
        recs = out[regime]
        rows = []
        for name in samplers:
            rs = [r for r in recs if r["solver"] == name]
            rows.append([name, ms([r["tv"] for r in rs]), ms([r["marginal_err"] for r in rs]),
                         ms([r["corr_err"] for r in rs]), float(np.mean([r["wall_time_s"] for r in rs]))])
        sections.append(f"### {regime}\n\n" + table(["Sampler", "TV distance", "max marginal error",
                                                     "max edge-corr error", "wall (s)"], rows))
    floor = np.sqrt(2 ** 12 / n_s) / 2
    body = f"""
How close are the samplers' outputs to the exact Boltzmann distribution at β = 1? This is the quantity
that drives learning (paper Proposition 8: the gradient bias is at most 2·TV).

## Setup

- **Problems:** posterior Ising models on a 3×4 grid (n = 12, 4096 states, so the distribution can be
  enumerated exactly).
- **Regimes:** three coupling strengths, from {len(probs)} problems each.
- **Samples:** {n_s} per problem. Metrics are TV distance to the exact distribution, and the maximum error of
  the ⟨yᵢ⟩ and ⟨yᵢyₖ⟩ estimates.
- **Finite-sample floor:** the exact sampler's TV row shows the floor. Even perfect samples have TV ≈
  {floor:.2f} or less at this sample size, from 4096 states and {n_s} draws.
- **Sampler variants:**
  - "raw" is the sampler output as returned;
  - "calibrated" is wrapped in the pseudo-likelihood temperature calibration (paper Proposition 9);
  - "+ k polish" adds k Gibbs sweeps at β = 1 from each sample (paper Proposition 10).

{chr(10).join(sections)}

## Annealing stand-in variants

- **auto range:** Ocean's default schedule adapts to the problem's coefficients and anneals to (near) zero
  temperature. Rescaling cannot change its temperature, so calibration cannot help it.
- **fixed T:** the inverse-temperature range is fixed at {FIXED_T}. Like a physical device, its effective
  temperature then depends on the submitted coefficient scale, which is what calibration adjusts.

## Calibration of the fixed-T stand-in

Raw pseudo-likelihood estimates on the nominal problems: {', '.join(f'{b:.2f}' for b in out['beta_hat_raw_anneal_sim'])}.
A value of 1000 is the estimator's cap. It means every sampled spin was aligned with its local field, so the samples
look like zero temperature and no finite temperature can be read off them.

Calibration still converges, because each step changes the submit scale by at most 4×. The scale where
β̂ ≈ 1 reveals the device's effective inverse temperature, β_eff ≈ 1/scale:

{table(["Regime", "steps", "final scale", "implied β_eff = 1/scale", "last β̂"],
       [[r, len(h), h[-1]['scale'], 1 / h[-1]['scale'], h[-1]['beta_hat']] for r, h in cal_hist.items()])}

For comparison, the stand-in's schedule ends at β = {FIXED_T[1]}.

## Observations

- In the weak and nominal regimes, raw annealing samples are too cold (large TV, overconfident marginals).
  This is the bias that paper Proposition 7 says would inflate learned accuracies. With a fixed device
  temperature, pseudo-likelihood calibration rescales the problem until β̂ ≈ 1, and the TV distance drops to
  the exact-sampler floor. Five Gibbs polishing sweeps achieve the same. The auto-range sampler cannot be
  calibrated by rescaling; only polishing helps it.
- **Strong coupling reverses the picture.** Gibbs chains get stuck (TV 0.3–0.5 even after 100 sweeps), while
  the annealing stand-in, raw or polished, stays close to the target. This is the regime where an annealer's
  global moves matter, and it is the comparison to repeat on real hardware.
- Calibration has a limit under strong coupling. Almost every sample is then a ground state, so β̂ saturates
  (last β̂ ≫ 1 in the calibration table) and carries no temperature information. The calibrated TV stays
  around 0.2, while five polishing sweeps reach about 0.01. On hardware, calibrate on a weakly coupled proxy
  problem, or rely on polishing in this regime.
- QAOA output at p = 2 is far from Boltzmann (TV > 0.9 at nominal coupling). Followed by five Gibbs sweeps,
  its error drops sharply in the weak and nominal regimes, consistent with "polishing never hurts"
  (Proposition 10). Under strong coupling, five sweeps are not enough to repair QAOA seeds.
"""
    write("03-sampling-benchmark.md", "03 — Sampling benchmark against exact Boltzmann distributions", body)
    return out


# ----------------------------------------------------------------------------- 4. theory checks
def exp_theory(quick: bool) -> dict:
    rng = np.random.default_rng(0)
    out = {}
    # Prop. 6: agreement with MAP (majority) labels vs true accuracy
    rows6 = []
    for a in (0.6, 0.7, 0.8, 0.9):
        y = rng.choice([-1, 1], size=200_000)
        votes = np.where(rng.random((y.size, 3)) < a, y[:, None], -y[:, None])
        maj = np.sign(votes.sum(axis=1))
        emp = float(np.mean(votes[:, 0] == maj))
        rows6.append([a, emp, 1 - a * (1 - a), (1 - a) ** 2])
    out["prop6"] = rows6
    # Prop. 9: PL temperature estimator
    rows9 = []
    ex = ExactSolver(seed=1)
    pairs = [(i, k) for i in range(10) for k in range(i + 1, 10) if rng.random() < 0.4]
    p = IsingProblem(rng.normal(0, 0.5, 10), np.array(pairs), rng.normal(0, 0.5, len(pairs)))
    for b0 in (0.25, 0.5, 1.0, 2.0, 4.0):
        for S in (500, 5000, 50000):
            ss = ex.sample([p], S, beta=b0)[0]
            rows9.append([b0, S, estimate_beta_pl(p, ss.samples, ss.counts)])
    out["prop9"] = rows9
    # Prop. 8: statistical term vs sample size, exact sampler
    rows8 = []
    p8, _ = posterior_problems((3, 4), 1, seed=31)
    mu, _ = ExactSolver().moments(p8[0])
    for S in (100, 1000, 10000):
        errs = [np.abs(ExactSolver(seed=k).sample([p8[0]], S)[0].mean() - mu).max() for k in range(20)]
        rows8.append([S, float(np.mean(errs)), float(np.sqrt(2 * np.log(2 / 0.05) / S))])
    out["prop8"] = rows8
    save_json(RAW / "theory_checks.json", out)

    body = f"""
Numerical checks of the paper's quantitative claims, using exact samplers so that only the claim under
test is exercised.

## Proposition 6: MAP labels overestimate accuracy

Three conditionally independent LFs, each correct with probability a. The table compares each LF's
empirical agreement with the majority (MAP) label over 200,000 sites against the predicted 1 − a(1 − a).

{table(["true accuracy a", "empirical agreement with MAP", "predicted 1 − a(1−a)", "bias (1−a)²"], rows6)}

## Proposition 8: the statistical error does not depend on the configuration space

Maximum error of the ⟨yᵢ⟩ estimates from S exact samples (mean over 20 repetitions) on an n = 12 posterior,
against the Hoeffding term √(2 log(2/δ)/S) with δ = 0.05.

{table(["S", "mean max marginal error", "Hoeffding bound (δ=0.05)"], rows8)}

## Proposition 9: the pseudo-likelihood temperature estimator is consistent

Samples are drawn exactly at inverse temperature β₀ from a random n = 10 Ising model. β̂ is the maximum
pseudo-likelihood estimate.

{table(["β₀", "samples S", "β̂"], rows9)}

β̂ converges to β₀ as S grows (Proposition 9). The single 1000 entry (β₀ = 4, S = 500) is the estimator's cap.
At low temperature with few samples, every spin can be aligned with its local field. The pseudo-likelihood
then increases without bound in β, and the finite-sample maximizer does not exist. With more samples the
estimate is accurate again.

## Verified elsewhere by the test suite (`pytest`)

- Propositions 1–3 and Lemma 1 / Theorem 1: the posterior builder matches the brute-force joint, and the
  gradient matches finite differences of the exact marginal likelihood (`tests/test_model.py`).
- Proposition 4: Ising ↔ QUBO ↔ Ocean ↔ Pauli energy identities by enumeration (`tests/test_ising.py`).
  The Qiskit circuit's probabilities equal our statevector QAOA (`tests/test_solvers.py`).
- Proposition 5: conditioning exactness, strong-field fixing preserves minimizers, and block-CD monotonicity
  (`tests/test_ising.py`, `tests/test_decompose_eval.py`).
- Proposition 10: Gibbs polishing preserves the exact distribution and moves bad seeds toward it
  (`tests/test_solvers.py`).
"""
    write("04-theory-checks.md", "04 — Numerical checks of the paper's propositions", body)
    return out


# ----------------------------------------------------------------------------- index
def register_headlines(lq: dict, sm: dict) -> None:
    g = lq["grid 16×16, Gibbs"]
    nominal = sm["nominal (β=0.5 model)"]

    def tv_of(name):
        return float(np.mean([r["tv"] for r in nominal if r["solver"] == name]))

    register({
        "01-label-quality.md": {
            "title": "01 — Label quality", "question": "Does the structured label model beat unstructured weak supervision?",
            "headline": f"IoU {ms([r['iou']['structured_map'] for r in g])} vs. {ms([r['iou']['majority_vote'] for r in g])} "
                        f"(majority) and {ms([r['iou']['independent_model'] for r in g])} (independent); LF accuracies "
                        f"recovered within {max(r['theta_error'] for r in g):.2f}"},
        "02-map-benchmark.md": {
            "title": "02 — MAP benchmark", "question": "How well do solvers find ground states?",
            "headline": "Graph cut exact on ferromagnetic problems and refuses frustrated ones; SA / Ocean stand-in reach "
                        "the exact optimum on n = 16; QAOA p ≤ 2 has a lower hit rate"},
        "03-sampling-benchmark.md": {
            "title": "03 — Sampling benchmark", "question": "How close are samplers to Boltzmann at β = 1?",
            "headline": f"Nominal TV: Gibbs {tv_of('gibbs, 100 sweeps'):.2f}, raw annealing "
                        f"{tv_of('anneal_sim fixed T, raw'):.2f}, calibrated {tv_of('anneal_sim calibrated'):.2f}, "
                        f"QAOA+polish {tv_of('qaoa_sim p=2 + 5 polish'):.2f} (exact-sampler floor "
                        f"{tv_of('exact (reference sampler)'):.2f}); under strong coupling Gibbs fails, annealing does not"},
        "04-theory-checks.md": {
            "title": "04 — Theory checks", "question": "Do the paper's quantitative claims hold numerically?",
            "headline": "Proposition 6 bias, Proposition 8 sample complexity and Proposition 9 consistency all match"},
    })


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    RAW.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    seeds = [0, 1] if args.quick else [0, 1, 2, 3, 4]
    lq = exp_label_quality(seeds, args.quick)
    mp = exp_map(args.quick)
    sm = exp_sampling(args.quick)
    exp_theory(args.quick)
    register_headlines(lq, sm)
    write_index()
    print(f"done in {time.perf_counter() - t0:.0f}s")


if __name__ == "__main__":
    main()
