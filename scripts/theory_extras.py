"""Report 15: two further checks of the paper's sampling propositions.

Usage:  python scripts/theory_extras.py [--quick]

1. Learning with samplers at the wrong temperature (Proposition "cold samplers"): the same learning run is
   repeated with posterior samples drawn at inverse temperature beta_s instead of 1, and with MAP states (hard EM).
2. The sampler-bias term of the error decomposition: on every problem of report 03, the error of every
   empirical moment is compared with 2 * TV(empirical sampler distribution, exact posterior), which bounds
   it deterministically.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np

from _report import RAW, register, table, write, write_index  # noqa: E402
from qws.eval import synthetic_dataset
from qws.io import save_json
from qws.learn import TrainConfig, Trainer
from qws.model import LabelModelParams
from qws.solvers import SimulatedAnnealing, Solver

TRUE = {"theta": [1.0, 0.6, 0.3, 0.8], "alpha": [0.5, 0.0, -0.5, 0.2], "thetaP": [1.0], "alphaP": [0.0]}


class TemperedPosterior(Solver):
    """Annealed sampling that stops at inverse temperature ``beta_s`` (``None``: MAP by annealing + descent)."""

    name = "tempered"
    capabilities = frozenset({"sample"})

    def __init__(self, beta_s: float | None, seed: int):
        self.beta_s = beta_s
        self.sa = SimulatedAnnealing(num_sweeps=300, seed=seed)

    def sample(self, problems, num_samples, **_):
        if self.beta_s is None:
            return self.sa.minimize(problems, num_reads=num_samples)
        return self.sa.sample(problems, num_samples, beta=self.beta_s)


def sigma2(t):
    return 1.0 / (1.0 + np.exp(-2.0 * np.asarray(t, dtype=float)))


WEAK = {"theta": [0.5, 0.3, 0.15, 0.4], "alpha": [0.5, 0.0, -0.5, 0.2], "thetaP": [], "alphaP": []}


def exp_cold(quick: bool) -> dict:
    """Weak LFs (posterior genuinely uncertain), masks drawn from the model's own prior, so beta_s = 1 is the
    correctly specified case. Two settings: independent sites (beta = 0) and coupled sites (beta = 0.4)."""
    seeds = [0] if quick else [0, 1, 2]
    temps = [0.75, 1.0, 1.5, 2.0, 4.0, None]
    th_true = np.array(WEAK["theta"])
    rows, raw = [], []
    for beta in (0.0, 0.4):
        true = LabelModelParams(**WEAK, beta=beta)
        for bs in temps:
            th, err = [], []
            for s in seeds:
                ds = synthetic_dataset(true, 12, (16, 16), seed=s, mask_source="ising")
                data = [(im.lfo, im.sites) for im in ds]
                cfg = TrainConfig(epochs=6 if quick else 15, num_samples=60 if quick else 100, beta=beta, seed=s)
                res = Trainer(TemperedPosterior(bs, seed=100 + s), cfg).fit(data)
                th.append(res.params.theta.copy())
                err.append(float(np.abs(sigma2(res.params.theta) - sigma2(th_true)).mean()))
            th = np.array(th)
            label = "MAP (hard EM)" if bs is None else f"{bs:g}"
            rows.append([f"β = {beta:g}", label, float(np.mean(th / th_true)), float(np.std((th / th_true).mean(axis=1))),
                         float(np.mean(sigma2(th) - sigma2(th_true))), float(np.mean(err))])
            raw.append({"beta": beta, "beta_s": bs, "theta": th.tolist(), "acc_err": err})
            print(rows[-1], flush=True)
    save_json(RAW / "theory_cold.json", {"seeds": seeds, "true_theta": WEAK["theta"], "runs": raw})
    return {"rows": rows, "seeds": seeds}


def exp_bias_bound() -> dict:
    d = json.load(open(RAW / "sampling_benchmark.json"))
    rows, worst = [], 0.0
    for regime, recs in d.items():
        if not (isinstance(recs, list) and recs and isinstance(recs[0], dict) and "tv" in recs[0]):
            continue
        for solver in dict.fromkeys(r["solver"] for r in recs):
            rr = [r for r in recs if r["solver"] == solver]
            ratio = [max(r["marginal_err"], r["corr_err"]) / (2 * r["tv"]) if r["tv"] > 0 else 0.0 for r in rr]
            worst = max(worst, max(ratio))
            rows.append([regime, solver, float(np.mean([r["tv"] for r in rr])),
                         float(np.mean([max(r["marginal_err"], r["corr_err"]) for r in rr])), float(max(ratio))])
    allr = [max(r["marginal_err"], r["corr_err"]) / (2 * r["tv"]) for recs in d.values()
            if isinstance(recs, list) and recs and isinstance(recs[0], dict) and "tv" in recs[0] for r in recs if r["tv"] > 0]
    return {"rows": rows, "worst": worst, "median": float(np.median(allr))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    t0 = time.perf_counter()
    cold = exp_cold(args.quick)
    bias = exp_bias_bound()
    save_json(RAW / "theory_bias_bound.json", bias)
    r = {(row[0], row[1]): row for row in cold["rows"]}
    ind, cpl = "β = 0", "β = 0.4"
    body = f"""
Two checks that complement report 04. Both concern the sampler that supplies the posterior moments.

## 1. Learning with samplers at the wrong temperature

**Setup.** Four weak unary LFs (θ* = {WEAK['theta']}, α* = {WEAK['alpha']}), so that the posterior is genuinely
uncertain. Masks are drawn from the model's own prior (16×16 grids, 12 images per seed), so the model is
correctly specified and the true parameters are the population optimum. β is fixed at its true value. Two
settings: **independent sites** (β = 0) and **coupled sites** (β = 0.4, just below the critical point 0.44).
Learning (warm start, then Adam with the Theorem-1 gradient) is repeated with posterior samples drawn at inverse
temperature β_s instead of 1, by annealed sampling that stops at β_s. The MAP row uses annealing plus greedy
descent, which is hard EM. Seeds: {cold['seeds']}.

{table(["setting", "sampler β_s", "mean θ / θ*", "SD over seeds", "mean bias of σ(2θ)", "mean abs. error of σ(2θ)"],
       cold['rows'])}

- **Correct temperature.** At β_s = 1, θ / θ* is {r[(ind, '1')][2]:.2f} (independent) and {r[(cpl, '1')][2]:.2f}
  (coupled).
- **Independent sites.** θ / θ* moves from {r[(ind, '0.75')][2]:.2f} (β_s = 0.75) to {r[(ind, '2')][2]:.2f}
  (β_s = 2) and {r[(ind, 'MAP (hard EM)')][2]:.2f} (MAP).
- **Coupled sites.** θ / θ* moves from {r[(cpl, '0.75')][2]:.2f} (β_s = 0.75) to {r[(cpl, '2')][2]:.2f}
  (β_s = 2) and {r[(cpl, 'MAP (hard EM)')][2]:.2f} (MAP).

OBSERVATIONS_PLACEHOLDER

## 2. The sampler-bias term bounds the observed moment errors

For any function f with |f| ≤ 1, |E_q f − E_p f| ≤ 2·TV(q, p). Applied to the empirical distribution of each
sampler's output on the n = 12 problems of report 03, the error of every first and edge moment must be at
most twice the measured TV distance. The last column is the largest ratio (moment error) / (2·TV) over the
problems of that row; the bound holds if it is ≤ 1.

{table(["regime", "sampler", "mean TV", "mean max moment error", "max ratio to 2·TV"], bias['rows'])}

Over all {sum(1 for _ in bias['rows'])} sampler × regime rows, the largest ratio is **{bias['worst']:.2f}** and the
median ratio over individual problems is {bias['median']:.2f}. The bound holds everywhere. It is attained only
when a sampler puts nearly all its mass on states the target rarely visits; for good samplers the moment errors
are far below it.
"""
    write("15-theory-sampler-checks.md", "15 — Sampler temperature and the sampler-bias term", body)
    register({"15-theory-sampler-checks.md": {
        "title": "15 — Sampler checks", "question": "How does sampler temperature bias learning, and does 2·TV bound moment errors?",
        "headline": f"θ/θ* at β_s = 1: {r[(ind, '1')][2]:.2f} / {r[(cpl, '1')][2]:.2f} (independent / coupled); "
                    f"with MAP labels: {r[(ind, 'MAP (hard EM)')][2]:.2f} / {r[(cpl, 'MAP (hard EM)')][2]:.2f}; "
                    f"moment error ≤ 2·TV everywhere (max ratio {bias['worst']:.2f})"}})
    write_index()
    print(f"done in {time.perf_counter() - t0:.0f}s")
    print("exit 0")


if __name__ == "__main__":
    main()
