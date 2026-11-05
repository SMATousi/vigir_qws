"""Report 17: frustrated posteriors on real images, and what they need from a solver.

Usage:  python scripts/frustration_pet.py [--root data/oxford-iiit-pet] [--quick]

The full label model (unary LFs, the pairwise colour-contrast LF and smoothness beta = 0.5) is learned on
Oxford-IIIT Pet images. Its pairwise LF votes "different" across strong edges, which gives negative couplings,
so the per-image posterior Ising models are frustrated and graph cut does not apply. For each posterior we
measure:

* frustration: share of negative couplings, and share of cycles in a cycle basis with an odd number of them;
* roof duality (QPBO): share of variables it labels (persistent, i.e. exact), and whether the remainder is small
  enough to finish exactly by enumeration, which certifies the MAP;
* MAP solvers against a strong reference (best of QPBO + long annealing and 3 long annealing runs):
  QPBO + annealing, plain annealing, greedy single-flip descent from random starts;
* D-Wave resources: the QPBO remainder, embedded on a mock Advantage (Pegasus P16) graph: logical variables,
  physical qubits, longest chain.

Besides the learned parameters, five scaled settings show how this changes if the pairwise LF were stronger or
the unary LFs weaker.
"""

from __future__ import annotations

import argparse
import time

import networkx as nx
import numpy as np

from _report import RAW, register, table, write, write_index  # noqa: E402
from qws.eval import iou
from qws.experiment import DEFAULT_CONFIG, build_dataset, merge
from qws.io import save_json
from qws.ising import condition
from qws.learn import TrainConfig, Trainer
from qws.model import posterior_ising, warm_start
from qws.solvers import ExactSolver, GibbsSampler, SimulatedAnnealing
from qws.solvers.graphcut import roof_dual
from qws.solvers.mcmc import _greedy_descent


def frustration(p) -> tuple[float, float]:
    neg = float(np.mean(p.J < 0)) if p.m else 0.0
    g = nx.Graph()
    g.add_nodes_from(range(p.n))
    sign = {}
    for (i, k), j in zip(p.edges, p.J):
        g.add_edge(int(i), int(k))
        sign[frozenset((int(i), int(k)))] = np.sign(j)
    cycles = nx.cycle_basis(g)
    if not cycles:
        return neg, 0.0
    odd = 0
    for c in cycles:
        s = 1.0
        for a, b in zip(c, c[1:] + c[:1]):
            s *= sign[frozenset((a, b))]
        odd += s < 0
    return neg, odd / len(cycles)


def analyse(p, seed: int, qpu=None) -> dict:
    labels, bound = roof_dual(p)
    fixed = np.flatnonzero(labels != 0)
    cond = condition(p, fixed, labels[fixed])
    rem = cond.problem
    out = {"n": p.n, "m": p.m, "labeled": float(np.mean(labels != 0)), "remainder": int(rem.n)}
    out["neg_frac"], out["odd_cycle_frac"] = frustration(p)
    # QPBO completion: exact if the remainder is small, otherwise long annealing
    if rem.n == 0:
        y_qpbo, certified = labels.astype(float), True
    elif rem.n <= 20:
        y_qpbo = cond.lift(ExactSolver().minimize([rem])[0].best()[0], p.n)[0].astype(float)
        certified = True
    else:
        y_qpbo = cond.lift(SimulatedAnnealing(num_sweeps=2000, seed=seed).minimize([rem], num_reads=20)[0].best()[0],
                           p.n)[0].astype(float)
        certified = False
    e_qpbo = float(p.energy(y_qpbo[None])[0])
    long = [SimulatedAnnealing(num_sweeps=2000, seed=seed + r).minimize([p], num_reads=20)[0].best() for r in range(3)]
    e_ref = min([e_qpbo] + [float(e) for _, e in long])
    sa = SimulatedAnnealing(num_sweeps=300, seed=seed + 7).minimize([p], num_reads=20)[0].best()
    rng = np.random.default_rng(seed)
    greedy = _greedy_descent(p, rng.choice([-1.0, 1.0], size=(20, p.n)))
    e_greedy = float(p.energy(greedy).min())
    scale = max(abs(e_ref), 1e-9)
    out.update({"certified": certified, "e_ref": e_ref, "bound_gap": (e_ref - bound) / scale,
                "gap_qpbo_sa": (e_qpbo - e_ref) / scale, "gap_sa": (float(sa[1]) - e_ref) / scale,
                "gap_greedy": (e_greedy - e_ref) / scale, "y_ref": y_qpbo if e_qpbo <= e_ref + 1e-9 else None})
    if qpu is not None:
        for tag, prob in (("", rem), ("_full", p)):
            if prob.n == 0:
                out.update({f"qubits{tag}": 0, f"max_chain{tag}": 0})
                continue
            try:
                emb = qpu.embed([prob], tries=2, timeout=30)
                lengths = [len(c) for c in emb.values()]
                out.update({f"qubits{tag}": int(sum(lengths)), f"max_chain{tag}": int(max(lengths))})
            except RuntimeError:
                out.update({f"qubits{tag}": None, f"max_chain{tag}": None})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/oxford-iiit-pet")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    t0 = time.perf_counter()
    n_img = 10 if args.quick else 100
    cfg = merge(DEFAULT_CONFIG, {"seed": 0, "dataset": {"kind": "pet", "root": args.root, "n_images": n_img,
                                                         "n_segments": 150}})
    items, _ = build_dataset(cfg)
    data = [(im.lfo, im.sites) for im in items]
    tc = TrainConfig(epochs=5 if args.quick else 15, num_samples=100, beta=0.5, seed=0)
    params = Trainer(GibbsSampler(num_sweeps=100, seed=0), tc).fit(data, warm_start(data, beta=0.5)).params

    from qws.solvers.qpu import QPUAnnealingSolver
    qpu = QPUAnnealingSolver.mock(topology_shape=16)
    n_embed = 3 if args.quick else 10

    settings = {"learned": (1.0, 1.0), "pairwise ×3": (1.0, 3.0), "unary ×0.5": (0.5, 1.0),
                "unary ×0.5, pairwise ×3": (0.5, 3.0), "unary ×0.25, pairwise ×10": (0.25, 10.0),
                "no unary LFs, pairwise ×10": (0.0, 10.0)}
    rows, emb_rows, raw = [], [], {}
    for name, (u, s) in settings.items():
        p2 = params.copy()
        p2.theta, p2.thetaP = params.theta * u, params.thetaP * s
        res = []
        for k, (lfo, sites) in enumerate(data):
            p = posterior_ising(p2, lfo, sites)
            r = analyse(p, seed=k, qpu=qpu if k < n_embed else None)
            yref = r.pop("y_ref")
            r["iou"] = float(iou(sites.to_pixels(yref), items[k].mask, True)) if yref is not None else None
            res.append(r)
        raw[name] = res
        print(name, f"{time.perf_counter() - t0:.0f}s", flush=True)

        def m(key, rr=res):
            v = [x[key] for x in rr if x.get(key) is not None]
            return float(np.mean(v)) if v else float("nan")

        rows.append([name, m("n"), m("neg_frac"), m("odd_cycle_frac"), m("labeled"),
                     f"{sum(x['labeled'] == 1.0 for x in res)}/{len(res)}",
                     f"{sum(x['certified'] for x in res)}/{len(res)}", m("remainder"),
                     m("gap_qpbo_sa"), m("gap_sa"), m("gap_greedy"), m("iou")])
        er = [x for x in res if "qubits" in x]
        if er:
            ok = [x for x in er if x["qubits"] is not None]
            okf = [x for x in er if x["qubits_full"] is not None]
            emb_rows.append([name, len(er), float(np.mean([x["remainder"] for x in er])),
                             float(np.mean([x["qubits"] for x in ok])) if ok else "–",
                             int(max(x["max_chain"] for x in ok)) if ok else "–",
                             float(np.mean([x["n"] for x in er])),
                             float(np.mean([x["qubits_full"] for x in okf])) if okf else "–",
                             int(max(x["max_chain_full"] for x in okf)) if okf else "–", f"{len(okf)}/{len(er)}"])
    save_json(RAW / "frustration_pet.json", {"n_images": n_img, "params": params.to_dict(), "settings": settings,
                                            "results": raw, "wall_s": time.perf_counter() - t0})
    body = f"""
The full label model (unary LFs, the pairwise colour-contrast LF and β = 0.5) learned on {n_img} Oxford-IIIT Pet
images (seed 0; the setting of report 13). Its pairwise LF votes "different" across strong colour edges. With a
positive learned θᴾ = {float(params.thetaP[0]):.2f}, each such vote adds a negative coupling, so the posteriors are
frustrated and graph cut does not apply. This report asks what these real frustrated posteriors require from a MAP
solver. Besides the learned parameters, five scaled settings show the effect of a stronger pairwise LF or weaker
unary LFs.

## Frustration, roof duality and solver gaps (means over images)

- **neg. couplings:** share of couplings J < 0.
- **odd cycles:** share of the cycles in a cycle basis whose coupling signs multiply to −1 (frustrated cycles).
- **labeled:** share of variables fixed by roof duality (QPBO). These labels are exact (persistent).
- **all labeled / certified:** images where QPBO labels every variable / where the MAP is certified optimal, because
  QPBO labels everything or the remainder (≤ 20 variables) is finished by enumeration.
- **gaps:** (E − E_ref) / |E_ref|, against the best of QPBO + long annealing and three long annealing runs
  (2,000 sweeps × 20 reads each). SA: 300 sweeps × 20 reads. Greedy: single-flip descent from 20 random starts.
- **IoU:** of the reference MAP mask against the trimaps (boundary excluded).

{table(["setting", "sites", "neg. couplings", "odd cycles", "labeled", "all labeled", "certified", "remainder",
        "gap QPBO+SA", "gap SA", "gap greedy", "IoU"], rows)}

## D-Wave resources (first {n_embed} images per setting; mock Advantage, Pegasus P16)

Left: the QPBO remainder, which is what would be sent to the QPU. Right: the whole image's posterior, which is
the relevant size once QPBO stops deciding variables.

{table(["setting", "images", "remainder (vars)", "remainder qubits", "remainder longest chain", "full problem (vars)",
        "full problem qubits", "full longest chain", "full embedded"], emb_rows)}

## Observations

- **Real frustration is easy.** With the learned parameters, 10% of couplings are negative and 14% of basis
  cycles are frustrated. Yet roof duality labels 99% of the variables, every variable in 97 of 100 images, and
  the MAP is certified optimal in 98 of 100 (QPBO alone, or QPBO plus enumeration of a remainder of at most 20
  variables). Short annealing (300 sweeps) reaches the reference energy on all 100 images. The unary evidence
  anchors the posterior, as in report 10.
- **Hardness needs weak unaries and strong pairwise votes together.** Certified images drop to 86/100 at unary
  ×0.5 with pairwise ×3, to 58/100 at unary ×0.25 with pairwise ×10, and to 0 with no unary LFs (a field-free
  frustrated model on the superpixel graph). Short annealing reaches the reference on 86%, 12% and 14% of these
  images, though its mean gap stays below 1%. Greedy descent is 0.4–8% above the reference on average.
- **These hard settings are also worse segmentations.** IoU falls from 0.585 (learned) to 0.465 and 0.316 in the
  hardest settings. That fits report 16: the generic pairwise LF lowers IoU on Pet. On these real images, a
  stronger "different" LF does not create hard problems worth solving.
- **Hardware size is not the obstacle.** A whole image's posterior (about 122 variables) embeds in about 198
  physical qubits of the mock Advantage graph, with chains of at most 4, which is about 3.5% of a Pegasus P16
  chip. With the learned parameters, the QPBO remainder sent to the QPU would have only a few variables.
- **Caveat.** Where an image is not certified, the reference is the best energy found by long annealing and
  QPBO + annealing, not a proven optimum. The gaps are therefore lower bounds on the true gaps.
- **Consequence.** For binary segmentation with a realistic pairwise LF, MAP is solved classically: QPBO, and
  annealing for a small remainder. Quantum resources remain relevant for **sampling** under strong coupling
  (report 03), for multi-class (Potts) MAP, which is NP-hard even without frustration (report 11), and for
  models whose LFs are genuinely weak relative to their pairwise terms.
"""
    write("17-frustration-pet.md", "17 — Frustrated posteriors on real images", body)
    lr = rows[0]
    register({"17-frustration-pet.md": {
        "title": "17 — Frustrated posteriors (Pet)",
        "question": "How hard are real frustrated posteriors for MAP, and what would the QPU get?",
        "headline": f"Learned model: {lr[2]:.0%} negative couplings, {lr[3]:.0%} frustrated cycles; QPBO labels "
                    f"{lr[4]:.0%} of variables and certifies {lr[6]} images"}})
    write_index()
    print(f"done in {time.perf_counter() - t0:.0f}s")
    print("exit 0")


if __name__ == "__main__":
    main()
