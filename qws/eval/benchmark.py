"""Benchmark runners (doc 06 §2–4): compare solvers on identical IsingProblems."""

from __future__ import annotations

import time
from typing import Any, Mapping, Sequence

import numpy as np

from ..ising import IsingProblem
from ..solvers import ExactSolver, GraphCutSolver, Solver, UnsupportedProblem
from .metrics import hit_rate, relative_energy_gap, tts99, tv_distance


def reference_energy(p: IsingProblem) -> tuple[float, str]:
    """Exact optimum when available: enumeration for small n, graph cut if ferromagnetic."""
    if p.n <= 20:
        return ExactSolver().minimize([p])[0].best()[1], "exact"
    if p.is_ferromagnetic():
        return GraphCutSolver().minimize([p])[0].best()[1], "graphcut"
    return float("nan"), "none"


def map_benchmark(problems: Sequence[IsingProblem], solvers: Mapping[str, Solver], **solver_kw: Any
                  ) -> list[dict[str, Any]]:
    """Energy gap, hit rate and TTS99 per (problem, solver). When no exact reference
    exists, the best energy found by any solver is used and flagged."""
    records = []
    for pi, p in enumerate(problems):
        ref, ref_kind = reference_energy(p)
        rows = []
        for name, solver in solvers.items():
            t0 = time.perf_counter()
            try:
                ss = solver.minimize([p], **solver_kw.get(name, {}))[0]
            except UnsupportedProblem as exc:
                rows.append({"problem": pi, "solver": name, "skipped": str(exc)})
                continue
            wall = time.perf_counter() - t0
            rows.append({"problem": pi, "solver": name, "n": p.n, "best_energy": ss.best()[1], "wall_time_s": wall,
                         "_ss": ss, **{k: v for k, v in ss.info.items() if k in ("kappa", "chain_break_fraction")}})
        if not np.isfinite(ref):
            found = [r["best_energy"] for r in rows if "best_energy" in r]
            ref, ref_kind = (min(found), "best_found") if found else (float("nan"), "none")
        for r in rows:
            ss = r.pop("_ss", None)
            if ss is not None:
                ph = hit_rate(ss.energies, ss.counts, ref)
                r.update(reference=ref, reference_kind=ref_kind, gap=relative_energy_gap(r["best_energy"], ref),
                         hit_rate=ph, tts99_s=tts99(r["wall_time_s"], ph))
            records.append(r)
    return records


def sampling_benchmark(problems: Sequence[IsingProblem], samplers: Mapping[str, Solver], num_samples: int = 1000,
                       **sampler_kw: Any) -> list[dict[str, Any]]:
    """TV distance and marginal error against the exact Boltzmann distribution (small n)."""
    exact = ExactSolver()
    records = []
    for pi, p in enumerate(problems):
        _, probs = exact.distribution(p)
        mu, corr = exact.moments(p)
        for name, sampler in samplers.items():
            t0 = time.perf_counter()
            ss = sampler.sample([p], num_samples, **sampler_kw.get(name, {}))[0]
            records.append({"problem": pi, "solver": name, "n": p.n, "wall_time_s": time.perf_counter() - t0,
                            "tv": tv_distance(ss.samples, ss.counts, probs),
                            "marginal_err": float(np.abs(ss.mean() - mu).max()),
                            "corr_err": float(np.abs(ss.edge_corr(p.edges) - corr).max()) if p.m else 0.0})
    return records
