"""Exact MAP for ferromagnetic (submodular) problems via s–t minimum cut (doc 04 §1),
and roof duality / QPBO for general problems (roadmap 4.3; doc 02 §6.3).

``GraphCutSolver`` refuses problems with any negative coupling instead of returning a
wrong answer. ``QPBOSolver`` handles any problem: it labels the persistent variables
exactly and completes the rest with a fallback solver.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from ..ising import IsingProblem, condition, ising_to_qubo
from .base import SampleSet, Solver, UnsupportedProblem


def solve_submodular_qubo(linear: np.ndarray, edges: np.ndarray, quadratic: np.ndarray) -> np.ndarray:
    """Exact minimizer x in {0,1}^n of sum a_i x_i + sum b_ik x_i x_k, requiring every b_ik <= 0."""
    import maxflow

    if np.any(quadratic > 1e-12):
        raise UnsupportedProblem("submodular QUBO requires all quadratic coefficients <= 0")
    n = linear.shape[0]
    unary = np.asarray(linear, dtype=float).copy()
    g = maxflow.Graph[float]()
    nodes = g.add_grid_nodes((n,))
    edges = np.asarray(edges).reshape(-1, 2)
    quad = np.asarray(quadratic, dtype=float)
    nz = quad != 0
    i, k, b = edges[nz, 0], edges[nz, 1], quad[nz]
    # b x_i x_k = b x_i + (-b) x_i (1 - x_k), with -b >= 0 paid when x_i = 1, x_k = 0:
    # unary b on x_i, plus a directed edge k -> i of capacity -b (vectorized)
    np.add.at(unary, i, b)
    if b.size:
        g.add_edges(nodes[k], nodes[i], -b, np.zeros_like(b))
    # x_i = 1 <=> node on the sink side. Terminal capacities: source cap is paid on the sink
    # side (cost a >= 0 for x = 1); sink cap is paid on the source side (cost -a for x = 0).
    g.add_grid_tedges(nodes, np.maximum(unary, 0.0), np.maximum(-unary, 0.0))
    g.maxflow()
    return g.get_grid_segments(nodes).astype(np.int8)


class GraphCutSolver(Solver):
    name = "graphcut"
    capabilities = frozenset({"minimize"})

    def minimize(self, problems: Sequence[IsingProblem], **_: Any) -> list[SampleSet]:
        return [self._solve(p) for p in problems]

    def _solve(self, p: IsingProblem) -> SampleSet:
        if not p.is_ferromagnetic():
            raise UnsupportedProblem("graphcut requires all couplings J >= 0 (submodular energy)")
        q = ising_to_qubo(p)  # quadratic = -4 J <= 0
        x = solve_submodular_qubo(q.linear, q.edges, q.quadratic)
        return SampleSet.from_samples(p, 2 * x - 1, info={"solver": self.name})


def roof_dual(p: IsingProblem) -> tuple[np.ndarray, float]:
    """Roof-duality partial labeling of an arbitrary Ising problem.

    Returns ``labels`` in {-1, 0, +1} (0 = unlabeled) and a lower bound on min E.
    With E(x) = c + sum a_i x_i + sum b_ij x_i x_j and copies y_i meant to equal 1 - x_i,
    the doubled energy

        E2 = c + 1/2 sum_i a_i (x_i + 1 - y_i)
               + 1/2 sum_{b<0} b (x_i x_j + (1 - y_i)(1 - y_j))
               + 1/2 sum_{b>0} b (x_i (1 - y_j) + (1 - y_i) x_j)

    equals E whenever y = 1 - x and is submodular in (x, y), so one max-flow minimizes
    it exactly. min E2 <= min E. Every variable with y_i = 1 - x_i in the minimizer is
    weakly persistent: some global minimizer of E agrees with all of them at once
    (roof duality; Hammer, Hansen & Simeone 1984; Boros & Hammer 2002).
    """
    q = ising_to_qubo(p)
    n = p.n
    lin = np.zeros(2 * n)  # x: 0..n-1, y: n..2n-1
    const = q.offset + 0.5 * q.linear.sum()
    lin[:n] += 0.5 * q.linear
    lin[n:] -= 0.5 * q.linear
    e_list, b_list = [], []
    for (i, j), b in zip(q.edges, q.quadratic):
        h = 0.5 * b
        if b < 0:  # x_i x_j and (1 - y_i)(1 - y_j) = 1 - y_i - y_j + y_i y_j
            e_list += [(i, j), (n + i, n + j)]
            b_list += [h, h]
            lin[n + i] -= h
            lin[n + j] -= h
            const += h
        elif b > 0:  # x_i (1 - y_j) and (1 - y_i) x_j
            e_list += [(i, n + j), (j, n + i)]
            b_list += [-h, -h]
            lin[i] += h
            lin[j] += h
    edges = np.array(e_list, dtype=np.int64).reshape(-1, 2)
    quad = np.array(b_list, dtype=float)
    z = solve_submodular_qubo(lin, edges, quad)
    x, y = z[:n], z[n:]
    bound = const + float(z @ lin) + (float((z[edges[:, 0]] * z[edges[:, 1]]) @ quad) if edges.size else 0.0)
    labels = np.where(y == 1 - x, 2 * x - 1, 0).astype(np.int8)
    return labels, bound


class QPBOSolver(Solver):
    """MAP via roof duality, completed by a fallback solver on the unlabeled variables.

    The persistent labels are exact. The completion is exact if the fallback is
    (``ExactSolver`` for small remainders); otherwise it is heuristic. ``info`` reports
    the labeled fraction and the roof-dual lower bound.
    """

    name = "qpbo"
    capabilities = frozenset({"minimize"})

    def __init__(self, fallback: Solver | None = None, exact_max_n: int = 20):
        self.fallback = fallback
        self.exact_max_n = exact_max_n

    def persistent(self, p: IsingProblem):
        """Conditioned problem on the unlabeled variables (preprocessing, doc 02 §6.3)."""
        labels, _ = roof_dual(p)
        fixed = np.flatnonzero(labels != 0)
        return condition(p, fixed, labels[fixed])

    def minimize(self, problems: Sequence[IsingProblem], **kw: Any) -> list[SampleSet]:
        from .exact import ExactSolver
        from .mcmc import SimulatedAnnealing

        out = []
        for p in problems:
            labels, bound = roof_dual(p)
            fixed = np.flatnonzero(labels != 0)
            cond = condition(p, fixed, labels[fixed])
            if cond.problem.n == 0:
                y = labels.copy()
                exact = True
            else:
                solver = self.fallback or (ExactSolver() if cond.problem.n <= self.exact_max_n
                                           else SimulatedAnnealing(seed=0))
                exact = isinstance(solver, ExactSolver)
                y = cond.lift(solver.minimize([cond.problem], **kw)[0].best()[0], p.n)[0]
            out.append(SampleSet.from_samples(p, y, info={"solver": self.name, "labeled_fraction": fixed.size / max(p.n, 1),
                                                          "lower_bound": bound, "completion_exact": exact}))
        return out
