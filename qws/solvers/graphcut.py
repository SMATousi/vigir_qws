"""Exact MAP for ferromagnetic (submodular) problems via s–t minimum cut (doc 04 §1).

Refuses problems with any negative coupling instead of returning a wrong answer.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from ..ising import IsingProblem, ising_to_qubo
from .base import SampleSet, Solver, UnsupportedProblem


class GraphCutSolver(Solver):
    name = "graphcut"
    capabilities = frozenset({"minimize"})

    def minimize(self, problems: Sequence[IsingProblem], **_: Any) -> list[SampleSet]:
        return [self._solve(p) for p in problems]

    def _solve(self, p: IsingProblem) -> SampleSet:
        import maxflow

        if not p.is_ferromagnetic():
            raise UnsupportedProblem("graphcut requires all couplings J >= 0 (submodular energy)")
        q = ising_to_qubo(p)  # quadratic = -4 J <= 0
        unary = q.linear.copy()
        g = maxflow.Graph[float]()
        nodes = g.add_nodes(p.n)
        # b x_i x_k = b x_i + (-b) x_i (1 - x_k), with -b >= 0 paid when x_i = 1, x_k = 0
        for (i, k), b in zip(q.edges, q.quadratic):
            if b == 0:
                continue
            unary[i] += b
            g.add_edge(nodes[k], nodes[i], -b, 0.0)
        # x_i = 1 <=> node on the sink side. tedge(i, cap_source, cap_sink):
        # cap_source is paid if i ends on the sink side, cap_sink if on the source side.
        for i, a in enumerate(unary):
            if a >= 0:
                g.add_tedge(nodes[i], a, 0.0)
            else:
                g.add_tedge(nodes[i], 0.0, -a)
        g.maxflow()
        x = np.array([g.get_segment(nodes[i]) for i in range(p.n)], dtype=np.int8)
        y = 2 * x - 1
        return SampleSet.from_samples(p, y, info={"solver": self.name})
