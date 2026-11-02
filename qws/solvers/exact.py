"""Exact enumeration backend: ground truth for small problems (doc 04 §1)."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from scipy.special import logsumexp

from ..ising import IsingProblem
from .base import SampleSet, Solver, UnsupportedProblem


def all_spins(n: int) -> np.ndarray:
    """All 2^n configurations, shape (2^n, n), int8; row z has y_i = 1 - 2 * bit_i(z)."""
    z = np.arange(2**n, dtype=np.int64)[:, None]
    bits = (z >> np.arange(n)) & 1
    return (1 - 2 * bits).astype(np.int8)


class ExactSolver(Solver):
    name = "exact"
    capabilities = frozenset({"minimize", "sample", "marginals", "log_partition"})

    def __init__(self, max_n: int = 20, seed: int | None = None):
        self.max_n = max_n
        self.rng = np.random.default_rng(seed)

    def _enumerate(self, p: IsingProblem) -> tuple[np.ndarray, np.ndarray]:
        if p.n > self.max_n:
            raise UnsupportedProblem(f"exact solver limited to n <= {self.max_n} (got {p.n})")
        Y = all_spins(p.n)
        return Y, p.energy(Y)

    def distribution(self, p: IsingProblem, beta: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
        """All configurations and their Boltzmann probabilities at inverse temperature beta."""
        Y, E = self._enumerate(p)
        logw = -beta * E
        return Y, np.exp(logw - logsumexp(logw))

    def log_partition(self, p: IsingProblem, beta: float = 1.0) -> float:
        _, E = self._enumerate(p)
        return float(logsumexp(-beta * E))

    def moments(self, p: IsingProblem, beta: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
        """Exact <y_i> and <y_i y_k> on p.edges."""
        Y, prob = self.distribution(p, beta)
        Yf = Y.astype(float)
        return prob @ Yf, prob @ (Yf[:, p.edges[:, 0]] * Yf[:, p.edges[:, 1]])

    def minimize(self, problems: Sequence[IsingProblem], tol: float = 1e-9, **_: Any) -> list[SampleSet]:
        out = []
        for p in problems:
            Y, E = self._enumerate(p)
            ground = E <= E.min() + tol
            out.append(SampleSet.from_samples(p, Y[ground], info={"solver": self.name, "degeneracy": int(ground.sum())}))
        return out

    def sample(self, problems: Sequence[IsingProblem], num_samples: int, beta: float = 1.0, **_: Any
               ) -> list[SampleSet]:
        out = []
        for p in problems:
            Y, prob = self.distribution(p, beta)
            counts = self.rng.multinomial(num_samples, prob)
            keep = counts > 0
            out.append(SampleSet.from_samples(p, Y[keep], counts[keep], {"solver": self.name, "beta": beta}))
        return out
