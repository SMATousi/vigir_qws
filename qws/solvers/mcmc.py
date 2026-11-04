"""Classical MCMC backends: Gibbs sampling (with persistent chains and polishing) and
simulated annealing. Both use heat-bath updates over graph-colour classes, vectorized
across chains."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from ..ising import IsingProblem
from .base import SampleSet, Solver, greedy_coloring


class _Kernel:
    """Precomputed colour classes and row blocks of the coupling matrix for one problem."""

    def __init__(self, p: IsingProblem):
        self.p = p
        self.classes = greedy_coloring(p)
        W = p.coupling_matrix
        self.rows = [W[idx, :] for idx in self.classes]

    def sweep(self, states: np.ndarray, beta: float, rng: np.random.Generator) -> None:
        """One heat-bath sweep in place; states is (R, n) float in {-1, +1}."""
        h = self.p.h
        for idx, rows in zip(self.classes, self.rows):
            m = h[idx][None, :] + (rows @ states.T).T
            p_up = 0.5 * (1.0 + np.tanh(beta * m))  # = sigmoid(2 beta m)
            states[:, idx] = np.where(rng.random(p_up.shape) < p_up, 1.0, -1.0)


def gibbs_polish(p: IsingProblem, samples: np.ndarray, sweeps: int, rng: np.random.Generator | None = None,
                 beta: float = 1.0) -> np.ndarray:
    """Run ``sweeps`` Gibbs sweeps at ``beta`` from each sample (quantum-seeded polishing).

    The kernel leaves Boltzmann(beta) invariant, so it never increases the TV distance
    to it (paper Proposition 10).
    """
    rng = rng or np.random.default_rng()
    states = np.atleast_2d(samples).astype(float)
    if sweeps > 0 and p.n:
        k = _Kernel(p)
        for _ in range(sweeps):
            k.sweep(states, beta, rng)
    return states.astype(np.int8)


class GibbsSampler(Solver):
    """Parallel heat-bath Gibbs chains.

    Non-persistent mode starts ``num_samples`` chains uniformly at random and returns
    their states after ``num_sweeps``. Persistent mode (PCD-style) keeps chains per
    ``problem.meta['chain_key']`` across calls, so later calls continue from the
    previous states.
    """

    name = "gibbs"
    capabilities = frozenset({"sample"})

    def __init__(self, num_sweeps: int = 200, persistent: bool = False, seed: int | None = None):
        self.num_sweeps = num_sweeps
        self.persistent = persistent
        self.rng = np.random.default_rng(seed)
        self._chains: dict[Any, np.ndarray] = {}

    def sample(self, problems: Sequence[IsingProblem], num_samples: int, beta: float = 1.0,
               num_sweeps: int | None = None, init: np.ndarray | None = None, **_: Any) -> list[SampleSet]:
        sweeps = self.num_sweeps if num_sweeps is None else num_sweeps
        out = []
        for p in problems:
            key = p.meta.get("chain_key")
            if init is not None:
                states = np.atleast_2d(init).astype(float).copy()
            elif self.persistent and key is not None and key in self._chains \
                    and self._chains[key].shape == (num_samples, p.n):
                states = self._chains[key].astype(float)
            else:
                states = self.rng.choice([-1.0, 1.0], size=(num_samples, p.n))
            k = _Kernel(p)
            for _ in range(sweeps):
                k.sweep(states, beta, self.rng)
            if self.persistent and key is not None:
                self._chains[key] = states.astype(np.int8)  # spins fit in int8: 8x less memory
            out.append(SampleSet.from_samples(p, states, info={"solver": self.name, "sweeps": sweeps,
                                                               "beta": beta, "colors": len(k.classes)}))
        return out


class SimulatedAnnealing(Solver):
    """Heat-bath simulated annealing on a geometric inverse-temperature schedule.

    ``minimize`` anneals to ``beta_max``; ``sample`` anneals to ``beta`` (default 1)
    and stops there (annealed sampling).
    """

    name = "sa"
    capabilities = frozenset({"minimize", "sample"})

    def __init__(self, num_sweeps: int = 500, beta_min: float = 0.05, beta_max: float = 20.0,
                 seed: int | None = None):
        self.num_sweeps, self.beta_min, self.beta_max = num_sweeps, beta_min, beta_max
        self.rng = np.random.default_rng(seed)

    def _anneal(self, p: IsingProblem, reads: int, beta_end: float, sweeps: int) -> np.ndarray:
        scale = max(np.abs(p.h).max(initial=0.0), np.abs(p.J).max(initial=0.0), 1e-12)
        betas = np.geomspace(min(self.beta_min / scale, beta_end), beta_end, sweeps)
        states = self.rng.choice([-1.0, 1.0], size=(reads, p.n))
        k = _Kernel(p)
        for b in betas:
            k.sweep(states, b, self.rng)
        return states

    def minimize(self, problems: Sequence[IsingProblem], num_reads: int = 20, num_sweeps: int | None = None,
                 **_: Any) -> list[SampleSet]:
        sweeps = num_sweeps or self.num_sweeps
        out = []
        for p in problems:
            scale = max(np.abs(p.h).max(initial=0.0), np.abs(p.J).max(initial=0.0), 1e-12)
            states = self._anneal(p, num_reads, self.beta_max / scale, sweeps)
            # zero-temperature cleanup: greedy single-flip descent
            states = _greedy_descent(p, states)
            out.append(SampleSet.from_samples(p, states, info={"solver": self.name, "sweeps": sweeps},
                                              aggregate=True))
        return out

    def sample(self, problems: Sequence[IsingProblem], num_samples: int, beta: float = 1.0,
               num_sweeps: int | None = None, **_: Any) -> list[SampleSet]:
        sweeps = num_sweeps or self.num_sweeps
        return [SampleSet.from_samples(p, self._anneal(p, num_samples, beta, sweeps),
                                       info={"solver": self.name, "beta": beta, "sweeps": sweeps})
                for p in problems]


def _greedy_descent(p: IsingProblem, states: np.ndarray, max_iter: int = 1000) -> np.ndarray:
    """Flip any spin whose flip lowers the energy until a local minimum is reached."""
    s = states.copy()
    k = _Kernel(p)
    for _ in range(max_iter):
        changed = False
        for idx, rows in zip(k.classes, k.rows):
            m = p.h[idx][None, :] + (rows @ s.T).T
            want = np.where(m > 0, 1.0, np.where(m < 0, -1.0, s[:, idx]))
            if np.any(want != s[:, idx]):
                changed = True
                s[:, idx] = want
        if not changed:
            break
    return s
