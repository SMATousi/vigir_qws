"""K-class (Potts) problems, the one-hot QUBO encoding and Potts samplers
(spec doc 07; paper Section 8 and Proposition 11; roadmap 4.1).

Energy convention (as for Ising, minimized, p ∝ exp(-E)):

    E(y) = offset - sum_i u[i, y_i] - sum_{(i,k)} Jt_ik 1[y_i = y_k],   y_i in {0..K-1}.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..ising import QUBO, IsingProblem, qubo_to_ising


@dataclass(frozen=True)
class PottsProblem:
    unary: np.ndarray   # (n, K)
    edges: np.ndarray   # (m, 2), i < k
    Jt: np.ndarray      # (m,)
    offset: float = 0.0

    @property
    def n(self) -> int:
        return int(self.unary.shape[0])

    @property
    def K(self) -> int:
        return int(self.unary.shape[1])

    def energy(self, labels: np.ndarray) -> np.ndarray:
        y = np.atleast_2d(labels)
        e = self.offset - self.unary[np.arange(self.n)[None, :], y].sum(axis=1)
        if self.edges.size:
            e = e - (y[:, self.edges[:, 0]] == y[:, self.edges[:, 1]]) @ self.Jt
        return e

    def penalty_bound(self) -> float:
        """max_{i,c} (|u_ic| + sum_k |Jt_ik|): any A above it makes every minimizer of the
        one-hot QUBO feasible (paper Proposition 11)."""
        deg = np.zeros(self.n)
        np.add.at(deg, self.edges[:, 0], np.abs(self.Jt))
        np.add.at(deg, self.edges[:, 1], np.abs(self.Jt))
        return float((np.abs(self.unary) + deg[:, None]).max())


# ------------------------------------------------------------------------------ one-hot encoding
def onehot_index(i: int | np.ndarray, c: int | np.ndarray, K: int):
    """Binary variable index of (site i, class c)."""
    return np.asarray(i) * K + np.asarray(c)


def to_onehot_qubo(p: PottsProblem, A: float | None = None, margin: float = 1e-3) -> tuple[QUBO, float]:
    """One-hot QUBO with n*K variables:

        E_K(x) = offset - sum u_ic x_ic - sum Jt_ik sum_c x_ic x_kc + A sum_i (1 - sum_c x_ic)^2.

    Expanding the penalty with x^2 = x: (1 - sum_c x_c)^2 = 1 - sum_c x_c + 2 sum_{c<d} x_c x_d.
    The default A is the Proposition-11 bound plus ``margin``.
    """
    K = p.K
    A = p.penalty_bound() + margin if A is None else A
    linear = (-p.unary - A).reshape(-1)
    e_list, q_list = [], []
    for i in range(p.n):
        for c, d in itertools.combinations(range(K), 2):
            e_list.append((onehot_index(i, c, K), onehot_index(i, d, K)))
            q_list.append(2.0 * A)
    for (i, k), jt in zip(p.edges, p.Jt):
        for c in range(K):
            e_list.append((onehot_index(i, c, K), onehot_index(k, c, K)))
            q_list.append(-jt)
    edges = np.array(e_list, dtype=np.int64).reshape(-1, 2)
    return QUBO(linear, edges, np.array(q_list, dtype=float), p.offset + A * p.n), A


def to_onehot_ising(p: PottsProblem, A: float | None = None, margin: float = 1e-3) -> tuple[IsingProblem, float]:
    """The one-hot QUBO as an IsingProblem, solvable by every binary backend (incl. the QPU path)."""
    q, A = to_onehot_qubo(p, A, margin)
    return qubo_to_ising(q, kind="onehot", meta={"K": p.K, "n_sites": p.n, "penalty": A}), A


def decode_onehot(spins: np.ndarray, p: PottsProblem) -> tuple[np.ndarray, np.ndarray]:
    """Spins of the one-hot model -> labels, plus a per-sample infeasibility flag.

    Infeasible sites are repaired by choosing, among their active classes (or among all
    classes if none is active), the one with the largest unary score (doc 07 §2).
    """
    x = (np.atleast_2d(spins) > 0).reshape(-1, p.n, p.K)
    nact = x.sum(axis=2)
    infeasible = (nact != 1).any(axis=1)
    scores = np.where(x, p.unary[None], -np.inf)
    scores = np.where((nact == 0)[..., None], p.unary[None], scores)
    return scores.argmax(axis=2), infeasible


# ------------------------------------------------------------------------------ exact and sampling
def all_labelings(n: int, K: int) -> np.ndarray:
    return np.array(list(itertools.product(range(K), repeat=n)), dtype=np.int64).reshape(-1, n)


def exact_potts(p: PottsProblem, beta: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """All K^n labelings and their Boltzmann probabilities (small problems only)."""
    if p.K**p.n > 2_000_000:
        raise ValueError("exact Potts enumeration too large")
    Y = all_labelings(p.n, p.K)
    lw = -beta * p.energy(Y)
    w = np.exp(lw - lw.max())
    return Y, w / w.sum()


def potts_coloring(p: PottsProblem) -> list[np.ndarray]:
    from ..solvers import greedy_coloring

    return greedy_coloring(IsingProblem(np.zeros(p.n), p.edges, np.ones(len(p.edges))))


class PottsGibbs:
    """Heat-bath Gibbs sampler for Potts models (colour classes updated in parallel), with
    an optional annealing schedule for MAP (Potts simulated annealing baseline)."""

    def __init__(self, num_sweeps: int = 100, seed: int | None = None):
        self.num_sweeps = num_sweeps
        self.rng = np.random.default_rng(seed)

    def _sweep(self, p: PottsProblem, Y: np.ndarray, classes, nbrs, beta: float) -> None:
        for idx in classes:
            # field[s, i, c] = u[i, c] + sum_k Jt_ik 1[y_k = c]
            field = np.repeat(p.unary[idx][None], Y.shape[0], axis=0)
            for pos, i in enumerate(idx):
                ks, js = nbrs[i]
                if ks.size:
                    np.add.at(field[:, pos, :], (np.arange(Y.shape[0])[:, None], Y[:, ks]), js[None, :])
            logits = beta * field
            logits -= logits.max(axis=2, keepdims=True)
            prob = np.exp(logits)
            prob /= prob.sum(axis=2, keepdims=True)
            u = self.rng.random(prob.shape[:2])[..., None]
            Y[:, idx] = (prob.cumsum(axis=2) < u).sum(axis=2).clip(max=p.K - 1)

    def _setup(self, p: PottsProblem):
        nbrs: dict[int, Any] = {i: ([], []) for i in range(p.n)}
        for (i, k), jt in zip(p.edges, p.Jt):
            nbrs[i][0].append(k)
            nbrs[i][1].append(jt)
            nbrs[k][0].append(i)
            nbrs[k][1].append(jt)
        nbrs = {i: (np.array(a, dtype=np.int64), np.array(b, dtype=float)) for i, (a, b) in nbrs.items()}
        classes = potts_coloring(p)
        return nbrs, classes

    def sample(self, p: PottsProblem, num_samples: int, beta: float = 1.0, init: np.ndarray | None = None) -> np.ndarray:
        nbrs, classes = self._setup(p)
        Y = self.rng.integers(0, p.K, size=(num_samples, p.n)) if init is None else np.array(init, copy=True)
        for _ in range(self.num_sweeps):
            self._sweep(p, Y, classes, nbrs, beta)
        return Y

    def anneal(self, p: PottsProblem, num_reads: int = 10, beta_max: float = 20.0) -> np.ndarray:
        """Potts simulated annealing; returns the best labeling found."""
        nbrs, classes = self._setup(p)
        scale = max(np.abs(p.unary).max(initial=0.0), np.abs(p.Jt).max(initial=0.0), 1e-12)
        Y = self.rng.integers(0, p.K, size=(num_reads, p.n))
        for b in np.geomspace(0.05 / scale, beta_max / scale, self.num_sweeps):
            self._sweep(p, Y, classes, nbrs, b)
        return Y[np.argmin(p.energy(Y))]
