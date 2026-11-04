"""Solver interface (spec doc 05 §3). Solvers only ever see an IsingProblem."""

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from ..ising import IsingProblem


class UnsupportedProblem(ValueError):
    """Raised instead of silently returning a wrong answer (doc 05 §3)."""


@dataclass
class SampleSet:
    """Samples for one problem. Energies are always recomputed in OUR convention."""

    samples: np.ndarray
    energies: np.ndarray
    counts: np.ndarray
    info: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_samples(cls, problem: IsingProblem, samples: np.ndarray, counts: np.ndarray | None = None,
                     info: dict[str, Any] | None = None, aggregate: bool = False) -> "SampleSet":
        s = np.atleast_2d(np.asarray(samples)).astype(np.int8)
        if s.shape[1] != problem.n:
            raise ValueError(f"samples have {s.shape[1]} variables, problem has {problem.n}")
        c = np.ones(s.shape[0], dtype=np.int64) if counts is None else np.asarray(counts, dtype=np.int64)
        if aggregate:
            s, inv = np.unique(s, axis=0, return_inverse=True)
            c = np.bincount(inv.reshape(-1), weights=c, minlength=s.shape[0]).astype(np.int64)
        info = dict(info or {})
        info.setdefault("problem_hash", problem.hash())
        return cls(s, problem.energy(s), c, info)

    @property
    def num_samples(self) -> int:
        return int(self.counts.sum())

    def best(self) -> tuple[np.ndarray, float]:
        k = int(np.argmin(self.energies))
        return self.samples[k], float(self.energies[k])

    def mean(self) -> np.ndarray:
        """Count-weighted <y_i>."""
        w = self.counts / self.counts.sum()
        return w @ self.samples.astype(float)

    def edge_corr(self, edges: np.ndarray) -> np.ndarray:
        """Count-weighted <y_i y_k> on the given edges."""
        w = self.counts / self.counts.sum()
        s = self.samples.astype(float)
        return w @ (s[:, edges[:, 0]] * s[:, edges[:, 1]])

    def expanded(self) -> np.ndarray:
        """Samples repeated by their counts, shape (num_samples, n)."""
        return np.repeat(self.samples, self.counts, axis=0)


class Solver(ABC):
    """Base class. Each call takes a sequence of problems so backends can batch."""

    name: str = "solver"
    capabilities: frozenset[str] = frozenset()

    def minimize(self, problems: Sequence[IsingProblem], **params: Any) -> list[SampleSet]:
        raise UnsupportedProblem(f"{self.name} does not support minimize")

    def sample(self, problems: Sequence[IsingProblem], num_samples: int, **params: Any) -> list[SampleSet]:
        raise UnsupportedProblem(f"{self.name} does not support sample")


_COLORING_CACHE: dict[tuple, list[np.ndarray]] = {}


def greedy_coloring(problem: IsingProblem) -> list[np.ndarray]:
    """Partition variables into independent sets (no edge inside a class).

    Spins in one class are conditionally independent given the rest, so a whole class
    can be updated in parallel without changing the Gibbs kernel's invariant law.
    Memoized by graph structure (n and a hash of the edge list): training revisits the
    same graphs every step, and the greedy pass is O(n) Python work.
    """
    import hashlib

    key = (problem.n, hashlib.sha1(np.ascontiguousarray(problem.edges).tobytes()).hexdigest())
    if key in _COLORING_CACHE:
        return _COLORING_CACHE[key]
    if len(_COLORING_CACHE) > 4096:
        _COLORING_CACHE.clear()
    _COLORING_CACHE[key] = classes = _greedy_coloring(problem)
    return classes


def _greedy_coloring(problem: IsingProblem) -> list[np.ndarray]:
    W = problem.coupling_matrix
    colors = -np.ones(problem.n, dtype=np.int64)
    order = np.argsort(-np.diff(W.indptr))  # largest degree first
    for v in order:
        used = set(colors[W.indices[W.indptr[v]:W.indptr[v + 1]]].tolist())
        c = 0
        while c in used:
            c += 1
        colors[v] = c
    return [np.flatnonzero(colors == c) for c in range(int(colors.max()) + 1)] if problem.n else []
