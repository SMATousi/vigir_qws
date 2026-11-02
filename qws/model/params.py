"""Label-model parameters (symmetric potentials, paper Definition 1)."""

from __future__ import annotations

from dataclasses import dataclass, field, fields, replace

import numpy as np

_VECTOR_FIELDS = ("alpha", "theta", "alphaP", "thetaP", "gamma")
_SCALAR_FIELDS = ("b", "beta")


@dataclass
class LabelModelParams:
    """Parameters Θ of the structured label model.

    Attributes:
        alpha, theta: Propensity and accuracy of each unary LF, shape ``(m,)``.
        alphaP, thetaP: Propensity and accuracy of each pairwise LF, shape ``(q,)``.
        gamma: Strength of each correlation clique in ``cliques``.
        b: Class bias. beta: Smoothness strength (>= 0).
        cliques: Groups of correlated unary LFs (each of size >= 2, disjoint). LFs not
            listed form singleton cliques. Feature: number of agreeing non-abstaining pairs.
    """

    alpha: np.ndarray
    theta: np.ndarray
    alphaP: np.ndarray = field(default_factory=lambda: np.zeros(0))
    thetaP: np.ndarray = field(default_factory=lambda: np.zeros(0))
    gamma: np.ndarray = field(default_factory=lambda: np.zeros(0))
    b: float = 0.0
    beta: float = 0.0
    cliques: tuple[tuple[int, ...], ...] = ()

    def __post_init__(self) -> None:
        for name in _VECTOR_FIELDS:
            setattr(self, name, np.asarray(getattr(self, name), dtype=float).reshape(-1).copy())
        self.b, self.beta = float(self.b), float(self.beta)
        self.cliques = tuple(tuple(int(j) for j in c) for c in self.cliques)
        if self.alpha.shape != self.theta.shape or self.alphaP.shape != self.thetaP.shape:
            raise ValueError("alpha/theta (and alphaP/thetaP) must have equal length")
        if self.gamma.shape[0] != len(self.cliques):
            raise ValueError("gamma must have one entry per clique")
        seen: set[int] = set()
        for c in self.cliques:
            if len(c) < 2 or seen & set(c) or min(c) < 0 or max(c) >= self.m:
                raise ValueError("cliques must be disjoint, of size >= 2, and index existing LFs")
            seen |= set(c)

    @property
    def m(self) -> int:
        return int(self.theta.shape[0])

    @property
    def q(self) -> int:
        return int(self.thetaP.shape[0])

    @classmethod
    def init(cls, m: int, q: int = 0, theta0: float = 0.5, thetaP0: float = 0.5, b: float = 0.0, beta: float = 0.0,
             coverage: np.ndarray | None = None, coverageP: np.ndarray | None = None,
             cliques: tuple[tuple[int, ...], ...] = ()) -> "LabelModelParams":
        """Positive accuracies (breaks the label-flip symmetry, doc 01 §6) and propensities
        matched to observed coverage when given."""
        theta = np.full(m, theta0)
        thetaP = np.full(q, thetaP0)
        return cls(alpha_from_coverage(coverage, theta), theta, alpha_from_coverage(coverageP, thetaP), thetaP,
                   np.zeros(len(cliques)), b, beta, cliques)

    def copy(self) -> "LabelModelParams":
        return replace(self, **{k: getattr(self, k).copy() for k in _VECTOR_FIELDS})

    def zeros_like(self) -> "LabelModelParams":
        return replace(self, **{k: np.zeros_like(getattr(self, k)) for k in _VECTOR_FIELDS}, b=0.0, beta=0.0)

    # vector view for optimizers
    def to_vector(self) -> np.ndarray:
        return np.concatenate([getattr(self, k) for k in _VECTOR_FIELDS] + [np.array([self.b, self.beta])])

    def from_vector(self, v: np.ndarray) -> "LabelModelParams":
        out, pos = self.copy(), 0
        for k in _VECTOR_FIELDS:
            size = getattr(self, k).shape[0]
            setattr(out, k, np.asarray(v[pos:pos + size], dtype=float).copy())
            pos += size
        out.b, out.beta = float(v[pos]), float(v[pos + 1])
        return out

    def to_dict(self) -> dict:
        d = {k: getattr(self, k).tolist() for k in _VECTOR_FIELDS}
        d.update(b=self.b, beta=self.beta, cliques=[list(c) for c in self.cliques])
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "LabelModelParams":
        return cls(**{f.name: d[f.name] for f in fields(cls) if f.name in d and f.name != "cliques"},
                   cliques=tuple(tuple(c) for c in d.get("cliques", ())))


def alpha_from_coverage(coverage: np.ndarray | None, theta: np.ndarray) -> np.ndarray:
    """Solve cov = 2 e^a cosh(t) / (1 + 2 e^a cosh(t)) for a (singleton closed form)."""
    if coverage is None:
        return np.zeros_like(theta)
    c = np.clip(np.asarray(coverage, dtype=float), 1e-3, 1 - 1e-3)
    return np.log(c / ((1 - c) * 2.0 * np.cosh(theta)))
