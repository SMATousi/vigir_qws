"""Canonical Ising problem object (spec doc 02 §1).

Energy convention (ours, used everywhere outside backend boundaries):

    E(y) = offset - sum_i h_i y_i - sum_{(i,k)} J_ik y_i y_k,   p(y) ∝ exp(-E(y)),

with spins y_i in {-1, +1} (+1 = foreground) and J > 0 ferromagnetic.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any

import numpy as np
import scipy.sparse as sp


@dataclass(frozen=True)
class IsingProblem:
    """An Ising model on variables ``0..n-1``.

    Attributes:
        h: Local fields, shape ``(n,)``.
        edges: Unordered pairs as an ``(m, 2)`` int array with ``i < k`` and no duplicates.
        J: Couplings aligned with ``edges``, shape ``(m,)``.
        offset: Constant energy term.
        kind: ``"posterior"``, ``"prior"`` or ``"generic"``.
        meta: Free-form metadata (image id, tile id, parameter snapshot, ...).
    """

    h: np.ndarray
    edges: np.ndarray
    J: np.ndarray
    offset: float = 0.0
    kind: str = "generic"
    meta: dict[str, Any] = field(default_factory=dict, compare=False)

    def __post_init__(self) -> None:
        h = np.asarray(self.h, dtype=float).reshape(-1)
        edges = np.asarray(self.edges, dtype=np.int64).reshape(-1, 2)
        J = np.asarray(self.J, dtype=float).reshape(-1)
        if edges.shape[0] != J.shape[0]:
            raise ValueError("edges and J must have the same length")
        if edges.size:
            if np.any(edges[:, 0] == edges[:, 1]):
                raise ValueError("self-loops are not allowed")
            if edges.min() < 0 or edges.max() >= h.shape[0]:
                raise ValueError("edge index out of range")
            lo, hi = edges.min(axis=1), edges.max(axis=1)
            edges = np.stack([lo, hi], axis=1)
            keys = lo * h.shape[0] + hi
            if np.unique(keys).size != keys.size:
                raise ValueError("duplicate edges; merge couplings before building the problem")
        object.__setattr__(self, "h", h)
        object.__setattr__(self, "edges", edges)
        object.__setattr__(self, "J", J)
        object.__setattr__(self, "offset", float(self.offset))

    # ------------------------------------------------------------------ basics
    @property
    def n(self) -> int:
        return int(self.h.shape[0])

    @property
    def m(self) -> int:
        return int(self.edges.shape[0])

    @classmethod
    def from_dicts(cls, h: dict[int, float], J: dict[tuple[int, int], float], n: int | None = None,
                   offset: float = 0.0, **kw: Any) -> "IsingProblem":
        """Build from dict form, merging ``(i,k)``/``(k,i)`` duplicates."""
        n = n if n is not None else 1 + max([*h.keys(), *(max(e) for e in J)], default=-1)
        hv = np.zeros(n)
        for i, v in h.items():
            hv[i] += v
        merged: dict[tuple[int, int], float] = {}
        for (i, k), v in J.items():
            key = (min(i, k), max(i, k))
            merged[key] = merged.get(key, 0.0) + v
        edges = np.array(list(merged.keys()), dtype=np.int64).reshape(-1, 2)
        return cls(hv, edges, np.array(list(merged.values())), offset, **kw)

    @cached_property
    def coupling_matrix(self) -> sp.csr_matrix:
        """Symmetric sparse matrix W with W_ik = W_ki = J_ik and zero diagonal."""
        i, k = self.edges[:, 0], self.edges[:, 1]
        W = sp.coo_matrix((np.concatenate([self.J, self.J]), (np.concatenate([i, k]), np.concatenate([k, i]))),
                          shape=(self.n, self.n))
        return W.tocsr()

    def energy(self, samples: np.ndarray) -> np.ndarray:
        """Energies of spin samples, shape ``(S, n)`` or ``(n,)``."""
        s = np.asarray(samples, dtype=float)
        single = s.ndim == 1
        s = np.atleast_2d(s)
        e = self.offset - s @ self.h
        if self.m:
            e = e - (s[:, self.edges[:, 0]] * s[:, self.edges[:, 1]]) @ self.J
        return e[0] if single else e

    def local_fields(self, samples: np.ndarray) -> np.ndarray:
        """m_i(y) = h_i + sum_k J_ik y_k for each sample, shape ``(S, n)``."""
        s = np.atleast_2d(np.asarray(samples, dtype=float))
        return self.h[None, :] + (self.coupling_matrix @ s.T).T

    def degree_abs_coupling(self) -> np.ndarray:
        """sum_k |J_ik| for each variable."""
        out = np.zeros(self.n)
        np.add.at(out, self.edges[:, 0], np.abs(self.J))
        np.add.at(out, self.edges[:, 1], np.abs(self.J))
        return out

    def is_ferromagnetic(self, tol: float = 0.0) -> bool:
        return bool(np.all(self.J >= -tol))

    def scaled(self, factor: float) -> "IsingProblem":
        """Problem with energy multiplied by ``factor`` (inverse-temperature scaling)."""
        return IsingProblem(self.h * factor, self.edges, self.J * factor, self.offset * factor, self.kind,
                            {**self.meta, "scale": self.meta.get("scale", 1.0) * factor})

    def dynamic_range(self) -> float:
        """max|coef| / min_{coef != 0}|coef| over h and J (doc 02 §7)."""
        c = np.abs(np.concatenate([self.h, self.J]))
        c = c[c > 0]
        return float(c.max() / c.min()) if c.size else 1.0

    def hash(self) -> str:
        """Content hash of (h, edges, J, offset) for provenance records."""
        m = hashlib.sha256()
        for arr in (self.h, self.edges, self.J, np.array([self.offset])):
            m.update(np.ascontiguousarray(arr).tobytes())
        return m.hexdigest()[:16]
