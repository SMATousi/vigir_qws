"""Exact reductions (doc 02 §6; paper Proposition 5)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .problem import IsingProblem


@dataclass(frozen=True)
class Conditioned:
    """A problem on the free variables after fixing some spins.

    ``free`` maps reduced index -> original index; ``fixed`` holds the fixed values
    on the original index set (0 where free).
    """

    problem: IsingProblem
    free: np.ndarray
    fixed_idx: np.ndarray
    fixed_val: np.ndarray

    def lift(self, reduced_samples: np.ndarray, n: int) -> np.ndarray:
        """Embed reduced samples back into full-length spin vectors."""
        r = np.atleast_2d(reduced_samples)
        out = np.empty((r.shape[0], n), dtype=np.int8)
        out[:, self.free] = r
        out[:, self.fixed_idx] = self.fixed_val
        return out


def condition(p: IsingProblem, fixed_idx: np.ndarray, fixed_val: np.ndarray) -> Conditioned:
    """Fix y_F = ybar_F; return the exact conditional Ising model on the rest.

    h'_i = h_i + sum_{k in F} J_ik ybar_k, with the F-only energy moved to the offset.
    """
    fixed_idx = np.asarray(fixed_idx, dtype=np.int64).reshape(-1)
    fixed_val = np.asarray(fixed_val, dtype=np.int8).reshape(-1)
    ybar = np.zeros(p.n)
    ybar[fixed_idx] = fixed_val
    is_fixed = np.zeros(p.n, dtype=bool)
    is_fixed[fixed_idx] = True
    free = np.flatnonzero(~is_fixed)
    remap = -np.ones(p.n, dtype=np.int64)
    remap[free] = np.arange(free.size)

    i, k = p.edges[:, 0], p.edges[:, 1]
    fi, fk = is_fixed[i], is_fixed[k]
    h = p.h.copy()
    # mixed edges -> fields on the free endpoint
    m1 = ~fi & fk
    np.add.at(h, i[m1], p.J[m1] * ybar[k[m1]])
    m2 = fi & ~fk
    np.add.at(h, k[m2], p.J[m2] * ybar[i[m2]])
    both = fi & fk
    offset = p.offset - float(p.h[fixed_idx] @ ybar[fixed_idx]) - float(p.J[both] @ (ybar[i[both]] * ybar[k[both]]))
    keep = ~fi & ~fk
    edges = np.stack([remap[i[keep]], remap[k[keep]]], axis=1)
    sub = IsingProblem(h[free], edges, p.J[keep], offset, p.kind, {**p.meta, "conditioned_on": int(fixed_idx.size)})
    return Conditioned(sub, free, fixed_idx, fixed_val)


def strong_field_fix(p: IsingProblem, max_rounds: int = 100) -> Conditioned:
    """Iteratively fix spins with |h_i| > sum_k |J_ik| to sign(h_i).

    Exact for MAP (every minimizer agrees on the fixed spins); NOT exact for sampling.
    """
    fixed_idx = np.empty(0, dtype=np.int64)
    fixed_val = np.empty(0, dtype=np.int8)
    current = condition(p, fixed_idx, fixed_val)
    for _ in range(max_rounds):
        q = current.problem
        strong = np.abs(q.h) > q.degree_abs_coupling()
        if not strong.any():
            break
        new_idx = current.free[strong]
        new_val = np.sign(q.h[strong]).astype(np.int8)
        fixed_idx = np.concatenate([fixed_idx, new_idx])
        fixed_val = np.concatenate([fixed_val, new_val])
        current = condition(p, fixed_idx, fixed_val)
    return current
