"""Conversions between our Ising convention and QUBO / SDK / Pauli forms (doc 02 §3–5).

All conversions are energy-preserving; each backend converts at its own boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .problem import IsingProblem


@dataclass(frozen=True)
class QUBO:
    """E(x) = offset + sum_i linear_i x_i + sum_{(i,k)} quadratic_ik x_i x_k, x in {0,1}."""

    linear: np.ndarray
    edges: np.ndarray
    quadratic: np.ndarray
    offset: float

    def energy(self, x: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(np.asarray(x, dtype=float))
        e = self.offset + x @ self.linear
        if self.edges.size:
            e = e + (x[:, self.edges[:, 0]] * x[:, self.edges[:, 1]]) @ self.quadratic
        return e


def spins_to_bits(y: np.ndarray) -> np.ndarray:
    """y in {-1,+1} -> x in {0,1} with x = (1 + y) / 2 (x = 1 is foreground)."""
    return ((np.asarray(y) + 1) // 2).astype(np.int8)


def bits_to_spins(x: np.ndarray) -> np.ndarray:
    return (2 * np.asarray(x, dtype=np.int8) - 1).astype(np.int8)


def ising_to_qubo(p: IsingProblem) -> QUBO:
    """Substitute y = 2x - 1 (paper Proposition 4)."""
    deg = np.zeros(p.n)
    np.add.at(deg, p.edges[:, 0], p.J)
    np.add.at(deg, p.edges[:, 1], p.J)
    linear = -2.0 * p.h + 2.0 * deg
    quadratic = -4.0 * p.J
    offset = p.offset + p.h.sum() - p.J.sum()
    return QUBO(linear, p.edges.copy(), quadratic, float(offset))


def qubo_to_ising(q: QUBO, **kw: Any) -> IsingProblem:
    """Inverse of :func:`ising_to_qubo` via x = (1 + y) / 2."""
    n = q.linear.shape[0]
    h = np.zeros(n)
    # x_i = (1 + y_i)/2 ; x_i x_k = (1 + y_i + y_k + y_i y_k)/4 ; E = -(...) in our convention
    h -= q.linear / 2.0
    np.add.at(h, q.edges[:, 0], -q.quadratic / 4.0)
    np.add.at(h, q.edges[:, 1], -q.quadratic / 4.0)
    J = -q.quadratic / 4.0
    offset = q.offset + q.linear.sum() / 2.0 + q.quadratic.sum() / 4.0
    return IsingProblem(h, q.edges.copy(), J, offset, **kw)


def to_dimod_bqm(p: IsingProblem):
    """D-Wave Ocean SPIN BQM. Ocean energy is sum h' s + sum J' s s + offset, so h' = -h, J' = -J."""
    import dimod

    linear = {i: -float(v) for i, v in enumerate(p.h)}
    quadratic = {(int(i), int(k)): -float(v) for (i, k), v in zip(p.edges, p.J)}
    return dimod.BinaryQuadraticModel(linear, quadratic, p.offset, dimod.SPIN)


def pauli_terms(p: IsingProblem) -> list[tuple[str, float]]:
    """Terms of H_C = offset I - sum h_i Z_i - sum J_ik Z_i Z_k as (label, coeff) pairs.

    Labels are big-endian over variables: character i refers to variable i. Qiskit's
    SparsePauliOp uses little-endian strings, so callers must reverse the label
    (handled in :mod:`qws.solvers.qaoa`).
    """
    n = p.n
    terms: list[tuple[str, float]] = []
    if p.offset:
        terms.append(("I" * n, p.offset))
    for i, v in enumerate(p.h):
        if v:
            lab = ["I"] * n
            lab[i] = "Z"
            terms.append(("".join(lab), -float(v)))
    for (i, k), v in zip(p.edges, p.J):
        if v:
            lab = ["I"] * n
            lab[i] = lab[k] = "Z"
            terms.append(("".join(lab), -float(v)))
    return terms
