"""Labeling-function interfaces and application (doc 01 §2.1).

Votes are int8 in {-1, 0, +1}: -1 background / "different", +1 foreground / "same",
0 abstain. Unary LFs vote per site; pairwise LFs vote per edge of the site graph
(v1 restricts E_r to the site graph's edges, see doc 08).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from ..graph import Sites


@dataclass(frozen=True)
class LFOutputs:
    """Vote matrices for one image.

    Attributes:
        L: Unary votes, ``(n_sites, m)`` int8.
        P: Pairwise votes on ``sites.edges``, ``(n_edges, q)`` int8.
    """

    L: np.ndarray
    P: np.ndarray
    unary_names: tuple[str, ...] = ()
    pairwise_names: tuple[str, ...] = ()

    @property
    def m(self) -> int:
        return int(self.L.shape[1])

    @property
    def q(self) -> int:
        return int(self.P.shape[1])

    def coverage(self) -> np.ndarray:
        """Fraction of non-abstaining votes per unary LF."""
        return (self.L != 0).mean(axis=0) if self.L.size else np.zeros(self.m)


class UnaryLF(ABC):
    name: str = "unary"

    @abstractmethod
    def votes(self, image: np.ndarray, sites: Sites) -> np.ndarray:
        """Per-site votes, shape ``(sites.n,)``."""


class PairwiseLF(ABC):
    name: str = "pairwise"

    @abstractmethod
    def votes(self, image: np.ndarray, sites: Sites) -> np.ndarray:
        """Per-edge votes on ``sites.edges``, shape ``(sites.m,)``."""


def lift_votes(pixel_votes: np.ndarray, sites: Sites, min_coverage: float = 0.5) -> np.ndarray:
    """Lift pixel votes to sites (open question Q3 default).

    A site takes the majority of its non-abstaining pixel votes if at least
    ``min_coverage`` of its pixels voted and there is no tie; otherwise it abstains.
    """
    flat = sites.labels.reshape(-1)
    v = np.asarray(pixel_votes).reshape(-1)
    n_pos = np.bincount(flat, weights=(v > 0), minlength=sites.n)
    n_neg = np.bincount(flat, weights=(v < 0), minlength=sites.n)
    total = np.maximum(np.bincount(flat, minlength=sites.n), 1)
    covered = (n_pos + n_neg) / total >= min_coverage
    return np.where(covered, np.sign(n_pos - n_neg), 0).astype(np.int8)


class PixelLF(UnaryLF):
    """Unary LF defined by a pixel-level function ``image -> (H, W) votes``, lifted to sites."""

    def __init__(self, fn: Callable[[np.ndarray], np.ndarray], name: str = "pixel_lf", min_coverage: float = 0.5):
        self.fn = fn
        self.name = name
        self.min_coverage = min_coverage

    def votes(self, image: np.ndarray, sites: Sites) -> np.ndarray:
        pv = np.asarray(self.fn(image))
        if pv.shape != sites.shape:
            raise ValueError(f"{self.name}: pixel votes shape {pv.shape} != image shape {sites.shape}")
        return lift_votes(pv, sites, self.min_coverage)


def apply_lfs(image: np.ndarray, sites: Sites, unary: Sequence[UnaryLF] = (),
              pairwise: Sequence[PairwiseLF] = ()) -> LFOutputs:
    L = np.zeros((sites.n, len(unary)), dtype=np.int8)
    for j, lf in enumerate(unary):
        L[:, j] = _check(lf.votes(image, sites), sites.n, lf.name)
    P = np.zeros((sites.m, len(pairwise)), dtype=np.int8)
    for r, lf in enumerate(pairwise):
        P[:, r] = _check(lf.votes(image, sites), sites.m, lf.name)
    return LFOutputs(L, P, tuple(lf.name for lf in unary), tuple(lf.name for lf in pairwise))


def _check(v: np.ndarray, size: int, name: str) -> np.ndarray:
    v = np.asarray(v).reshape(-1)
    if v.shape[0] != size:
        raise ValueError(f"LF {name!r} returned {v.shape[0]} votes, expected {size}")
    if not np.isin(v, (-1, 0, 1)).all():
        raise ValueError(f"LF {name!r} returned values outside {{-1, 0, 1}}")
    return v.astype(np.int8)
