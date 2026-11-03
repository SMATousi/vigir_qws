"""Effective-temperature estimation and calibration (doc 03 §4; paper Proposition 9)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
from scipy.optimize import minimize_scalar

from ..ising import IsingProblem
from ..solvers import SampleSet, Solver


def estimate_beta_pl(p: IsingProblem, samples: np.ndarray, counts: np.ndarray | None = None,
                     beta_max: float = 1e3) -> float:
    """Maximum pseudo-likelihood estimate of the inverse temperature of ``samples``
    relative to ``p``: argmax_b sum_s sum_i log sigmoid(b * 2 y_i m_i(y)).

    Concave in b (Proposition 9). Returns ``beta_max`` when the samples are
    consistent with zero temperature (every spin aligned with its local field).
    """
    y = np.atleast_2d(samples).astype(float)
    w = np.ones(y.shape[0]) if counts is None else np.asarray(counts, dtype=float)
    c = 2.0 * y * p.local_fields(y)  # (S, n)
    c, ww = c.reshape(-1), np.repeat(w, y.shape[1])
    keep = c != 0
    c, ww = c[keep], ww[keep]
    if c.size == 0:
        return float("nan")
    if np.all(c > 0):
        return beta_max

    def neg_ll(b: float) -> float:
        return float(ww @ np.logaddexp(0.0, -b * c))  # -log sigmoid(b c)

    res = minimize_scalar(neg_ll, bounds=(0.0, beta_max), method="bounded", options={"xatol": 1e-6})
    return float(res.x)


@dataclass
class CalibratedSampler(Solver):
    """Wraps a sampler and adapts the submit-time scale so that the estimated effective
    inverse temperature relative to the target problem approaches 1.

    Each call submits ``p.scaled(scale)``, estimates beta_hat of the returned samples
    w.r.t. the unscaled ``p``, and updates ``scale <- scale / beta_hat**damping``.
    Each update changes the scale by at most a factor ``max_step``. This matters because
    beta_hat saturates at its cap when every spin is aligned with its local field (zero
    temperature), and an unclipped step would then overshoot by orders of magnitude.
    The scale stays within ``[min_scale, max_scale]``. The device's coefficient bounds are enforced by
    the inner solver; when the target is unreachable, the beta_hat history shows it.
    """

    inner: Solver
    scale: float = 1.0
    damping: float = 0.5
    min_scale: float = 1e-3
    max_scale: float = 1e3
    max_step: float = 4.0
    history: list[dict[str, float]] = field(default_factory=list)

    name = "calibrated"
    capabilities = frozenset({"sample"})

    def sample(self, problems: Sequence[IsingProblem], num_samples: int, **kw: Any) -> list[SampleSet]:
        raw = self.inner.sample([p.scaled(self.scale) for p in problems], num_samples, **kw)
        out, betas = [], []
        for p, ss in zip(problems, raw):
            b = estimate_beta_pl(p, ss.samples, ss.counts)
            betas.append(b)
            info = {**ss.info, "submit_scale": self.scale, "beta_hat": b}
            out.append(SampleSet.from_samples(p, ss.samples, ss.counts, info))
        finite = [b for b in betas if np.isfinite(b) and b > 0]
        if finite:
            beta_hat = float(np.exp(np.mean(np.log(finite))))
            self.history.append({"scale": self.scale, "beta_hat": beta_hat})
            step = float(np.clip(beta_hat**self.damping, 1.0 / self.max_step, self.max_step))
            self.scale = float(np.clip(self.scale / step, self.min_scale, self.max_scale))
        return out
