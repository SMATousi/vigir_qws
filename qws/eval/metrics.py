"""Metrics (doc 06 §3)."""

from __future__ import annotations

import numpy as np


def iou(pred: np.ndarray, gt: np.ndarray) -> float:
    """Foreground IoU for masks in {-1,+1} (or {0,1})."""
    p, g = np.asarray(pred) > 0, np.asarray(gt) > 0
    union = np.logical_or(p, g).sum()
    return float(np.logical_and(p, g).sum() / union) if union else 1.0


def dice(pred: np.ndarray, gt: np.ndarray) -> float:
    p, g = np.asarray(pred) > 0, np.asarray(gt) > 0
    denom = p.sum() + g.sum()
    return float(2 * np.logical_and(p, g).sum() / denom) if denom else 1.0


def accuracy(pred: np.ndarray, gt: np.ndarray) -> float:
    return float(np.mean((np.asarray(pred) > 0) == (np.asarray(gt) > 0)))


def brier(prob_fg: np.ndarray, gt: np.ndarray) -> float:
    return float(np.mean((np.asarray(prob_fg) - (np.asarray(gt) > 0)) ** 2))


def ece(prob_fg: np.ndarray, gt: np.ndarray, bins: int = 10) -> float:
    """Expected calibration error of P(foreground)."""
    p = np.asarray(prob_fg, dtype=float).reshape(-1)
    g = (np.asarray(gt) > 0).reshape(-1).astype(float)
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    total = 0.0
    for b in range(bins):
        sel = idx == b
        if sel.any():
            total += sel.mean() * abs(p[sel].mean() - g[sel].mean())
    return float(total)


def relative_energy_gap(found: float, reference: float) -> float:
    return float((found - reference) / max(abs(reference), 1e-12))


def hit_rate(energies: np.ndarray, counts: np.ndarray, ground: float, tol: float = 1e-6) -> float:
    e, c = np.asarray(energies), np.asarray(counts)
    return float(c[e <= ground + tol].sum() / c.sum())


def tts99(t_run: float, p_hit: float) -> float:
    """Time to solution at 99% confidence: t * log(0.01) / log(1 - p)."""
    if p_hit <= 0:
        return float("inf")
    if p_hit >= 0.99:
        return float(t_run)
    return float(t_run * np.log(0.01) / np.log(1 - p_hit))


def config_index(y: np.ndarray) -> np.ndarray:
    """Integer index z with bit_i(z) = (1 - y_i)/2, consistent with solvers.all_spins."""
    bits = (1 - np.atleast_2d(y).astype(np.int64)) // 2
    return bits @ (1 << np.arange(bits.shape[1], dtype=np.int64))


def tv_distance(samples: np.ndarray, counts: np.ndarray, exact_probs: np.ndarray) -> float:
    """TV distance between an empirical sample distribution and exact probabilities
    indexed as in :func:`config_index`."""
    emp = np.zeros_like(exact_probs)
    np.add.at(emp, config_index(samples), np.asarray(counts, dtype=float))
    emp /= emp.sum()
    return float(0.5 * np.abs(emp - exact_probs).sum())
