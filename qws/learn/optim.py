"""Minimal Adam optimizer (gradient ascent)."""

from __future__ import annotations

import numpy as np


class Adam:
    def __init__(self, lr: float = 0.05, b1: float = 0.9, b2: float = 0.999, eps: float = 1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.m: np.ndarray | None = None
        self.v: np.ndarray | None = None
        self.t = 0

    def step(self, x: np.ndarray, grad: np.ndarray) -> np.ndarray:
        """Return x moved uphill along ``grad``."""
        if self.m is None or self.m.shape != x.shape:
            self.m, self.v, self.t = np.zeros_like(x), np.zeros_like(x), 0
        self.t += 1
        self.m = self.b1 * self.m + (1 - self.b1) * grad
        self.v = self.b2 * self.v + (1 - self.b2) * grad**2
        mhat = self.m / (1 - self.b1**self.t)
        vhat = self.v / (1 - self.b2**self.t)
        return x + self.lr * mhat / (np.sqrt(vhat) + self.eps)
