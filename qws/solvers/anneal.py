"""Quantum-annealing backend through the D-Wave Ocean SDK (doc 04 §2).

``AnnealingSolver`` wraps any (unstructured) dimod sampler, e.g. Ocean's classical
SimulatedAnnealingSampler as a stand-in. The real QPU path, with explicit embedding,
an embedding cache and parallel embedding, is :mod:`qws.solvers.qpu`.
"""

from __future__ import annotations

import time
from typing import Any, Sequence

import numpy as np

from ..ising import IsingProblem, to_dimod_bqm
from .base import SampleSet, Solver


class AnnealingSolver(Solver):
    """Annealing via any dimod sampler.

    Args:
        sampler: A dimod sampler. On QPU stacks it is an embedding composite.
        h_range, j_range: Hardware coefficient ranges. On QPUs these are read from
            ``sampler.child.properties`` at runtime when not given (never hard-coded).
        chain_strength_factor: Chain strength = factor * max|coef| (QPU only).
        chain_break_threshold: Calls above this chain-break fraction are flagged
            low-fidelity.
        sample_kwargs: Passed through to ``sampler.sample`` (annealing_time,
            num_spin_reversal_transforms, ...).
    """

    name = "anneal"
    capabilities = frozenset({"minimize", "sample"})

    def __init__(self, sampler: Any, h_range: tuple[float, float] | None = None,
                 j_range: tuple[float, float] | None = None, chain_strength_factor: float | None = None,
                 chain_break_threshold: float = 0.1, **sample_kwargs: Any):
        self.sampler = sampler
        self.h_range, self.j_range = h_range, j_range
        self.chain_strength_factor = chain_strength_factor
        self.chain_break_threshold = chain_break_threshold
        self.sample_kwargs = sample_kwargs
        self._embeddings: dict[str, Any] = {}

    # ------------------------------------------------------------------ scaling
    def _ranges(self) -> tuple[tuple[float, float] | None, tuple[float, float] | None]:
        h_range, j_range = self.h_range, self.j_range
        child = getattr(self.sampler, "child", None)
        props = getattr(child, "properties", None) or {}
        if h_range is None and "h_range" in props:
            h_range = tuple(props["h_range"])
        if j_range is None and "j_range" in props:
            j_range = tuple(props["j_range"])
        return h_range, j_range

    def scale_factor(self, p: IsingProblem) -> float:
        """kappa = max(max|h| / h_max, max|J| / J_max) (doc 02 §7); 1 when ranges are unknown."""
        h_range, j_range = self._ranges()
        kappa = 0.0
        if h_range is not None and p.n:
            kappa = max(kappa, np.abs(p.h).max() / min(abs(h_range[0]), abs(h_range[1])))
        if j_range is not None and p.m:
            kappa = max(kappa, np.abs(p.J).max() / min(abs(j_range[0]), abs(j_range[1])))
        return float(kappa) if kappa > 0 else 1.0

    # ------------------------------------------------------------------ calls
    def _run(self, p: IsingProblem, num_reads: int, **kw: Any) -> SampleSet:
        kappa = self.scale_factor(p)
        bqm = to_dimod_bqm(p.scaled(1.0 / kappa))  # Ocean sign convention: h' = -h, J' = -J
        params = {**self.sample_kwargs, **kw, "num_reads": num_reads}
        if self.chain_strength_factor is not None:
            params.setdefault("chain_strength",
                              self.chain_strength_factor * max(np.abs(bqm.linear_values()).max(initial=0),
                                                               np.abs(bqm.quadratic_values()).max(initial=0), 1e-9))
            params.setdefault("return_embedding", True)
        t0 = time.perf_counter()
        ss = self.sampler.sample(bqm, **params)
        wall = time.perf_counter() - t0
        order = [ss.variables.index(i) for i in range(p.n)]
        samples = np.asarray(ss.record.sample)[:, order]
        counts = np.asarray(ss.record.num_occurrences)
        info: dict[str, Any] = {"solver": self.name, "kappa": kappa, "wall_time_s": wall,
                                "sampler": type(self.sampler).__name__}
        if "chain_break_fraction" in ss.record.dtype.names:
            cbf = float(np.average(ss.record.chain_break_fraction, weights=counts))
            info["chain_break_fraction"] = cbf
            info["low_fidelity"] = cbf > self.chain_break_threshold
        if "timing" in ss.info:
            info["timing"] = dict(ss.info["timing"])
        emb = ss.info.get("embedding_context", {}).get("embedding")
        if emb is not None:
            lengths = [len(c) for c in emb.values()]
            info["embedding"] = {"qubits": int(sum(lengths)), "max_chain": int(max(lengths)),
                                 "mean_chain": float(np.mean(lengths))}
        return SampleSet.from_samples(p, samples, counts, info)

    def sample(self, problems: Sequence[IsingProblem], num_samples: int, **kw: Any) -> list[SampleSet]:
        return [self._run(p, num_samples, **kw) for p in problems]

    def minimize(self, problems: Sequence[IsingProblem], num_reads: int = 100, **kw: Any) -> list[SampleSet]:
        return [self._run(p, num_reads, **kw) for p in problems]


def simulated_annealing_stand_in(**kw: Any) -> AnnealingSolver:
    """Ocean's classical SimulatedAnnealingSampler behind the same interface: exercises
    sign conversion, variable ordering and metadata without QPU access.

    By default neal chooses its inverse-temperature range from the problem's
    coefficients, so rescaling the problem does not change the sample temperature.
    Pass a fixed ``beta_range=(lo, hi)`` to mimic a device with a fixed physical
    temperature, where the submit-time scale controls the effective temperature
    (this is what temperature calibration relies on)."""
    from neal import SimulatedAnnealingSampler

    return AnnealingSolver(SimulatedAnnealingSampler(), **kw)
