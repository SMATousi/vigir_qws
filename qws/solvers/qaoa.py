"""Gate-based QAOA backends (doc 04 §3).

Basis convention (doc 02 §4): y_i is the eigenvalue of Z_i, so measured bit 0 gives
y_i = +1 (foreground). Qubit i is variable i; Qiskit count strings are little-endian
(qubit 0 is the rightmost character).
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from scipy.optimize import minimize as sp_minimize

from ..ising import IsingProblem
from .base import SampleSet, Solver, UnsupportedProblem
from .exact import all_spins


def linear_ramp(p_layers: int, delta: float = 0.75) -> np.ndarray:
    """Linear-ramp initial angles [gammas..., betas...] (annealing-inspired warm start)."""
    t = (np.arange(p_layers) + 0.5) / p_layers
    return np.concatenate([delta * t, delta * (1 - t)])


def _norm(p: IsingProblem) -> float:
    return float(max(np.abs(p.h).max(initial=0.0), np.abs(p.J).max(initial=0.0), 1e-12))


class QAOASimulator(Solver):
    """Exact statevector QAOA for small problems (n <= max_n).

    The cost is normalized by max|coef|, so angles are O(1). ``minimize`` returns the
    best of ``shots`` measurements; ``sample`` returns raw measurement samples, which
    are NOT Boltzmann-distributed (use them as seeds for Gibbs polishing).
    """

    name = "qaoa_sim"
    capabilities = frozenset({"minimize", "sample"})

    def __init__(self, p_layers: int = 2, max_n: int = 16, optimizer: str = "COBYLA", maxiter: int = 200,
                 seed: int | None = None):
        self.p_layers, self.max_n, self.optimizer, self.maxiter = p_layers, max_n, optimizer, maxiter
        self.rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------ statevector
    def cost_diagonal(self, p: IsingProblem) -> np.ndarray:
        if p.n > self.max_n:
            raise UnsupportedProblem(f"statevector QAOA limited to n <= {self.max_n} (got {p.n})")
        return p.energy(all_spins(p.n)) / _norm(p)  # index z: bit i of z = qubit i

    @staticmethod
    def statevector(diag: np.ndarray, n: int, angles: np.ndarray) -> np.ndarray:
        k = angles.shape[0] // 2
        gammas, betas = angles[:k], angles[k:]
        psi = np.full(2**n, 2 ** (-n / 2), dtype=complex)
        for g, b in zip(gammas, betas):
            psi = psi * np.exp(-1j * g * diag)
            c, s = np.cos(b), -1j * np.sin(b)
            for i in range(n):  # exp(-i b X_i)
                v = psi.reshape(2 ** (n - i - 1), 2, 2**i)
                psi = (c * v + s * v[:, ::-1, :]).reshape(-1)
        return psi

    def optimize_angles(self, p: IsingProblem, init: np.ndarray | None = None) -> tuple[np.ndarray, float]:
        diag = self.cost_diagonal(p)

        def f(a: np.ndarray) -> float:
            return float(np.abs(self.statevector(diag, p.n, a)) ** 2 @ diag)

        x0 = linear_ramp(self.p_layers) if init is None else init
        res = sp_minimize(f, x0, method=self.optimizer, options={"maxiter": self.maxiter})
        return res.x, float(res.fun) * _norm(p)

    def probabilities(self, p: IsingProblem, angles: np.ndarray) -> np.ndarray:
        return np.abs(self.statevector(self.cost_diagonal(p), p.n, angles)) ** 2

    # ------------------------------------------------------------------ solver API
    def _measure(self, p: IsingProblem, shots: int, angles: np.ndarray | None) -> SampleSet:
        if angles is None:
            angles, exp_e = self.optimize_angles(p)
        else:
            exp_e = float(self.probabilities(p, angles) @ (self.cost_diagonal(p) * _norm(p)))
        probs = self.probabilities(p, angles)
        counts = self.rng.multinomial(shots, probs / probs.sum())
        keep = counts > 0
        Y = all_spins(p.n)[keep]
        return SampleSet.from_samples(p, Y, counts[keep], {"solver": self.name, "p_layers": self.p_layers,
                                                           "angles": angles.tolist(), "expected_energy": exp_e})

    def minimize(self, problems: Sequence[IsingProblem], shots: int = 1024, angles: np.ndarray | None = None,
                 **_: Any) -> list[SampleSet]:
        return [self._measure(p, shots, angles) for p in problems]

    def sample(self, problems: Sequence[IsingProblem], num_samples: int, angles: np.ndarray | None = None,
               **_: Any) -> list[SampleSet]:
        return [self._measure(p, num_samples, angles) for p in problems]


def qaoa_circuit(p: IsingProblem, angles: np.ndarray, measure: bool = True):
    """Qiskit circuit for QAOA with cost H_C / max|coef| (paper Proposition 4).

    exp(-i g (-h_i) Z_i) = RZ(-2 g h_i);  exp(-i g (-J) Z Z) = RZZ(-2 g J);  exp(-i b X) = RX(2 b).
    """
    from qiskit import QuantumCircuit

    s = _norm(p)
    k = angles.shape[0] // 2
    qc = QuantumCircuit(p.n)
    qc.h(range(p.n))
    for g, b in zip(angles[:k], angles[k:]):
        for i, hv in enumerate(p.h / s):
            if hv:
                qc.rz(-2 * g * hv, i)
        for (i, j), jv in zip(p.edges, p.J / s):
            if jv:
                qc.rzz(-2 * g * jv, int(i), int(j))
        qc.rx(2 * b, range(p.n))
    if measure:
        qc.measure_all()
    return qc


def counts_to_spins(counts: dict[str, int], n: int) -> tuple[np.ndarray, np.ndarray]:
    """Little-endian Qiskit count strings -> spins (bit 0 -> +1)."""
    keys = list(counts)
    bits = np.array([[int(key.replace(" ", "")[-1 - i]) for i in range(n)] for key in keys], dtype=np.int8)
    return (1 - 2 * bits).astype(np.int8), np.array([counts[k] for k in keys])


class QAOAQiskit(Solver):
    """QAOA executed as a Qiskit circuit.

    Two execution modes:

    * ``backend.run`` (default: the Aer simulator), for local simulation;
    * a **SamplerV2 primitive** (``sampler=``), which is how IBM Quantum hardware is
      accessed today (``qiskit_ibm_runtime.SamplerV2``). When a ``backend`` is also given,
      the circuit is first transpiled to that backend's instruction-set architecture (ISA).
      The SamplerV2 path is tested offline with ``qiskit.primitives.StatevectorSampler``.

    Angles come from the statevector optimizer when n is small; otherwise a linear ramp
    is used (angle transfer). Real IBM hardware needs an account (spec doc 08, 3.4).
    """

    name = "qaoa_qiskit"
    capabilities = frozenset({"minimize", "sample"})

    def __init__(self, backend: Any | None = None, p_layers: int = 2, optimization_level: int = 1,
                 seed: int | None = None, sampler: Any | None = None):
        self.backend = backend
        self.sampler = sampler
        self.p_layers = p_layers
        self.optimization_level = optimization_level
        self.seed = seed
        self._sim = QAOASimulator(p_layers=p_layers, seed=seed)

    @classmethod
    def ibm(cls, backend_name: str | None = None, **kw: Any) -> "QAOAQiskit":  # pragma: no cover - needs account
        """IBM Quantum via qiskit-ibm-runtime SamplerV2 (credentials from the saved account
        or QISKIT_IBM_TOKEN). ``backend_name`` defaults to the least busy real device."""
        from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2

        service = QiskitRuntimeService()
        backend = service.backend(backend_name) if backend_name else service.least_busy(operational=True,
                                                                                        simulator=False)
        return cls(backend=backend, sampler=SamplerV2(mode=backend), **kw)

    def _backend(self):
        if self.backend is None:
            from qiskit_aer import AerSimulator

            self.backend = AerSimulator(seed_simulator=self.seed)
        return self.backend

    def _counts(self, qc, shots: int) -> tuple[dict[str, int], dict[str, Any]]:
        from qiskit import transpile

        if self.sampler is not None:
            if self.backend is not None:  # ISA circuit for real hardware
                qc = transpile(qc, self.backend, optimization_level=self.optimization_level,
                               seed_transpiler=self.seed)
            result = self.sampler.run([qc], shots=shots).result()[0]
            counts = result.data.meas.get_counts()  # measure_all() names the register "meas"
            return counts, {"mode": "SamplerV2", "sampler": type(self.sampler).__name__, "depth": qc.depth(),
                            "backend": str(self.backend) if self.backend is not None else None}
        backend = self._backend()
        qc = transpile(qc, backend, optimization_level=self.optimization_level, seed_transpiler=self.seed)
        counts = backend.run(qc, shots=shots).result().get_counts()
        return counts, {"mode": "backend.run", "backend": str(backend), "depth": qc.depth()}

    def _run(self, p: IsingProblem, shots: int, angles: np.ndarray | None) -> SampleSet:
        if angles is None:
            angles = self._sim.optimize_angles(p)[0] if p.n <= self._sim.max_n else linear_ramp(self.p_layers)
        counts, info = self._counts(qaoa_circuit(p, angles), shots)
        Y, c = counts_to_spins(counts, p.n)
        return SampleSet.from_samples(p, Y, c, {"solver": self.name, "angles": np.asarray(angles).tolist(), **info},
                                      aggregate=True)

    def minimize(self, problems: Sequence[IsingProblem], shots: int = 1024, angles: np.ndarray | None = None,
                 **_: Any) -> list[SampleSet]:
        return [self._run(p, shots, angles) for p in problems]

    def sample(self, problems: Sequence[IsingProblem], num_samples: int, angles: np.ndarray | None = None,
               **_: Any) -> list[SampleSet]:
        return [self._run(p, num_samples, angles) for p in problems]
