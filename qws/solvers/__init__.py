"""Solver backends. Quantum SDK imports are lazy so the classical core runs without extras."""

from .base import SampleSet, Solver, UnsupportedProblem, greedy_coloring
from .exact import ExactSolver, all_spins
from .graphcut import GraphCutSolver, QPBOSolver, roof_dual, solve_submodular_qubo
from .mcmc import GibbsSampler, SimulatedAnnealing, gibbs_polish

__all__ = [
    "SampleSet", "Solver", "UnsupportedProblem", "greedy_coloring", "ExactSolver", "all_spins",
    "GraphCutSolver", "QPBOSolver", "roof_dual", "solve_submodular_qubo", "GibbsSampler", "SimulatedAnnealing", "gibbs_polish", "get_solver",
]


def get_solver(name: str, **kw):
    """Construct a solver by its config name (doc 04 §1)."""
    if name == "exact":
        return ExactSolver(**kw)
    if name == "graphcut":
        return GraphCutSolver()
    if name == "qpbo":
        return QPBOSolver(**kw)
    if name == "gibbs":
        return GibbsSampler(**kw)
    if name == "sa":
        return SimulatedAnnealing(**kw)
    if name == "anneal_sim":
        from .anneal import simulated_annealing_stand_in

        return simulated_annealing_stand_in(**kw)
    if name == "anneal_qpu":  # pragma: no cover - needs D-Wave credentials
        from .qpu import QPUAnnealingSolver

        return QPUAnnealingSolver.dwave(**kw)
    if name == "anneal_mock":
        from .qpu import QPUAnnealingSolver

        return QPUAnnealingSolver.mock(**kw)
    if name == "qaoa_sim":
        from .qaoa import QAOASimulator

        return QAOASimulator(**kw)
    if name == "qaoa_qiskit":
        from .qaoa import QAOAQiskit

        return QAOAQiskit(**kw)
    raise ValueError(f"unknown solver {name!r}")
