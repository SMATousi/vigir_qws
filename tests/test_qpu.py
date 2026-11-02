"""Real-hardware checks (roadmap Phase 3). Deselected by default (``-m 'not qpu'``).

Run with ``pytest -m qpu`` once credentials exist:
- D-Wave: ``DWAVE_API_TOKEN``
- IBM:    ``QISKIT_IBM_TOKEN`` and ``QWS_IBM_BACKEND``
"""

import os

import numpy as np
import pytest

from qws.solvers import ExactSolver

from .conftest import random_problem

pytestmark = pytest.mark.qpu


@pytest.mark.skipif(not os.environ.get("DWAVE_API_TOKEN"), reason="needs a D-Wave Leap token")
def test_dwave_qpu_small_problem():
    from qws.solvers.anneal import AnnealingSolver

    p = random_problem(12, np.random.default_rng(0))
    solver = AnnealingSolver.dwave_qpu()
    ss = solver.minimize([p], num_reads=200)[0]
    assert ss.best()[1] == pytest.approx(ExactSolver().minimize([p])[0].best()[1])
    assert "timing" in ss.info and ss.info["chain_break_fraction"] < 0.5


@pytest.mark.skipif(not (os.environ.get("QISKIT_IBM_TOKEN") and os.environ.get("QWS_IBM_BACKEND")),
                    reason="needs an IBM Quantum token and backend name")
def test_ibm_qaoa_small_problem():
    from qiskit_ibm_runtime import QiskitRuntimeService

    from qws.solvers.qaoa import QAOAQiskit

    backend = QiskitRuntimeService().backend(os.environ["QWS_IBM_BACKEND"])
    p = random_problem(5, np.random.default_rng(0))
    ss = QAOAQiskit(backend=backend, p_layers=1).minimize([p], shots=4000)[0]
    assert ss.num_samples == 4000
