"""Offline tests of the D-Wave QPU path against MockDWaveSampler (Pegasus topology)."""

import numpy as np
import pytest

pytest.importorskip("dwave.system")
pytest.importorskip("minorminer")

from qws.ising import IsingProblem  # noqa: E402
from qws.learn import CalibratedSampler  # noqa: E402
from qws.solvers import ExactSolver  # noqa: E402
from qws.solvers.qpu import QPUAnnealingSolver, graph_key  # noqa: E402

from .conftest import random_problem  # noqa: E402


@pytest.fixture(scope="module")
def qpu():
    return QPUAnnealingSolver.mock(topology_shape=6)


def test_reads_ranges_and_topology(qpu):
    assert qpu.h_range == (-4.0, 4.0) and qpu.j_range == (-1.0, 1.0)
    assert qpu.child.properties["topology"]["type"] == "pegasus"


def test_scale_factor_only_shrinks(qpu):
    small = IsingProblem(np.array([0.5]), np.zeros((0, 2)), np.zeros(0))
    assert qpu.scale_factor(small) == 1.0
    p = IsingProblem(np.array([8.0, 0.0]), np.array([[0, 1]]), np.array([0.5]))
    assert qpu.scale_factor(p) == pytest.approx(2.0)  # max|h| / h_max = 8 / 4
    q = IsingProblem(np.zeros(2), np.array([[0, 1]]), np.array([2.0]))
    assert qpu.scale_factor(q) == pytest.approx(qpu.csf * 2.0)  # chains must fit |J_min| = 1


def test_finds_exact_optimum_and_packs(qpu, rng):
    problems = [random_problem(10, rng) for _ in range(3)]
    out = qpu.minimize(problems, num_reads=100)
    ex = ExactSolver()
    for p, ss in zip(problems, out):
        assert ss.best()[1] == pytest.approx(ex.minimize([p])[0].best()[1])
        assert ss.info["packed_with"] == 3 and ss.num_samples == 100
        assert ss.info["chain_strength"] <= abs(qpu.j_range[0]) + 1e-12
        assert "qpu_access_time_us" in ss.info
    assert qpu.calls[-1]["problems"] == 3


def test_embedding_cache_by_graph_shape(qpu, rng):
    p = random_problem(8, rng)
    qpu.minimize([p], num_reads=10)
    n_cached = len(qpu.embeddings)
    # same graph, different coefficients -> cache hit
    q = IsingProblem(p.h * 2 + 1, p.edges, -p.J, p.offset)
    assert graph_key(q) == graph_key(p)
    qpu.minimize([q], num_reads=10)
    assert len(qpu.embeddings) == n_cached


def test_serial_mode(rng):
    qpu = QPUAnnealingSolver.mock(topology_shape=4, parallel=False)
    out = qpu.minimize([random_problem(6, rng) for _ in range(2)], num_reads=20)
    assert all(ss.info["packed_with"] == 1 for ss in out) and len(qpu.calls) == 2


def test_calibration_through_mock_qpu(rng):
    """The mock samples at a fixed inverse temperature (beta_range end = 4) in hardware
    units, so calibration should converge to a submit scale of about 1/4."""
    p = random_problem(10, rng, density=0.3, scale=0.3)
    cal = CalibratedSampler(QPUAnnealingSolver.mock(topology_shape=4), damping=1.0)
    for _ in range(8):
        cal.sample([p], 2000)
    assert cal.history[-1]["beta_hat"] == pytest.approx(1.0, abs=0.15)
    assert 1 / cal.scale == pytest.approx(4.0, rel=0.2)
