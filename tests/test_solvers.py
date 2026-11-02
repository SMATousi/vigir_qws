import numpy as np
import pytest

from qws.ising import IsingProblem
from qws.solvers import (
    ExactSolver, GibbsSampler, GraphCutSolver, SampleSet, SimulatedAnnealing, UnsupportedProblem, all_spins,
    gibbs_polish, greedy_coloring,
)

from .conftest import random_problem


def test_exact_single_spin_analytic():
    p = IsingProblem(np.array([0.7]), np.zeros((0, 2)), np.zeros(0))
    mu, _ = ExactSolver().moments(p)
    assert mu[0] == pytest.approx(np.tanh(0.7))
    assert ExactSolver().log_partition(p) == pytest.approx(np.log(2 * np.cosh(0.7)))


def test_exact_minimize_returns_all_ground_states():
    p = IsingProblem(np.zeros(2), np.array([[0, 1]]), np.array([1.0]))  # degenerate: ++ and --
    ss = ExactSolver().minimize([p])[0]
    assert ss.info["degeneracy"] == 2 and ss.best()[1] == pytest.approx(-1.0)


def test_exact_size_guard():
    with pytest.raises(UnsupportedProblem):
        ExactSolver(max_n=4).minimize([random_problem(5, np.random.default_rng(0))])


def test_graphcut_matches_exact_on_ferromagnetic(rng):
    ex, gc = ExactSolver(), GraphCutSolver()
    for _ in range(25):
        p = random_problem(int(rng.integers(2, 13)), rng, ferro=True)
        assert gc.minimize([p])[0].best()[1] == pytest.approx(ex.minimize([p])[0].best()[1], abs=1e-9)


def test_graphcut_refuses_frustrated(rng):
    p = IsingProblem(np.zeros(3), np.array([[0, 1], [1, 2]]), np.array([1.0, -1.0]))
    with pytest.raises(UnsupportedProblem):
        GraphCutSolver().minimize([p])


def test_coloring_is_proper(rng):
    p = random_problem(30, rng, density=0.2)
    classes = greedy_coloring(p)
    color = np.empty(p.n, int)
    for c, idx in enumerate(classes):
        color[idx] = c
    assert np.all(color[p.edges[:, 0]] != color[p.edges[:, 1]])
    assert sorted(np.concatenate(classes).tolist()) == list(range(p.n))


@pytest.mark.slow
def test_gibbs_marginals_match_exact(rng):
    p = random_problem(8, rng, density=0.4, scale=0.6)
    mu, corr = ExactSolver().moments(p)
    ss = GibbsSampler(num_sweeps=60, seed=1).sample([p], 4000)[0]
    np.testing.assert_allclose(ss.mean(), mu, atol=0.06)
    np.testing.assert_allclose(ss.edge_corr(p.edges), corr, atol=0.06)


def test_gibbs_persistent_chains_reused(rng):
    p = random_problem(5, rng)
    p = IsingProblem(p.h, p.edges, p.J, p.offset, meta={"chain_key": "a"})
    g = GibbsSampler(num_sweeps=2, persistent=True, seed=0)
    g.sample([p], 10)
    first = g._chains["a"].copy()
    g.sample([p], 10, num_sweeps=0)
    np.testing.assert_array_equal(g._chains["a"], first)


def test_polish_preserves_exact_distribution(rng):
    p = random_problem(6, rng, scale=0.7)
    ex = ExactSolver(seed=2)
    seeds = ex.sample([p], 4000)[0].expanded()
    polished = gibbs_polish(p, seeds, sweeps=3, rng=np.random.default_rng(3))
    mu, _ = ex.moments(p)
    np.testing.assert_allclose(polished.mean(axis=0), mu, atol=0.06)


def test_polish_from_bad_seeds_moves_toward_target(rng):
    p = random_problem(6, rng, scale=0.7)
    mu, _ = ExactSolver().moments(p)
    bad = np.ones((3000, p.n), dtype=np.int8)
    err0 = np.abs(bad.mean(axis=0) - mu).max()
    err = np.abs(gibbs_polish(p, bad, 20, np.random.default_rng(0)).mean(axis=0) - mu).max()
    assert err < err0 and err < 0.08


def test_sa_finds_ground_state(rng):
    sa, ex = SimulatedAnnealing(num_sweeps=300, seed=0), ExactSolver()
    for _ in range(10):
        p = random_problem(12, rng)
        assert sa.minimize([p], num_reads=10)[0].best()[1] == pytest.approx(ex.minimize([p])[0].best()[1])


def test_sampleset_energies_recomputed(doc02_example):
    ss = SampleSet.from_samples(doc02_example, np.array([[1, 1, -1], [1, 1, -1], [1, 1, 1]]), aggregate=True)
    assert ss.num_samples == 3 and ss.best()[1] == pytest.approx(-3.0)
    assert ss.mean().tolist() == pytest.approx([1, 1, -1 / 3])


# ---------------------------------------------------------------- annealing via Ocean (simulated stand-in)
def test_anneal_stand_in_sign_and_ordering(rng):
    pytest.importorskip("neal")
    from qws.solvers.anneal import simulated_annealing_stand_in

    solver = simulated_annealing_stand_in()
    ex = ExactSolver()
    for _ in range(5):
        p = random_problem(10, rng)
        ss = solver.minimize([p], num_reads=50)[0]
        assert ss.best()[1] == pytest.approx(ex.minimize([p])[0].best()[1])
        assert ss.info["kappa"] == 1.0 and ss.num_samples == 50


def test_anneal_scale_factor_uses_ranges(doc02_example):
    from qws.solvers.anneal import AnnealingSolver

    s = AnnealingSolver(sampler=None, h_range=(-2.0, 2.0), j_range=(-1.0, 1.0))
    p = doc02_example.scaled(4.0)  # max|h| = 6, max|J| = 2
    assert s.scale_factor(p) == pytest.approx(3.0)


# ---------------------------------------------------------------- QAOA
def test_qaoa_statevector_normalized_and_mixer(rng):
    from qws.solvers.qaoa import QAOASimulator, linear_ramp

    p = random_problem(5, rng)
    q = QAOASimulator(p_layers=2)
    psi = q.statevector(q.cost_diagonal(p), p.n, linear_ramp(2))
    assert np.abs(psi).sum() > 0 and (np.abs(psi) ** 2).sum() == pytest.approx(1.0)
    # zero angles leave the uniform superposition
    probs = q.probabilities(p, np.zeros(4))
    np.testing.assert_allclose(probs, np.full(32, 1 / 32))


def test_qaoa_improves_expected_energy_and_finds_ground(rng):
    from qws.solvers.qaoa import QAOASimulator

    p = random_problem(6, rng)
    q = QAOASimulator(p_layers=2, seed=0)
    uniform = p.energy(all_spins(6)).mean()
    angles, exp_e = q.optimize_angles(p)
    assert exp_e < uniform
    ss = q.minimize([p], shots=2048, angles=angles)[0]
    assert ss.best()[1] == pytest.approx(ExactSolver().minimize([p])[0].best()[1])


def test_qiskit_circuit_matches_statevector(rng):
    pytest.importorskip("qiskit")
    from qiskit.quantum_info import Statevector

    from qws.solvers.qaoa import QAOASimulator, qaoa_circuit

    p = random_problem(4, rng)
    angles = np.array([0.4, 0.9, 0.7, 0.2])
    sv = Statevector(qaoa_circuit(p, angles, measure=False)).probabilities()  # index bit i = qubit i
    ours = QAOASimulator().probabilities(p, angles)
    np.testing.assert_allclose(sv, ours, atol=1e-9)


def test_qiskit_aer_run_and_bit_order(rng):
    pytest.importorskip("qiskit_aer")
    from qws.solvers.qaoa import QAOAQiskit, counts_to_spins

    y, c = counts_to_spins({"001": 3, "110": 1}, 3)  # qubit 0 is the rightmost bit
    assert y.tolist() == [[-1, 1, 1], [1, -1, -1]] and c.tolist() == [3, 1]
    p = random_problem(5, rng)
    ss = QAOAQiskit(p_layers=1, seed=7).minimize([p], shots=2000)[0]
    assert ss.num_samples == 2000
    assert ss.best()[1] == pytest.approx(ExactSolver().minimize([p])[0].best()[1])
