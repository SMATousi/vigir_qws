import itertools

import numpy as np
import pytest
from scipy.special import logsumexp

from qws.graph import grid_sites
from qws.multiclass import (
    MulticlassParams, MulticlassVotes, PottsGibbs, PottsProblem, all_labelings, decode_onehot, exact_potts,
    fit_multiclass, gradient_multiclass, multiclass_dataset, posterior_potts, sample_moments, to_onehot_ising,
    to_onehot_qubo,
)
from qws.solvers import ExactSolver, SimulatedAnnealing, all_spins


def random_potts(rng, n=4, K=3, density=0.6, scale=1.0):
    pairs = [(i, k) for i in range(n) for k in range(i + 1, n) if rng.random() < density]
    return PottsProblem(rng.normal(0, scale, (n, K)), np.array(pairs, dtype=np.int64).reshape(-1, 2),
                        rng.normal(0, scale, len(pairs)))


def test_onehot_energy_identity_on_feasible(rng):
    p = random_potts(rng)
    q, A = to_onehot_qubo(p)
    for y in all_labelings(p.n, p.K)[::5]:
        x = np.zeros(p.n * p.K)
        x[np.arange(p.n) * p.K + y] = 1
        assert q.energy(x)[0] == pytest.approx(p.energy(y)[0])


def test_penalty_bound_makes_minimizers_feasible(rng):
    """Paper Proposition 11, checked by exhaustive enumeration of all 2^(nK) binary states."""
    ex = ExactSolver(max_n=15)
    for _ in range(10):
        p = random_potts(rng, n=5, K=3)
        ising, A = to_onehot_ising(p)
        ground = ex.minimize([ising])[0]
        labels, infeasible = decode_onehot(ground.samples, p)
        assert not infeasible.any()
        assert p.energy(labels).min() == pytest.approx(p.energy(all_labelings(5, 3)).min())


def test_too_small_penalty_can_break_feasibility():
    p = PottsProblem(np.array([[3.0, 2.9, -5.0]]), np.zeros((0, 2), np.int64), np.zeros(0))
    ising, _ = to_onehot_ising(p, A=0.5)
    labels, infeasible = decode_onehot(ExactSolver().minimize([ising])[0].samples, p)
    assert infeasible.all() and labels[0, 0] == 0  # repaired to the best active class


def test_decode_repair():
    p = PottsProblem(np.array([[0.0, 1.0, 0.5], [2.0, 0.0, 0.0]]), np.zeros((0, 2), np.int64), np.zeros(0))
    spins = np.array([[1, -1, 1, -1, -1, -1]])  # site 0: classes 0 and 2 active; site 1: none active
    labels, infeasible = decode_onehot(spins, p)
    assert labels.tolist() == [[2, 0]] and infeasible.tolist() == [True]


def test_potts_gibbs_matches_exact(rng):
    p = random_potts(rng, n=4, K=3, scale=0.6)
    Y, prob = exact_potts(p)
    exact_marg = np.stack([(prob[:, None] * (Y == c)).sum(0) for c in range(3)], axis=1)
    S = PottsGibbs(num_sweeps=40, seed=0).sample(p, 4000)
    probs, _ = sample_moments(p, S)
    np.testing.assert_allclose(probs, exact_marg, atol=0.04)


def test_potts_anneal_and_qubo_routes_agree(rng):
    for _ in range(5):
        p = random_potts(rng, n=6, K=3)
        best = p.energy(all_labelings(6, 3)).min()
        assert p.energy(PottsGibbs(num_sweeps=300, seed=1).anneal(p, num_reads=10))[0] == pytest.approx(best)
        ising, _ = to_onehot_ising(p)
        labels, _ = decode_onehot(SimulatedAnnealing(num_sweeps=400, seed=0).minimize([ising], num_reads=100)[0].best()[0], p)
        assert p.energy(labels)[0] == pytest.approx(best)


# --------------------------------------------------------------- brute-force gradient check
def _joint_logpot(params, sites, L, P, y):
    s = params.b[y].sum() + params.beta * sites.weights @ (y[sites.edges[:, 0]] == y[sites.edges[:, 1]])
    s += ((L != 0) * params.alpha).sum() + ((L == (y[:, None] + 1)) * params.theta).sum()
    sp = np.where(y[sites.edges[:, 0]] == y[sites.edges[:, 1]], 1, -1)
    s += ((P != 0) * params.alphaP).sum() + ((P * sp[:, None]) * params.thetaP).sum()
    return s


def _brute_loglik(params, sites, L0, P0):
    Ys = all_labelings(sites.n, params.K)
    num = logsumexp([_joint_logpot(params, sites, L0, P0, y) for y in Ys])
    allL = list(itertools.product(range(params.K + 1), repeat=sites.n * params.m))
    allP = list(itertools.product((-1, 0, 1), repeat=sites.m * params.q))
    den = logsumexp([_joint_logpot(params, sites, np.array(Lf).reshape(sites.n, params.m),
                                   np.array(Pf).reshape(sites.m, params.q), y)
                     for Lf in allL for Pf in allP for y in Ys])
    return num - den


def test_multiclass_gradient_matches_finite_differences():
    sites = grid_sites(np.array([[0.0, 0.4]]))
    params = MulticlassParams(3, alpha=[0.2, -0.3], theta=[0.9, 0.5], alphaP=[0.1], thetaP=[0.6], b=[0.1, -0.2, 0.0],
                              beta=0.4)
    votes = MulticlassVotes(np.array([[1, 0], [3, 3]]), np.array([[-1]]))
    pot = posterior_potts(params, votes, sites)
    Y, prob = exact_potts(pot)
    probs = np.stack([(prob[:, None] * (Y == c)).sum(0) for c in range(3)], axis=1)
    same = np.array([prob @ (Y[:, 0] == Y[:, 1])])
    g = gradient_multiclass(params, [(votes, sites)], [probs], [same])
    eps = 1e-5
    for name in ("alpha", "theta", "alphaP", "thetaP"):
        for k in range(getattr(params, name).size):
            up, dn = params.copy(), params.copy()
            getattr(up, name)[k] += eps
            getattr(dn, name)[k] -= eps
            fd = (_brute_loglik(up, sites, votes.L, votes.P) - _brute_loglik(dn, sites, votes.L, votes.P)) / (2 * eps)
            assert getattr(g, name)[k] == pytest.approx(fd, abs=1e-6), (name, k)


def test_posterior_potts_matches_joint():
    sites = grid_sites(np.array([[0.0, 0.4, 0.9]]))
    params = MulticlassParams(3, alpha=[0.2, -0.3], theta=[0.9, 0.5], alphaP=[0.1], thetaP=[0.6], b=[0.1, -0.2, 0.0],
                              beta=0.4)
    votes = MulticlassVotes(np.array([[1, 0], [3, 3], [0, 2]]), np.array([[-1], [1]]))
    Ys = all_labelings(3, 3)
    s = np.array([_joint_logpot(params, sites, votes.L, votes.P, y) for y in Ys])
    _, prob = exact_potts(posterior_potts(params, votes, sites))
    np.testing.assert_allclose(prob, np.exp(s - logsumexp(s)), atol=1e-12)


@pytest.mark.slow
def test_multiclass_learning_recovers_accuracies():
    true = MulticlassParams(3, alpha=[0.5, 0.0, 0.3], theta=[1.2, 0.7, 1.5], alphaP=[0.0], thetaP=[0.8], beta=0.5)
    ds = multiclass_dataset(true, 8, (12, 12), seed=0)
    batch = [(v, s) for v, s, _ in ds]
    init = MulticlassParams(3, alpha=np.zeros(3), theta=np.full(3, 0.5), alphaP=[0.0], thetaP=[0.5], beta=0.5)
    fit = fit_multiclass(batch, init, epochs=60, num_samples=50, sweeps=15)
    np.testing.assert_allclose(fit.theta, true.theta, atol=0.25)
    np.testing.assert_allclose(fit.thetaP, true.thetaP, atol=0.25)
