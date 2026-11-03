"""Model tests against brute-force enumeration of the full joint p(Λ, Π, y)."""

import itertools

import numpy as np
import pytest
from scipy.special import logsumexp

from qws.graph import grid_sites
from qws.lf import LFOutputs
from qws.model import (
    LabelModelParams, PosteriorMoments, agreement_feature, gradient, model_moments, posterior_ising, prior_ising,
    warm_start,
)
from qws.solvers import ExactSolver, all_spins


def _unary_scores(params, Ls, y):
    """Unary-LF and clique log-potentials for a stack of vote matrices Ls (K, n, m)."""
    s = ((Ls != 0) * params.alpha).sum((1, 2)) + (Ls * y[None, :, None] * params.theta).sum((1, 2))
    s = s + ((Ls != 0) * y[None, :, None] * params.omega).sum((1, 2)) + (Ls * params.zeta).sum((1, 2))
    for c, g in zip(params.cliques, params.gamma):
        sub = Ls[:, :, list(c)].reshape(-1, len(c))
        s = s + g * agreement_feature(sub).reshape(Ls.shape[0], -1).sum(1)
    return s


def _pair_scores(params, sites, Ps, y):
    if not params.q:
        return np.zeros(Ps.shape[0])
    yy = y[sites.edges[:, 0]] * y[sites.edges[:, 1]]
    return ((Ps != 0) * params.alphaP).sum((1, 2)) + (Ps * yy[None, :, None] * params.thetaP).sum((1, 2))


def joint_score(params, sites, L, P, y):
    """log-potential of the joint model (1) for one configuration (no normalizer)."""
    s0 = params.b * y.sum() + params.beta * sites.weights @ (y[sites.edges[:, 0]] * y[sites.edges[:, 1]])
    return float(s0 + _unary_scores(params, L[None], y)[0] + _pair_scores(params, sites, P[None], y)[0])


def brute_loglik(params, sites, L0, P0):
    """Marginal log-likelihood by full enumeration of (Λ, Π, y)."""
    n, m, q, me = sites.n, params.m, params.q, sites.m
    Ys = all_spins(n)
    num = logsumexp([joint_score(params, sites, L0, P0, y) for y in Ys])
    allL = np.array(list(itertools.product((-1, 0, 1), repeat=n * m)), dtype=float).reshape(-1, n, m)
    allP = (np.array(list(itertools.product((-1, 0, 1), repeat=me * q)), dtype=float).reshape(-1, me, q)
            if q else np.zeros((1, me, 0)))
    terms = []
    for y in Ys:
        s0 = params.b * y.sum() + params.beta * sites.weights @ (y[sites.edges[:, 0]] * y[sites.edges[:, 1]])
        full = _unary_scores(params, allL, y)[:, None] + _pair_scores(params, sites, allP, y)[None, :]
        terms.append(s0 + full.ravel())
    return num - logsumexp(np.concatenate(terms))


@pytest.fixture
def tiny():
    sites = grid_sites(np.array([[0.0, 0.3, 1.0]]))  # 3 sites, 2 edges
    params = LabelModelParams(alpha=[0.3, -0.2], theta=[0.9, 0.4], alphaP=[0.1], thetaP=[0.7], b=0.2, beta=0.6)
    L0 = np.array([[1, 0], [1, -1], [-1, -1]], dtype=np.int8)
    P0 = np.array([[1], [-1]], dtype=np.int8)
    return sites, params, LFOutputs(L0, P0)


def test_posterior_builder_matches_joint(tiny):
    sites, params, lfo = tiny
    Ys = all_spins(3)
    s = np.array([joint_score(params, sites, lfo.L.astype(float), lfo.P.astype(float), y) for y in Ys])
    brute = np.exp(s - logsumexp(s))
    _, ising = ExactSolver().distribution(posterior_ising(params, lfo, sites))
    np.testing.assert_allclose(ising, brute, atol=1e-12)


def test_gradient_matches_finite_differences(tiny):
    sites, params, lfo = tiny
    ex = ExactSolver()
    post = PosteriorMoments(*ex.moments(posterior_ising(params, lfo, sites)))
    prior = PosteriorMoments(*ex.moments(prior_ising(params, sites)))
    g = gradient(params, [(lfo, sites)], [post], [prior], learn_b=True, learn_beta=True).to_vector()
    v = params.to_vector()
    eps = 1e-5
    fd = np.zeros_like(v)
    for k in range(v.size):
        up, dn = v.copy(), v.copy()
        up[k] += eps
        dn[k] -= eps
        fd[k] = (brute_loglik(params.from_vector(up), sites, lfo.L.astype(float), lfo.P.astype(float))
                 - brute_loglik(params.from_vector(dn), sites, lfo.L.astype(float), lfo.P.astype(float))) / (2 * eps)
    np.testing.assert_allclose(g, fd, atol=1e-6)


def test_clique_gradient_matches_finite_differences():
    sites = grid_sites(np.array([[0.0, 1.0]]))  # 2 sites, 1 edge
    params = LabelModelParams(alpha=[0.2, 0.0, -0.3], theta=[0.8, 0.5, 0.3], gamma=[0.4], cliques=((0, 1),),
                              beta=0.5)
    L0 = np.array([[1, 1, -1], [0, 1, 1]], dtype=np.int8)
    lfo = LFOutputs(L0, np.zeros((1, 0), np.int8))
    ex = ExactSolver()
    post = PosteriorMoments(*ex.moments(posterior_ising(params, lfo, sites)))
    g = gradient(params, [(lfo, sites)], [post]).to_vector()
    v = params.to_vector()
    eps = 1e-5
    for k in range(v.size - 2):  # b and beta are not learned here
        up, dn = v.copy(), v.copy()
        up[k] += eps
        dn[k] -= eps
        fd = (brute_loglik(params.from_vector(up), sites, L0.astype(float), lfo.P.astype(float))
              - brute_loglik(params.from_vector(dn), sites, L0.astype(float), lfo.P.astype(float))) / (2 * eps)
        assert g[k] == pytest.approx(fd, abs=1e-6)


def test_closed_form_moments():
    params = LabelModelParams(alpha=[0.3], theta=[0.9], alphaP=[-0.4], thetaP=[0.6])
    mm = model_moments(params)
    z = 1 + 2 * np.exp(0.3) * np.cosh(0.9)
    assert mm.agree[0] == pytest.approx(2 * np.exp(0.3) * np.sinh(0.9) / z)
    assert mm.cover[0, 0] == pytest.approx(2 * np.exp(0.3) * np.cosh(0.9) / z)
    assert mm.vote[1, 0] == pytest.approx(-mm.vote[0, 0]) and mm.cover[1, 0] == pytest.approx(mm.cover[0, 0])
    assert mm.prior_field == pytest.approx(0.0)
    zp = 1 + 2 * np.exp(-0.4) * np.cosh(0.6)
    assert mm.agreeP[0] == pytest.approx(2 * np.exp(-0.4) * np.sinh(0.6) / zp)
    # conditional accuracy given a vote is sigmoid(2 theta)
    assert mm.agree[0] / mm.cover[0, 0] == pytest.approx(np.tanh(0.9))


def test_prior_independent_of_lf_params(tiny):
    sites, params, _ = tiny
    other = params.copy()
    other.theta, other.alpha = np.array([2.0, -1.0]), np.array([1.0, 3.0])
    np.testing.assert_allclose(prior_ising(params, sites).h, prior_ising(other, sites).h)
    np.testing.assert_allclose(prior_ising(params, sites).J, prior_ising(other, sites).J)


def test_params_roundtrip():
    p = LabelModelParams(alpha=[0.1, 0.2, 0.3], theta=[1, 2, 3], gamma=[0.5], cliques=((0, 2),), b=0.1, beta=0.4)
    assert np.allclose(p.from_vector(p.to_vector()).to_vector(), p.to_vector())
    q = LabelModelParams.from_dict(p.to_dict())
    assert q.cliques == p.cliques and np.allclose(q.to_vector(), p.to_vector())
    with pytest.raises(ValueError):
        LabelModelParams(alpha=[0, 0], theta=[1, 1], gamma=[0.1], cliques=((0,),))


def test_warm_start_recovers_independent_model():
    from qws.eval import synthetic_dataset

    true = LabelModelParams(alpha=[0.5, 0.0, -0.5, 0.2], theta=[1.0, 0.6, 0.3, 0.8])
    ds = synthetic_dataset(true, n_images=30, shape=(16, 16), smoothness=0.01, fg_fraction=0.5, seed=3)
    p = warm_start([(im.lfo, im.sites) for im in ds], iters=600)
    np.testing.assert_allclose(p.theta, true.theta, atol=0.12)
    np.testing.assert_allclose(p.alpha, true.alpha, atol=0.12)


@pytest.fixture
def tiny_cc():
    """Class-conditional unary LFs, one correlation clique, one pairwise LF."""
    sites = grid_sites(np.array([[0.0, 0.3]]))  # 2 sites, 1 edge
    params = LabelModelParams(alpha=[0.3, -0.2, 0.1], theta=[0.9, 0.4, 0.6], omega=[0.5, -0.3, 0.2],
                              zeta=[-0.4, 0.2, 0.1], gamma=[0.3], cliques=((0, 2),), alphaP=[0.1], thetaP=[0.7],
                              b=0.2, beta=0.6)
    L0 = np.array([[1, 0, 1], [-1, -1, 0]], dtype=np.int8)
    P0 = np.array([[-1]], dtype=np.int8)
    return sites, params, LFOutputs(L0, P0)


def test_class_conditional_posterior_and_prior_match_brute_force(tiny_cc):
    sites, params, lfo = tiny_cc
    Ys = all_spins(2)
    s = np.array([joint_score(params, sites, lfo.L.astype(float), lfo.P.astype(float), y) for y in Ys])
    _, post = ExactSolver().distribution(posterior_ising(params, lfo, sites))
    np.testing.assert_allclose(post, np.exp(s - logsumexp(s)), atol=1e-12)
    # prior marginal p(y): brute-force sum over all votes vs the prior Ising model (Proposition 2)
    allL = np.array(list(itertools.product((-1, 0, 1), repeat=2 * 3)), dtype=float).reshape(-1, 2, 3)
    allP = np.array(list(itertools.product((-1, 0, 1), repeat=1)), dtype=float).reshape(-1, 1, 1)
    logp = []
    for y in Ys:
        s0 = params.b * y.sum() + params.beta * sites.weights @ (y[sites.edges[:, 0]] * y[sites.edges[:, 1]])
        full = _unary_scores(params, allL, y)[:, None] + _pair_scores(params, sites, allP, y)[None, :]
        logp.append(s0 + logsumexp(full))
    brute = np.exp(np.array(logp) - logsumexp(logp))
    _, prior = ExactSolver().distribution(prior_ising(params, sites))
    np.testing.assert_allclose(prior, brute, atol=1e-12)
    assert not params.is_symmetric


def test_class_conditional_gradient_matches_finite_differences(tiny_cc):
    sites, params, lfo = tiny_cc
    ex = ExactSolver()
    post = PosteriorMoments(*ex.moments(posterior_ising(params, lfo, sites)))
    prior = PosteriorMoments(*ex.moments(prior_ising(params, sites)))
    g = gradient(params, [(lfo, sites)], [post], [prior], learn_b=True, learn_beta=True).to_vector()
    v = params.to_vector()
    eps = 1e-5
    L, P = lfo.L.astype(float), lfo.P.astype(float)
    for k in range(v.size):
        up, dn = v.copy(), v.copy()
        up[k] += eps
        dn[k] -= eps
        fd = (brute_loglik(params.from_vector(up), sites, L, P) - brute_loglik(params.from_vector(dn), sites, L, P)) / (2 * eps)
        assert g[k] == pytest.approx(fd, abs=1e-6), k


def test_gradient_requires_prior_when_class_conditional(tiny_cc):
    sites, params, lfo = tiny_cc
    post = PosteriorMoments(np.zeros(2), np.zeros(1))
    with pytest.raises(ValueError):
        gradient(params, [(lfo, sites)], [post])
