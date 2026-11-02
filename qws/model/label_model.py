"""Structured label model: Ising builders, model moments and gradients.

Implements doc 01 §3 and the paper's Propositions 1–3 and Theorem 1 for symmetric
potentials. Pairwise LFs vote on the site graph's edges (E_r = E, doc 08).
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from ..graph import Sites
from ..ising import IsingProblem
from ..lf import LFOutputs
from .params import LabelModelParams


# --------------------------------------------------------------------------- Ising builders
def posterior_ising(params: LabelModelParams, lfo: LFOutputs, sites: Sites, **meta) -> IsingProblem:
    """Posterior p(y | Λ, Π) as an Ising model (paper Proposition 1)."""
    _check_shapes(params, lfo, sites)
    h = params.b + lfo.L.astype(float) @ params.theta
    J = params.beta * sites.weights
    if params.q:
        J = J + lfo.P.astype(float) @ params.thetaP
    return IsingProblem(h, sites.edges, J, 0.0, "posterior", dict(meta))


def prior_ising(params: LabelModelParams, sites: Sites, **meta) -> IsingProblem:
    """Prior marginal p(y) (Proposition 3: LF parameters drop out for symmetric potentials)."""
    return IsingProblem(np.full(sites.n, params.b), sites.edges, params.beta * sites.weights, 0.0, "prior",
                        dict(meta))


def _check_shapes(params: LabelModelParams, lfo: LFOutputs, sites: Sites) -> None:
    if lfo.L.shape != (sites.n, params.m):
        raise ValueError(f"L has shape {lfo.L.shape}, expected {(sites.n, params.m)}")
    if lfo.P.shape != (sites.m, params.q):
        raise ValueError(f"P has shape {lfo.P.shape}, expected {(sites.m, params.q)}")


# --------------------------------------------------------------------------- clique enumeration
@lru_cache(maxsize=None)
def _vote_table(size: int) -> np.ndarray:
    return np.array(list(itertools.product((-1, 0, 1), repeat=size)), dtype=float)


def agreement_feature(votes: np.ndarray) -> np.ndarray:
    """phi_c = number of pairs j<l in the clique with equal non-zero votes (flip-invariant)."""
    v = np.atleast_2d(votes)
    out = np.zeros(v.shape[0])
    for a, b in itertools.combinations(range(v.shape[1]), 2):
        out += (v[:, a] == v[:, b]) & (v[:, a] != 0)
    return out


def all_cliques(params: LabelModelParams) -> list[tuple[tuple[int, ...], float]]:
    """Every unary LF's clique with its gamma (singletons have gamma 0)."""
    in_clique = {j for c in params.cliques for j in c}
    out = [(c, float(g)) for c, g in zip(params.cliques, params.gamma)]
    out += [((j,), 0.0) for j in range(params.m) if j not in in_clique]
    return out


@dataclass(frozen=True)
class ModelMoments:
    """Per-site (unary) and per-edge (pairwise) expectations under the model.

    By symmetry (paper Proposition 3) none of these depend on the law of y.
    """

    agree: np.ndarray       # E[Λ_j y], (m,)
    cover: np.ndarray       # E[1(Λ_j != 0)], (m,)
    clique_feat: np.ndarray  # E[phi_c], (len(params.cliques),)
    agreeP: np.ndarray      # E[Π_r y_i y_k], (q,)
    coverP: np.ndarray      # E[1(Π_r != 0)], (q,)


def model_moments(params: LabelModelParams) -> ModelMoments:
    agree = np.zeros(params.m)
    cover = np.zeros(params.m)
    feat = np.zeros(len(params.cliques))
    clique_pos = {c: k for k, c in enumerate(params.cliques)}
    for c, g in all_cliques(params):
        idx = list(c)
        table = _vote_table(len(c))  # λ for the clique; condition on y = +1 (valid by symmetry)
        logw = table @ params.theta[idx] + (table != 0) @ params.alpha[idx]
        phi = agreement_feature(table) if len(c) > 1 else np.zeros(len(table))
        logw = logw + g * phi
        w = np.exp(logw - logw.max())
        w /= w.sum()
        agree[idx] = w @ table
        cover[idx] = w @ (table != 0)
        if c in clique_pos:
            feat[clique_pos[c]] = w @ phi
    zP = 1.0 + 2.0 * np.exp(params.alphaP) * np.cosh(params.thetaP)
    agreeP = 2.0 * np.exp(params.alphaP) * np.sinh(params.thetaP) / zP
    coverP = 2.0 * np.exp(params.alphaP) * np.cosh(params.thetaP) / zP
    return ModelMoments(agree, cover, feat, agreeP, coverP)


# --------------------------------------------------------------------------- gradient (Theorem 1)
@dataclass(frozen=True)
class PosteriorMoments:
    """Estimated <y_i> (n,) and <y_i y_k> on sites.edges (m,) for one image."""

    mean: np.ndarray
    edge_corr: np.ndarray

    @classmethod
    def from_samples(cls, samples: np.ndarray, edges: np.ndarray, weights: np.ndarray | None = None
                     ) -> "PosteriorMoments":
        s = np.atleast_2d(samples).astype(float)
        w = np.ones(s.shape[0]) if weights is None else np.asarray(weights, dtype=float)
        w = w / w.sum()
        return cls(w @ s, w @ (s[:, edges[:, 0]] * s[:, edges[:, 1]]))


def gradient(params: LabelModelParams, batch: list[tuple[LFOutputs, Sites]], post: list[PosteriorMoments],
             prior: list[PosteriorMoments] | None = None, learn_b: bool = False,
             learn_beta: bool = False) -> LabelModelParams:
    """Gradient of the marginal log-likelihood (paper Lemma 1 / Theorem 1), as raw sums.

    ``prior`` (moments of the prior Ising model per image) is required only when
    learning ``b`` or ``beta``.
    """
    mm = model_moments(params)
    g = params.zeros_like()
    clique_cols = [list(c) for c in params.cliques]
    for (lfo, sites), pm in zip(batch, post, strict=True):
        n, m_edges = sites.n, sites.m
        L = lfo.L.astype(float)
        g.theta += L.T @ pm.mean - n * mm.agree
        g.alpha += (lfo.L != 0).sum(axis=0) - n * mm.cover
        for k, cols in enumerate(clique_cols):
            g.gamma[k] += agreement_feature(lfo.L[:, cols]).sum() - n * mm.clique_feat[k]
        if params.q:
            P = lfo.P.astype(float)
            g.thetaP += P.T @ pm.edge_corr - m_edges * mm.agreeP
            g.alphaP += (lfo.P != 0).sum(axis=0) - m_edges * mm.coverP
    if learn_b or learn_beta:
        if prior is None:
            raise ValueError("prior moments are required to learn b or beta")
        for (lfo, sites), pm, pr in zip(batch, post, prior, strict=True):
            if learn_b:
                g.b += pm.mean.sum() - pr.mean.sum()
            if learn_beta:
                g.beta += sites.weights @ pm.edge_corr - sites.weights @ pr.edge_corr
    return g


# --------------------------------------------------------------------------- warm start
def independent_posterior(params: LabelModelParams, lfo: LFOutputs) -> PosteriorMoments:
    """Exact posterior moments of the beta = 0, no-pairwise model: <y_i> = tanh(h_i)."""
    mean = np.tanh(params.b + lfo.L.astype(float) @ params.theta)
    return PosteriorMoments(mean, np.zeros(0))


def warm_start(batch: list[tuple[LFOutputs, Sites]], q: int | None = None, iters: int = 300, lr: float = 0.05,
               theta0: float = 0.5, cliques: tuple[tuple[int, ...], ...] = (), b: float = 0.0,
               beta: float = 0.0) -> LabelModelParams:
    """Fit unary LF parameters with the independent model (closed-form posterior), then
    attach pairwise LFs initialised at ``theta0`` and the given ``b``, ``beta``."""
    from ..learn.optim import Adam

    lfo0 = batch[0][0]
    m = lfo0.m
    q = lfo0.q if q is None else q
    cov = np.mean(np.concatenate([lfo.L != 0 for lfo, _ in batch]), axis=0)
    covP = np.mean(np.concatenate([lfo.P != 0 for lfo, _ in batch]), axis=0) if q else None
    params = LabelModelParams.init(m, q, theta0, theta0, b, 0.0, cov, covP, cliques)
    unary = params.copy()
    unary.alphaP, unary.thetaP = np.zeros(0), np.zeros(0)
    stripped = [(LFOutputs(lfo.L, np.zeros((sites.m, 0), np.int8)), sites) for lfo, sites in batch]
    n_total = sum(s.n for _, s in batch)
    opt = Adam(lr)
    for _ in range(iters):
        post = [independent_posterior(unary, lfo) for lfo, _ in stripped]
        g = gradient(unary, stripped, post)
        mask = unary.zeros_like()
        mask.theta[:], mask.alpha[:], mask.gamma[:] = 1, 1, 1
        vec = opt.step(unary.to_vector(), g.to_vector() * mask.to_vector() / n_total)
        unary = project(unary.from_vector(vec))
    params.alpha, params.theta, params.gamma = unary.alpha, unary.theta, unary.gamma
    params.beta = beta
    return params


def project(params: LabelModelParams, eps: float = 1e-3) -> LabelModelParams:
    """beta >= 0 and mean accuracy > 0 (label-flip symmetry breaking)."""
    p = params.copy()
    p.beta = max(p.beta, 0.0)
    if p.m and p.theta.mean() < eps:
        p.theta = p.theta + (eps - p.theta.mean())
    return p
