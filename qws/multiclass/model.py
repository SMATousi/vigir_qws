"""K-class structured label model (spec doc 07; roadmap 4.1).

Votes Λ_ij in {0, 1..K} (0 = abstain, c = "class c-1"); pairwise votes Π in {-1, 0, +1}.
Symmetric potentials:

    psi_j(λ, y)   = alpha_j 1[λ != 0] + theta_j 1[λ = y + 1]
    psi^P_r(π, y_i, y_k) = alpha^P_r 1[π != 0] + theta^P_r π (2 1[y_i = y_k] - 1)

plus class biases b_c and Potts smoothness beta w_ik 1[y_i = y_k]. As in the binary case
(Proposition 3), the per-LF normalizers do not depend on y:
Z_j = 1 + e^{alpha+theta} + (K-1) e^{alpha}. The prior is then the Potts model (b, beta), and
every LF model moment is a closed-form constant, so learning needs only the posterior
marginals P(y_i = c) and the edge agreement probabilities P(y_i = y_k).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import gaussian_filter

from ..graph import Sites, grid_sites
from .potts import PottsGibbs, PottsProblem


@dataclass
class MulticlassParams:
    K: int
    alpha: np.ndarray
    theta: np.ndarray
    alphaP: np.ndarray = field(default_factory=lambda: np.zeros(0))
    thetaP: np.ndarray = field(default_factory=lambda: np.zeros(0))
    b: np.ndarray = field(default_factory=lambda: np.zeros(0))
    beta: float = 0.0

    def __post_init__(self) -> None:
        for k in ("alpha", "theta", "alphaP", "thetaP", "b"):
            setattr(self, k, np.asarray(getattr(self, k), dtype=float).reshape(-1).copy())
        if self.b.size == 0:
            self.b = np.zeros(self.K)

    @property
    def m(self) -> int:
        return int(self.theta.size)

    @property
    def q(self) -> int:
        return int(self.thetaP.size)

    def copy(self) -> "MulticlassParams":
        return MulticlassParams(self.K, self.alpha, self.theta, self.alphaP, self.thetaP, self.b, self.beta)


@dataclass(frozen=True)
class MulticlassVotes:
    L: np.ndarray   # (n, m) int in {0..K}
    P: np.ndarray   # (n_edges, q) int in {-1, 0, 1}


def posterior_potts(params: MulticlassParams, votes: MulticlassVotes, sites: Sites) -> PottsProblem:
    """Posterior p(y | Λ, Π) as a Potts model:
    u_ic = b_c + sum_j theta_j 1[Λ_ij = c+1];  Jt_ik = beta w_ik + sum_r 2 theta^P_r Π_ikr
    (the -theta^P Π part of (2 1[=] - 1) is constant in y and is dropped)."""
    K = params.K
    u = np.tile(params.b, (sites.n, 1))
    for j in range(params.m):
        c = votes.L[:, j]
        voted = c > 0
        u[np.flatnonzero(voted), c[voted] - 1] += params.theta[j]
    Jt = params.beta * sites.weights
    if params.q:
        Jt = Jt + 2.0 * votes.P.astype(float) @ params.thetaP
    return PottsProblem(u, sites.edges, Jt)


def model_constants(params: MulticlassParams) -> dict[str, np.ndarray]:
    """Closed-form model moments: E[1(Λ_j = y+1)], E[1(Λ_j != 0)], E[Π s'], E[1(Π != 0)]."""
    K = params.K
    ea, eat = np.exp(params.alpha), np.exp(params.alpha + params.theta)
    Z = 1 + eat + (K - 1) * ea
    zP = 1 + 2 * np.exp(params.alphaP) * np.cosh(params.thetaP)
    return {"correct": eat / Z, "cover": (eat + (K - 1) * ea) / Z,
            "agreeP": 2 * np.exp(params.alphaP) * np.sinh(params.thetaP) / zP,
            "coverP": 2 * np.exp(params.alphaP) * np.cosh(params.thetaP) / zP}


def gradient_multiclass(params: MulticlassParams, batch: list[tuple[MulticlassVotes, Sites]],
                        post_probs: list[np.ndarray], post_same: list[np.ndarray]) -> MulticlassParams:
    """Theorem-1 gradient for the K-class model (b, beta fixed).

    ``post_probs[n]``: (n_sites, K) posterior P(y_i = c); ``post_same[n]``: (n_edges,) P(y_i = y_k).
    """
    mc = model_constants(params)
    g = MulticlassParams(params.K, np.zeros(params.m), np.zeros(params.m), np.zeros(params.q), np.zeros(params.q),
                         np.zeros(params.K), 0.0)
    for (votes, sites), probs, same in zip(batch, post_probs, post_same, strict=True):
        n = sites.n
        for j in range(params.m):
            c = votes.L[:, j]
            voted = c > 0
            g.theta[j] += probs[np.flatnonzero(voted), c[voted] - 1].sum() - n * mc["correct"][j]
            g.alpha[j] += voted.sum() - n * mc["cover"][j]
        if params.q:
            s_prime = 2 * same - 1
            g.thetaP += votes.P.astype(float).T @ s_prime - sites.m * mc["agreeP"]
            g.alphaP += (votes.P != 0).sum(axis=0) - sites.m * mc["coverP"]
    return g


def sample_moments(p: PottsProblem, samples: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Posterior class probabilities (n, K) and edge agreement probabilities (m,) from samples."""
    probs = np.stack([(samples == c).mean(axis=0) for c in range(p.K)], axis=1)
    same = (samples[:, p.edges[:, 0]] == samples[:, p.edges[:, 1]]).mean(axis=0) if p.edges.size else np.zeros(0)
    return probs, same


def fit_multiclass(batch: list[tuple[MulticlassVotes, Sites]], params: MulticlassParams, epochs: int = 40,
                   lr: float = 0.05, num_samples: int = 100, sweeps: int = 30, seed: int = 0) -> MulticlassParams:
    """Adam ascent on the K-class likelihood with Potts Gibbs sampling of the posteriors."""
    from ..learn.optim import Adam

    sampler = PottsGibbs(num_sweeps=sweeps, seed=seed)
    opt = Adam(lr)
    chains: dict[int, np.ndarray] = {}
    n_tot = sum(s.n for _, s in batch)
    p = params.copy()
    for _ in range(epochs):
        probs, sames = [], []
        for k, (votes, sites) in enumerate(batch):
            pot = posterior_potts(p, votes, sites)
            Y = sampler.sample(pot, num_samples, init=chains.get(k))
            chains[k] = Y
            pr, sm = sample_moments(pot, Y)
            probs.append(pr)
            sames.append(sm)
        g = gradient_multiclass(p, batch, probs, sames)
        vec = np.concatenate([p.alpha, p.theta, p.alphaP, p.thetaP])
        gv = np.concatenate([g.alpha, g.theta, g.alphaP, g.thetaP]) / n_tot
        vec = opt.step(vec, gv)
        m, q = p.m, p.q
        p.alpha, p.theta = vec[:m], vec[m:2 * m]
        p.alphaP, p.thetaP = vec[2 * m:2 * m + q], vec[2 * m + q:]
        if p.theta.mean() < 1e-3:  # label-permutation symmetry breaking, as in the binary case
            p.theta = p.theta + (1e-3 - p.theta.mean())
    return p


# ------------------------------------------------------------------------------ synthetic data
def make_multiclass_mask(shape: tuple[int, int], K: int, rng: np.random.Generator, smoothness: float = 3.0
                         ) -> np.ndarray:
    fields = np.stack([gaussian_filter(rng.standard_normal(shape), smoothness) for _ in range(K)])
    return fields.argmax(axis=0)


def sample_multiclass_votes(params: MulticlassParams, y: np.ndarray, edges: np.ndarray, rng: np.random.Generator
                            ) -> MulticlassVotes:
    """Votes from the symmetric conditional law: abstain, the correct class, or any wrong class."""
    n, K = y.shape[0], params.K
    L = np.zeros((n, params.m), dtype=np.int64)
    for j in range(params.m):
        w = np.array([1.0, np.exp(params.alpha[j] + params.theta[j])] + [np.exp(params.alpha[j])] * (K - 1))
        kind = rng.choice(K + 1, size=n, p=w / w.sum())  # 0 abstain, 1 correct, 2.. wrong class offsets
        wrong = (y + kind - 1) % K
        L[:, j] = np.where(kind == 0, 0, np.where(kind == 1, y + 1, wrong + 1))
    P = np.zeros((edges.shape[0], params.q), dtype=np.int64)
    s = np.where(y[edges[:, 0]] == y[edges[:, 1]], 1, -1)
    for r in range(params.q):
        w = np.array([1.0, np.exp(params.alphaP[r] + params.thetaP[r]), np.exp(params.alphaP[r] - params.thetaP[r])])
        k = rng.choice(3, size=edges.shape[0], p=w / w.sum())
        P[:, r] = np.where(k == 0, 0, np.where(k == 1, s, -s))
    return MulticlassVotes(L, P)


def multiclass_dataset(params: MulticlassParams, n_images: int, shape: tuple[int, int], seed: int = 0):
    """List of (votes, sites, y) on pixel grids with contrast weights from a noisy class image."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_images):
        y = make_multiclass_mask(shape, params.K, rng)
        image = y / max(params.K - 1, 1) + 0.15 * rng.standard_normal(shape)
        sites = grid_sites(image)
        yv = y.reshape(-1)
        out.append((sample_multiclass_votes(params, yv, sites.edges, rng), sites, yv))
    return out
