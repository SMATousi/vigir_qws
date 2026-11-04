"""Hybrid learning loop (doc 03 §3; paper Algorithm 1) and inference helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Sequence

import numpy as np

from ..graph import Sites
from ..lf import LFOutputs
from ..model import (
    LabelModelParams, PosteriorMoments, alpha_from_coverage, gradient, posterior_ising, prior_ising, project,
    warm_start,
)
from ..solvers import SampleSet, Solver, gibbs_polish
from .optim import Adam

Data = Sequence[tuple[LFOutputs, Sites]]


@dataclass
class TrainConfig:
    epochs: int = 30
    lr: float = 0.05
    num_samples: int = 200
    batch_size: int | None = None  # None: full batch
    polish_sweeps: int = 0
    mode: Literal["gradient", "em"] = "gradient"
    learn_b: bool = False
    learn_beta: bool = False
    class_conditional: bool = False  # learn omega, zeta (class-dependent coverage / vote bias)
    beta: float = 1.0  # smoothness used when params are warm-started (fixed in v1, doc 01 §6)
    warm_start_iters: int = 300
    tol: float = 1e-4
    seed: int | None = 0


@dataclass
class TrainResult:
    params: LabelModelParams
    history: list[dict[str, Any]] = field(default_factory=list)


class Trainer:
    """Learns LF parameters with any sampling backend.

    Each step builds per-image posterior Ising models, draws samples from the solver
    (quantum or classical), optionally polishes them with Gibbs sweeps, and forms the
    Theorem-1 gradient. In ``em`` mode, the M-step uses the closed forms
    theta = artanh(agree / cover) and alpha from coverage (singleton cliques only).
    """

    def __init__(self, solver: Solver, config: TrainConfig | None = None):
        self.solver = solver
        self.config = config or TrainConfig()
        self.rng = np.random.default_rng(self.config.seed)
        self.clamps: dict[int, np.ndarray] = {}
        self.min_positive: dict[int, int] = {}
        self.constraint_stats: dict = {"checked": 0, "all_rejected": 0, "accepted_fraction": []}

    # ------------------------------------------------------------------ moments
    def _post_moments(self, params: LabelModelParams, batch: Data, idx: Sequence[int]) -> list[PosteriorMoments]:
        """Posterior moments per image, honouring optional constraints (doc 11 §6):

        * ``clamps[i]`` (values in {-1, 0, +1}, 0 = free): clamped sites are fixed. Fully
          clamped items get exact moments without sampling. Partially clamped items are
          sampled on the exact conditional Ising model of their free sites.
        * ``min_positive[i] = k``: samples with fewer than k sites at +1 are rejected (exact
          conditioning on "at least k positives"). If every sample is rejected, the samples
          with the most positives are kept, and the event is counted in ``constraint_stats``.
        """
        from ..ising import condition

        out: list[PosteriorMoments | None] = [None] * len(batch)
        problems, owners, conds = [], [], []
        for pos, (i, (lfo, sites)) in enumerate(zip(idx, batch)):
            p = posterior_ising(params, lfo, sites, chain_key=("post", int(i)))
            clamp = self.clamps.get(int(i))
            if clamp is not None and np.all(clamp != 0):
                c = clamp.astype(float)
                out[pos] = PosteriorMoments(c, c[sites.edges[:, 0]] * c[sites.edges[:, 1]])
                continue
            if clamp is not None and np.any(clamp != 0):
                fixed = np.flatnonzero(clamp != 0)
                cond = condition(p, fixed, clamp[fixed])
                sub = cond.problem
                sub = type(sub)(sub.h, sub.edges, sub.J, sub.offset, sub.kind, {**sub.meta, "chain_key": ("post", int(i))})
                problems.append(sub)
                conds.append(cond)
            else:
                problems.append(p)
                conds.append(None)
            owners.append(pos)
        sets = self.solver.sample(problems, self.config.num_samples) if problems else []
        for pos, p, ss, cond in zip(owners, problems, sets, conds):
            ss = self._polish(p, ss)
            full = cond.lift(ss.samples, batch[pos][1].n) if cond is not None else ss.samples
            counts = ss.counts
            k = self.min_positive.get(int(idx[pos]))
            if k:
                npos = (full > 0).sum(axis=1)
                ok = npos >= k
                self.constraint_stats["checked"] += 1
                if not ok.any():
                    self.constraint_stats["all_rejected"] += 1
                    ok = npos == npos.max()
                self.constraint_stats["accepted_fraction"].append(float(counts[ok].sum() / counts.sum()))
                full, counts = full[ok], counts[ok]
            w = counts / counts.sum()
            fs = full.astype(float)
            e = batch[pos][1].edges
            out[pos] = PosteriorMoments(w @ fs, w @ (fs[:, e[:, 0]] * fs[:, e[:, 1]]))
        return out  # type: ignore[return-value]

    def _prior_moments(self, params: LabelModelParams, batch: Data) -> list[PosteriorMoments]:
        """Moments of the prior Ising model. It depends only on the graph (and parameters),
        so images whose site graphs are identical share one sampling call (doc 03 §2.2)."""
        groups: dict[str, list[int]] = {}
        for k, (_, sites) in enumerate(batch):
            key = _graph_signature(sites)
            groups.setdefault(key, []).append(k)
        out: list[PosteriorMoments | None] = [None] * len(batch)
        reps = [members[0] for members in groups.values()]
        problems = [prior_ising(params, batch[k][1], chain_key=("prior", key))
                    for k, key in zip(reps, groups)]
        sets = self.solver.sample(problems, self.config.num_samples)
        for p, ss, members in zip(problems, sets, groups.values()):
            ss = self._polish(p, ss)
            mom = PosteriorMoments(ss.mean(), ss.edge_corr(batch[members[0]][1].edges))
            for k in members:
                out[k] = mom
        return out  # type: ignore[return-value]

    def _polish(self, p, ss: SampleSet) -> SampleSet:
        if self.config.polish_sweeps <= 0:
            return ss
        polished = gibbs_polish(p, ss.expanded(), self.config.polish_sweeps, self.rng)
        return SampleSet.from_samples(p, polished, info={**ss.info, "polish_sweeps": self.config.polish_sweeps})

    # ------------------------------------------------------------------ fit
    def fit(self, data: Data, params: LabelModelParams | None = None, clamps: dict[int, np.ndarray] | None = None,
            min_positive: dict[int, int] | None = None) -> TrainResult:
        """Fit on ``data``. ``clamps`` and ``min_positive`` are keyed by item index; see
        ``_post_moments`` (used for location-level labels in the gully branch, doc 11 §6)."""
        cfg = self.config
        data = list(data)
        self.clamps = clamps or {}
        self.min_positive = min_positive or {}
        self.constraint_stats = {"checked": 0, "all_rejected": 0, "accepted_fraction": []}
        if params is None:
            params = warm_start(data, iters=cfg.warm_start_iters, beta=cfg.beta)
        if cfg.mode == "em" and (params.cliques or cfg.class_conditional or cfg.learn_b or cfg.learn_beta):
            raise ValueError("EM mode supports the symmetric singleton model with fixed b, beta; use mode='gradient'")
        need_prior = cfg.learn_b or cfg.learn_beta or cfg.class_conditional or not params.is_symmetric
        opt = Adam(cfg.lr)
        result = TrainResult(params.copy())
        bs = cfg.batch_size or len(data)
        for epoch in range(cfg.epochs):
            order = self.rng.permutation(len(data))
            delta = 0.0
            for start in range(0, len(data), bs):
                idx = order[start:start + bs]
                batch = [data[i] for i in idx]
                post = self._post_moments(params, batch, idx)
                if cfg.mode == "em":
                    new = self._m_step(params, batch, post)
                else:
                    prior = self._prior_moments(params, batch) if need_prior else None
                    g = gradient(params, batch, post, prior, cfg.learn_b, cfg.learn_beta)
                    if not cfg.learn_b:
                        g.b = 0.0
                    if not cfg.learn_beta:
                        g.beta = 0.0
                    if not cfg.class_conditional:
                        g.omega[:] = 0.0
                        g.zeta[:] = 0.0
                    n_sites = sum(s.n for _, s in batch)
                    new = project(params.from_vector(opt.step(params.to_vector(), g.to_vector() / n_sites)))
                delta = max(delta, float(np.max(np.abs(new.to_vector() - params.to_vector()))))
                params = new
            result.history.append({"epoch": epoch, "max_param_change": delta, "params": params.to_dict(),
                                   **self._solver_diagnostics()})
            if delta < cfg.tol:
                break
        result.params = params
        return result

    def _m_step(self, params: LabelModelParams, batch: Data, post: list[PosteriorMoments]) -> LabelModelParams:
        p = params.copy()
        n = sum(s.n for _, s in batch)
        agree = sum(lfo.L.astype(float).T @ pm.mean for (lfo, _), pm in zip(batch, post)) / n
        cover = sum((lfo.L != 0).sum(axis=0) for lfo, _ in batch) / n
        p.theta = np.arctanh(np.clip(agree / np.maximum(cover, 1e-9), -0.999, 0.999))
        p.alpha = alpha_from_coverage(cover, p.theta)
        if p.q:
            me = sum(s.m for _, s in batch)
            agreeP = sum(lfo.P.astype(float).T @ pm.edge_corr for (lfo, _), pm in zip(batch, post)) / me
            coverP = sum((lfo.P != 0).sum(axis=0) for lfo, _ in batch) / me
            p.thetaP = np.arctanh(np.clip(agreeP / np.maximum(coverP, 1e-9), -0.999, 0.999))
            p.alphaP = alpha_from_coverage(coverP, p.thetaP)
        return project(p)

    def _solver_diagnostics(self) -> dict[str, Any]:
        hist = getattr(self.solver, "history", None)
        return {"beta_hat": hist[-1]["beta_hat"], "submit_scale": hist[-1]["scale"]} if hist else {}


def _graph_signature(sites: Sites) -> str:
    import hashlib

    m = hashlib.sha256(np.array([sites.n], dtype=np.int64).tobytes())
    m.update(np.ascontiguousarray(sites.edges).tobytes())
    m.update(np.ascontiguousarray(np.round(sites.weights, 12)).tobytes())
    return m.hexdigest()[:16]


# ---------------------------------------------------------------------- inference
def infer_map(params: LabelModelParams, data: Data, solver: Solver, **kw: Any) -> list[np.ndarray]:
    """Hard masks (per site) via the solver's minimize (Corollary 1: MAP = ground state)."""
    problems = [posterior_ising(params, lfo, sites) for lfo, sites in data]
    return [ss.best()[0] for ss in solver.minimize(problems, **kw)]


def infer_marginals(params: LabelModelParams, data: Data, solver: Solver, num_samples: int = 500,
                    polish_sweeps: int = 0, seed: int | None = None, **kw: Any) -> list[np.ndarray]:
    """Soft masks mu_i = P(y_i = +1) estimated from samples."""
    rng = np.random.default_rng(seed)
    problems = [posterior_ising(params, lfo, sites) for lfo, sites in data]
    out = []
    for p, ss in zip(problems, solver.sample(problems, num_samples, **kw)):
        s = gibbs_polish(p, ss.expanded(), polish_sweeps, rng) if polish_sweeps else ss.expanded()
        out.append(0.5 * (1.0 + s.mean(axis=0)))
    return out
