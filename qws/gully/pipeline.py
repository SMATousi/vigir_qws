"""End-to-end weak segmentation of ephemeral gullies (spec doc 11 §6–7).

Steps:

1. **prepare**: per location, load the 6-year stack and the cached terrain, apply the gully
   LFs, and build the spatiotemporal site graph. LF votes are cached on disk.
2. **stage 1, exact independent MIL label model** (decision G8): learn every LF parameter
   (alpha, theta and, by default, the class-conditional omega, zeta) and the class bias b with
   beta = 0, where everything is exact:
   - free sites have posterior P(y_i = +1) = sigma(2 h_i);
   - negative locations are clamped to background (decision G3);
   - positive locations carry the multiple-instance constraint "at least one EG pixel over
     the 6 years", whose exact marginals are p_i / (1 - prod_j (1 - p_j));
   - the prior marginal is tanh(b + F), with F the prior field of Proposition 2.
3. **stage 2, structured model**: the stage-1 parameters with orientation-aware smoothness
   beta and temporal coupling tau (chosen on dev locations). Optional extra sampling-based
   training of the symmetric parameters (``structured_epochs``) uses clamps and rejection for
   the MIL constraint.
4. **infer** on any locations (test labels are never used): MAP masks by graph cut (the
   posterior is ferromagnetic) and marginals by Gibbs sampling.
5. **evaluate** location-level detection: scores derived from the masks against the location
   labels (AUC, best F1), compared with baselines. Pixel metrics, if annotations exist, are in
   ``qws.gully.annotate``.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ..data.dem3dep import Terrain
from ..data.gully import GullyLocation, load_location, read_labels
from ..graph import Sites
from ..learn import TrainConfig, Trainer
from ..learn.optim import Adam
from ..lf import LFOutputs
from ..model import LabelModelParams, PosteriorMoments, gradient, posterior_ising
from ..solvers import GibbsSampler, GraphCutSolver
from .graph import spatiotemporal_sites
from .lfs import GullyLF, apply_gully_lfs, default_gully_lfs


@dataclass
class GullyConfig:
    root: str = "data/gully"
    use_terrain: bool = True
    orientation: bool = True     # orientation-aware couplings (G6); False = isotropic
    tau: float = 0.5             # temporal coupling weight
    beta: float = 0.5            # smoothness strength (fixed)
    class_conditional: bool = True  # stage 1 learns omega, zeta (class-dependent coverage / vote bias)
    mil: bool = True             # exact "at least one EG pixel" constraint on positive locations in stage 1
    epochs: int = 0              # structured_epochs: optional stage-2 sampling-based training (symmetric only)
    lr: float = 0.05
    num_samples: int = 12        # posterior chains per location
    sweeps: int = 4              # Gibbs sweeps per training step (persistent chains)
    warm_iters: int = 400
    warm_tol: float = 2e-4       # stop stage 1 when no parameter moves more than this per step
    marg_samples: int = 16       # Gibbs chains for marginals (initialized at the MAP state)
    marg_sweeps: int = 20
    batch_size: int | None = 64
    seed: int = 0


@dataclass
class GullyItem:
    loc: GullyLocation
    terrain: Terrain | None
    sites: Sites
    lfo: LFOutputs

    @property
    def tile(self) -> int:
        return self.loc.tile


def _lf_key(lfs: list[GullyLF], cfg: GullyConfig) -> str:
    desc = json.dumps([(lf.name, sorted(vars(lf).items())) for lf in lfs], default=str)
    return hashlib.sha1(desc.encode()).hexdigest()[:10]


def prepare(tiles: list[int], cfg: GullyConfig, lfs: list[GullyLF] | None = None) -> list[GullyItem]:
    root = Path(cfg.root)
    labels = read_labels(root)
    lfs = lfs or default_gully_lfs(cfg.use_terrain)
    key = _lf_key(lfs, cfg)
    cache = root / "cache" / f"lf_{key}"
    cache.mkdir(parents=True, exist_ok=True)
    items = []
    for tile in tiles:
        loc = load_location(root, tile, labels.get(tile))
        tpath = root / "terrain" / f"{tile}.npz"
        terrain = Terrain.load(tpath) if (cfg.use_terrain and tpath.exists()) else None
        cpath = cache / f"{tile}.npy"
        if cpath.exists():
            L = np.load(cpath)
        else:
            L = apply_gully_lfs(loc, terrain, lfs)
            np.save(cpath, L)
        sites = spatiotemporal_sites(loc.images, cfg.tau, cfg.orientation)
        lfo = LFOutputs(L, np.zeros((sites.m, 0), np.int8), tuple(lf.name for lf in lfs))
        items.append(GullyItem(loc, terrain, sites, lfo))
    return items


def negative_clamps(items: list[GullyItem]) -> dict[int, np.ndarray]:
    return {k: -np.ones(it.sites.n, dtype=np.int8) for k, it in enumerate(items) if it.loc.label == 0}


def _mil_correct(h: np.ndarray) -> np.ndarray:
    """Exact posterior means of independent sites conditioned on "at least one y_i = +1".

    p_i = sigma(2 h_i); P(none) = prod_j (1 - p_j); P(y_i = +1 | at least one) = p_i / (1 - P(none)).
    Computed in log space for stability.
    """
    p = 0.5 * (1.0 + np.tanh(h))
    log_none = np.sum(-np.logaddexp(0.0, 2.0 * h))  # log prod (1 - p_j), with 1 - p = sigma(-2h)
    q = np.clip(p / max(-np.expm1(log_none), 1e-300), 0.0, 1.0)
    return 2.0 * q - 1.0


def warm_start(items: list[GullyItem], cfg: GullyConfig) -> LabelModelParams:
    """Stage 1: exact independent label model with clamped negatives and the MIL constraint.

    Learns (alpha, theta, b) and, if ``cfg.class_conditional``, (omega, zeta). At beta = 0:
    - free sites have <y_i> = tanh(h_i), with h_i = b + sum_j theta_j L_ij + sum_j omega_j 1[L_ij != 0];
    - positive locations (MIL) use the exact conditional means of ``_mil_correct``;
    - negative locations are clamped (<y_i> = -1);
    - the prior moment is tanh(b + F), with F = sum_c 1/2 log(Z_c(+1) / Z_c(-1)) (Proposition 2).

    Vectorized over the concatenated vote matrix of all locations. It computes the same
    gradient as ``qws.model.gradient`` (singleton cliques; checked in tests/test_gully.py).
    """
    m = items[0].lfo.m
    data = Stage1Data.build(items)
    p = LabelModelParams.init(m, 0, 0.5, coverage=data.colC / data.N, b=-2.0)
    opt = Adam(cfg.lr)
    for _ in range(cfg.warm_iters):
        g = stage1_gradient(p, data, cfg)
        old = p.to_vector()
        p = p.from_vector(opt.step(old, g.to_vector() / data.N))
        if p.theta.mean() < 1e-3:
            p.theta = p.theta + (1e-3 - p.theta.mean())
        if np.max(np.abs(p.to_vector() - old)) < cfg.warm_tol:
            break
    return p


@dataclass
class Stage1Data:
    """Concatenated votes of all locations, for the vectorized stage-1 gradient."""

    L: np.ndarray
    C: np.ndarray
    sizes: np.ndarray
    starts: np.ndarray
    neg: np.ndarray
    pos_items: list[int]
    colL: np.ndarray  # column sums, constant across iterations (computed once)
    colC: np.ndarray

    @property
    def N(self) -> int:
        return int(self.L.shape[0])

    @classmethod
    def build(cls, items: list[GullyItem]) -> "Stage1Data":
        L = np.concatenate([it.lfo.L for it in items]).astype(np.float32)
        sizes = np.array([it.sites.n for it in items])
        lab = np.repeat([it.loc.label for it in items], sizes)
        C = (L != 0).astype(np.float32)
        return cls(L, C, sizes, np.concatenate([[0], np.cumsum(sizes)[:-1]]), lab == 0,
                   [k for k, it in enumerate(items) if it.loc.label == 1],
                   L.sum(0, dtype=np.float64), C.sum(0, dtype=np.float64))


def stage1_gradient(p: LabelModelParams, d: Stage1Data, cfg: GullyConfig) -> LabelModelParams:
    """Exact marginal-likelihood gradient of the independent model with clamps and MIL."""
    from ..model import model_moments

    mm = model_moments(p)
    h = np.float32(p.b) + d.L @ p.theta.astype(np.float32) + d.C @ p.omega.astype(np.float32)
    mean = np.tanh(h)
    mean[d.neg] = -1.0
    if cfg.mil:
        for k in d.pos_items:
            sl = slice(d.starts[k], d.starts[k] + d.sizes[k])
            mean[sl] = _mil_correct(h[sl].astype(np.float64))
    # keep the big products in float32: mixing float32 L with a float64 vector makes NumPy
    # materialize a float64 copy of the whole (N x m) matrix on every call
    mean = mean.astype(np.float32)
    p1 = d.N * 0.5 * (1.0 + np.tanh(p.b + mm.prior_field))  # sum_i P_prior(y_i = +1)
    p0 = d.N - p1
    g = p.zeros_like()
    g.theta = (d.L.T @ mean).astype(np.float64) - (p1 * mm.vote[0] - p0 * mm.vote[1])
    g.alpha = d.colC - (p1 * mm.cover[0] + p0 * mm.cover[1])
    if cfg.class_conditional:
        g.omega = (d.C.T @ mean).astype(np.float64) - (p1 * mm.cover[0] - p0 * mm.cover[1])
        g.zeta = d.colL - (p1 * mm.vote[0] + p0 * mm.vote[1])
    g.b = float(mean.sum(dtype=np.float64) - (p1 - p0))
    return g


def train(items: list[GullyItem], cfg: GullyConfig) -> tuple[LabelModelParams, LabelModelParams, dict]:
    """Stage 1 (exact) and optional stage-2 sampling-based training.

    Returns (stage-1 params, structured params with beta = cfg.beta, stats)."""
    t0 = time.perf_counter()
    warm = warm_start(items, cfg)
    t_warm = time.perf_counter() - t0
    params = warm.copy()
    params.beta = cfg.beta
    stats: dict[str, Any] = {"stage1_time_s": t_warm, "structured_epochs": cfg.epochs}
    if cfg.epochs > 0:
        if not params.is_symmetric:
            raise ValueError("stage-2 sampling training supports symmetric potentials only (set class_conditional=False)")
        data = [(it.lfo, it.sites) for it in items]
        tc = TrainConfig(epochs=cfg.epochs, lr=cfg.lr, num_samples=cfg.num_samples, batch_size=cfg.batch_size,
                         beta=cfg.beta, tol=0.0, seed=cfg.seed)
        trainer = Trainer(GibbsSampler(num_sweeps=cfg.sweeps, persistent=True, seed=cfg.seed), tc)
        res = trainer.fit(data, params, clamps=negative_clamps(items),
                          min_positive={k: 1 for k, it in enumerate(items) if it.loc.label == 1})
        params = res.params
        cs = trainer.constraint_stats
        stats.update(stage2_time_s=time.perf_counter() - t0 - t_warm, mil_checked=cs["checked"],
                     mil_all_rejected=cs["all_rejected"],
                     mil_mean_accepted=float(np.mean(cs["accepted_fraction"])) if cs["accepted_fraction"] else None)
    return warm, params, stats


@dataclass
class Prediction:
    tile: int
    map_mask: np.ndarray          # (6, H, W) bool, structured MAP
    marginal: np.ndarray | None   # (6, H, W) P(EG)
    indep_mask: np.ndarray        # (6, H, W) bool, independent-model MAP (warm-start params, beta = 0)
    majority: np.ndarray          # (6, H, W) bool, LF majority vote
    scores: dict[str, float] = field(default_factory=dict)
    indep_marginal: np.ndarray | None = None  # (6, H, W) P(EG) under the stage-1 (independent) model


def _stack(it: GullyItem, flat: np.ndarray) -> np.ndarray:
    T, H, W = it.loc.images.shape[:3]
    return np.asarray(flat).reshape(T, H, W)


def infer(items: list[GullyItem], structured: LabelModelParams, warm: LabelModelParams, cfg: GullyConfig,
          marginals: bool = True) -> list[Prediction]:
    gc = GraphCutSolver()
    gibbs = GibbsSampler(num_sweeps=cfg.marg_sweeps, seed=cfg.seed)
    out = []
    for it in items:
        p = posterior_ising(structured, it.lfo, it.sites)
        y = gc.minimize([p])[0].best()[0]
        marg = None
        if marginals:
            ss = gibbs.sample([p], cfg.marg_samples, init=np.tile(y, (cfg.marg_samples, 1)))
            marg = _stack(it, 0.5 * (1 + ss[0].mean()))
        h0 = warm.b + it.lfo.L.astype(float) @ warm.theta + (it.lfo.L != 0).astype(float) @ warm.omega
        vs = it.lfo.L.astype(int).sum(axis=1)
        pred = Prediction(it.tile, _stack(it, y > 0), marg, _stack(it, h0 > 0), _stack(it, vs > 0))
        pred.indep_marginal = _stack(it, 0.5 * (1 + np.tanh(h0)))
        pred.scores = location_scores(pred, it)
        out.append(pred)
    return out


def _largest_component(mask2d: np.ndarray) -> int:
    from scipy.ndimage import label

    lab, n = label(mask2d, structure=np.ones((3, 3)))
    return int(np.bincount(lab.ravel())[1:].max()) if n else 0


def location_scores(pred: Prediction, it: GullyItem) -> dict[str, float]:
    """Location-level detection scores (higher = more likely to contain an EG)."""
    s = {
        "structured_frac": float(pred.map_mask.mean()),
        "structured_largest_cc": float(max(_largest_component(m) for m in pred.map_mask)),
        "independent_frac": float(pred.indep_mask.mean()),
        "majority_frac": float(pred.majority.mean()),
    }
    if pred.marginal is not None:
        s["structured_marginal_mass"] = float(pred.marginal.mean())
        s["structured_top100_marginal"] = float(np.sort(pred.marginal.ravel())[-100:].mean())
        s["structured_largest_cc_p0.5"] = float(max(_largest_component(m > 0.5) for m in pred.marginal))
    if pred.indep_marginal is not None:
        s["independent_marginal_mass"] = float(pred.indep_marginal.mean())
        s["independent_top100_marginal"] = float(np.sort(pred.indep_marginal.ravel())[-100:].mean())
    L = it.lfo.L
    for j, name in enumerate(it.lfo.unary_names):
        s[f"lf_{name}_pos_frac"] = float(np.mean(L[:, j] == 1))
    if it.terrain is not None:
        s["terrain_max_log_acc"] = float(np.nanmax(it.terrain.log_acc))
    return s


def roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Mann–Whitney AUC with tie handling."""
    from scipy.stats import rankdata

    pos, neg = labels == 1, labels == 0
    if not pos.any() or not neg.any():
        return float("nan")
    r = rankdata(scores)
    return float((r[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * neg.sum()))


def best_f1(scores: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    """Best F1 over thresholds and its threshold (an optimistic diagnostic; report with AUC)."""
    best, thr = 0.0, float("nan")
    for t in np.unique(scores):
        pred = scores >= t
        tp = np.sum(pred & (labels == 1))
        fp = np.sum(pred & (labels == 0))
        fn = np.sum(~pred & (labels == 1))
        f1 = 2 * tp / max(2 * tp + fp + fn, 1)
        if f1 > best:
            best, thr = f1, float(t)
    return best, thr


def evaluate_locations(preds: list[Prediction], items: list[GullyItem]) -> dict[str, dict[str, float]]:
    labels = np.array([it.loc.label for it in items])
    names = sorted({k for p in preds for k in p.scores})
    out = {}
    for name in names:
        sc = np.array([p.scores.get(name, np.nan) for p in preds])
        ok = np.isfinite(sc)
        f1, thr = best_f1(sc[ok], labels[ok])
        out[name] = {"auc": roc_auc(sc[ok], labels[ok]), "best_f1": f1, "threshold": thr, "n": int(ok.sum())}
    return out


def config_dict(cfg: GullyConfig) -> dict[str, Any]:
    return asdict(cfg)
