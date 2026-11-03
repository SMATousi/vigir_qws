"""Synthetic data generator (doc 06 §1).

Votes are sampled from the label model's own conditional law given the true mask
(paper Lemma 5), so the generating parameters are known and recoverable.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter

from ..graph import Sites, grid_sites, superpixel_sites
from ..lf import LFOutputs
from ..model import LabelModelParams, all_cliques
from ..model.label_model import clique_logweights


@dataclass
class SyntheticImage:
    image: np.ndarray       # (H, W) float
    mask: np.ndarray        # (H, W) int8 in {-1, +1}
    sites: Sites
    y: np.ndarray           # (n,) true site labels
    lfo: LFOutputs


def make_mask(shape: tuple[int, int], rng: np.random.Generator, smoothness: float = 4.0,
              fg_fraction: float = 0.4) -> np.ndarray:
    """Spatially coherent random blobs: smoothed noise thresholded at a quantile."""
    field = gaussian_filter(rng.standard_normal(shape), smoothness)
    return np.where(field > np.quantile(field, 1 - fg_fraction), 1, -1).astype(np.int8)


def make_image(mask: np.ndarray, rng: np.random.Generator, contrast: float = 1.0, noise: float = 0.3) -> np.ndarray:
    return contrast * (mask > 0) + noise * rng.standard_normal(mask.shape)


def sample_votes(params: LabelModelParams, y: np.ndarray, edges: np.ndarray, rng: np.random.Generator) -> LFOutputs:
    """Draw Λ and Π from p(Λ, Π | y) (paper Lemma 5), including class-conditional potentials."""
    n = y.shape[0]
    L = np.zeros((n, params.m), dtype=np.int8)
    for c, g in all_cliques(params):
        idx = list(c)
        table, lw_pos, lw_neg, _ = clique_logweights(params, c, g)
        for label, lw in ((1, lw_pos), (-1, lw_neg)):
            sel = np.flatnonzero(y == label)
            w = np.exp(lw - lw.max())
            w /= w.sum()
            L[np.ix_(sel, idx)] = table[rng.choice(len(table), size=sel.size, p=w)].astype(np.int8)
    P = np.zeros((edges.shape[0], params.q), dtype=np.int8)
    s = y[edges[:, 0]] * y[edges[:, 1]]
    for r in range(params.q):
        logw = np.array([0.0, params.alphaP[r] + params.thetaP[r], params.alphaP[r] - params.thetaP[r]])
        w = np.exp(logw - logw.max())
        w /= w.sum()  # outcomes (0, agree with s, disagree with s)
        k = rng.choice(3, size=edges.shape[0], p=w)
        P[:, r] = np.where(k == 0, 0, np.where(k == 1, s, -s)).astype(np.int8)
    return LFOutputs(L, P, tuple(f"lf{j}" for j in range(params.m)), tuple(f"plf{r}" for r in range(params.q)))


def ising_mask(shape: tuple[int, int], b: float, beta: float, rng: np.random.Generator, sweeps: int = 500
               ) -> np.ndarray:
    """A mask drawn (approximately, by Gibbs sampling) from the prior Ising model
    p(y) ∝ exp(b sum y + beta sum_{4-nbr} y_i y_k) with uniform weights. Use it to
    test identifiability of (b, beta): the generating prior is then exactly the model's."""
    from ..ising import IsingProblem
    from ..solvers import GibbsSampler

    st = grid_sites(np.zeros(shape), sigma=np.inf)
    p = IsingProblem(np.full(st.n, b), st.edges, np.full(st.m, beta))
    y = GibbsSampler(num_sweeps=sweeps, seed=int(rng.integers(2**31))).sample([p], 1)[0].samples[0]
    return y.reshape(shape).astype(np.int8)


def synthetic_dataset(true_params: LabelModelParams, n_images: int = 10, shape: tuple[int, int] = (16, 16),
                      sites: str = "grid", n_segments: int = 100, smoothness: float = 3.0,
                      fg_fraction: float = 0.4, seed: int | None = 0, mask_source: str = "blobs"
                      ) -> list[SyntheticImage]:
    """Synthetic images with votes from the label model's conditional law.

    ``mask_source="blobs"``: smoothed-noise masks, contrast-sensitive weights.
    ``mask_source="ising"``: masks from the prior Ising model with the true (b, beta) and
    uniform weights (grid sites only), so the whole generative model is the label model.
    """
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_images):
        if mask_source == "ising":
            mask = ising_mask(shape, true_params.b, true_params.beta, rng)
        else:
            mask = make_mask(shape, rng, smoothness, fg_fraction)
        image = make_image(mask, rng)
        if mask_source == "ising":
            st = grid_sites(image, sigma=np.inf)
        else:
            st = grid_sites(image) if sites == "grid" else superpixel_sites(image, n_segments=n_segments)
        y = np.where(st.from_pixels(mask) >= 0, 1, -1).astype(np.int8)
        out.append(SyntheticImage(image, mask, st, y, sample_votes(true_params, y, st.edges, rng)))
    return out
