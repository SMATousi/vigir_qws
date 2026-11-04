"""Decomposition for problems larger than the hardware (doc 04 §5; paper Proposition 5(b)–(c))."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..graph import Sites
from ..ising import IsingProblem, condition
from ..solvers import SampleSet, Solver


def grid_tiles(sites: Sites, tile: int) -> list[np.ndarray]:
    """Partition sites into square tiles of ``tile`` x ``tile`` pixels by site centroid."""
    rc = np.floor(sites.centroids / tile).astype(np.int64)
    keys = rc[:, 0] * (rc[:, 1].max() + 1) + rc[:, 1]
    return [np.flatnonzero(keys == k) for k in np.unique(keys)]


def bfs_tiles(p: IsingProblem, max_size: int) -> list[np.ndarray]:
    """Greedy connected partition of the coupling graph into blocks of at most ``max_size``."""
    W = p.coupling_matrix
    unassigned = np.ones(p.n, dtype=bool)
    tiles = []
    for seed in range(p.n):
        if not unassigned[seed]:
            continue
        block, frontier = [], [seed]
        unassigned[seed] = False
        while frontier and len(block) < max_size:
            v = frontier.pop(0)
            block.append(v)
            for u in W.indices[W.indptr[v]:W.indptr[v + 1]]:
                if unassigned[u] and len(block) + len(frontier) < max_size:
                    unassigned[u] = False
                    frontier.append(u)
        for v in frontier:  # return unused frontier vertices to the pool
            unassigned[v] = True
        tiles.append(np.array(sorted(block), dtype=np.int64))
    return tiles


def _complement(n: int, tile: np.ndarray) -> np.ndarray:
    mask = np.ones(n, dtype=bool)
    mask[tile] = False
    return np.flatnonzero(mask)


def block_coordinate_descent(p: IsingProblem, tiles: list[np.ndarray], solver: Solver,
                             init: np.ndarray | None = None, max_sweeps: int = 20,
                             seed: int | None = None, **solver_kw: Any) -> tuple[np.ndarray, list[float]]:
    """MAP by sweeping tiles; each tile is solved conditioned on the rest.

    A tile update is accepted only if it does not increase the global energy, so the
    energy trace is monotone even with heuristic solvers (Proposition 5(c)).
    """
    rng = np.random.default_rng(seed)
    y = rng.choice(np.array([-1, 1], dtype=np.int8), size=p.n) if init is None else np.asarray(init, np.int8).copy()
    trace = [float(p.energy(y))]
    for _ in range(max_sweeps):
        improved = False
        for tile in tiles:
            rest = _complement(p.n, tile)
            cond = condition(p, rest, y[rest])
            best, _ = solver.minimize([cond.problem], **solver_kw)[0].best()
            cand = y.copy()
            cand[cond.free] = best
            e_new, e_old = float(p.energy(cand)), trace[-1]
            if e_new < e_old - 1e-12:
                y, improved = cand, True
                trace.append(e_new)
        if not improved:
            break
    return y, trace


def block_gibbs(p: IsingProblem, tiles: list[np.ndarray], sampler: Solver, num_samples: int,
                sweeps: int = 50, seed: int | None = None, **sampler_kw: Any) -> SampleSet:
    """Block Gibbs: resample each tile from its exact conditional (Proposition 5(c)).

    Runs ``num_samples`` independent chains; each tile update draws one sample per chain
    from the conditioned problem with the given sampler.
    """
    rng = np.random.default_rng(seed)
    Y = rng.choice(np.array([-1, 1], dtype=np.int8), size=(num_samples, p.n))
    for _ in range(sweeps):
        for tile in tiles:
            rest = _complement(p.n, tile)
            for c in range(num_samples):
                cond = condition(p, rest, Y[c, rest])
                ss = sampler.sample([cond.problem], 1, **sampler_kw)[0]
                Y[c, cond.free] = ss.expanded()[0]
    return SampleSet.from_samples(p, Y, info={"solver": f"block_gibbs[{getattr(sampler, 'name', '?')}]",
                                             "tiles": len(tiles), "sweeps": sweeps})
