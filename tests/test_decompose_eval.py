import numpy as np
import pytest

from qws.cli import main
from qws.decompose import bfs_tiles, block_coordinate_descent, block_gibbs, grid_tiles
from qws.eval import (
    config_index, dice, ece, hit_rate, iou, map_benchmark, sampling_benchmark, synthetic_dataset, tts99, tv_distance,
)
from qws.graph import grid_sites
from qws.ising import IsingProblem
from qws.model import LabelModelParams
from qws.solvers import ExactSolver, GibbsSampler, SimulatedAnnealing, all_spins

from .conftest import random_problem


def _grid_problem(rng, side=4, frustrated=True):
    s = grid_sites(rng.random((side, side)))
    J = rng.normal(0, 1, s.m) if frustrated else rng.uniform(0, 1, s.m)
    return IsingProblem(rng.normal(0, 0.5, s.n), s.edges, J), s


def test_tiles_partition(rng):
    p, s = _grid_problem(rng, 6)
    for tiles in (grid_tiles(s, 3), bfs_tiles(p, 5)):
        allv = np.sort(np.concatenate(tiles))
        assert allv.tolist() == list(range(p.n))
    assert max(len(t) for t in bfs_tiles(p, 5)) <= 5


def test_block_coordinate_descent_monotone_and_optimal(rng):
    ex = ExactSolver()
    for _ in range(5):
        p, s = _grid_problem(rng, 4)
        y, trace = block_coordinate_descent(p, grid_tiles(s, 2), ex, max_sweeps=30, seed=0)
        assert all(b <= a + 1e-12 for a, b in zip(trace, trace[1:]))
        assert p.energy(y) == pytest.approx(trace[-1])
        # block-local optimum: no single tile can be improved given the rest
        from qws.ising import condition

        for tile in grid_tiles(s, 2):
            rest = np.setdiff1d(np.arange(p.n), tile)
            sub = condition(p, rest, y[rest])
            assert ex.minimize([sub.problem])[0].best()[1] >= trace[-1] - 1e-9
        # one tile covering everything is exact
        y_all, _ = block_coordinate_descent(p, [np.arange(p.n)], ex, seed=0)
        assert p.energy(y_all) == pytest.approx(ex.minimize([p])[0].best()[1])


@pytest.mark.slow
def test_block_gibbs_matches_exact(rng):
    p, s = _grid_problem(rng, 3)
    p = p.scaled(0.5)
    mu, _ = ExactSolver().moments(p)
    ss = block_gibbs(p, grid_tiles(s, 2), ExactSolver(seed=1), num_samples=300, sweeps=4, seed=2)
    np.testing.assert_allclose(ss.mean(), mu, atol=0.12)


def test_segmentation_metrics():
    gt = np.array([1, 1, -1, -1])
    assert iou(gt, gt) == 1.0 and dice(gt, gt) == 1.0
    assert iou(np.array([1, -1, -1, -1]), gt) == pytest.approx(0.5)
    assert dice(np.array([1, -1, -1, -1]), gt) == pytest.approx(2 / 3)
    assert ece(np.array([1.0, 1.0, 0.0, 0.0]), gt) == pytest.approx(0.0)


def test_optimization_metrics():
    assert hit_rate(np.array([-3.0, -2.0]), np.array([3, 1]), -3.0) == pytest.approx(0.75)
    assert tts99(1.0, 0.0) == float("inf") and tts99(2.0, 0.995) == 2.0
    assert tts99(1.0, 0.5) == pytest.approx(np.log(0.01) / np.log(0.5))


def test_tv_distance_and_indexing(rng):
    Y = all_spins(4)
    assert np.array_equal(config_index(Y), np.arange(16))
    p = random_problem(4, rng)
    ex = ExactSolver(seed=0)
    Yall, probs = ex.distribution(p)
    assert tv_distance(Yall, probs * 1e6, probs) == pytest.approx(0.0, abs=1e-9)
    ss = ex.sample([p], 50000)[0]
    assert tv_distance(ss.samples, ss.counts, probs) < 0.03


def test_benchmarks_run(rng):
    probs = [random_problem(8, rng) for _ in range(2)]
    recs = map_benchmark(probs, {"sa": SimulatedAnnealing(num_sweeps=100, seed=0), "exact": ExactSolver()})
    assert {r["solver"] for r in recs} == {"sa", "exact"}
    assert all(r["gap"] == pytest.approx(0.0) for r in recs if r["solver"] == "exact")
    srecs = sampling_benchmark(probs[:1], {"gibbs": GibbsSampler(num_sweeps=30, seed=0)}, num_samples=500)
    assert 0 <= srecs[0]["tv"] <= 1


def test_synthetic_vote_statistics():
    true = LabelModelParams(alpha=[0.5, -0.5], theta=[1.0, 0.4], alphaP=[0.2], thetaP=[0.8])
    ds = synthetic_dataset(true, n_images=20, shape=(16, 16), seed=0)
    L = np.concatenate([im.lfo.L for im in ds])
    y = np.concatenate([im.y for im in ds])
    for j in range(2):
        voted = L[:, j] != 0
        acc = np.mean(L[voted, j] == y[voted])
        assert acc == pytest.approx(1 / (1 + np.exp(-2 * true.theta[j])), abs=0.02)


def test_structured_beats_baselines_on_coherent_data():
    """M1 acceptance criterion (doc 06 §5)."""
    from qws.experiment import run_experiment

    rep = run_experiment({"dataset": {"n_images": 4, "shape": [12, 12]}, "train": {"epochs": 5, "num_samples": 40},
                          "solver": {"name": "gibbs", "num_sweeps": 40}})
    assert rep["iou"]["structured_map"] > rep["iou"]["majority_vote"]
    assert rep["iou"]["structured_map"] > rep["iou"]["independent_model"]


def test_cli_demo_and_run(tmp_path, capsys):
    assert main(["demo", "--epochs", "2", "--images", "2", "--size", "8", "--out", str(tmp_path / "r.json")]) == 0
    assert "structured_map" in capsys.readouterr().out and (tmp_path / "r.json").exists()
    cfg = tmp_path / "c.yaml"
    cfg.write_text("dataset: {n_images: 2, shape: [8, 8]}\ntrain: {epochs: 1}\nsolver: {name: sa, num_sweeps: 30}\n")
    assert main(["run", str(cfg)]) == 0


def test_config_merge_replaces_solver_spec_on_name_change():
    from qws.experiment import DEFAULT_CONFIG, merge

    cfg = merge(DEFAULT_CONFIG, {"solver": {"name": "anneal_mock"}, "train": {"epochs": 3}})
    assert cfg["solver"] == {"name": "anneal_mock"}
    assert cfg["train"]["epochs"] == 3 and cfg["train"]["lr"] == DEFAULT_CONFIG["train"]["lr"]
    same = merge(DEFAULT_CONFIG, {"solver": {"num_sweeps": 7}})
    assert same["solver"] == {"name": "gibbs", "num_sweeps": 7}
