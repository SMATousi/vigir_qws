"""Gully branch tests on a synthetic mini-dataset in the exact dataset layout (GeoTIFF + JSON)."""

import itertools
import json

import numpy as np
import pytest

tifffile = pytest.importorskip("tifffile")

from qws.data.gully import load_gully, make_split, read_labels, spatial_clusters, spatial_split, tile_bounds  # noqa: E402
from qws.data.hydro import fill_depressions, flow_accumulation  # noqa: E402

SCALE = 9.765625e-06


def _write_tile(path, img, west, north):
    tifffile.imwrite(path, img, photometric="rgb", extratags=[
        (33550, "d", 3, (SCALE, SCALE, 0.0), False), (33922, "d", 6, (0.0, 0.0, 0.0, west, north, 0.0), False)])


def _synthetic_location(rng, gully: bool, size=48):
    """6 years: textured field; a sinuous dark gully in some years if ``gully``; a straight
    field edge in every year (a permanent feature)."""
    imgs = []
    rr, cc = np.indices((size, size))
    path = np.abs(cc - (size / 2 + 6 * np.sin(rr / 7.0))) < 1.2
    for t in range(6):
        base = 0.35 + 0.05 * rng.standard_normal((size, size))
        img = np.stack([base * 0.9, base * 1.1, base * 0.7], -1)
        img[4:6, :] = 0.75  # straight bright field edge, all years
        if gully and t in (1, 2, 4):
            img[path] *= 0.55
        imgs.append(np.clip(img, 0, 1))
    return (np.stack(imgs) * 255).astype(np.uint8), path


@pytest.fixture(scope="module")
def gully_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("gully")
    (root / "GD_Images").mkdir()
    rng = np.random.default_rng(0)
    labels, paths = {}, {}
    for k in range(12):
        tile = 100 + k
        gully = k % 2 == 0
        imgs, path = _synthetic_location(rng, gully)
        west, north = -93.85 + 0.03 * (k // 2), 42.26  # pairs of tiles share a ~2 km cell
        for t in range(6):
            _write_tile(root / "GD_Images" / f"rgb_{t}_tile_{tile}.tif", imgs[t], west, north)
        labels[str(tile)] = {"label": "1" if gully else "0"}
        paths[tile] = path
    (root / "GT_L6_Tol.json").write_text(json.dumps(labels))
    return root, paths


def test_loader_and_bounds(gully_root):
    root, _ = gully_root
    locs = load_gully(root)
    assert len(locs) == 12 and locs[0].images.shape == (6, 48, 48, 3)
    assert 0 <= locs[0].images.min() and locs[0].images.max() <= 1
    w, s, e, n = tile_bounds(root / "GD_Images" / "rgb_0_tile_100.tif")
    assert n == pytest.approx(42.26) and e - w == pytest.approx(48 * SCALE)
    assert sorted(read_labels(root).values()).count(1) == 6


def test_spatial_split_keeps_clusters_together(gully_root):
    root, _ = gully_root
    labels = read_labels(root)
    bounds = {t: tile_bounds(root / "GD_Images" / f"rgb_0_tile_{t}.tif") for t in labels}
    clusters = spatial_clusters(bounds, cell_deg=0.02)
    dev, test = spatial_split(labels, clusters, seed=0, trials=500)
    assert sorted(dev + test) == sorted(labels) and dev and test
    assert not {clusters[t] for t in dev} & {clusters[t] for t in test}
    s = make_split(root, seed=0)
    assert set(s["dev"]) | set(s["test"]) == set(labels)


def test_hydrology_known_answers():
    plane = -np.arange(30, dtype=float)[:, None] + 0 * np.arange(10)[None, :]
    assert flow_accumulation(plane)[:, 5].tolist() == list(range(1, 31))
    r, c = np.indices((40, 21))
    valley = np.abs(c - 10) - 0.1 * r
    assert flow_accumulation(valley)[-1, 10] == valley.size
    pit = valley.copy()
    pit[20, 10] -= 5
    filled = fill_depressions(pit)
    assert filled[20, 10] > pit[20, 10] and flow_accumulation(pit)[-1, 10] == valley.size


def test_mil_correction_matches_enumeration():
    from qws.gully.pipeline import _mil_correct

    h = np.array([-1.2, 0.3, -2.0, -0.4, 0.8])
    Y = np.array(list(itertools.product([-1, 1], repeat=5)))
    w = np.exp(Y @ h)
    keep = (Y == 1).any(axis=1)
    exact = (w[keep] @ Y[keep]) / w[keep].sum()
    np.testing.assert_allclose(_mil_correct(h), exact, atol=1e-12)


def test_orientation_aware_weights_follow_lines():
    from qws.gully.graph import orientation_field, spatiotemporal_sites

    img = np.full((32, 32, 3), 0.5)
    img[:, 15:17] = 0.1  # vertical dark line
    line_dir, coh = orientation_field(img)
    assert coh[16, 14] > 0.5 and abs(np.cos(line_dir[16, 14])) < 0.3  # direction ~ vertical
    s = spatiotemporal_sites(np.stack([img] * 2), tau=0.7)
    assert s.n == 2 * 32 * 32 and np.all(s.weights >= 0)
    w = dict(zip(map(tuple, s.edges.tolist()), s.weights))
    # next to the line: the edge along it (vertical) is stronger than the edge across it (horizontal)
    along = w[(10 * 32 + 14, 11 * 32 + 14)]
    across = w[(10 * 32 + 13, 10 * 32 + 14)]
    assert along > across
    assert w[(5 * 32 + 5, 32 * 32 + 5 * 32 + 5)] == pytest.approx(0.7)  # temporal edge
    iso = spatiotemporal_sites(np.stack([img] * 2), orientation=False)
    assert iso.edges is s.edges  # shared structure


def test_lfs_on_synthetic_gully(gully_root):
    from qws.gully.lfs import PermanentLineLF, StraightLineLF, ThinLineLF

    root, paths = gully_root
    loc = load_gully(root, [100])[0]
    thin = ThinLineLF(q_pos=0.9).votes(loc, None)
    path = paths[100]
    assert thin.shape == (6, 48, 48)
    assert (thin[1][path] == 1).mean() > (thin[0][path] == 1).mean()  # gully visible in year 1, not year 0
    perm = PermanentLineLF(q=0.9).votes(loc, None)
    assert (perm[:, 4:6, :] == -1).mean() > 0.5        # the all-year field edge is permanent
    assert (perm[1][path] == -1).mean() < 0.2           # the ephemeral gully is not
    straight = StraightLineLF(min_length=30).votes(loc, None)
    assert (straight[0, 3:8, :] == -1).mean() > 0.3


def test_stage1_matches_reference_gradient(gully_root):
    from qws.gully.pipeline import GullyConfig, Stage1Data, _mil_correct, prepare, stage1_gradient
    from qws.model import LabelModelParams, PosteriorMoments, gradient, model_moments

    root, _ = gully_root
    cfg = GullyConfig(root=str(root), use_terrain=False)
    items = prepare([100, 101, 102], cfg)
    m = items[0].lfo.m
    p = LabelModelParams.init(m, 0, 0.5, coverage=np.full(m, 0.3), b=-2.0)
    p.omega, p.zeta = np.linspace(-0.3, 0.3, m), np.linspace(0.2, -0.2, m)
    F = model_moments(p).prior_field
    post, prior = [], []
    for it in items:
        h = p.b + it.lfo.L.astype(float) @ p.theta + (it.lfo.L != 0).astype(float) @ p.omega
        mean = -np.ones(it.sites.n) if it.loc.label == 0 else _mil_correct(h)
        post.append(PosteriorMoments(mean, np.zeros(it.sites.m)))
        prior.append(PosteriorMoments(np.full(it.sites.n, np.tanh(p.b + F)), np.zeros(it.sites.m)))
    ref = gradient(p, [(it.lfo, it.sites) for it in items], post, prior, learn_b=True)
    vec = stage1_gradient(p, Stage1Data.build(items), cfg)
    for k in ("theta", "alpha", "omega", "zeta"):
        np.testing.assert_allclose(getattr(vec, k), getattr(ref, k), rtol=1e-4, atol=1e-2)  # float32 sums
    assert vec.b == pytest.approx(ref.b, rel=1e-4)


def test_pipeline_end_to_end(gully_root):
    from qws.gully.pipeline import GullyConfig, evaluate_locations, infer, prepare, train

    root, _ = gully_root
    cfg = GullyConfig(root=str(root), use_terrain=False, warm_iters=60, marg_samples=4, marg_sweeps=3)
    items = prepare(sorted(read_labels(root)), cfg)
    warm, params, stats = train(items[:8], cfg)
    assert params.beta == cfg.beta and stats["structured_epochs"] == 0
    preds = infer(items[8:], params, warm, cfg)
    assert preds[0].map_mask.shape == (6, 48, 48) and preds[0].marginal.shape == (6, 48, 48)
    ev = evaluate_locations(preds, items[8:])
    assert "structured_largest_cc" in ev and 0 <= ev["structured_largest_cc"]["auc"] <= 1


def test_trainer_clamps_and_min_positive():
    from qws.graph import grid_sites
    from qws.learn import TrainConfig, Trainer
    from qws.lf import LFOutputs
    from qws.model import LabelModelParams
    from qws.solvers import ExactSolver

    sites = grid_sites(np.zeros((2, 3)))
    L = np.array([[1], [-1], [0], [0], [-1], [1]], dtype=np.int8)
    lfo = LFOutputs(L, np.zeros((sites.m, 0), np.int8))
    params = LabelModelParams(alpha=[0.0], theta=[0.8], b=-1.0, beta=0.3)
    tr = Trainer(ExactSolver(seed=0), TrainConfig(num_samples=200_000))
    tr.clamps = {0: -np.ones(6, np.int8)}
    tr.min_positive = {1: 2}
    tr.constraint_stats = {"checked": 0, "all_rejected": 0, "accepted_fraction": []}
    m0, m1 = tr._post_moments(params, [(lfo, sites), (lfo, sites)], [0, 1])
    assert np.all(m0.mean == -1) and np.all(m0.edge_corr == 1)
    # exact check of the constrained mean by enumeration
    from qws.model import posterior_ising
    from qws.solvers import all_spins

    p = posterior_ising(params, lfo, sites)
    Y = all_spins(6)
    w = np.exp(-p.energy(Y))
    keep = (Y == 1).sum(1) >= 2
    exact = (w[keep] @ Y[keep]) / w[keep].sum()
    np.testing.assert_allclose(m1.mean, exact, atol=0.02)
    assert tr.constraint_stats["checked"] == 1


def test_annotations_and_pixel_metrics(tmp_path):
    from skimage.io import imsave

    from qws.gully.annotate import IGNORE, load_annotations, load_geojson_annotations, pixel_metrics

    shape = (6, 32, 32)
    m = np.zeros((32, 32), np.uint8)
    m[:, 10:12] = 1
    m[0:5, :] = 255
    imsave(tmp_path / "7_y2.png", m, check_contrast=False)
    ann = load_annotations(tmp_path, 7, (0, 0, 1, 1), shape)
    assert ann[0].min() == IGNORE and ann[2, 10, 10] == 1 and ann[2, 0, 0] == IGNORE
    pred = np.zeros(shape, bool)
    pred[2, :, 11:13] = True
    met = pixel_metrics(pred, ann, tol=1.0)
    assert 0 < met["iou"] < 1 and met["tol_precision"] == pytest.approx(1.0) and met["tol_recall"] == pytest.approx(1.0)
    pytest.importorskip("shapely")
    bounds = (-93.85, 42.26, -93.85 + 32 * SCALE, 42.26 + 32 * SCALE)
    lon = bounds[0] + 16 * SCALE
    fc = {"type": "FeatureCollection", "features": [{"type": "Feature", "properties": {"year": 3},
          "geometry": {"type": "LineString", "coordinates": [[lon, bounds[1] + 1e-6], [lon, bounds[3] - 1e-6]]}}]}
    (tmp_path / "8.geojson").write_text(json.dumps(fc))
    g = load_geojson_annotations(tmp_path, 8, bounds, shape, width_m=3.0)
    assert g[3].sum() >= 32 and g[3][:, 14:19].sum() == g[3].sum() and np.all(g[0] == 0)
