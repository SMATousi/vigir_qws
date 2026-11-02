import numpy as np
import pytest

from qws.graph import grid_sites, sites_from_labels, superpixel_sites
from qws.lf import BoxLF, EdgeContrastLF, ScribbleLF, ThresholdLF, apply_lfs, lift_votes


def test_grid_sites_counts():
    img = np.random.default_rng(0).random((5, 7))
    s4 = grid_sites(img)
    assert s4.n == 35 and s4.m == 5 * 6 + 4 * 7
    s8 = grid_sites(img, connectivity=8)
    assert s8.m == s4.m + 2 * 4 * 6
    assert np.all((s4.weights > 0) & (s4.weights <= 1))
    assert np.all(s4.edges[:, 0] < s4.edges[:, 1])
    flat = grid_sites(img, sigma=np.inf)
    assert np.all(flat.weights == 1)


def test_pixel_site_round_trip():
    img = np.random.default_rng(1).random((12, 12, 3))
    s = superpixel_sites(img, n_segments=20)
    vals = np.arange(s.n, dtype=float)
    assert np.allclose(s.from_pixels(s.to_pixels(vals)), vals)
    assert s.edges.size and np.all(s.edges[:, 0] < s.edges[:, 1])


def test_region_adjacency():
    labels = np.array([[0, 0, 1], [2, 2, 1], [2, 2, 1]])
    s = sites_from_labels(np.zeros((3, 3)), labels)
    assert sorted(map(tuple, s.edges.tolist())) == [(0, 1), (0, 2), (1, 2)]


def test_lift_votes_majority_and_coverage():
    labels = np.array([[0, 0, 1, 1]])
    s = sites_from_labels(np.zeros((1, 4)), labels)
    assert lift_votes(np.array([[1, 1, -1, 0]]), s).tolist() == [1, -1]  # site 1: 50% coverage, majority -1
    assert lift_votes(np.array([[1, -1, -1, -1]]), s).tolist() == [0, -1]  # site 0 tie -> abstain
    assert lift_votes(np.array([[1, 0, 0, 0]]), s, min_coverage=0.6).tolist() == [0, 0]


def test_apply_lfs_shapes_and_values():
    img = np.zeros((6, 6))
    img[2:5, 2:5] = 1.0
    s = grid_sites(img)
    out = apply_lfs(img, s, [ThresholdLF(0.3, 0.7), BoxLF((1, 1, 5, 5), margin=0), ScribbleLF(np.zeros((6, 6)))],
                    [EdgeContrastLF(0.1, 0.5)])
    assert out.L.shape == (36, 3) and out.P.shape == (s.m, 1)
    assert set(np.unique(out.L)) <= {-1, 0, 1}
    assert out.L[s.labels[3, 3], 0] == 1 and out.L[s.labels[0, 0], 0] == -1
    assert out.L[s.labels[0, 0], 1] == -1 and out.L[s.labels[3, 3], 1] == 0
    assert (out.P == -1).any() and (out.P == 1).any()


def test_bad_lf_output_rejected():
    class Bad(ThresholdLF):
        def votes(self, image, sites):
            return np.full(sites.n, 2)

    with pytest.raises(ValueError):
        apply_lfs(np.zeros((2, 2)), grid_sites(np.zeros((2, 2))), [Bad(0, 1)])
