import numpy as np
import pytest

from qws.data import list_names, load_pet, trimap_to_mask
from qws.data.petlike import make_petlike, write_petlike_dataset
from qws.eval import iou
from qws.graph import superpixel_sites
from qws.lf import (
    BorderBackgroundLF, CenterForegroundLF, ColorContrastLF, ColorModelLF, apply_lfs, default_natural_lfs,
    spectral_residual_saliency,
)


@pytest.fixture(scope="module")
def pet_root(tmp_path_factory):
    return write_petlike_dataset(tmp_path_factory.mktemp("pet"), n_images=6, shape=(48, 64), seed=1)


def test_trimap_binarization():
    assert trimap_to_mask(np.array([[1, 2, 3]])).tolist() == [[1, -1, 0]]


def test_layout_and_splits(pet_root):
    assert (pet_root / "images").is_dir() and (pet_root / "annotations" / "trimaps").is_dir()
    tv, te = list_names(pet_root, "trainval"), list_names(pet_root, "test")
    assert len(tv) == 3 and len(te) == 3 and not set(tv) & set(te)
    assert len(list_names(pet_root, None)) == 6


def test_load_pet_resizes_and_keeps_labels(pet_root):
    samples = load_pet(pet_root, "trainval", max_side=32)
    assert len(samples) == 3
    for s in samples:
        assert max(s.image.shape[:2]) == 32 and s.image.shape[2] == 3
        assert 0.0 <= s.image.min() and s.image.max() <= 1.0
        assert set(np.unique(s.mask)) <= {-1, 0, 1} and (s.mask == 0).any()  # nearest-neighbour keeps the band
    assert len(load_pet(pet_root, "trainval", limit=2)) == 2


def test_ignore_zero_metric():
    gt = np.array([1, 0, -1, 1])
    pred = np.array([1, 1, -1, -1])
    assert iou(pred, gt, ignore_zero=True) == pytest.approx(0.5)
    assert iou(pred, gt) != pytest.approx(0.5)


def test_generic_lfs_are_informative():
    rng = np.random.default_rng(0)
    img, tri = make_petlike((64, 64), rng)
    image, mask = img / 255.0, trimap_to_mask(tri)
    sites = superpixel_sites(image, n_segments=80)
    y = np.where(sites.from_pixels(mask) > 0, 1, -1)
    out = apply_lfs(image, sites, [BorderBackgroundLF(), CenterForegroundLF(), ColorModelLF()], [ColorContrastLF()])
    for j in range(3):
        voted = out.L[:, j] != 0
        assert voted.any() and np.mean(out.L[voted, j] == y[voted]) > 0.7
    yy = y[sites.edges[:, 0]] * y[sites.edges[:, 1]]
    voted = out.P[:, 0] != 0
    assert np.mean(out.P[voted, 0] == yy[voted]) > 0.7
    sal = spectral_residual_saliency(image)
    assert sal.shape == image.shape[:2] and sal.min() >= 0 and sal.max() <= 1
    u, p = default_natural_lfs()
    assert len(u) == 4 and len(p) == 1


def test_petlike_pipeline_end_to_end(tmp_path):
    from qws.experiment import run_experiment

    rep = run_experiment({"dataset": {"kind": "petlike", "root": str(tmp_path / "pl"), "n_images": 4,
                                      "n_segments": 80}, "train": {"epochs": 3, "num_samples": 30},
                          "solver": {"name": "gibbs", "num_sweeps": 20}, "map_solver": {"name": "sa", "num_sweeps": 100}})
    assert rep["true_params"] is None and rep["theta_error"] is None
    assert rep["iou"]["structured_map"] > rep["iou"]["majority_vote"]
    assert len(rep["lf_accuracy"]["names"]) == 4
