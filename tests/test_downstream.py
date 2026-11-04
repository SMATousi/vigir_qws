import numpy as np
import pytest

torch = pytest.importorskip("torch")

from qws.downstream.unet import UNetConfig, build_unet, noise_aware_loss, predict_unet, train_unet  # noqa: E402


def test_noise_aware_loss_properties():
    z = torch.tensor([0.3, -1.2, 2.0])
    hard = torch.tensor([1.0, 0.0, 1.0])
    bce = torch.nn.functional.binary_cross_entropy_with_logits(z, hard)
    assert float(noise_aware_loss(z, hard)) == pytest.approx(float(bce))
    # for soft labels the minimizer is z = logit(mu)
    mu = torch.tensor([0.2, 0.7, 0.5])
    zz = torch.logit(mu).clone().requires_grad_(True)
    noise_aware_loss(zz, mu).backward()
    assert torch.allclose(zz.grad, torch.zeros(3), atol=1e-7)
    # masked pixels do not count
    valid = torch.tensor([1.0, 0.0, 1.0])
    assert float(noise_aware_loss(z, hard, valid)) == pytest.approx(
        float(torch.nn.functional.binary_cross_entropy_with_logits(z[[0, 2]], hard[[0, 2]])))


def test_unet_shapes():
    net = build_unet(3, base=4, depth=2)
    assert net(torch.zeros(2, 3, 32, 32)).shape == (2, 32, 32)


def test_unet_learns_simple_task():
    rng = np.random.default_rng(0)
    imgs, masks = [], []
    for _ in range(12):
        m = np.zeros((24, 24))
        r, c = rng.integers(4, 12, 2)
        m[r:r + 10, c:c + 10] = 1
        imgs.append(np.clip(np.stack([m * 0.8 + 0.1] * 3, -1) + 0.05 * rng.standard_normal((24, 24, 3)), 0, 1))
        masks.append(m)
    cfg = UNetConfig(epochs=30, base=4, depth=2, size=24, batch_size=4, lr=1e-2)
    model, hist = train_unet(imgs, masks, cfg)
    assert hist[-1] < hist[0]
    pred = predict_unet(model, imgs[:2], cfg.size)
    acc = np.mean([np.mean((p > 0.5) == (m > 0.5)) for p, m in zip(pred, masks[:2])])
    assert pred[0].shape == (24, 24) and acc > 0.9


def test_downstream_pipeline_smoke(tmp_path):
    from qws.downstream.pipeline import run_downstream

    out = run_downstream({"dataset": {"kind": "petlike", "root": str(tmp_path / "pl"), "n_images": 4, "generate": 8,
                                      "n_segments": 60}, "train": {"epochs": 2, "num_samples": 20},
                          "solver": {"name": "gibbs", "num_sweeps": 10}, "map_solver": {"name": "sa", "num_sweeps": 50}},
                         UNetConfig(epochs=2, base=4, depth=2, size=32))
    assert set(out["signals"]) == {"structured_soft", "independent_soft", "majority_hard", "ground_truth"}
    assert all(0 <= v["test_iou"] <= 1 for v in out["signals"].values())
