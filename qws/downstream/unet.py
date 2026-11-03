"""Downstream segmentation network trained on label-model soft masks (decision R10; roadmap 5.4).

A deliberately small U-Net and the noise-aware loss of doc 03 §7: the expected
binary cross-entropy under the soft labels mu,

    L = -mean_valid[ mu log sigma(z) + (1 - mu) log(1 - sigma(z)) ],

which equals the cross-entropy against a hard label drawn from the label model's
marginal. Requires the ``downstream`` extra (PyTorch).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


def _torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise ImportError("the downstream trainer needs PyTorch: pip install -e '.[downstream]'") from exc
    return torch


def build_unet(in_channels: int = 3, base: int = 16, depth: int = 3):
    """U-Net with ``depth`` down/up levels, 3x3 conv blocks, 2x2 max-pooling,
    transposed-conv upsampling with skip connections, and a 1-channel logit head."""
    torch = _torch()
    nn = torch.nn

    def block(cin: int, cout: int):
        return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                             nn.Conv2d(cout, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))

    class UNet(nn.Module):
        def __init__(self):
            super().__init__()
            chans = [base * 2**k for k in range(depth + 1)]
            self.down = nn.ModuleList([block(in_channels, chans[0])] +
                                      [block(chans[k], chans[k + 1]) for k in range(depth)])
            self.up = nn.ModuleList([nn.ConvTranspose2d(chans[k + 1], chans[k], 2, stride=2)
                                     for k in reversed(range(depth))])
            self.dec = nn.ModuleList([block(2 * chans[k], chans[k]) for k in reversed(range(depth))])
            self.head = nn.Conv2d(chans[0], 1, 1)
            self.pool = nn.MaxPool2d(2)

        def forward(self, x):
            skips = []
            for k, d in enumerate(self.down):
                x = d(x if k == 0 else self.pool(x))
                skips.append(x)
            for up, dec, skip in zip(self.up, self.dec, reversed(skips[:-1])):
                x = dec(torch.cat([up(x), skip], dim=1))
            return self.head(x)[:, 0]

    return UNet()


def noise_aware_loss(logits, soft, valid=None):
    """Expected BCE under soft labels; ``valid`` (bool) excludes pixels from the mean."""
    torch = _torch()
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, soft, reduction="none")
    if valid is not None:
        return (loss * valid).sum() / valid.sum().clamp(min=1)
    return loss.mean()


@dataclass
class UNetConfig:
    epochs: int = 40
    lr: float = 3e-3
    batch_size: int = 8
    base: int = 16
    depth: int = 3
    size: int = 64          # images are resized to size x size for training and inference
    seed: int = 0
    augment: bool = True    # random horizontal flips


def _resize_stack(arrs: Sequence[np.ndarray], size: int, order: int) -> np.ndarray:
    from skimage.transform import resize

    return np.stack([resize(a, (size, size) + a.shape[2:], order=order, preserve_range=True,
                            anti_aliasing=order > 0) for a in arrs]).astype(np.float32)


def train_unet(images: Sequence[np.ndarray], soft_masks: Sequence[np.ndarray], cfg: UNetConfig | None = None,
               valid_masks: Sequence[np.ndarray] | None = None):
    """Train on images (H, W, 3) in [0, 1] with soft foreground probabilities (H, W)."""
    torch = _torch()
    cfg = cfg or UNetConfig()
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    X = torch.from_numpy(_resize_stack(images, cfg.size, 1)).permute(0, 3, 1, 2)
    Y = torch.from_numpy(np.clip(_resize_stack(soft_masks, cfg.size, 1), 0, 1))
    V = (torch.from_numpy(_resize_stack([v.astype(float) for v in valid_masks], cfg.size, 0)) > 0.5).float() \
        if valid_masks is not None else None
    model = build_unet(X.shape[1], cfg.base, cfg.depth)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(cfg.epochs, 1))
    history = []
    for _ in range(cfg.epochs):
        model.train()
        order = rng.permutation(len(X))
        total = 0.0
        for start in range(0, len(X), cfg.batch_size):
            idx = torch.from_numpy(order[start:start + cfg.batch_size])
            xb, yb = X[idx], Y[idx]
            vb = V[idx] if V is not None else None
            if cfg.augment and rng.random() < 0.5:
                xb, yb = xb.flip(-1), yb.flip(-1)
                vb = vb.flip(-1) if vb is not None else None
            loss = noise_aware_loss(model(xb), yb, vb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.detach()) * len(idx)
        sched.step()
        history.append(total / len(X))
    model.eval()
    return model, history


def predict_unet(model, images: Sequence[np.ndarray], size: int) -> list[np.ndarray]:
    """Foreground probabilities at each image's original resolution."""
    torch = _torch()
    from skimage.transform import resize

    X = torch.from_numpy(_resize_stack(images, size, 1)).permute(0, 3, 1, 2)
    with torch.no_grad():
        P = torch.sigmoid(model(X)).numpy()
    return [resize(p, img.shape[:2], order=1, preserve_range=True) for p, img in zip(P, images)]
