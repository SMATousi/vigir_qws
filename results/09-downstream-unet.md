# 09 — Downstream U-Net trained on weak labels

End-task evaluation (decision R10): a small U-Net is trained on weak labels for the training images and
evaluated on held-out test images. Data is the generated Pet-layout set of report 08, **not real Pet**.

## Setup

- **Split:** 60 training images (Pet `trainval` list) and 60 test images (`test` list).
- **Network:** U-Net with depth 3 and 16 base channels, on 64×64 inputs; Adam with cosine schedule,
  100 epochs, horizontal-flip augmentation.
- **Loss:** noise-aware expected binary cross-entropy under the soft labels (doc 03 §7).
- **Metric:** test IoU at original resolution, excluding trimap boundary pixels.
- **Wall time:** 140 s on CPU.

| training signal | IoU of the training labels | U-Net test IoU | final train loss |
|---|---|---|---|
| structured label model, soft (this work) | 0.797 | 0.745 | 0.234 |
| independent label model, soft | 0.679 | 0.749 | 0.337 |
| majority vote, hard | 0.512 | 0.394 | 0.265 |
| ground truth (upper bound) | 1.000 | 0.733 | 0.103 |

For reference, the label model applied directly (transductively) to the test images reaches IoU
**0.819**.

## Observations

- The structured label model gives by far the best *training labels*, well above the independent model and
  majority vote.
- On this small synthetic dataset, the U-Net itself is the bottleneck. Even trained on ground truth, it reaches
  only about the test IoU it reaches from either label model's soft masks. The label-quality advantage does not
  carry through to the end task here. Majority-vote labels, in contrast, clearly hurt.
- The label model applied directly beats every U-Net on these images. The meaningful end-task comparison needs
  real data with learnable appearance (Oxford-IIIT Pet), more images, and a stronger network. The
  pipeline is ready for that.
