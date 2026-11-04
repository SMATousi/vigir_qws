# 14 — Downstream U-Net on real Oxford-IIIT Pet

End-task evaluation on the **real Oxford-IIIT Pet dataset** (decision R10). A small U-Net is trained on weak
labels for 400 random `trainval` images, and evaluated on 400 random images of the
official `test` list. This repeats report 09, which used generated Pet-layout data.

## Setup

- **Network:** U-Net with depth 3 and 16 base channels, on
  64×64 inputs; Adam with cosine schedule, 60 epochs,
  horizontal-flip augmentation.
- **Loss:** noise-aware expected binary cross-entropy under the soft labels (doc 03 §7).
- **Metric:** test IoU at the label resolution (longer side 128 px), excluding trimap boundary pixels.
- **Wall time:** 758 s on CPU.

| training signal | IoU of the training labels | U-Net test IoU | final train loss |
|---|---|---|---|
| structured label model, soft (this work) | 0.625 | 0.621 | 0.209 |
| independent label model, soft | 0.584 | 0.621 | 0.311 |
| majority vote, hard | 0.500 | 0.485 | 0.164 |
| ground truth (upper bound) | 1.000 | 0.743 | 0.074 |

For reference, the label model applied directly (transductively) to the test images reaches IoU
**0.613**.

## Observations

- Training-label quality: structured 0.625, independent
  0.584, majority 0.500.
- U-Net test IoU: structured 0.621, independent
  0.621, majority 0.485; trained on ground
  truth 0.743.
- Better training labels do not carry over to the network here: structured and independent soft labels give
  the same U-Net test IoU, even though the structured training labels are better. Majority-vote labels are
  clearly worse, and ground truth is well ahead. The U-Net trained on structured labels matches the label
  model applied directly to the test images (0.613), so it generalizes the weak labels to new images without
  losing quality.
- The network is small and trained on CPU at 64 px, so absolute numbers are far from the state of the art. The
  comparison between training signals is what this report measures.
