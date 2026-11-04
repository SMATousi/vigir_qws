# 13 — Label model on real Oxford-IIIT Pet images

The label model on the **real Oxford-IIIT Pet dataset** (Parkhi et al., CVPR 2012; decision R5): 200 random
`trainval` images per seed, 3 seed(s). This is the same configuration as report 08, which ran on
generated Pet-layout data; only the data changed. Pixel labels are used **only for evaluation**.

## Setup

- **Data:** 200 images per seed (longer side 128 px), about 150 SLIC superpixels each. Trimap boundary pixels
  are excluded from every metric.
- **LFs:** generic only, none uses ground truth:
  - border → background;
  - centre ellipse → foreground;
  - border-vs-centre colour-histogram model;
  - spectral-residual saliency;
  - pairwise colour contrast.
- **Model:** β = 0.5, 15 epochs, Gibbs sampling, MAP by simulated annealing.
- **Wall time:** 341 s on CPU for all seeds.

## Label quality (IoU per seed)

| seed | IoU oracle | IoU majority | IoU independent | IoU structured | ECE structured |
|---|---|---|---|---|---|
| 0 | 0.965 | 0.518 | 0.590 | 0.634 | 0.150 |
| 1 | 0.963 | 0.484 | 0.583 | 0.614 | 0.153 |
| 2 | 0.963 | 0.480 | 0.553 | 0.599 | 0.165 |

Mean ± SD over seeds: oracle 0.963 ± 0.001, majority 0.494 ± 0.017,
independent 0.576 ± 0.016, **structured 0.616 ± 0.014**.

"Oracle" is the best mask the superpixels can represent (each superpixel set to its majority ground-truth
label). It bounds every method that labels superpixels.

## What the label model learned about each LF (seed 0, without ground truth)

The empirical accuracy (measured against site ground truth) is a diagnostic only. The implied accuracy σ(2θ)
is the model's own estimate of the probability that a cast vote is correct.

| LF | coverage | empirical accuracy | learned θ | implied accuracy σ(2θ) |
|---|---|---|---|---|
| border_bg | 0.246 | 0.897 | 1.703 | 0.968 |
| center_fg | 0.089 | 0.865 | 1.097 | 0.900 |
| color_model | 0.720 | 0.873 | 1.905 | 0.978 |
| saliency | 0.613 | 0.687 | 0.441 | 0.707 |

Pairwise colour-contrast LF: learned θᴾ = 1.17.

## Observations

- On real photographs, the structured model reaches IoU 0.616, against 0.576 for the
  independent model and 0.494 for majority vote. It closes 10% of the gap between the
  independent model and the superpixel oracle.
- The label model ranks the LFs correctly without labels: saliency, the weakest LF, gets by far the lowest
  weight. It is, however, **overconfident**: the implied accuracies σ(2θ) exceed the measured ones by about
  0.02–0.11 (table above), and the expected calibration error of the structured marginals is 0.156.
- The remaining gap is most likely due to the generic LFs themselves. They encode a "centred object, distinct
  from the border" prior, which fails for off-centre pets, pets touching the border, and backgrounds coloured
  like the animal. Better LFs, not a better solver, are the lever on this dataset.
- Compared with report 08 (generated data, same pipeline), the ordering structured > independent > majority
  holds, but the margins are smaller on real photographs.
