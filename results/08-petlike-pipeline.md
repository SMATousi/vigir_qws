# 08 — Real-image pipeline on Pet-format data

The real-image pipeline (decision R5): JPEG decoding, Pet trimaps with an ignore band, resizing, SLIC
superpixels, generic LFs, and ignore-masked metrics. It runs end to end on a **generated dataset in the exact
Oxford-IIIT Pet file layout** (`qws.data.petlike`), because the real dataset's hosts are blocked from this
environment. **These are not Oxford-IIIT Pet results.** The images are synthetic textured blobs on textured
backgrounds, with overlapping colour distributions.

## Setup

- **Data:** 30 images per seed (128 px longer side), ~150 SLIC superpixels each.
- **LFs:** generic only, none uses ground truth:
  - border → background;
  - centre ellipse → foreground;
  - border-vs-centre colour-histogram model;
  - spectral-residual saliency;
  - pairwise colour contrast.
- **Model:** β = 0.5, 15 epochs, Gibbs sampling, MAP by simulated annealing.
- **Metrics:** boundary (trimap = 3) pixels are excluded.

## Label quality (per seed)

| seed | IoU oracle | IoU majority | IoU independent | IoU structured | ECE structured |
|---|---|---|---|---|---|
| 0 | 0.926 | 0.533 | 0.704 | 0.802 | 0.058 |
| 1 | 0.926 | 0.562 | 0.711 | 0.810 | 0.051 |
| 2 | 0.933 | 0.430 | 0.599 | 0.749 | 0.086 |

Mean: oracle 0.928, majority 0.508, independent 0.671, **structured 0.787**.

## What the label model learned about each LF (seed 0, without ground truth)

The empirical accuracy (measured against site ground truth) is a diagnostic only. The implied accuracy
σ(2θ) is the model's own estimate of the probability that a cast vote is correct.

| LF | coverage | empirical accuracy | learned θ | implied accuracy σ(2θ) |
|---|---|---|---|---|
| border_bg | 0.227 | 0.981 | 3.074 | 0.998 |
| center_fg | 0.090 | 0.906 | 2.554 | 0.994 |
| color_model | 0.822 | 0.970 | 2.152 | 0.987 |
| saliency | 0.653 | 0.476 | -0.079 | 0.461 |

Pairwise colour-contrast LF: learned θᴾ = 1.06.

## Observations

- The structured model beats both baselines by a wide margin, and closes roughly half of the gap between the
  independent model and the superpixel oracle. The remainder reflects errors in the generic LFs themselves.
- Without labels, the label model correctly ranks the LFs. It assigns high weight to the reliable priors and
  near-zero weight to saliency, which is close to chance on these images. This is the core promise of data
  programming, working on real-format images.
- Next step (blocked by network policy): run the same command on the real dataset, with `kind: pet` and the
  path to the downloaded data.
