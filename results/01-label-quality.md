# 01 — Label quality: structured vs. unstructured weak supervision

Does the structured label model give better segmentation labels than unstructured weak supervision?
This is evaluation question Q1 in [spec/06](../spec/06-evaluation.md).

## Setup

- **Data:** synthetic. Masks are smoothed random blobs (smoothness 3, 40% foreground), and images are the mask
  plus Gaussian noise. There are 8 images per run and 5 seeds (0, 1, 2, 3, 4).
- **Labeling functions:** votes are drawn from the label model's own generative law
  (paper Lemma 5) with known parameters:
  - 4 unary LFs with θ = [1.0, 0.6, 0.3, 0.8] and α = [0.5, 0.0, -0.5, 0.2];
  - 1 pairwise "same/different" LF with θᴾ = [1.0].
  The pairwise LF makes the posterior **frustrated** (it has negative couplings).
- **Model:** smoothness β = 0.5 (fixed, open question Q4). Warm start from the independent model, then
  20 epochs of Adam (gradient mode) or EM, with 100 samples per image per step.
- **Inference:** the hard mask comes from simulated annealing, since graph cuts refuse frustrated problems.
  Soft masks come from 200 Gibbs samples.
- **Baselines:** per-site majority vote, and the independent (Snorkel-style, β = 0, no pairwise LF)
  label model.
- **Oracle:** the true per-site labels projected back to pixels. This is the best IoU any method can reach
  with that site representation; it is below 1 for superpixels that straddle the mask boundary.

## Summary (mean ± std over seeds)

| Configuration | IoU oracle | IoU majority | IoU independent | IoU structured | ECE structured | max |θ−θ*| | |θᴾ−θᴾ*| | train time (s) |
|---|---|---|---|---|---|---|---|---|
| grid 16×16, Gibbs | 1.000 ± 0.000 | 0.709 ± 0.005 | 0.752 ± 0.009 | 0.979 ± 0.006 | 0.012 ± 0.002 | 0.070 ± 0.025 | 0.163 ± 0.025 | 6.864 ± 0.038 |
| superpixels (~60) on 32×32, Gibbs | 0.769 ± 0.017 | 0.586 ± 0.023 | 0.618 ± 0.032 | 0.753 ± 0.015 | 0.109 ± 0.008 | 0.085 ± 0.023 | 0.166 ± 0.048 | 2.166 ± 0.030 |
| grid 16×16, anneal_sim (fixed T) + calibration | 1.000 ± 0.000 | 0.709 ± 0.005 | 0.752 ± 0.009 | 0.979 ± 0.006 | 0.009 ± 0.003 | 0.064 ± 0.032 | 0.161 ± 0.025 | 8.472 ± 0.024 |
| grid 16×16, EM mode, Gibbs | 1.000 ± 0.000 | 0.709 ± 0.005 | 0.752 ± 0.009 | 0.983 ± 0.007 | 0.012 ± 0.002 | 0.062 ± 0.028 | 0.031 ± 0.024 | 3.486 ± 0.076 |

## Learned vs. true parameters (grid 16×16, Gibbs, seed 0)

| Parameter | True | Learned |
|---|---|---|
| θ_0 | 1.000 | 1.016 |
| θ_1 | 0.600 | 0.635 |
| θ_2 | 0.300 | 0.328 |
| θ_3 | 0.800 | 0.716 |
| α_0 | 0.500 | 0.545 |
| α_1 | 0 | -0.035 |
| α_2 | -0.500 | -0.557 |
| α_3 | 0.200 | 0.265 |
| θᴾ_0 (pairwise) | 1.000 | 1.192 |
| αᴾ_0 (pairwise) | 0 | -0.139 |

## Per-seed results

| Configuration | Seed | IoU majority | IoU independent | IoU structured | max |θ−θ*| | Epochs |
|---|---|---|---|---|---|---|
| grid 16×16, Gibbs | 0 | 0.714 | 0.753 | 0.979 | 0.084 | 20 |
| grid 16×16, Gibbs | 1 | 0.710 | 0.758 | 0.982 | 0.031 | 20 |
| grid 16×16, Gibbs | 2 | 0.710 | 0.761 | 0.982 | 0.096 | 20 |
| grid 16×16, Gibbs | 3 | 0.701 | 0.742 | 0.969 | 0.077 | 20 |
| grid 16×16, Gibbs | 4 | 0.709 | 0.744 | 0.985 | 0.062 | 20 |
| superpixels (~60) on 32×32, Gibbs | 0 | 0.610 | 0.640 | 0.757 | 0.062 | 20 |
| superpixels (~60) on 32×32, Gibbs | 1 | 0.582 | 0.619 | 0.754 | 0.068 | 20 |
| superpixels (~60) on 32×32, Gibbs | 2 | 0.606 | 0.655 | 0.772 | 0.117 | 20 |
| superpixels (~60) on 32×32, Gibbs | 3 | 0.576 | 0.603 | 0.751 | 0.099 | 20 |
| superpixels (~60) on 32×32, Gibbs | 4 | 0.555 | 0.572 | 0.730 | 0.081 | 20 |
| grid 16×16, anneal_sim (fixed T) + calibration | 0 | 0.714 | 0.753 | 0.979 | 0.082 | 20 |
| grid 16×16, anneal_sim (fixed T) + calibration | 1 | 0.710 | 0.758 | 0.982 | 0.014 | 20 |
| grid 16×16, anneal_sim (fixed T) + calibration | 2 | 0.710 | 0.761 | 0.982 | 0.096 | 20 |
| grid 16×16, anneal_sim (fixed T) + calibration | 3 | 0.701 | 0.742 | 0.969 | 0.077 | 20 |
| grid 16×16, anneal_sim (fixed T) + calibration | 4 | 0.709 | 0.744 | 0.985 | 0.054 | 20 |
| grid 16×16, EM mode, Gibbs | 0 | 0.714 | 0.753 | 0.981 | 0.070 | 10 |
| grid 16×16, EM mode, Gibbs | 1 | 0.710 | 0.758 | 0.988 | 0.019 | 10 |
| grid 16×16, EM mode, Gibbs | 2 | 0.710 | 0.761 | 0.985 | 0.098 | 10 |
| grid 16×16, EM mode, Gibbs | 3 | 0.701 | 0.742 | 0.971 | 0.066 | 10 |
| grid 16×16, EM mode, Gibbs | 4 | 0.709 | 0.744 | 0.988 | 0.057 | 10 |

## Observations

- The structured model beats both baselines by a wide margin in every configuration and every seed. This is
  the M1 acceptance criterion of spec/06 §5.
- Unary LF accuracies are recovered without ground truth, to within about 0.1.
- The pairwise accuracy θᴾ is recovered within about 0.03 by EM, but only within about 0.16 by gradient
  ascent after the same budget. Its gradient comes only from the edge correlations ⟨yᵢyₖ⟩, and Adam has not
  converged on it after 20 epochs. EM's closed-form M-step (θ = artanh(agree/cover)) jumps there directly.
- On superpixels the structured model closes most of the gap between the baselines and the oracle; the
  remaining gap reflects coarser sites and noisier lifted votes.
- The `anneal_sim` path (Ocean interface, fixed-temperature stand-in, pseudo-likelihood calibration) reaches the
  same quality as Gibbs. This exercises the code path a real QPU would use.
- Caveat: votes follow the model's own generative law, so this is a best case for the label model.
  Real LFs will deviate from it (open question Q1: dataset choice).
