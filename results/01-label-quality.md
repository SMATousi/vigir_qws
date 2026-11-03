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
| grid 16×16, Gibbs | 1.000 ± 0.000 | 0.723 ± 0.006 | 0.761 ± 0.007 | 0.978 ± 0.007 | 0.012 ± 0.002 | 0.057 ± 0.016 | 0.162 ± 0.031 | 7.217 ± 0.124 |
| superpixels (~60) on 32×32, Gibbs | 0.769 ± 0.017 | 0.605 ± 0.023 | 0.633 ± 0.029 | 0.760 ± 0.018 | 0.106 ± 0.009 | 0.097 ± 0.026 | 0.160 ± 0.058 | 2.226 ± 0.080 |
| grid 16×16, anneal_sim (fixed T) + calibration | 1.000 ± 0.000 | 0.723 ± 0.006 | 0.761 ± 0.007 | 0.978 ± 0.007 | 0.009 ± 0.002 | 0.059 ± 0.016 | 0.160 ± 0.031 | 78.948 ± 6.599 |
| grid 16×16, EM mode, Gibbs | 1.000 ± 0.000 | 0.723 ± 0.006 | 0.761 ± 0.007 | 0.981 ± 0.005 | 0.011 ± 0.001 | 0.060 ± 0.014 | 0.037 ± 0.017 | 3.634 ± 0.045 |

## Learned vs. true parameters (grid 16×16, Gibbs, seed 0)

| Parameter | True | Learned |
|---|---|---|
| θ_0 | 1.000 | 1.055 |
| θ_1 | 0.600 | 0.590 |
| θ_2 | 0.300 | 0.288 |
| θ_3 | 0.800 | 0.746 |
| α_0 | 0.500 | 0.511 |
| α_1 | 0 | -0.081 |
| α_2 | -0.500 | -0.613 |
| α_3 | 0.200 | 0.259 |
| θᴾ_0 (pairwise) | 1.000 | 1.193 |
| αᴾ_0 (pairwise) | 0 | -0.140 |

## Per-seed results

| Configuration | Seed | IoU majority | IoU independent | IoU structured | max |θ−θ*| | Epochs |
|---|---|---|---|---|---|---|
| grid 16×16, Gibbs | 0 | 0.716 | 0.753 | 0.978 | 0.055 | 20 |
| grid 16×16, Gibbs | 1 | 0.725 | 0.768 | 0.983 | 0.035 | 20 |
| grid 16×16, Gibbs | 2 | 0.729 | 0.769 | 0.971 | 0.051 | 20 |
| grid 16×16, Gibbs | 3 | 0.717 | 0.755 | 0.972 | 0.066 | 20 |
| grid 16×16, Gibbs | 4 | 0.728 | 0.761 | 0.987 | 0.077 | 20 |
| superpixels (~60) on 32×32, Gibbs | 0 | 0.610 | 0.653 | 0.780 | 0.118 | 20 |
| superpixels (~60) on 32×32, Gibbs | 1 | 0.598 | 0.623 | 0.758 | 0.104 | 20 |
| superpixels (~60) on 32×32, Gibbs | 2 | 0.616 | 0.666 | 0.775 | 0.067 | 20 |
| superpixels (~60) on 32×32, Gibbs | 3 | 0.631 | 0.629 | 0.751 | 0.073 | 20 |
| superpixels (~60) on 32×32, Gibbs | 4 | 0.570 | 0.592 | 0.734 | 0.124 | 20 |
| grid 16×16, anneal_sim (fixed T) + calibration | 0 | 0.716 | 0.753 | 0.978 | 0.057 | 20 |
| grid 16×16, anneal_sim (fixed T) + calibration | 1 | 0.725 | 0.768 | 0.983 | 0.037 | 20 |
| grid 16×16, anneal_sim (fixed T) + calibration | 2 | 0.729 | 0.769 | 0.971 | 0.056 | 20 |
| grid 16×16, anneal_sim (fixed T) + calibration | 3 | 0.717 | 0.755 | 0.972 | 0.066 | 20 |
| grid 16×16, anneal_sim (fixed T) + calibration | 4 | 0.728 | 0.761 | 0.987 | 0.079 | 20 |
| grid 16×16, EM mode, Gibbs | 0 | 0.716 | 0.753 | 0.980 | 0.061 | 10 |
| grid 16×16, EM mode, Gibbs | 1 | 0.725 | 0.768 | 0.986 | 0.042 | 10 |
| grid 16×16, EM mode, Gibbs | 2 | 0.729 | 0.769 | 0.977 | 0.056 | 10 |
| grid 16×16, EM mode, Gibbs | 3 | 0.717 | 0.755 | 0.977 | 0.062 | 10 |
| grid 16×16, EM mode, Gibbs | 4 | 0.728 | 0.761 | 0.987 | 0.080 | 10 |

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
