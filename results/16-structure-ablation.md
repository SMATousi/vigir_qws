# 16 — Joint structured learning vs. learn-then-smooth

Does learning the LF parameters *with* the spatial prior beat the simpler pipeline of learning them
independently (as Snorkel does) and smoothing the masks afterwards? Four variants share the same images, LF
votes and smoothness β:

| variant | learning | inference |
|---|---|---|
| independent | β = 0 | β = 0 |
| learn-then-smooth | β = 0 | MAP with β, unary LFs |
| joint (unary) | with β | MAP with β, unary LFs |
| joint (full) | with β and the pairwise LF | MAP with β and the pairwise LF |

independent → learn-then-smooth is the effect of smoothing at inference. **learn-then-smooth → joint (unary)
is the effect of joint learning**, the question of this report. joint (unary) → joint (full) is the effect of the
pairwise LF. Each β is reported, so no variant benefits from a β tuned for it. Ground truth is used only for
evaluation. MAP is exact (graph cut) for the unary-only variants and simulated annealing for the full model.
The last column is the mean IoU gain of joint (unary) over learn-then-smooth, and how many runs it wins.

## Synthetic, report 01 setting (votes from the model's law; θ* = [1.0, 0.6, 0.3, 0.8], θᴾ* = 1.0)

16×16 grids, 8 images, seeds [0, 1, 2, 3, 4].

| β | independent | learn-then-smooth | joint (unary) | joint (full) | joint − smooth (wins) |
|---|---|---|---|---|---|
| 0.25 | 0.761 ± 0.006 | 0.874 ± 0.009 | 0.864 ± 0.010 | 0.952 ± 0.004 | -0.010 (0/5) |
| 0.5 | 0.761 ± 0.006 | 0.937 ± 0.012 | 0.928 ± 0.014 | 0.978 ± 0.006 | -0.009 (0/5) |
| 1 | 0.761 ± 0.006 | 0.976 ± 0.006 | 0.976 ± 0.006 | 0.991 ± 0.004 | +0.000 (2/5) |

Mean absolute error of the implied accuracies σ(2θ) of the unary LFs:

| β | independent | learn-then-smooth | joint (unary) | joint (full) |
|---|---|---|---|---|
| 0.25 | 0.011 | 0.011 | 0.016 | 0.010 |
| 0.5 | 0.011 | 0.011 | 0.014 | 0.009 |
| 1 | 0.011 | 0.011 | 0.010 | 0.010 |

## Synthetic, weak LFs (θ* = [0.5, 0.3, 0.15, 0.4], θᴾ* = 0.5)

Same setting with weaker LFs, so that the posterior is uncertain. Seeds [0, 1, 2, 3, 4].

| β | independent | learn-then-smooth | joint (unary) | joint (full) | joint − smooth (wins) |
|---|---|---|---|---|---|
| 0.25 | 0.519 ± 0.011 | 0.727 ± 0.006 | 0.668 ± 0.033 | 0.730 ± 0.024 | -0.059 (0/5) |
| 0.5 | 0.519 ± 0.011 | 0.879 ± 0.017 | 0.860 ± 0.015 | 0.878 ± 0.010 | -0.019 (0/5) |
| 1 | 0.519 ± 0.011 | 0.934 ± 0.015 | 0.936 ± 0.014 | 0.960 ± 0.005 | +0.002 (4/5) |

Mean absolute error of the implied accuracies σ(2θ):

| β | independent | learn-then-smooth | joint (unary) | joint (full) |
|---|---|---|---|---|
| 0.25 | 0.028 | 0.028 | 0.041 | 0.016 |
| 0.5 | 0.028 | 0.028 | 0.027 | 0.014 |
| 1 | 0.028 | 0.028 | 0.012 | 0.011 |

## Real images: Oxford-IIIT Pet

200 random `trainval` images per seed, about 150 SLIC superpixels each, the generic LFs of report 13;
trimap boundary pixels excluded. Seeds [0, 1, 2].

| β | independent | learn-then-smooth | joint (unary) | joint (full) | joint − smooth (wins) |
|---|---|---|---|---|---|
| 0.25 | 0.576 ± 0.016 | 0.626 ± 0.011 | 0.624 ± 0.013 | 0.604 ± 0.014 | -0.002 (0/3) |
| 0.5 | 0.576 ± 0.016 | 0.632 ± 0.011 | 0.631 ± 0.011 | 0.616 ± 0.014 | -0.001 (1/3) |
| 1 | 0.576 ± 0.016 | 0.643 ± 0.013 | 0.639 ± 0.016 | 0.621 ± 0.012 | -0.003 (1/3) |

## Observations

- **Smoothing at inference is where the gain comes from.** Learn-then-smooth beats the independent model
  by 0.11–0.22 IoU on the synthetic data, 0.21–0.42 with weak LFs, and 0.050–0.067 on Pet.
- **Joint learning of the unary LFs adds nothing on top.** Joint (unary) is never more than 0.002 IoU better
  than learn-then-smooth, and it is worse at small β (by up to 0.059 with weak LFs at β = 0.25). It wins 8 of
  39 runs, all at β ≥ 0.5.
- **Why.** If the LFs are conditionally independent given each site's label, the independent model maximizes
  the sum of per-site marginal log-likelihoods. That is a composite likelihood, which estimates θ consistently
  whatever the spatial dependence between sites, provided the per-site model (including the class prior) is
  right. Joint learning can then only improve efficiency, and it adds a risk: when β is wrong, the
  posterior moments are biased, and so is θ. That matches the accuracy tables. Joint (unary) estimates the
  implied accuracies better than the independent model only at β = 1 with weak LFs (error 0.012 against
  0.028), and worse at β = 0.25 (0.041 against 0.028).
- **The pairwise LF is what needs structured learning.** Its parameter cannot be learned without the
  structured posterior. On synthetic data, where its votes follow the model, it adds 0.015–0.088 IoU
  (0.018–0.062 with weak LFs), and the full model also estimates the unary accuracies best (error
  0.009–0.016). On Pet, the generic colour-contrast pairwise LF **lowers** IoU by 0.015–0.020.
- **Best real-image result.** On Pet the best variant is learn-then-smooth with β = 1 (IoU 0.643). That is
  above the full model of report 13 (0.616), which used the pairwise LF and β = 0.5.
- **Consequence for the quantum role.** For unary LFs that are conditionally independent given the label,
  learning needs no structured sampling: the independent model is in closed form, and MAP with β ≥ 0 is a
  graph cut. Structured sampling, the paper's target for annealers, is needed when the model has terms that
  only the structured posterior identifies: pairwise LFs, learned β and b, LF dependencies, and
  location-level constraints such as those of report 12.
