# 15 — Sampler temperature and the sampler-bias term

Two checks that complement report 04. Both concern the sampler that supplies the posterior moments.

## 1. Learning with samplers at the wrong temperature

**Setup.** Four weak unary LFs (θ* = [0.5, 0.3, 0.15, 0.4], α* = [0.5, 0.0, -0.5, 0.2]), so that the posterior is genuinely
uncertain. Masks are drawn from the model's own prior (16×16 grids, 12 images per seed), so the model is
correctly specified and the true parameters are the population optimum. β is fixed at its true value. Two
settings: **independent sites** (β = 0) and **coupled sites** (β = 0.4, just below the critical point 0.44).
Learning (warm start, then Adam with the Theorem-1 gradient) is repeated with posterior samples drawn at inverse
temperature β_s instead of 1, by annealed sampling that stops at β_s. The MAP row uses annealing plus greedy
descent, which is hard EM. Seeds: [0, 1, 2].

| setting | sampler β_s | mean θ / θ* | SD over seeds | mean bias of σ(2θ) | mean abs. error of σ(2θ) |
|---|---|---|---|---|---|
| β = 0 | 0.75 | 0.023 | 0.019 | -0.159 | 0.159 |
| β = 0 | 1 | 0.914 | 0.098 | -0.010 | 0.034 |
| β = 0 | 1.5 | 1.125 | 0.168 | 0.016 | 0.086 |
| β = 0 | 2 | 1.296 | 0.229 | 0.041 | 0.084 |
| β = 0 | 4 | 1.436 | 0.319 | 0.060 | 0.093 |
| β = 0 | MAP (hard EM) | 1.422 | 0.333 | 0.058 | 0.094 |
| β = 0.4 | 0.75 | 0.632 | 0.126 | -0.060 | 0.060 |
| β = 0.4 | 1 | 0.925 | 0.027 | -0.008 | 0.017 |
| β = 0.4 | 1.5 | 1.032 | 0.018 | 0.007 | 0.016 |
| β = 0.4 | 2 | 1.027 | 0.008 | 0.005 | 0.013 |
| β = 0.4 | 4 | 0.979 | 0.023 | -0.002 | 0.014 |
| β = 0.4 | MAP (hard EM) | 0.985 | 0.022 | -0.002 | 0.012 |

- **Correct temperature.** At β_s = 1, θ / θ* is 0.91 (independent) and 0.92
  (coupled).
- **Independent sites.** θ / θ* moves from 0.02 (β_s = 0.75) to 1.30
  (β_s = 2) and 1.42 (MAP).
- **Coupled sites.** θ / θ* moves from 0.63 (β_s = 0.75) to 1.03
  (β_s = 2) and 0.99 (MAP).

- **Independent sites: the predicted direction, monotone.** Colder sampling inflates θ steadily, up to about
  1.4 × θ* at β_s = 4 and with MAP labels, and the implied accuracies are then overestimated by about 0.06. A
  hotter sampler (β_s = 0.75) is worse still: learning collapses to the trivial fixed point θ ≈ 0.
- **Coupled sites: small and not monotone.** Colder sampling changes θ by at most 3% (1.03 × θ* at β_s = 1.5–2,
  0.98 at β_s = 4 and with MAP labels), and the implied-accuracy error stays at or below 0.016 for every
  β_s ≥ 1. A likely reason is that near the critical point the coupled posterior is already sharp, so cooling it
  further changes the moments little. The hotter sampler still deflates θ (0.63 × θ*).
- **At β_s = 1** θ is slightly underestimated (0.91–0.92 × θ*): 12 images and 15 epochs from the warm start.
- **Reading.** The proposition's statements hold: a sampler at β_s returns the posterior of the model at β_s·Θ.
  What that does to the learned parameters depends on the model. Without coupling, cold samplers inflate them as
  the proposition's title says; with coupling the effect can be small and change sign. Too-hot samplers are at
  least as harmful as too-cold ones, which again argues for calibrating the temperature rather than lowering it.

## 2. The sampler-bias term bounds the observed moment errors

For any function f with |f| ≤ 1, |E_q f − E_p f| ≤ 2·TV(q, p). Applied to the empirical distribution of each
sampler's output on the n = 12 problems of report 03, the error of every first and edge moment must be at
most twice the measured TV distance. The last column is the largest ratio (moment error) / (2·TV) over the
problems of that row; the bound holds if it is ≤ 1.

| regime | sampler | mean TV | mean max moment error | max ratio to 2·TV |
|---|---|---|---|---|
| weak coupling (J×0.3) | exact (reference sampler) | 0.041 | 0.025 | 0.360 |
| weak coupling (J×0.3) | gibbs, 10 sweeps | 0.042 | 0.022 | 0.426 |
| weak coupling (J×0.3) | gibbs, 100 sweeps | 0.043 | 0.022 | 0.407 |
| weak coupling (J×0.3) | anneal_sim, auto range (cold) | 0.566 | 0.658 | 0.765 |
| weak coupling (J×0.3) | anneal_sim fixed T, raw | 0.462 | 0.456 | 0.574 |
| weak coupling (J×0.3) | anneal_sim fixed T + 5 polish | 0.044 | 0.024 | 0.506 |
| weak coupling (J×0.3) | anneal_sim calibrated | 0.044 | 0.029 | 0.644 |
| weak coupling (J×0.3) | qaoa_sim p=2 raw | 0.375 | 0.420 | 0.651 |
| weak coupling (J×0.3) | qaoa_sim p=2 + 5 polish | 0.044 | 0.024 | 0.506 |
| nominal (β=0.5 model) | exact (reference sampler) | 0.010 | 0.011 | 0.703 |
| nominal (β=0.5 model) | gibbs, 10 sweeps | 0.023 | 0.040 | 0.942 |
| nominal (β=0.5 model) | gibbs, 100 sweeps | 0.009 | 0.012 | 0.843 |
| nominal (β=0.5 model) | anneal_sim, auto range (cold) | 0.127 | 0.206 | 0.961 |
| nominal (β=0.5 model) | anneal_sim fixed T, raw | 0.118 | 0.187 | 0.954 |
| nominal (β=0.5 model) | anneal_sim fixed T + 5 polish | 0.011 | 0.014 | 0.840 |
| nominal (β=0.5 model) | anneal_sim calibrated | 0.010 | 0.011 | 0.761 |
| nominal (β=0.5 model) | qaoa_sim p=2 raw | 0.891 | 1.195 | 0.856 |
| nominal (β=0.5 model) | qaoa_sim p=2 + 5 polish | 0.086 | 0.152 | 0.939 |
| strong coupling (J×3) | exact (reference sampler) | 0.001 | 0.002 | 1.000 |
| strong coupling (J×3) | gibbs, 10 sweeps | 0.474 | 0.853 | 0.983 |
| strong coupling (J×3) | gibbs, 100 sweeps | 0.323 | 0.621 | 1.000 |
| strong coupling (J×3) | anneal_sim, auto range (cold) | 0.037 | 0.074 | 1.000 |
| strong coupling (J×3) | anneal_sim fixed T, raw | 0.032 | 0.064 | 1.000 |
| strong coupling (J×3) | anneal_sim fixed T + 5 polish | 0.024 | 0.049 | 1.000 |
| strong coupling (J×3) | anneal_sim calibrated | 0.167 | 0.252 | 1.000 |
| strong coupling (J×3) | qaoa_sim p=2 raw | 0.998 | 1.261 | 0.775 |
| strong coupling (J×3) | qaoa_sim p=2 + 5 polish | 0.450 | 0.780 | 0.903 |

Over all 27 sampler × regime rows, the largest ratio is **1.00** and the
median ratio over individual problems is 0.63. The bound holds everywhere, and it is tight in the
strong-coupling regime for every sampler, including the exact one. There the target concentrates on two
mirror-image basins, and any misallocation of mass between them shifts every spin at once, so the largest moment
error equals 2·TV exactly. In the weak regime the errors are about 0.4–0.8 of the bound.
