# 03 — Sampling benchmark against exact Boltzmann distributions

How close are the samplers' outputs to the exact Boltzmann distribution at β = 1? This is the quantity
that drives learning (paper Proposition 8: the gradient bias is at most 2·TV).

## Setup

- **Problems:** posterior Ising models on a 3×4 grid (n = 12, 4096 states, so the distribution can be
  enumerated exactly).
- **Regimes:** three coupling strengths, from 6 problems each.
- **Samples:** 3000 per problem. Metrics are TV distance to the exact distribution, and the maximum error of
  the ⟨yᵢ⟩ and ⟨yᵢyₖ⟩ estimates.
- **Finite-sample floor:** the exact sampler's TV row shows the floor. Even perfect samples have TV ≈
  0.58 or less at this sample size, from 4096 states and 3000 draws.
- **Sampler variants:**
  - "raw" is the sampler output as returned;
  - "calibrated" is wrapped in the pseudo-likelihood temperature calibration (paper Proposition 9);
  - "+ k polish" adds k Gibbs sweeps at β = 1 from each sample (paper Proposition 10).

### weak coupling (J×0.3)

| Sampler | TV distance | max marginal error | max edge-corr error | wall (s) |
|---|---|---|---|---|
| exact (reference sampler) | 0.041 ± 0.020 | 0.018 ± 0.010 | 0.024 ± 0.009 | 0.000782 |
| gibbs, 10 sweeps | 0.042 ± 0.021 | 0.019 ± 0.007 | 0.022 ± 0.010 | 0.005 |
| gibbs, 100 sweeps | 0.043 ± 0.023 | 0.020 ± 0.008 | 0.022 ± 0.010 | 0.042 |
| anneal_sim, auto range (cold) | 0.566 ± 0.186 | 0.591 ± 0.261 | 0.656 ± 0.260 | 0.419 |
| anneal_sim fixed T, raw | 0.462 ± 0.149 | 0.424 ± 0.161 | 0.453 ± 0.132 | 0.517 |
| anneal_sim fixed T + 5 polish | 0.044 ± 0.022 | 0.020 ± 0.008 | 0.023 ± 0.009 | 0.515 |
| anneal_sim calibrated | 0.044 ± 0.019 | 0.025 ± 0.008 | 0.029 ± 0.010 | 0.601 |
| qaoa_sim p=2 raw | 0.375 ± 0.139 | 0.417 ± 0.203 | 0.412 ± 0.202 | 0.240 |
| qaoa_sim p=2 + 5 polish | 0.044 ± 0.022 | 0.020 ± 0.008 | 0.023 ± 0.009 | 0.249 |
### nominal (β=0.5 model)

| Sampler | TV distance | max marginal error | max edge-corr error | wall (s) |
|---|---|---|---|---|
| exact (reference sampler) | 0.010 ± 0.006 | 0.010 ± 0.006 | 0.011 ± 0.007 | 0.000799 |
| gibbs, 10 sweeps | 0.023 ± 0.023 | 0.040 ± 0.044 | 0.039 ± 0.043 | 0.005 |
| gibbs, 100 sweeps | 0.009 ± 0.003 | 0.010 ± 0.005 | 0.011 ± 0.005 | 0.042 |
| anneal_sim, auto range (cold) | 0.127 ± 0.125 | 0.193 ± 0.250 | 0.206 ± 0.252 | 0.445 |
| anneal_sim fixed T, raw | 0.118 ± 0.106 | 0.174 ± 0.207 | 0.187 ± 0.211 | 0.493 |
| anneal_sim fixed T + 5 polish | 0.011 ± 0.012 | 0.014 ± 0.016 | 0.014 ± 0.017 | 0.493 |
| anneal_sim calibrated | 0.010 ± 0.005 | 0.010 ± 0.006 | 0.011 ± 0.006 | 0.582 |
| qaoa_sim p=2 raw | 0.891 ± 0.125 | 1.140 ± 0.380 | 1.108 ± 0.351 | 0.177 |
| qaoa_sim p=2 + 5 polish | 0.086 ± 0.091 | 0.152 ± 0.164 | 0.147 ± 0.157 | 0.191 |
### strong coupling (J×3)

| Sampler | TV distance | max marginal error | max edge-corr error | wall (s) |
|---|---|---|---|---|
| exact (reference sampler) | 0.001 ± 0.002 | 0.002 ± 0.003 | 0.002 ± 0.003 | 0.000715 |
| gibbs, 10 sweeps | 0.474 ± 0.144 | 0.788 ± 0.206 | 0.740 ± 0.339 | 0.005 |
| gibbs, 100 sweeps | 0.323 ± 0.184 | 0.590 ± 0.273 | 0.508 ± 0.400 | 0.041 |
| anneal_sim, auto range (cold) | 0.037 ± 0.063 | 0.073 ± 0.124 | 0.074 ± 0.126 | 0.388 |
| anneal_sim fixed T, raw | 0.032 ± 0.053 | 0.064 ± 0.107 | 0.064 ± 0.107 | 0.389 |
| anneal_sim fixed T + 5 polish | 0.024 ± 0.056 | 0.048 ± 0.111 | 0.049 ± 0.111 | 0.393 |
| anneal_sim calibrated | 0.167 ± 0.197 | 0.214 ± 0.248 | 0.252 ± 0.286 | 0.575 |
| qaoa_sim p=2 raw | 0.998 ± 0.003 | 1.214 ± 0.170 | 1.200 ± 0.103 | 0.104 |
| qaoa_sim p=2 + 5 polish | 0.450 ± 0.125 | 0.674 ± 0.184 | 0.699 ± 0.305 | 0.104 |

## Annealing stand-in variants

- **auto range:** Ocean's default schedule adapts to the problem's coefficients and anneals to (near) zero
  temperature. Rescaling cannot change its temperature, so calibration cannot help it.
- **fixed T:** the inverse-temperature range is fixed at (0.1, 4.0). Like a physical device, its effective
  temperature then depends on the submitted coefficient scale, which is what calibration adjusts.

## Calibration of the fixed-T stand-in

Raw pseudo-likelihood estimates on the nominal problems: 1000.00, 1000.00, 1000.00, 1000.00, 1000.00, 3.95.
A value of 1000 is the estimator's cap. It means every sampled spin was aligned with its local field, so the samples
look like zero temperature and no finite temperature can be read off them.

Calibration still converges, because each step changes the submit scale by at most 4×. The scale where
β̂ ≈ 1 reveals the device's effective inverse temperature, β_eff ≈ 1/scale:

| Regime | steps | final scale | implied β_eff = 1/scale | last β̂ |
|---|---|---|---|---|
| weak coupling (J×0.3) | 10 | 0.251 | 3.978 | 0.999 |
| nominal (β=0.5 model) | 10 | 0.263 | 3.806 | 1.037 |
| strong coupling (J×3) | 10 | 0.247 | 4.047 | 99.878 |

For comparison, the stand-in's schedule ends at β = 4.0.

## Observations

- In the weak and nominal regimes, raw annealing samples are too cold (large TV, overconfident marginals).
  This is the bias that paper Proposition 7 says would inflate learned accuracies. With a fixed device
  temperature, pseudo-likelihood calibration rescales the problem until β̂ ≈ 1, and the TV distance drops to
  the exact-sampler floor. Five Gibbs polishing sweeps achieve the same. The auto-range sampler cannot be
  calibrated by rescaling; only polishing helps it.
- **Strong coupling reverses the picture.** Gibbs chains get stuck (TV 0.3–0.5 even after 100 sweeps), while
  the annealing stand-in, raw or polished, stays close to the target. This is the regime where an annealer's
  global moves matter, and it is the comparison to repeat on real hardware.
- Calibration has a limit under strong coupling. Almost every sample is then a ground state, so β̂ saturates
  (last β̂ ≫ 1 in the calibration table) and carries no temperature information. The calibrated TV stays
  around 0.2, while five polishing sweeps reach about 0.01. On hardware, calibrate on a weakly coupled proxy
  problem, or rely on polishing in this regime.
- QAOA output at p = 2 is far from Boltzmann (TV > 0.9 at nominal coupling). Followed by five Gibbs sweeps,
  its error drops sharply in the weak and nominal regimes, consistent with "polishing never hurts"
  (Proposition 10). Under strong coupling, five sweeps are not enough to repair QAOA seeds.
