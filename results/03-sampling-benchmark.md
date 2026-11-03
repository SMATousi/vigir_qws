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
| exact (reference sampler) | 0.043 ± 0.012 | 0.023 ± 0.014 | 0.027 ± 0.013 | 0.000717 |
| gibbs, 10 sweeps | 0.036 ± 0.009 | 0.016 ± 0.007 | 0.021 ± 0.007 | 0.005 |
| gibbs, 100 sweeps | 0.039 ± 0.012 | 0.017 ± 0.009 | 0.026 ± 0.013 | 0.043 |
| anneal_sim, auto range (cold) | 0.542 ± 0.132 | 0.682 ± 0.264 | 0.715 ± 0.260 | 0.426 |
| anneal_sim fixed T, raw | 0.430 ± 0.118 | 0.457 ± 0.161 | 0.491 ± 0.176 | 0.525 |
| anneal_sim fixed T + 5 polish | 0.041 ± 0.015 | 0.020 ± 0.007 | 0.022 ± 0.006 | 0.516 |
| anneal_sim calibrated | 0.039 ± 0.014 | 0.015 ± 0.008 | 0.020 ± 0.007 | 0.606 |
| qaoa_sim p=2 raw | 0.490 ± 0.201 | 0.625 ± 0.354 | 0.624 ± 0.348 | 0.214 |
| qaoa_sim p=2 + 5 polish | 0.041 ± 0.015 | 0.020 ± 0.007 | 0.022 ± 0.006 | 0.218 |
### nominal (β=0.5 model)

| Sampler | TV distance | max marginal error | max edge-corr error | wall (s) |
|---|---|---|---|---|
| exact (reference sampler) | 0.009 ± 0.004 | 0.010 ± 0.008 | 0.011 ± 0.008 | 0.00074 |
| gibbs, 10 sweeps | 0.010 ± 0.006 | 0.013 ± 0.013 | 0.015 ± 0.010 | 0.005 |
| gibbs, 100 sweeps | 0.006 ± 0.002 | 0.006 ± 0.004 | 0.006 ± 0.003 | 0.040 |
| anneal_sim, auto range (cold) | 0.092 ± 0.029 | 0.142 ± 0.051 | 0.148 ± 0.059 | 0.434 |
| anneal_sim fixed T, raw | 0.092 ± 0.029 | 0.142 ± 0.050 | 0.148 ± 0.059 | 0.476 |
| anneal_sim fixed T + 5 polish | 0.008 ± 0.003 | 0.010 ± 0.006 | 0.010 ± 0.006 | 0.489 |
| anneal_sim calibrated | 0.007 ± 0.004 | 0.007 ± 0.006 | 0.008 ± 0.007 | 0.578 |
| qaoa_sim p=2 raw | 0.917 ± 0.058 | 1.109 ± 0.184 | 1.145 ± 0.234 | 0.189 |
| qaoa_sim p=2 + 5 polish | 0.042 ± 0.041 | 0.058 ± 0.059 | 0.065 ± 0.074 | 0.194 |
### strong coupling (J×3)

| Sampler | TV distance | max marginal error | max edge-corr error | wall (s) |
|---|---|---|---|---|
| exact (reference sampler) | 0.002 ± 0.003 | 0.003 ± 0.006 | 0.003 ± 0.006 | 0.00075 |
| gibbs, 10 sweeps | 0.467 ± 0.103 | 0.777 ± 0.079 | 0.777 ± 0.223 | 0.006 |
| gibbs, 100 sweeps | 0.375 ± 0.120 | 0.628 ± 0.150 | 0.646 ± 0.302 | 0.043 |
| anneal_sim, auto range (cold) | 0.057 ± 0.074 | 0.099 ± 0.141 | 0.113 ± 0.149 | 0.417 |
| anneal_sim fixed T, raw | 0.052 ± 0.071 | 0.094 ± 0.138 | 0.104 ± 0.142 | 0.405 |
| anneal_sim fixed T + 5 polish | 0.014 ± 0.030 | 0.023 ± 0.045 | 0.029 ± 0.060 | 0.402 |
| anneal_sim calibrated | 0.180 ± 0.230 | 0.238 ± 0.306 | 0.280 ± 0.366 | 0.567 |
| qaoa_sim p=2 raw | 0.992 ± 0.014 | 1.350 ± 0.169 | 1.161 ± 0.074 | 0.125 |
| qaoa_sim p=2 + 5 polish | 0.487 ± 0.184 | 0.762 ± 0.246 | 0.702 ± 0.391 | 0.134 |

## Annealing stand-in variants

- **auto range:** Ocean's default schedule adapts to the problem's coefficients and anneals to (near) zero
  temperature. Rescaling cannot change its temperature, so calibration cannot help it.
- **fixed T:** the inverse-temperature range is fixed at (0.1, 4.0). Like a physical device, its effective
  temperature then depends on the submitted coefficient scale, which is what calibration adjusts.

## Calibration of the fixed-T stand-in

Raw pseudo-likelihood estimates on the nominal problems: 1000.00, 1000.00, 1000.00, 1000.00, 1000.00, 1000.00.
A value of 1000 is the estimator's cap. It means every sampled spin was aligned with its local field, so the samples
look like zero temperature and no finite temperature can be read off them.

Calibration still converges, because each step changes the submit scale by at most 4×. The scale where
β̂ ≈ 1 reveals the device's effective inverse temperature, β_eff ≈ 1/scale:

| Regime | steps | final scale | implied β_eff = 1/scale | last β̂ |
|---|---|---|---|---|
| weak coupling (J×0.3) | 10 | 0.248 | 4.031 | 1.001 |
| nominal (β=0.5 model) | 10 | 0.242 | 4.134 | 0.961 |
| strong coupling (J×3) | 10 | 0.248 | 4.039 | 100.061 |

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
