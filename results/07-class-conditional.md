# 07 — Class-conditional labeling functions

LF potentials whose accuracy and coverage differ between foreground and background (decision R9), in the
parametrization ψ_j(λ,y) = α_j·1[λ≠0] + θ_j·λy + ω_j·1[λ≠0]·y + ζ_j·λ ([spec/09 §3](../spec/09-implementation-notes.md)).
Here ω is class-dependent coverage and ζ is a vote bias. The model terms of the gradient now depend on the
prior marginals P(yᵢ = ±1), so the trainer samples the prior Ising model, whose field gains
Σ_c ½ log(Z_c(+1)/Z_c(−1)) (paper Proposition 2).

## (a) Identifiability with exact moments

The coupling is β = 0, so posterior and prior moments are exact closed forms and there is no sampling noise.
There are 200,000 sites, with labels drawn from the model's own prior marginal.

| parameter | true | learned |
|---|---|---|
| theta_0 | 1.000 | 0.973 |
| theta_1 | 0.600 | 0.597 |
| theta_2 | 0.300 | 0.303 |
| theta_3 | 0.800 | 0.799 |
| alpha_0 | 0.500 | 0.523 |
| alpha_1 | 0 | -0.003 |
| alpha_2 | -0.500 | -0.494 |
| alpha_3 | 0.200 | 0.203 |
| omega_0 | 0.800 | 0.786 |
| omega_1 | -0.600 | -0.600 |
| omega_2 | 0 | -0.002 |
| omega_3 | 0.400 | 0.396 |
| zeta_0 | 0.300 | 0.322 |
| zeta_1 | 0 | 0.007 |
| zeta_2 | -0.400 | -0.398 |
| zeta_3 | 0 | 0.004 |

## (b) Coupled model: symmetric vs. class-conditional fit

- **Data:** masks from the full prior marginal (Ising with field b + prior field, β = 0.3), so the generative
  process is exactly the class-conditional label model. One pairwise LF; 24 images of 16×16.
- **Fit:** both models trained with persistent Gibbs chains for 120 epochs. MAP masks by
  simulated annealing.

| fitted model | max |θ−θ*| | max |ω−ω*| | max |ζ−ζ*| | IoU (MAP) | train time (s) |
|---|---|---|---|---|---|
| symmetric model | 0.391 | 0.800 | 0.400 | 0.985 | 41.825 |
| class-conditional model | 0.159 | 0.398 | 0.174 | 0.990 | 43.979 |

## Observations

- With exact moments, every parameter (θ, α, ω, ζ) is recovered essentially exactly, so the
  class-conditional model is identifiable.
- In the coupled, finite-data setting, ζ (vote bias) is recovered well. ω (class-dependent coverage) is
  recovered only partially, because its signal is the small difference in coverage between classes, estimated
  from a few thousand sites with sampling noise. More data, or longer training, narrows the gap.
- On segmentation quality (IoU), the two models are close on this data. The class-conditional model mainly
  matters for calibrated soft labels and for LFs with strongly asymmetric behaviour.
