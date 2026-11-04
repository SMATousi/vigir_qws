# 06 — Learning the smoothness β and class bias b

Can the smoothness β and the class bias b be learned from LF votes alone (decision R8)? This is the
negative phase of the gradient, sampled from the prior Ising model (doc 03 §2; paper Theorem 1 remark).

## Setup

- **Data:** masks drawn from the prior Ising model itself (`mask_source="ising"`, 16×16 grids, uniform
  weights), at a known (b, β). The whole generative process is therefore exactly the label model, which makes
  this a pure identifiability test.
- **LFs:** 12 images per run, the 4 unary LFs of report 01, and one pairwise LF with θᴾ = 0.9.
- **Training:** warm start at β = 0.1, b = 0, then 60 epochs of Adam on all parameters with
  persistent Gibbs chains (prior moments shared per graph shape). Seeds: [0, 1, 2].
- **Reference point:** the 2-D Ising critical point is β_c = ½ ln(1 + √2) ≈ 0.441.

## Results (mean ± std over seeds)

| true β | true b | learned β | learned b | max |θ − θ*| |
|---|---|---|---|---|
| 0.200 | 0 | 0.205 ± 0.003 | -0.003 ± 0.012 | 0.047 ± 0.020 |
| 0.350 | 0.100 | 0.339 ± 0.019 | 0.120 ± 0.029 | 0.055 ± 0.013 |
| 0.600 | -0.100 | 0.555 ± 0.032 | -0.434 ± 0.024 | 0.031 ± 0.008 |

## Observations

- β is recovered well below and above the critical point.
- b is well identified only at weak coupling. In the ordered phase (β > β_c), each image is spontaneously
  dominated by one class, so 12 images carry little information about the bias, and the estimate is noisy.
  Use more images, or fix b, when coupling is strong.
- Unary accuracies θ stay accurate while β and b are learned jointly.
