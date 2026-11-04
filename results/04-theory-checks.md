# 04 — Numerical checks of the paper's propositions

Numerical checks of the paper's quantitative claims, using exact samplers so that only the claim under
test is exercised.

## Proposition 6: MAP labels overestimate accuracy

Three conditionally independent LFs, each correct with probability a. The table compares each LF's
empirical agreement with the majority (MAP) label over 200,000 sites against the predicted 1 − a(1 − a).

| true accuracy a | empirical agreement with MAP | predicted 1 − a(1−a) | bias (1−a)² |
|---|---|---|---|
| 0.600 | 0.760 | 0.760 | 0.160 |
| 0.700 | 0.790 | 0.790 | 0.090 |
| 0.800 | 0.840 | 0.840 | 0.040 |
| 0.900 | 0.911 | 0.910 | 0.010 |

## Proposition 8: the statistical error does not depend on the configuration space

Maximum error of the ⟨yᵢ⟩ estimates from S exact samples (mean over 20 repetitions) on an n = 12 posterior,
against the Hoeffding term √(2 log(2/δ)/S) with δ = 0.05.

| S | mean max marginal error | Hoeffding bound (δ=0.05) |
|---|---|---|
| 100 | 0.042 | 0.272 |
| 1000 | 0.018 | 0.086 |
| 10000 | 0.004 | 0.027 |

## Proposition 9: the pseudo-likelihood temperature estimator is consistent

Samples are drawn exactly at inverse temperature β₀ from a random n = 10 Ising model. β̂ is the maximum
pseudo-likelihood estimate.

| β₀ | samples S | β̂ |
|---|---|---|
| 0.250 | 500 | 0.224 |
| 0.250 | 5000 | 0.254 |
| 0.250 | 50000 | 0.247 |
| 0.500 | 500 | 0.484 |
| 0.500 | 5000 | 0.504 |
| 0.500 | 50000 | 0.502 |
| 1.000 | 500 | 0.977 |
| 1.000 | 5000 | 1.007 |
| 1.000 | 50000 | 0.996 |
| 2.000 | 500 | 2.091 |
| 2.000 | 5000 | 2.031 |
| 2.000 | 50000 | 1.997 |
| 4.000 | 500 | 1000.000 |
| 4.000 | 5000 | 4.464 |
| 4.000 | 50000 | 4.015 |

β̂ converges to β₀ as S grows (Proposition 9). The single 1000 entry (β₀ = 4, S = 500) is the estimator's cap.
At low temperature with few samples, every spin can be aligned with its local field. The pseudo-likelihood
then increases without bound in β, and the finite-sample maximizer does not exist. With more samples the
estimate is accurate again.

## Verified elsewhere by the test suite (`pytest`)

- Propositions 1–3 and Lemma 1 / Theorem 1: the posterior builder matches the brute-force joint, and the
  gradient matches finite differences of the exact marginal likelihood (`tests/test_model.py`).
- Proposition 4: Ising ↔ QUBO ↔ Ocean ↔ Pauli energy identities by enumeration (`tests/test_ising.py`).
  The Qiskit circuit's probabilities equal our statevector QAOA (`tests/test_solvers.py`).
- Proposition 5: conditioning exactness, strong-field fixing preserves minimizers, and block-CD monotonicity
  (`tests/test_ising.py`, `tests/test_decompose_eval.py`).
- Proposition 10: Gibbs polishing preserves the exact distribution and moves bad seeds toward it
  (`tests/test_solvers.py`).
