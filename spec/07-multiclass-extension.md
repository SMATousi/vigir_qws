# 07 — Multi-class Extension (Planned, v2)

**Status:** design sketch only. Its purpose is to make sure the binary
design in docs 01–05 does not block this extension.

## 1. Model

- Labels $y_i \in \{1,\dots,K\}$; LF votes $\Lambda_{ij} \in \{0, 1,\dots,K\}$, where $0$ means abstain.
- Unary potential (symmetric): $\psi_j(\lambda, y) = \alpha_j\mathbb{1}[\lambda\ne0] + \theta_j\mathbb{1}[\lambda = y]$.
- Smoothness: a Potts prior $\beta w_{ik}\mathbb{1}[y_i = y_k]$, plus class biases $b_c$.
- Pairwise LFs keep their same/different meaning, contributing $\theta^P_r \Pi_{ikr}(2\mathbb{1}[y_i=y_k]-1)$.

MAP for $K \ge 3$ is NP-hard even when $\beta \ge 0$, so multi-class is a
genuine quantum target even **without** frustration.

## 2. One-hot QUBO encoding (default)

Use binary variables $x_{ic} = \mathbb{1}[y_i = c]$, so there are $NK$ logical variables:

$$
E(x) = -\sum_{i,c}\Big(b_c + \sum_j \theta_j\mathbb{1}[\Lambda_{ij}=c]\Big)x_{ic}
\;-\;\sum_{(i,k)}\tilde J_{ik}\sum_c x_{ic}x_{kc}
\;+\;A\sum_i\Big(1-\sum_c x_{ic}\Big)^2
$$

- $\tilde J_{ik}$ collects the smoothness and pairwise-LF terms: same-label
  agreement with weight $\beta w_{ik}+2\sum_r\theta^P_r\Pi_{ikr}$, up to constants.
- The penalty $A$ MUST be large enough that no infeasible state beats the
  best feasible state. The baseline bound is $A >$ (the largest energy
  decrease available from violating one site's one-hot constraint).
  Taking that bound too loosely ruins the dynamic range on hardware
  (doc 02 §7), so $A$ is a tuned parameter, and the fraction of infeasible
  reads MUST be reported.
- Decoding: an infeasible site is repaired by choosing its argmax-field class.

## 3. Alternative encodings

| Encoding | Qubits per site | Notes |
|----------|-----------------|-------|
| One-hot | $K$ | Simple; needs the penalty; $K(K-1)/2$ intra-site couplers |
| Domain-wall | $K-1$ | Fewer qubits, and reported to behave better on annealers; more complex decoding |
| Binary (log) | $\lceil\log_2 K\rceil$ | Best for gate-based QAOA; costs become higher-order and need quadratization for annealers |

The gate-based backend MAY use a constraint-preserving XY mixer, which avoids
penalties entirely.

## 4. Impact on the binary design

To keep this extension possible, the binary implementation MUST:
- keep `IsingProblem` generic (arbitrary variable ids, not bare site ids), so
  $(i, c)$ pairs fit;
- keep the encoding/decoding step as a separate layer between `model` and
  `solvers`;
- avoid assuming $\pm 1$ labels outside the binary encoder.

Learning works the same way as in doc 03: the posterior and prior over
one-hot variables are QUBOs, and the sufficient statistics are per-class
indicators.
