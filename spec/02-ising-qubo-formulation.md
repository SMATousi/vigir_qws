# 02 — Ising / QUBO Formulation

This document gives the exact, hardware-agnostic encoding of the label
model's problems (doc 01) as Ising and QUBO problems. Every backend in doc 04
consumes the `IsingProblem` defined here.

## 1. Canonical problem object

An `IsingProblem` holds:

| Field | Type | Meaning |
|-------|------|---------|
| `h` | map site → float | Local fields |
| `J` | map (i,k), i<k → float | Couplings (upper-triangular, no self-loops) |
| `offset` | float | Constant energy term |
| `kind` | enum `{posterior, prior}` | Which distribution it encodes (doc 01 §3.2 / §3.4) |
| `fixed` | map site → ±1 | Variables already fixed by preprocessing or boundary conditions (§6) |
| `meta` | dict | Image id, site-to-pixel map, tile id, θ snapshot, etc. |

Energy, in our convention:

$$
E(y) = \text{offset} - \sum_i h_i y_i - \sum_{i<k} J_{ik}\, y_i y_k,
\qquad p(y) \propto e^{-E(y)} .
$$

The label model is written in natural units, so the **target sampling
distribution is at inverse temperature exactly 1**.

## 2. Mapping the tasks

| Task (doc 01 §4) | Ising problem | Required solver output |
|------------------|---------------|------------------------|
| T1 MAP | Posterior $(h, J)$ | Lowest-energy state found |
| T2 Marginals | Posterior $(h, J)$ | Samples ≈ Boltzmann at β = 1, then $\hat\mu_i = \tfrac12(1 + \overline{y_i})$ |
| T3 Learning, positive phase | Posterior $(h, J)$ for each image in the minibatch | Samples ≈ Boltzmann at β = 1 |
| T3 Learning, negative phase | Prior $(h^{\text{pr}}, J^{\text{pr}})$ | Samples ≈ Boltzmann at β = 1 |

The coefficient formulas are in doc 01 §3.2 and §3.4.

## 3. Ising → QUBO

Substitute $y_i = 2x_i - 1$ with $x_i \in \{0,1\}$:

$$
E(x) = c + \sum_i Q_{ii}\, x_i + \sum_{i<k} Q_{ik}\, x_i x_k
$$

$$
Q_{ii} = -2h_i + 2\sum_{k \in \mathcal{N}(i)} J_{ik},\qquad
Q_{ik} = -4 J_{ik},\qquad
c = \text{offset} + \sum_i h_i - \sum_{i<k} J_{ik}
$$

The inverse map is $x_i = (1 + y_i)/2$, so $x_i = 1$ is foreground. The
conversion MUST be energy-preserving. Unit tests MUST check this by full
enumeration on random problems with up to 12 sites.

## 4. Ising → gate-based cost Hamiltonian

$$
H_C = \text{offset}\cdot I - \sum_i h_i Z_i - \sum_{i<k} J_{ik} Z_i Z_k
$$

**Basis convention:** $y_i$ is the eigenvalue of $Z_i$. Measured bit $0$ maps
to $y_i = +1$ (foreground) and bit $1$ to $y_i = -1$. The QAOA backend owns
this conversion, and nothing outside the backend may assume a bit order.
Mind the little-endian qubit ordering used by some SDKs.

## 5. Sign conventions at hardware boundaries

SDKs disagree on signs. Each backend MUST convert at its boundary and nowhere
else.

| Target | Their energy | Conversion from ours |
|--------|-------------|----------------------|
| D-Wave Ocean `BinaryQuadraticModel` (SPIN) | $\sum h'_i s_i + \sum J'_{ik} s_i s_k$ | $h' = -h$, $J' = -J$ |
| D-Wave Ocean BQM (BINARY) | $\sum Q'_{ik} x_i x_k$ | $Q' = Q$ (§3), offset carried separately |
| Qiskit `SparsePauliOp` | operator as written | Coefficients as in §4 |

Conventions MUST be checked against the SDK version actually pinned. A
round-trip test (ours → SDK → energy of random states) is mandatory for each
backend.

## 6. Preprocessing and conditioning

### 6.1 Strong-field fixing (exact)

If $|h_i| > \sum_k |J_{ik}|$, then every ground state has $y_i = \operatorname{sign}(h_i)$.
For **MAP** the builder MAY fix such sites and fold them into neighbours'
fields (§6.2). On confident images this can remove most sites. For
**sampling** this fixing is **not exact**; it MAY be used only as an
explicitly flagged approximation.

### 6.2 Conditioning on fixed sites

Fixing $y_k = \bar y_k$ for $k \in F$ leaves a reduced problem on $S = V \setminus F$:

$$
h'_i = h_i + \sum_{k \in F} J_{ik}\,\bar y_k,\qquad
\text{offset}' = \text{offset} - \sum_{k\in F} h_k \bar y_k - \sum_{k<l \in F} J_{kl}\,\bar y_k \bar y_l
$$

The same rule gives the boundary conditions for tiles in decomposition
(doc 04 §5). Conditioning on fixed values is exact for both MAP and sampling
of the *conditional* distribution.

### 6.3 Persistency (optional)

For MAP, roof duality (QPBO) MAY be run to fix the persistent variables of
frustrated problems before sending the rest to quantum hardware.

## 7. Scaling and precision

Hardware accepts bounded coefficients, $|h| \le h_{\max}$ and $|J| \le J_{\max}$,
and adds analog control noise. The bounds MUST be read from the device's
reported properties at runtime, never hard-coded.

- **Scale factor** $\kappa = \max\big(\max_i |h_i| / h_{\max},\ \max_{ik} |J_{ik}| / J_{\max}\big)$.
  Submit $h/\kappa$ and $J/\kappa$.
- **MAP**: scaling does not move the argmin. It does reduce the gaps between
  states relative to noise. The builder MUST log the **dynamic range**
  $\max|\text{coef}| / \min_{\text{coef}\neq 0}|\text{coef}|$ and warn when it
  exceeds a configurable threshold.
- **Sampling**: the device samples roughly from $\exp(-\beta_{\text{eff}} E/\kappa)$,
  with an unknown $\beta_{\text{eff}}$. The backend MUST report $\kappa$, and
  the learning loop MUST estimate and correct the effective temperature
  (doc 03 §4).

## 8. Worked example

Three sites in a line, with edges (1,2) and (2,3), $\beta w = 0.5$ on both,
$b = 0$, and symmetric LFs with $\theta = (1.0, 0.5)$:

| Site | $\Lambda_{i1}$ | $\Lambda_{i2}$ | $h_i = \sum_j \theta_j \Lambda_{ij}$ |
|------|---|---|---|
| 1 | +1 | +1 | 1.5 |
| 2 | +1 | −1 | 0.5 |
| 3 | −1 | 0 | −1.0 |

Ising: $J_{12} = J_{23} = 0.5$. The MAP state is $y = (+1, +1, -1)$ with
$E = -3.0$. The runners-up are $(+,+,+)$ and $(+,-,-)$, both at $E = -2.0$.

QUBO (§3): $Q_{11} = -2$, $Q_{22} = 1$, $Q_{33} = 3$, $Q_{12} = Q_{23} = -2$,
$c = 0$. Check: $x = (1,1,0)$ gives $-2 + 1 - 2 = -3$ ✓.

This example MUST be a fixture in the test suite.
