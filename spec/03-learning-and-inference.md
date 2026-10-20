# 03 — Joint Learning and Inference

This document specifies the hybrid loop. A classical optimizer updates the
label-model parameters, and an Ising solver (quantum or classical) supplies
the expectations that the gradient needs (doc 01 §4).

## 1. Overview

```
            ┌──────────────── classical ────────────────┐
 images ──► apply LFs ──► Λ, Π ──► build posterior Ising ─┼──► solver.sample ──► ⟨y_i⟩, ⟨y_i y_k⟩ (posterior)
                                                           │
                  graph shape ──► build prior Ising ───────┼──► solver.sample ──► ⟨y_i⟩, ⟨y_i y_k⟩ (prior)
                                                           │
                      gradient = posterior stats − model stats ──► update θ ──► repeat
            └──────────────────────────────────────────┘
 after training:  posterior Ising ──► solver.minimize (T1: hard mask)
                                  └─► solver.sample   (T2: soft mask)
```

## 2. Gradients per parameter

Write $\langle\cdot\rangle_{\text{post}}$ for expectations under
$p_\theta(y \mid \Lambda, \Pi)$ and $\langle\cdot\rangle_{\text{pr}}$ for
expectations under the prior marginal $p_\theta(y)$. Sums run over the sites
and edges of every image in the minibatch.

| Parameter | Positive phase (data) | Negative phase (model) |
|-----------|----------------------|------------------------|
| $\theta_j$ (accuracy) | $\sum_i \Lambda_{ij}\langle y_i\rangle_{\text{post}}$ | $\sum_i \mathbb{E}[\Lambda_{ij} y_i]$ |
| $\alpha_j$ (propensity) | $\sum_i \mathbb{1}[\Lambda_{ij}\ne0]$ | $\sum_i \mathbb{E}[\mathbb{1}[\Lambda_{ij}\ne0]]$ |
| $\gamma_c$ (correlation) | $\sum_i \phi_c(\Lambda_{ic})$ | $\sum_i \mathbb{E}[\phi_c(\Lambda_{ic})]$ |
| $\theta^P_r$ (pairwise accuracy) | $\sum_{(i,k)\in E_r}\Pi_{ikr}\langle y_iy_k\rangle_{\text{post}}$ | $\sum_{(i,k)\in E_r}\mathbb{E}[\Pi_{ikr}\,y_iy_k]$ |
| $b$ (class bias) | $\sum_i\langle y_i\rangle_{\text{post}}$ | $\sum_i\langle y_i\rangle_{\text{pr}}$ |
| $\beta$ (smoothness) | $\sum_{E} w_{ik}\langle y_iy_k\rangle_{\text{post}}$ | $\sum_{E} w_{ik}\langle y_iy_k\rangle_{\text{pr}}$ |

How the negative-phase LF terms are computed:

- **Symmetric potentials.** The LF terms have **closed forms** that do not
  depend on $y$ (doc 01 §3.4).
- **Class-conditional potentials, or LF cliques.** The model moment is
  $\mathbb{E}[\cdot] = \sum_{y\in\pm1} p_{\text{pr}}(y_i{=}y)\,\mathbb{E}[\cdot \mid y_i{=}y]$.
  The conditional part is exact (a sum over at most $3^{|c|}$ terms), and
  $p_{\text{pr}}(y_i)$ comes from prior samples.

### 2.1 Key simplification for v1

Take symmetric LF potentials, with $b$ and $\beta$ fixed as hyperparameters.
Then **learning needs only posterior sampling**, because every negative-phase
term is closed form. The only problem sent to the quantum solver is the
posterior Ising model of each image. v1 MUST support this mode first.

### 2.2 Prior problem reuse

The prior Ising model depends only on the graph and on $(b, \beta,$ the LF
parameters$)$, never on the votes. All images that share a graph shape (for
example, equal-size pixel grids) share one prior problem per step. The
implementation SHOULD sample it once per step and reuse the samples.

## 3. Algorithm

```
Input: images {I_n}, LFs, graph builder, solver, hyperparameters
1.  Λ_n, Π_n ← apply LFs to every image (cached, sparse int8)
2.  θ ← warm start: fit the independent label model (β = 0, closed-form/EM),
        so the initial θ_j > 0 on average (doc 01 §6)
3.  for epoch = 1..T:
4.     for minibatch B ⊂ {1..D}:
5.        P_n ← build_posterior_ising(θ, Λ_n, Π_n, G_n)     for n ∈ B
6.        S_n ← solver.sample(P_n, num_samples=S)           (batched in one call where supported, doc 04 §6)
7.        if learning b or β (or class-conditional): R ← solver.sample(build_prior_ising(θ, G), S)
8.        g ← positive_stats(S_n) − negative_stats(θ, R)    (§2)
9.        θ ← optimizer_step(θ, g − ∇reg(θ)); project(θ)    (β ≥ 0; mean θ_j > 0)
10. Inference, per image:
11.    ŷ_n ← solver.minimize(P_n)                           (T1)
12.    μ_n ← mean(solver.sample(P_n, S'))                   (T2)
Output: θ, {ŷ_n}, {μ_n}
```

- `optimizer_step` defaults to Adam, with L2 regularization on $\theta$.
- **EM alternative.** With symmetric potentials, the M-step for each
  $\theta_j$ solves the scalar equation
  $N_j\,\frac{2e^{\alpha_j}\sinh\theta_j}{1+2e^{\alpha_j}\cosh\theta_j} = \sum_i \Lambda_{ij}\mu_i$
  ($N_j$ = number of sites), which is a 1-D root find. EM MAY be offered as a
  solver-call-efficient mode, because it needs one round of posterior
  sampling per M-step.

## 4. Effective temperature and sample fidelity

Quantum samplers do not return exact $\beta = 1$ Boltzmann samples. The loop
MUST handle this explicitly.

1. **Estimate.** For each returned sample set, estimate the effective inverse
   temperature $\hat\beta$ relative to the submitted problem. The default is
   maximum pseudo-likelihood over the scalar $\beta'$:
   $\max_{\beta'} \sum_{s}\sum_i \log\sigma\big(2\beta' y^{(s)}_i (h_i + \textstyle\sum_k J_{ik} y^{(s)}_k)\big)$.
   This is classical and cheap.
2. **Correct the scale.** Adjust the submit-time scale $\kappa$ (doc 02 §7)
   across iterations so that $\hat\beta/\kappa \to 1$. The correction is
   limited by the device's coefficient ranges, and the backend MUST report
   when the target is unreachable.
3. **Polish (optional, recommended).** Run $k$ sweeps of classical Gibbs at
   $\beta = 1$, starting from each quantum sample. This *quantum-seeded MCMC*
   makes the sample distribution converge to the exact target as $k$ grows.
   The quantum device then supplies diverse, low-energy starting points that
   are hard to reach by local moves.
4. **Report.** Every experiment MUST report results both raw ($k = 0$) and
   polished, so the quantum contribution can be measured.

QAOA output is not a Boltzmann distribution. In v1, QAOA serves T1 (MAP) and
seeds polished sampling (step 3). Using it directly as a sampler is a research
item (see open questions).

## 5. Budget and variance

| Knob | Meaning | Default (to tune) |
|------|---------|-------------------|
| `S` | Samples per problem per step | 100–1000 |
| `|B|` | Images per minibatch | Enough to fill one device call (doc 04 §6) |
| `k` | Gibbs polish sweeps | 0 (raw), 1, 10 |
| `T` | Epochs | Until $\lVert\Delta\theta\rVert$ < tol or budget exhausted |

The classical baseline sampler uses **persistent chains** (PCD-style), so its
comparison against quantum samplers runs at equal wall-clock time and equal
sample budget.

## 6. LF correlation structure

The LF outputs $\Lambda$ are observed, so correlation-structure learning is
entirely classical. v1 accepts **user-declared** cliques. Optionally, the
structure is learned with L1-regularized pseudo-likelihood on $\Lambda$, in the
style of Bach et al. (2017). This happens once before step 3 and is not a
quantum workload.

## 7. Outputs and downstream use

- **Hard masks** $\hat y_n$, and **soft masks** $\mu_n \in [0,1]^N$ projected
  back from sites to pixels.
- **Learned parameters** $\theta$: per-LF accuracies and propensities, and
  $b$, $\beta$.
- Soft masks are meant for a downstream segmentation network trained with a
  noise-aware loss (expected cross-entropy under $\mu$). That network is
  outside this repo's core.

## 8. Diagnostics and stopping

The exact log-likelihood is intractable, so the loop logs:
- Parameter trajectories and $\lVert\Delta\theta\rVert$.
- The pseudo-likelihood of the posterior samples, and $\hat\beta$ per call (§4).
- Mean energy and the best energy found per problem.
- On synthetic data (doc 06): the error between $\theta$ and the generating
  parameters.
- On the labelled validation split: IoU/Dice of $\hat y$, and calibration of $\mu$.
