# Quantum Data Programming for Image Segmentation: An Ising Formulation of Structured Label Models

**Draft v0.1. Theory only; experiments are pending.**
**Authors:** [to be added]

> Notes for authors: every claim marked *(verify)* and every reference must
> be checked before submission. Proofs are written out in full so they can be
> moved to an appendix when this is converted to LaTeX.

---

## Abstract

Data programming trains models from noisy, programmatic labeling functions
(LFs) in place of hand labels. A generative label model infers LF accuracies
without ground truth. Classical formulations treat data points as
independent, which suits classification but ignores the spatial structure
that defines image segmentation.

We introduce a *structured* label model for binary segmentation that
combines:
- unary LFs, which vote on sites;
- pairwise LFs, which vote on whether two sites share a label;
- a contrast-sensitive Markov random field prior.

We prove that both the posterior over masks and the prior marginal over
masks are Ising models, with fields and couplings given in closed form by
the LF votes and parameters. As a result:
1. maximum a posteriori (MAP) segmentation is an Ising ground-state problem;
2. maximum-marginal-likelihood learning of LF accuracies reduces to
   estimating low-order moments of Boltzmann distributions over these
   Ising models.

For symmetric LF potentials, the model-side term of the gradient is
available in closed form. The only hard computational primitive left is
sampling the per-image posterior Ising model at unit temperature.

We further show that:
- learning from zero-temperature (MAP) labels systematically overestimates
  LF accuracies;
- a sampler that runs too cold is equivalent to evaluating the data term at
  inflated parameters;
- the gradient error splits into a sampler-bias term, bounded by total
  variation distance, and a statistical term that does not depend on the
  $2^N$ size of the configuration space.

These results motivate a hybrid algorithm in which a quantum annealer serves
as a temperature-calibrated Boltzmann sampler for learning, and as a
ground-state solver for inference. Gate-based QAOA is an alternative backend
for small subproblems. Finally, we map where the formulation is classically
tractable (ferromagnetic MAP via graph cuts) and where it is a genuine
target for quantum hardware: frustrated couplings induced by pairwise LFs,
partition-function-dependent learning, and multi-class segmentation.

---

## 1. Introduction

Pixel-level annotation is among the most expensive kinds of supervision in
computer vision. Data programming [Ratner et al. 2016; Ratner et al. 2017]
replaces hand labels with *labeling functions*: heuristics, rules, weak
detectors, or annotator scribbles that each emit a noisy vote or abstain. A
generative *label model* learns the LFs' accuracies from their agreements
and disagreements alone, and outputs probabilistic training labels.

Standard label models assume data points are conditionally independent. For
segmentation this assumption is wrong in an essential way: the labels of
neighbouring sites are strongly dependent, and spatial coherence is what
separates a segmentation from a collection of per-pixel guesses. Adding this
dependence through a Markov random field (MRF) [Geman & Geman 1984] makes
the label model *structured*. It also makes its core computations
intractable in general. MAP inference becomes NP-hard once couplings have
mixed signs [Barahona 1982], and the partition function needed for learning
is #P-hard to compute exactly.

The structured label model has exactly the algebraic form that quantum
annealers implement natively [Kadowaki & Nishimori 1998; Johnson et al.
2011] and that gate-based variational algorithms target [Farhi et al. 2014].
This paper develops that connection with mathematical precision.

**Contributions.**
1. A structured data-programming label model for binary segmentation, with
   unary and pairwise LFs and a contrast-sensitive smoothness prior (§3).
2. Exact reductions of the posterior and the prior marginal to Ising models,
   and of MAP segmentation to an Ising ground-state problem (§4,
   Propositions 1–3).
3. A learning theorem: for symmetric LF potentials, every LF-parameter
   gradient needs only first and edge moments of the per-image posterior
   Ising models (Theorem 1).
4. An analysis of sampler quality for learning:
   - zero-temperature (MAP) labels overestimate LF accuracies
     (Proposition 6);
   - miscalibrated sampler temperature is equivalent to parameter inflation
     (Proposition 7);
   - gradient error decomposes into a total-variation bias plus a
     dimension-free statistical term (Proposition 8) (§5).
5. A hybrid classical–quantum algorithm with temperature calibration,
   quantum-seeded MCMC polishing, and exact decomposition for problems
   larger than the hardware (§6–7).
6. A complexity map stating where quantum resources can, and cannot,
   plausibly help (§4.4).

We make no claims of empirical quantum advantage here. The paper fixes the
mathematical object and the evaluation protocol in advance of hardware
experiments (§9).

---

## 2. Background

### 2.1 Data programming

Let $\lambda_1,\dots,\lambda_m$ be LFs that emit votes in $\{-1,0,+1\}$, where
$0$ means abstain. A label model $p_\theta(\Lambda, y)$ couples the observed
votes $\Lambda$ with the latent true labels $y$. It is fitted by maximizing the
marginal likelihood $\sum_y p_\theta(\Lambda, y)$
[Dawid & Skene 1979; Ratner et al. 2016]. Correlation structure among LFs can
be learned from $\Lambda$ alone [Bach et al. 2017].

### 2.2 Ising models and Boltzmann distributions

An Ising model on spins $s \in \{-1,+1\}^N$ has energy
$E(s) = -\sum_i h_i s_i - \sum_{i<k} J_{ik} s_i s_k$ and Boltzmann distribution
$p_\beta(s) = e^{-\beta E(s)}/Z_\beta$. Couplings with $J_{ik} \ge 0$ are
*ferromagnetic*. Mixed signs can produce *frustration*.

For binary variables, ferromagnetic energies are submodular, and their
minimization reduces to an s–t minimum cut [Greig et al. 1989; Kolmogorov &
Zabih 2004]. Ground states of general spin glasses on non-planar graphs are
NP-hard [Barahona 1982]. Computing $Z$ is #P-hard in general, though
approximation schemes exist for ferromagnetic cases
[Jerrum & Sinclair 1993].

### 2.3 Quantum annealing and QAOA

A quantum annealer evolves under
$H(t) = -A(t)\sum_i \sigma^x_i + B(t)\,H_{\text{Ising}}$, with $A$ decreasing
and $B$ increasing over the anneal [Kadowaki & Nishimori 1998]. In practice,
readouts are approximately Boltzmann-distributed at an effective,
instance-dependent temperature [Amin 2015; Benedetti et al. 2016]. This
property has been exploited for training Boltzmann machines
[Adachi & Henderson 2015; Benedetti et al. 2016].

QAOA [Farhi et al. 2014; Hadfield et al. 2019] prepares
$\prod_{l=1}^p e^{-i\beta_l H_M} e^{-i\gamma_l H_C}|+\rangle^{\otimes N}$ and
optimizes the angles classically. Here the cost Hamiltonian $H_C$ is the
Ising energy written in Pauli-$Z$ operators.

[Related work on quantum annealing and QUBO formulations for image
segmentation, and on quantum approaches to weak supervision, is to be
surveyed.]

---

## 3. A Structured Label Model for Segmentation

**Sites and graph.** An image is partitioned into sites $V = \{1,\dots,N\}$,
which are pixels or superpixels. They are connected by an adjacency graph
$G = (V, E)$ with affinities $w_{ik} \ge 0$, for example the contrast-sensitive
weights $w_{ik} = \exp(-\lVert I_i - I_k\rVert^2/2\sigma^2)$. Latent labels are
$y \in \{-1,+1\}^N$, with $+1$ meaning foreground.

**Labeling functions.**
- *Unary* LFs $\lambda_j$, $j = 1..m$, emit $\Lambda_{ij} \in \{-1,0,+1\}$ at every site.
- *Pairwise* LFs $\pi_r$, $r = 1..q$, emit $\Pi_{ikr} \in \{-1,0,+1\}$ on pairs
  $(i,k) \in E_r$. Here $+1$ means "same label", $-1$ means "different", and
  $0$ means abstain. An edge detector is a natural example.

**Joint model.** With parameters $\Theta$:

$$
p_\Theta(\Lambda,\Pi,y) = \frac{1}{Z_\Theta}\exp\Big(
\sum_i b\,y_i + \sum_{(i,k)\in E}\beta w_{ik}\,y_iy_k
+ \sum_{i,j}\psi_j(\Lambda_{ij},y_i)
+ \sum_i\sum_{c\in\mathcal{C}}\chi_c(\Lambda_{ic})
+ \sum_r\sum_{(i,k)\in E_r}\psi^P_r(\Pi_{ikr},y_iy_k)\Big)
\tag{1}
$$

- $\psi_j$ is a $3\times2$ log-potential table.
- $\chi_c$ couples the votes of a small clique $c$ of correlated LFs.
- $\psi^P_r$ is a $3\times2$ table in $(\pi, s)$, with $s = y_iy_k$.

Parameters are tied across sites and images.

**Definition 1 (symmetric potentials).** Write $\mathbb{1}$ for the indicator function. A potential is *symmetric* if
- $\psi_j(\lambda,y) = \alpha_j\mathbb{1}[\lambda\neq0] + \theta_j\lambda y$,
- $\psi^P_r(\pi,s) = \alpha^P_r\mathbb{1}[\pi\neq0] + \theta^P_r\pi s$, and
- each $\chi_c$ is invariant under the global vote flip $\lambda \mapsto -\lambda$.

Symmetric potentials mean an LF's accuracy and coverage do not depend on the
true class. This is the standard data-programming assumption.

---

## 4. Ising Reductions

**Notation.** For $f : \{\pm1\} \to \mathbb{R}$, write
$f(y) = \bar f + \tilde f\,y$, with $\bar f = \tfrac12(f(1)+f(-1))$ and
$\tilde f = \tfrac12(f(1)-f(-1))$.

### 4.1 The posterior is an Ising model

**Proposition 1.** For any potentials in (1),
$p_\Theta(y\mid\Lambda,\Pi) \propto \exp(-E_{\text{post}}(y))$, where

$$
E_{\text{post}}(y) = -\sum_i h_iy_i - \sum_{i<k}J_{ik}y_iy_k,\quad
h_i = b + \sum_j\tilde\psi_j(\Lambda_{ij},\cdot),\quad
J_{ik} = \beta w_{ik}\mathbb{1}[(i,k)\in E] + \sum_{r:(i,k)\in E_r}\tilde\psi^P_r(\Pi_{ikr},\cdot).
$$

For symmetric potentials, $h_i = b + \sum_j\theta_j\Lambda_{ij}$ and
$J_{ik} = \beta w_{ik} + \sum_r\theta^P_r\Pi_{ikr}$.

*Proof.* Conditioning on $(\Lambda, \Pi)$ removes every factor that does not
depend on $y$ (the $\chi_c$ terms and $Z_\Theta$). Decompose each remaining
factor as $\bar f + \tilde f\,y_i$ (or $\bar f + \tilde f\,y_iy_k$ for pairwise
factors). The constants $\bar f$ cancel on normalization. Collecting the
linear and bilinear terms gives $h$ and $J$. In the symmetric case,
$\tilde\psi_j(\lambda,\cdot) = \theta_j\lambda$ and
$\tilde\psi^P_r(\pi,\cdot) = \theta^P_r\pi$. ∎

**Corollary 1 (MAP = ground state).**
$\arg\max_y p_\Theta(y\mid\Lambda,\Pi) = \arg\min_y E_{\text{post}}(y)$,
because $x \mapsto e^{-x}$ is strictly decreasing and the normalizer does not
depend on $y$. ∎

**Remark (source of frustration).** With $\beta \ge 0$ and no pairwise LFs,
$J \ge 0$ and MAP is solvable exactly in polynomial time by minimum cut. A
"different" vote ($\Pi_{ikr} = -1$, with $\theta^P_r > 0$) contributes a
negative coupling. Cycles containing an odd number of net-negative couplings
are frustrated.

### 4.2 The prior marginal is an Ising model

**Proposition 2.** Let $Z_c(y) = \sum_{\lambda\in\{-1,0,1\}^{|c|}}\exp\big(\sum_{j\in c}\psi_j(\lambda_j,y) + \chi_c(\lambda)\big)$,
treating an uncorrelated LF as a singleton clique, and let
$Z^P_r(s) = \sum_\pi\exp\psi^P_r(\pi,s)$. Then
$p_\Theta(y) \propto \exp\big(\sum_i h^{\text{pr}}_iy_i + \sum_{i<k}J^{\text{pr}}_{ik}y_iy_k\big)$, with

$$
h^{\text{pr}}_i = b + \tfrac12\sum_c\log\frac{Z_c(+1)}{Z_c(-1)},\qquad
J^{\text{pr}}_{ik} = \beta w_{ik}\mathbb{1}[(i,k)\in E] + \tfrac12\sum_{r:(i,k)\in E_r}\log\frac{Z^P_r(+1)}{Z^P_r(-1)}.
$$

*Proof.* Given $y$, the vote variables of different sites, and of different
pairs, appear in disjoint factors. The sum over $(\Lambda, \Pi)$ therefore
factorizes into $\prod_i\prod_c Z_c(y_i)\cdot\prod_r\prod_{(i,k)\in E_r}Z^P_r(y_iy_k)$.
Each factor is a positive function of a single $\pm1$ argument, so
$\log Z(y) = \overline{\log Z} + \widetilde{\log Z}\,y$. Collecting terms
gives the result. ∎

**Proposition 3 (symmetric potentials decouple).** Under Definition 1,
$Z_c(+1) = Z_c(-1)$ and $Z^P_r(+1) = Z^P_r(-1)$. Hence
$p_\Theta(y) \propto \exp(b\sum_iy_i + \beta\sum_E w_{ik}y_iy_k)$, independent of
all LF parameters. Moreover, for a singleton clique,

$$
\mathbb{E}[\Lambda_{ij}y_i] = \frac{2e^{\alpha_j}\sinh\theta_j}{1+2e^{\alpha_j}\cosh\theta_j},\qquad
\mathbb{E}\,\mathbb{1}[\Lambda_{ij}\neq0] = \frac{2e^{\alpha_j}\cosh\theta_j}{1+2e^{\alpha_j}\cosh\theta_j},
\tag{2}
$$

and analogously for pairwise LFs, with $(\alpha^P_r, \theta^P_r)$.

*Proof.* Substitute $\lambda \mapsto -\lambda$ in the sum defining $Z_c(-1)$.
Symmetry of $\psi_j$ under $(\lambda,y) \mapsto (-\lambda,-y)$, and of $\chi_c$
under $\lambda \mapsto -\lambda$, maps it term by term onto $Z_c(+1)$. The
pairwise case is identical. For (2), condition on $y_i$: the three outcomes
of $\Lambda_{ij}$ have weights $1$ and $e^{\alpha_j\pm\theta_j}$. ∎

### 4.3 Encodings for hardware

**Proposition 4 (QUBO and Pauli forms).**
- Under $y_i = 2x_i - 1$, $E_{\text{post}}$ equals the QUBO
  $c + \sum_iQ_{ii}x_i + \sum_{i<k}Q_{ik}x_ix_k$, with
  $Q_{ii} = -2h_i + 2\sum_{k}J_{ik}$, $Q_{ik} = -4J_{ik}$, and
  $c = \sum_ih_i - \sum_{i<k}J_{ik}$.
- Under $y_i \mapsto Z_i$, $E_{\text{post}}$ is the diagonal operator
  $H_C = -\sum_ih_iZ_i - \sum_{i<k}J_{ik}Z_iZ_k$, whose computational-basis
  eigenvalues are the energies.

*Proof.* Direct substitution, using
$y_iy_k = 4x_ix_k - 2x_i - 2x_k + 1$. ∎

**Proposition 5 (exact reductions).**
1. *Strong-field fixing.* If $|h_i| > \sum_k|J_{ik}|$, then every minimizer of
   $E_{\text{post}}$ has $y_i = \operatorname{sign}(h_i)$.
2. *Conditioning.* Fixing $y_F = \bar y_F$ yields an Ising model on
   $V \setminus F$ with fields $h'_i = h_i + \sum_{k\in F}J_{ik}\bar y_k$. Its
   Boltzmann distribution equals the exact conditional
   $p(y_{V\setminus F}\mid y_F = \bar y_F)$.

*Proof.*
1. Flipping $y_i$ changes the energy by
   $\Delta = 2y_i\big(h_i + \sum_kJ_{ik}y_k\big)$. If
   $y_i = -\operatorname{sign}(h_i)$, then
   $\Delta \le -2|h_i| + 2\sum_k|J_{ik}| < 0$, contradicting minimality.
2. Substitute and collect terms; the terms involving only $F$ are constant. ∎

Part 2 justifies exact *tiling*. A large problem can be solved by block
coordinate descent (for MAP), or sampled by block Gibbs, with each block
being a hardware-sized conditional Ising model.

### 4.4 Complexity map

| Regime | MAP (T1) | Marginals / learning (T2, T3) |
|--------|----------|-------------------------------|
| $\beta = 0$, no pairwise LFs | $O(Nm)$ | Closed form |
| $J \ge 0$ | Polynomial (min-cut) | #P-hard exactly in general; MCMC slows near criticality |
| Mixed-sign $J$ | NP-hard in general | #P-hard; MCMC mixing can be exponentially slow |
| $K \ge 3$ classes (§8) | NP-hard | #P-hard |

Quantum resources are a candidate only in the last three rows, and for MAP
only in the last two. Ferromagnetic binary MAP serves as an exactly solvable
**validation** case.

---

## 5. Learning

### 5.1 The gradient

Write (1) as an exponential family
$p_\Theta \propto \exp(\Theta^\top\phi(\Lambda,\Pi,y))$. The marginal
log-likelihood of $D$ images is

$$
\ell(\Theta) = \sum_{n=1}^D\Big[\log\sum_y e^{\Theta^\top\phi(\Lambda^{(n)},\Pi^{(n)},y)} - \log Z_\Theta^{(n)}\Big].
$$

Both terms are log-partition functions over $2^{N_n}$ (or more)
configurations.

**Lemma 1.**
$\nabla_\Theta\ell = \sum_n\big(\mathbb{E}_{p_\Theta(y\mid\Lambda^{(n)},\Pi^{(n)})}[\phi] - \mathbb{E}_{p_\Theta}[\phi]\big)$.

*Proof.* $\nabla_\Theta\log\sum_u e^{\Theta^\top\phi(u)} = \sum_u\phi(u)\,e^{\Theta^\top\phi(u)}/\sum_{u'}e^{\Theta^\top\phi(u')}$,
which is the expectation of $\phi$ under the corresponding normalized
distribution. ∎

$\ell$ is a difference of convex functions and is not concave in general, so
gradient methods reach stationary points. At a stationary point, the
posterior moments match the model moments.

**Theorem 1 (posterior sampling suffices).** Assume symmetric potentials
(Definition 1), with $b$ and $\beta$ held fixed. Then for every LF parameter:

$$
\frac{\partial\ell}{\partial\theta_j} = \sum_n\sum_i\Big(\Lambda^{(n)}_{ij}\,\langle y_i\rangle^{(n)}_{\text{post}} - \mathbb{E}[\Lambda_{ij}y_i]\Big),\qquad
\frac{\partial\ell}{\partial\theta^P_r} = \sum_n\sum_{(i,k)\in E_r}\Big(\Pi^{(n)}_{ikr}\,\langle y_iy_k\rangle^{(n)}_{\text{post}} - \mathbb{E}[\Pi_{ikr}y_iy_k]\Big).
$$

- The second term of each is given in closed form by (2).
- The gradients for $\alpha_j$, $\alpha^P_r$ and $\gamma_c$ involve no
  $y$-dependent posterior quantity at all.
- The only quantities that are hard to compute are the first moments
  $\langle y_i\rangle_{\text{post}}$ and edge moments
  $\langle y_iy_k\rangle_{\text{post}}$ of the per-image posterior Ising models
  of Proposition 1.

*Proof.* Apply Lemma 1. The sufficient statistic for $\theta_j$ is
$\sum_i\Lambda_{ij}y_i$, which is linear in $y$ given $\Lambda$, so its posterior
expectation is $\sum_i\Lambda_{ij}\langle y_i\rangle$. The model expectation
factorizes per site by Proposition 2. By Proposition 3 it does not depend on
the distribution of $y$, and it equals (2). The pairwise case is analogous.
The statistics for $\alpha_j$, $\alpha^P_r$ and $\gamma_c$ depend only on
observed votes in the data term, and their model terms are per-site sums
computable in $O(3^{|c|})$. ∎

When $b$ and $\beta$ are learned too, their gradients additionally need
moments of the prior Ising model of Proposition 3. That model is shared by
all images with the same graph.

### 5.2 Why zero-temperature labels are not enough

A tempting shortcut replaces the posterior moments with the values at the
MAP state ("hard EM"). This approximation is biased.

**Proposition 6 (MAP agreement overestimates accuracy).** Take $\beta = 0$
and three conditionally independent, non-abstaining LFs. Each is correct
with probability $a \in (\tfrac12, 1)$, and the class prior is uniform. Then
the MAP label is the majority vote, and the probability that a given LF
agrees with it is $1 - a(1-a) > a$.

*Proof.* Let $A$ be the event that LF 1 is correct. LF 1 agrees with the
majority iff at least one of the other two LFs votes the same way as LF 1.
- Given $A$, the probability is $1-(1-a)^2$.
- Given not $A$, it is $1-a^2$.

So the agreement is $a(1-(1-a)^2) + (1-a)(1-a^2) = 1 - a(1-a)$. Finally,
$1 - a(1-a) - a = (1-a)^2 > 0$. ∎

For $a = 0.7$, accuracy estimated from MAP labels is $0.79$. The bias exists
because every LF took part in choosing the labels it is then scored against.

**Proposition 7 (cold samplers inflate parameters).** Assume symmetric
potentials, and let a sampler draw from
$q(y) \propto \exp(-\beta_s E_{\text{post}}(y))$ instead of the target
$\beta_s = 1$. Then $q$ is exactly the posterior of the model with
$(\theta, \theta^P, b, \beta)$ all multiplied by $\beta_s$. So the data term of
Lemma 1, evaluated under $q$, is the data term at inflated parameters.
- For $\beta_s > 1$, the posterior moments are overconfident.
- As $\beta_s \to \infty$, $q$ concentrates on the minimizers of
  $E_{\text{post}}$, and learning reduces to hard EM.

*Proof.* By Proposition 1, $h$ and $J$ are linear and homogeneous in
$(b, \theta, \beta, \theta^P)$, so $\beta_s E_{\text{post}}$ is the posterior
energy at the scaled parameters. The limit is the standard zero-temperature
concentration of a Boltzmann distribution on its ground states. ∎

The practical consequence is that sampler temperature must be calibrated,
not merely low.

### 5.3 Sample complexity of the gradient

For one image, write $g_j$ for the per-site-normalized data term of
Theorem 1, $g_j = \frac1N\sum_i\Lambda_{ij}\langle y_i\rangle$, and $\hat g_j$
for its estimate from $S$ samples drawn i.i.d. from a distribution $q$.

**Proposition 8 (error decomposition).** With probability at least
$1 - \delta$:

$$
|\hat g_j - g_j| \le \underbrace{2\,\mathrm{TV}(q, p_{\text{post}})}_{\text{sampler bias}} + \underbrace{\sqrt{2\log(2/\delta)/S}}_{\text{statistical error}}.
$$

The edge-moment terms satisfy the same bound.

*Proof.* The statistic $f(y) = \frac1N\sum_i\Lambda_{ij}y_i$ lies in $[-1,1]$. Then:
- $|\mathbb{E}_qf - \mathbb{E}_pf| \le 2\,\mathrm{TV}(q,p)$ for any function
  with values in $[-1,1]$.
- Hoeffding's inequality, applied to $S$ i.i.d. draws of a variable with
  range 2, gives the second term. ∎

The statistical term does not depend on $N$, or on the $2^N$ size of the
configuration space. **Learning is therefore limited by sampler fidelity,
not by enumeration.** This is the precise sense in which a high-fidelity
sampler can stand in for intractable summation. For MCMC samples, which are
correlated, $S$ is replaced by an effective sample size.

---

## 6. Hybrid Classical–Quantum Algorithm

**Algorithm 1 (learning).**
1. Compute $\Lambda$ and $\Pi$ for all images.
2. Warm-start $\Theta$ from the independent model ($\beta = 0$), which is
   closed-form/EM.
3. For each minibatch:
   1. build the posterior Ising models (Proposition 1);
   2. submit them to the sampler;
   3. estimate $\hat\beta_s$ and recalibrate (§6.1);
   4. optionally polish (§6.2);
   5. form the gradient (Theorem 1) and take a projected ascent step
     (enforcing $\beta \ge 0$ and $\overline{\theta} > 0$, which breaks the
     label-flip symmetry).
4. **Inference:** MAP via the ground-state solver (Corollary 1), and soft
   masks $\mu_i = \tfrac12(1+\langle y_i\rangle)$ via the sampler.

### 6.1 Temperature calibration

Hardware coefficients are rescaled by $\kappa$ to fit device ranges, so a
device returning samples approximately $\propto e^{-\beta_{\text{dev}}E/\kappa}$
has $\beta_s = \beta_{\text{dev}}/\kappa$. We estimate $\beta_s$ from the
returned samples by maximum pseudo-likelihood [Besag 1975]:

$$
\hat\beta_s = \arg\max_{\beta'}\sum_{s}\sum_i\log\sigma\Big(2\beta'y^{(s)}_i\big(h_i + \textstyle\sum_kJ_{ik}y^{(s)}_k\big)\Big),
$$

This is a one-dimensional concave problem, since each term is the
log-sigmoid of a linear function of $\beta'$. It is a consistent estimator
when the samples are i.i.d. Boltzmann at some temperature. We then adjust
$\kappa$ so that $\hat\beta_s \to 1$, subject to the device's coefficient
bounds [cf. Benedetti et al. 2016].

### 6.2 Quantum-seeded polishing

Starting from each quantum sample, we run $k$ sweeps of Gibbs sampling at
$\beta = 1$. The Gibbs kernel leaves $p_{\text{post}}$ invariant, so it never increases
the total-variation distance to it. Because the kernel is ergodic, that
distance goes to zero as $k$ grows, shrinking the bias term of
Proposition 8. The rate is model-dependent.

The division of labour is:
- the annealer supplies starting points spread across distant low-energy
  basins, which single-site MCMC crosses slowly;
- local Gibbs moves correct the within-basin proportions.

### 6.3 Gate-based variant

QAOA with cost $H_C$ (Proposition 4) serves as a ground-state heuristic for
subproblems of roughly $10$–$10^2$ sites, and as a seed generator for §6.2.
QAOA output distributions are not Boltzmann distributions. Using them
directly for learning is left open.

---

## 7. Mapping to Hardware

- **Problem size.** Superpixels reduce $N$ to roughly $10^2$–$10^3$.
  Region-adjacency and grid graphs are sparse, with degree about 4–8, which
  suits minor-embedding [Choi 2008] with short chains. Embeddings for
  repeated graph shapes are computed once and reused.
- **Larger images.** These use the exact conditioning of Proposition 5(2):
  block coordinate descent for MAP and block Gibbs for sampling. Non-adjacent
  blocks are submitted in parallel.
- **Precision.** Precision is limited by the device's coefficient ranges and
  analog noise. Strong-field fixing (Proposition 5(1)) and roof duality
  [Boros & Hammer 2002] remove easy variables before submission and improve
  the dynamic range.
- **Sign conventions.** Hardware SDKs differ in sign convention, so the
  conversion must be explicit.

---

## 8. Multi-class Extension

For $K$ classes, use one-hot variables $x_{ic} = \mathbb{1}[y_i = c]$, a Potts
smoothness prior, and the penalty $A\sum_i(1-\sum_cx_{ic})^2$. This gives a
QUBO with $NK$ variables. A penalty $A$ larger than the maximal energy
decrease achievable by violating one site's constraint makes the ground
state feasible *(verify the tight bound)*.

- Domain-wall encodings use $K-1$ qubits per site [Chancellor 2019].
- XY mixers preserve feasibility in QAOA [Hadfield et al. 2019].

Potts MAP is NP-hard for $K \ge 3$, even with only ferromagnetic couplings
[Boykov et al. 2001]. Multi-class is therefore a quantum target even without
frustration. Theorem 1 extends with per-class indicator statistics.

---

## 9. Discussion and Planned Evaluation

**Limitations.**
- $\ell$ is non-concave, so all guarantees are local.
- Identifiability of $\beta$ from votes alone is weak; we fix it on a small
  validation set.
- Consistency of maximum-likelihood estimation for the structured model,
  with dependent sites, is not established here.
- Annealer samples are only approximately Boltzmann. Their fairness among
  degenerate ground states is known to be imperfect [Matsuda et al. 2009].
- No quantum advantage is claimed.

**Planned evaluation.** We pre-register the comparison:
- *Synthetic data* sweeping $N$, $\beta$, the frustration fraction, and LF
  quality. Here we measure exact-distribution total-variation distance,
  marginal error, and parameter-recovery error for small $N$.
- *Real data*: small binary datasets, scored with IoU/Dice and calibration.

Quantum samplers, raw and polished, are compared with persistent-chain Gibbs
[Tieleman 2008] at equal wall-clock and sample budgets. Ground-state solvers
are compared with min-cut (ferromagnetic case), QPBO, and simulated
annealing, using time-to-solution and energy gap. Timing reports QPU time
separately from end-to-end time.

---

## 10. Conclusion

Spatially structured data programming for segmentation reduces exactly to
Ising models:
- MAP segmentation is a ground-state problem;
- LF-accuracy learning, under symmetric potentials, needs only unit-temperature
  moments of per-image posterior Ising models.

The quality of learning is governed by sampler fidelity and calibration, not
by the exponential size of the mask space. This identifies a precise,
testable role for quantum annealers and QAOA in weak supervision.

---

## References *(all to verify)*

- Adachi, S. H., & Henderson, M. P. (2015). Application of quantum annealing to training of deep neural networks. arXiv:1510.06356.
- Amin, M. H. (2015). Searching for quantum speedup in quasistatic quantum annealers. *Physical Review A*, 92, 052323.
- Bach, S. H., He, B., Ratner, A., & Ré, C. (2017). Learning the structure of generative models without labeled data. *ICML*.
- Barahona, F. (1982). On the computational complexity of Ising spin glass models. *J. Phys. A*, 15, 3241.
- Benedetti, M., Realpe-Gómez, J., Biswas, R., & Perdomo-Ortiz, A. (2016). Estimation of effective temperatures in quantum annealers for sampling applications. *Physical Review A*, 94, 022308.
- Besag, J. (1975). Statistical analysis of non-lattice data. *The Statistician*, 24(3), 179–195.
- Boros, E., & Hammer, P. L. (2002). Pseudo-Boolean optimization. *Discrete Applied Mathematics*, 123, 155–225.
- Boykov, Y., Veksler, O., & Zabih, R. (2001). Fast approximate energy minimization via graph cuts. *IEEE TPAMI*, 23(11).
- Chancellor, N. (2019). Domain wall encoding of discrete variables for quantum annealing and QAOA. *Quantum Science and Technology*, 4, 045004.
- Choi, V. (2008). Minor-embedding in adiabatic quantum computation: I. The parameter setting problem. *Quantum Information Processing*, 7, 193–209.
- Dawid, A. P., & Skene, A. M. (1979). Maximum likelihood estimation of observer error-rates using the EM algorithm. *Applied Statistics*, 28(1), 20–28.
- Farhi, E., Goldstone, J., & Gutmann, S. (2014). A quantum approximate optimization algorithm. arXiv:1411.4028.
- Geman, S., & Geman, D. (1984). Stochastic relaxation, Gibbs distributions, and the Bayesian restoration of images. *IEEE TPAMI*, 6(6).
- Greig, D. M., Porteous, B. T., & Seheult, A. H. (1989). Exact maximum a posteriori estimation for binary images. *JRSS B*, 51(2), 271–279.
- Hadfield, S., et al. (2019). From the quantum approximate optimization algorithm to a quantum alternating operator ansatz. *Algorithms*, 12(2), 34.
- Jerrum, M., & Sinclair, A. (1993). Polynomial-time approximation algorithms for the Ising model. *SIAM J. Computing*, 22(5).
- Johnson, M. W., et al. (2011). Quantum annealing with manufactured spins. *Nature*, 473, 194–198.
- Kadowaki, T., & Nishimori, H. (1998). Quantum annealing in the transverse Ising model. *Physical Review E*, 58, 5355.
- Kolmogorov, V., & Zabih, R. (2004). What energy functions can be minimized via graph cuts? *IEEE TPAMI*, 26(2).
- Matsuda, Y., Nishimori, H., & Katzgraber, H. G. (2009). Ground-state statistics from annealing algorithms: quantum versus classical approaches. *New Journal of Physics*, 11, 073021.
- Ratner, A., De Sa, C., Wu, S., Selsam, D., & Ré, C. (2016). Data programming: Creating large training sets, quickly. *NeurIPS*.
- Ratner, A., Bach, S. H., Ehrenberg, H., Fries, J., Wu, S., & Ré, C. (2017). Snorkel: Rapid training data creation with weak supervision. *PVLDB*, 11(3).
- Tieleman, T. (2008). Training restricted Boltzmann machines using approximations to the likelihood gradient. *ICML*.
- Younes, L. (1989). Parametric inference for imperfectly observed Gibbsian fields. *Probability Theory and Related Fields*, 82, 625–645.
