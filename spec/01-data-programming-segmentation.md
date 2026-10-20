# 01 — Data Programming for Segmentation (Classical Model)

This document defines the probabilistic label model. Documents 02–04 turn it
into quantum-solvable problems.

## 1. Background: data programming

In data programming (Ratner et al., 2016, the basis of Snorkel), users write
**labeling functions** instead of hand-labelling data. Each LF is a heuristic
that votes on a label or abstains. A generative **label model** learns the LF
accuracies and correlations from LF agreements and disagreements alone, with
no ground truth. It then produces probabilistic training labels.

Classical data programming treats data points as **independent**. In
segmentation the data points are sites in an image, and their labels are
strongly coupled. This repo's model adds that coupling explicitly, as a
structured label model.

## 2. Setup and notation

| Symbol | Meaning |
|--------|---------|
| $D$ | Number of images in the dataset; image index $n$ (suppressed when unambiguous) |
| $V = \{1,\dots,N\}$ | Sites (pixels or superpixels) of one image |
| $G = (V, E)$ | Site adjacency graph: a 4- or 8-neighbour grid for pixels, the region adjacency graph for superpixels |
| $w_{ik} \ge 0$ | Edge affinity, e.g., contrast-sensitive $w_{ik} = \exp(-\lVert I_i - I_k\rVert^2 / 2\sigma^2)$ |
| $y_i \in \{-1,+1\}$ | Latent true label of site $i$ ($+1$ = foreground) |
| $\lambda_j$, $j = 1..m$ | **Unary LFs** |
| $\Lambda_{ij} \in \{-1, 0, +1\}$ | Vote of LF $j$ on site $i$ ($0$ = abstain) |
| $\pi_r$, $r = 1..q$ | **Pairwise LFs** |
| $E_r \subseteq V \times V$ | Site pairs that pairwise LF $r$ can vote on |
| $\Pi_{ikr} \in \{-1, 0, +1\}$ | Vote of pairwise LF $r$ on pair $(i,k)$: $+1$ = same label, $-1$ = different, $0$ = abstain |
| $C$ | Set of correlated unary-LF cliques (user-declared or learned; doc 03 §6) |

### 2.1 Labeling functions (generic interface)

- A **unary LF** maps (image, site) → $\{-1, 0, +1\}$.
  Examples: an intensity threshold, a colour prior, a pretrained detector's box
  ("inside box → abstain or $+1$; far outside → $-1$"), a user scribble.
- A **pairwise LF** maps (image, site pair) → $\{-1, 0, +1\}$.
  Examples: an edge detector ("a strong edge between $i$ and $k$ → different"),
  or texture similarity ("very similar → same").
- On superpixel sites, a pixel-level LF MUST be lifted to the superpixel. The
  default is the majority of non-abstaining pixel votes, abstaining on ties or
  low coverage (see open questions).

## 3. The structured label model

### 3.1 Joint distribution

For one image, with parameters $\theta$ collecting every quantity below:

$$
p_\theta(\Lambda, \Pi, y) = \frac{1}{Z_\theta}\exp\Big(
\underbrace{\textstyle\sum_i b\, y_i + \sum_{(i,k)\in E} \beta\, w_{ik}\, y_i y_k}_{\text{spatial prior}}
+ \underbrace{\textstyle\sum_i \sum_j \psi_j(\Lambda_{ij}, y_i)}_{\text{unary LFs}}
+ \underbrace{\textstyle\sum_i \sum_{c \in C} \chi_c(\Lambda_{ic})}_{\text{LF correlations}}
+ \underbrace{\textstyle\sum_r \sum_{(i,k)\in E_r} \psi^{P}_r(\Pi_{ikr},\, y_i y_k)}_{\text{pairwise LFs}}
\Big)
$$

The factors are:

- **Unary LF potential** $\psi_j(\lambda, y)$, a $3 \times 2$ log-potential table.
  - *Symmetric* (default, Snorkel-style):
    $\psi_j(\lambda, y) = \alpha_j \mathbb{1}[\lambda \ne 0] + \theta_j \lambda y$.
    Here $\alpha_j$ is the labelling propensity and $\theta_j$ the accuracy.
  - *Class-conditional*: a general table, so accuracy and coverage MAY differ
    between foreground and background.
- **Correlation potential** $\chi_c(\Lambda_{ic})$ on the joint vote of a
  clique $c$ of correlated LFs, e.g.
  $\gamma_{jl}\mathbb{1}[\Lambda_{ij} = \Lambda_{il} \ne 0]$.
  It does not involve $y$.
- **Pairwise LF potential** $\psi^P_r(\pi, s)$, where $s = y_i y_k \in \{\pm 1\}$.
  - Symmetric: $\psi^P_r(\pi, s) = \alpha^P_r \mathbb{1}[\pi \ne 0] + \theta^P_r \pi s$.
- **Spatial prior**: a class bias $b$ and a smoothness strength $\beta \ge 0$
  (a ferromagnetic Ising prior).

**Parameter tying.** LF parameters are shared across all sites and all
images. That sharing is what makes them learnable without labels.

### 3.2 Posterior over masks is an Ising model

Given the observed votes, every term that does not involve $y$ cancels, and so
do the parts of $\psi_j$ that are even in $y$. Using
$f(y) = \tfrac{f(+1)+f(-1)}{2} + \tfrac{f(+1)-f(-1)}{2}\,y$ for any $f$ on $\{\pm1\}$:

$$
p_\theta(y \mid \Lambda, \Pi) \propto \exp\!\big(-E_{\text{post}}(y)\big),\qquad
E_{\text{post}}(y) = -\sum_i h_i y_i - \sum_{(i,k)} J_{ik}\, y_i y_k
$$

$$
h_i = b + \sum_j \tfrac{1}{2}\big[\psi_j(\Lambda_{ij},+1) - \psi_j(\Lambda_{ij},-1)\big]
\;\overset{\text{sym}}{=}\; b + \sum_j \theta_j \Lambda_{ij}
$$

$$
J_{ik} = \beta\, w_{ik}\,\mathbb{1}[(i,k)\in E] + \sum_{r:\,(i,k)\in E_r} \theta^P_r\, \Pi_{ikr}
$$

**This is the central structural fact of the project.** The posterior is an
Ising model on the graph $E \cup \bigcup_r E_r$. Its local fields come from
the unary LF votes; its couplings come from smoothness and the pairwise LF
votes. Its maximum-probability state is the minimum-energy state of an Ising
Hamiltonian. Doc 02 makes this exact.

### 3.3 Where frustration comes from

With $\beta \ge 0$ and no pairwise LFs, every $J_{ik} \ge 0$, so the model is
ferromagnetic (submodular). A pairwise LF voting "different"
($\Pi_{ikr} = -1$) with $\theta^P_r > 0$ adds a **negative** coupling. When
such votes form cycles that disagree with the smoothness prior, the model is
**frustrated**. These votes are realistic in practice; edge detectors produce
them naturally. They also make MAP inference NP-hard in general.

### 3.4 Prior marginal over masks is also an Ising model

Summing out the LF outputs gives the marginal $p_\theta(y)$, which learning
needs (doc 03). LF outputs are conditionally independent across sites given
$y$, so the sum factorizes per site (and per pair):

$$
p_\theta(y) \propto \exp\!\Big(\sum_i h^{\text{pr}}_i y_i + \sum_{(i,k)} J^{\text{pr}}_{ik} y_i y_k\Big)
$$

$$
h^{\text{pr}}_i = b + \tfrac12 \sum_{\text{cliques } c} \log\frac{Z_c(+1)}{Z_c(-1)},\qquad
Z_c(y) = \sum_{\lambda \in \{-1,0,1\}^{|c|}} \exp\Big(\sum_{j\in c}\psi_j(\lambda_j, y) + \chi_c(\lambda)\Big)
$$

$$
J^{\text{pr}}_{ik} = \beta w_{ik}\mathbb{1}[(i,k)\in E] + \tfrac12 \sum_{r:(i,k)\in E_r} \log\frac{Z^P_r(+1)}{Z^P_r(-1)},\qquad
Z^P_r(s) = \sum_{\pi} \exp \psi^P_r(\pi, s)
$$

Here an uncorrelated LF counts as a clique of size one.

- Each $Z_c$ is a sum over $3^{|c|}$ terms, so cliques MUST stay small
  (default bound: $|c| \le 3$).
- **Symmetric potentials** give $Z_c(+1) = Z_c(-1)$ and $Z^P_r(+1) = Z^P_r(-1)$.
  The LF terms then drop out completely, leaving the prior Ising model with
  fields $b$ and couplings $\beta w_{ik}$. The LF moments also have closed forms:

  $$
  \mathbb{E}[\Lambda_{ij} y_i] = \frac{2 e^{\alpha_j}\sinh\theta_j}{1 + 2e^{\alpha_j}\cosh\theta_j},\qquad
  \mathbb{E}[\mathbb{1}[\Lambda_{ij}\neq 0]] = \frac{2e^{\alpha_j}\cosh\theta_j}{1 + 2e^{\alpha_j}\cosh\theta_j}
  $$

## 4. The three computational tasks

For each image, given $\theta$ and the votes:

| Task | Definition | Output |
|------|-----------|--------|
| **T1 — MAP inference** | $\hat y = \arg\min_y E_{\text{post}}(y)$ | Hard mask |
| **T2 — Marginal inference / sampling** | $\mu_i = p_\theta(y_i = +1 \mid \Lambda, \Pi)$, or samples $y \sim p_\theta(y \mid \Lambda,\Pi)$ | Soft mask (probabilistic training labels) |
| **T3 — Learning** | $\hat\theta = \arg\max_\theta \sum_n \log \sum_{y} p_\theta(\Lambda^{(n)}, \Pi^{(n)}, y)$ | LF accuracies, correlations, $b$, $\beta$ |

The learning objective has the usual latent-variable gradient: the
expectation of the sufficient statistics $\phi$ under the posterior, minus
their expectation under the model:

$$
\nabla_\theta \ell = \sum_n \Big( \mathbb{E}_{y \sim p_\theta(y\mid\Lambda^{(n)},\Pi^{(n)})}[\phi(\Lambda^{(n)},\Pi^{(n)},y)]
- \mathbb{E}_{(\Lambda,\Pi,y)\sim p_\theta}[\phi(\Lambda,\Pi,y)] \Big)
$$

Both expectations reduce to **sampling Ising models**: the posterior (§3.2)
and the prior marginal (§3.4). Doc 03 covers the details.

## 5. Complexity summary

| Case | T1 (MAP) | T2 / T3 (marginals, partition function) |
|------|----------|----------------------------------------|
| $\beta = 0$, no pairwise LFs | $O(Nm)$; sites independent | Closed form |
| $J_{ik} \ge 0$ everywhere | Polynomial (s–t min-cut) | #P-hard to compute exactly in general; MCMC mixes slowly near the critical coupling |
| Mixed-sign $J$ (frustrated) | NP-hard in general (non-planar graphs) | #P-hard; MCMC can mix very poorly |
| Multi-class, $K \ge 3$ (doc 07) | NP-hard (Potts) | #P-hard |

**Consequence for the project:**
- Graph cuts are the exact **reference** for ferromagnetic MAP, used to
  validate quantum solvers.
- The **targets for quantum solvers** are the frustrated MAP problems, all
  the sampling problems (T2 and T3), and multi-class.

## 6. Identifiability and assumptions

- **Label-flip symmetry.** Flipping every $y$ together with the signs of
  $\theta_j$ and $b$ leaves the likelihood of $\Lambda$ unchanged. We break it
  by assuming LFs are better than random on average: $\theta_j$ is initialized
  positive and its mean is constrained to be positive. This follows standard
  data-programming practice.
- **Smoothness identifiability.** $\beta$ is only weakly identifiable from
  votes alone. In v1, $\beta$ MAY be fixed as a hyperparameter chosen on a
  small labelled validation set. Learning $\beta$ is a v1 stretch goal.
- **Conditional independence.** LFs are assumed conditionally independent
  given $y$, except within the declared cliques $C$.
- **Image independence.** Images are independent given the shared $\theta$.
