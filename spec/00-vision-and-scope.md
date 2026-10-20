# 00 — Vision and Scope

## 1. Goal

Build a system that:

1. Takes **weak supervision** for image segmentation, as a set of noisy,
   partial, possibly correlated labeling functions (LFs).
2. Combines the LFs with a **data programming** label model. The model is
   structured: it couples neighbouring sites, because segmentation masks are
   spatially coherent.
3. Writes the label model's hard optimization and sampling problems as
   **Ising / QUBO Hamiltonians**.
4. Solves those Hamiltonians on **real quantum hardware**: annealers, and
   gate-based devices through QAOA. Classical solvers act as baselines and
   fallbacks.
5. Outputs **segmentation masks**, as a hard MAP mask and per-site soft
   probabilities, that can train a downstream segmentation network.

## 2. Why this is hard classically

A binary mask over $N$ sites has $2^N$ possible configurations. A single
$64 \times 64$ image has $2^{4096}$. Data programming needs three things over
this space:

- **Inference**: the most probable mask given the LF votes (MAP).
- **Marginals**: per-site posterior probabilities, used as soft labels.
- **Learning**: LF accuracies estimated with no ground truth. The likelihood
  gradient needs expectations under the model, which requires the partition
  function, a sum over all $2^N$ masks for every image.

Without spatial coupling, sites are independent and all of this is trivial.
That gives *per-pixel classification*, not segmentation. Coupling sites with
a Markov random field (MRF) prior makes the problem structured, and that is
where the hardness comes from.

## 3. Where quantum can genuinely help (honest scoping)

The project MUST NOT claim quantum advantage where classical methods are
already exact and efficient. The spec separates the cases:

| Problem variant | Classical status | Role of quantum |
|-----------------|------------------|-----------------|
| Binary MAP, all couplings $J_{ik} \ge 0$ (ferromagnetic, submodular) | **Exact in polynomial time** (s–t min-cut / graph cuts) | Validation only: graph cuts give the true optimum to check quantum solutions against |
| Binary MAP with **mixed-sign couplings** (e.g., from pairwise "these differ" LFs, doc 01 §3.3) | NP-hard in general (spin-glass ground state on non-planar graphs) | **Target**: annealing / QAOA as heuristic optimizers |
| Posterior **sampling and marginals**, partition-function-dependent **learning** | #P-hard to do exactly in general; MCMC can mix slowly at strong coupling | **Target**: the annealer as an approximate Boltzmann sampler |
| **Multi-class** ($K \ge 3$) Potts MAP | NP-hard; α-expansion is only approximate | **Target** (doc 07) |

The scientific question the repo answers is: *for which problem sizes,
coupling regimes, and LF structures do quantum solvers match or beat strong
classical baselines on solution quality, sample quality, and time?* A
negative result, measured rigorously, is still a valid result.

## 4. Scope

**In scope (v1):**
- Binary segmentation over pixel grids and superpixel graphs.
- Unary LFs (per-site votes) and pairwise LFs (same/different votes on site pairs).
- Symmetric and class-conditional LF accuracy parameters, plus an optional LF-correlation structure.
- Ising/QUBO construction, scaling, and decomposition for large images.
- Backends: exact enumeration (tiny problems), classical heuristics, quantum annealer, QAOA (simulator and hardware).
- Joint learning and inference loop (doc 03).
- A benchmark harness (doc 06).

**Planned (v2):** multi-class segmentation (doc 07); higher-order (region-level) LFs via quadratization.

**Non-goals:**
- Training state-of-the-art downstream segmentation networks. We produce training labels; the downstream model is a plug-in.
- Designing domain-specific LFs. The LF interface is generic.
- Quantum machine learning of image features. Quantum hardware is used only for the label model's combinatorial core.

## 5. Milestones

| ID | Milestone | Exit criterion |
|----|-----------|----------------|
| M0 | Spec agreed | This folder reviewed; open questions resolved or deferred explicitly |
| M1 | Classical reference implementation | Label model, Ising builder, exact/SA/graph-cut/Gibbs solvers; learning recovers known parameters on synthetic data |
| M2 | Quantum simulation | QAOA on a simulator and annealing on a simulated sampler run end-to-end through the same solver interface |
| M3 | Real hardware, small | Superpixel-scale problems (≈10²–10³ sites) solved on real annealing hardware, and small subproblems (≈10–100 sites) on gate hardware, all benchmarked against M1 |
| M4 | Real hardware, decomposed | Full images via decomposition (doc 04 §5), with an end-to-end segmentation quality report |
| M5 | Multi-class | Doc 07 implemented and benchmarked |
