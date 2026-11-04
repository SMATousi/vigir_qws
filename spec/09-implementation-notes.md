# 09 — Implementation Notes

How the specification (docs 00–08) is realized in `qws/`. This document covers the formulas actually
implemented, conventions, algorithms, numerical choices and known limitations. Every non-obvious choice is
also recorded with its rationale in the [decision log (doc 10)](10-decision-log.md). Measured behaviour is in
[`results/`](../results/README.md).

## 1. Package map

```
qws/
  ising/       problem.py   IsingProblem (h, edges, J, offset), energies, local fields, scaling, hashing
               convert.py   Ising <-> QUBO, Ocean BQM (sign flip), Pauli terms
               reduce.py    exact conditioning; iterative strong-field fixing
  graph/       sites.py     pixel grid (4/8-nbr), SLIC superpixels, region adjacency, contrast weights
  lf/          base.py      LF interfaces, apply_lfs, majority lifting
               library.py   threshold / box / scribble / edge-contrast LFs
               natural.py   generic natural-image LFs (border, centre, colour model, saliency, colour contrast)
  model/       params.py    LabelModelParams (alpha, theta, alphaP, thetaP, gamma, omega, zeta, b, beta)
               label_model.py  posterior/prior builders, clique tables, moments, gradient, warm start
  solvers/     base.py      Solver, SampleSet, UnsupportedProblem, greedy colouring
               exact.py     enumeration (MAP, sampling, marginals, log Z)
               mcmc.py      heat-bath Gibbs (persistent chains, polishing), simulated annealing
               graphcut.py  submodular QUBO max-flow, GraphCutSolver, roof duality, QPBOSolver
               anneal.py    AnnealingSolver over any dimod sampler (Ocean SA stand-in)
               qpu.py       QPUAnnealingSolver on a structured D-Wave sampler (real or mock)
               qaoa.py      QAOASimulator (statevector), qaoa_circuit, QAOAQiskit (backend.run / SamplerV2)
  decompose/   tiling.py    grid/BFS tiles, block coordinate descent, block Gibbs
  learn/       optim.py     Adam;  temperature.py  PL estimator, CalibratedSampler;  trainer.py  Trainer, inference
  multiclass/  potts.py     PottsProblem, one-hot QUBO, decoding, exact Potts, Potts Gibbs/annealing
               model.py     K-class label model, gradient, learning, synthetic data
  data/        pet.py       Oxford-IIIT Pet loader/downloader;  petlike.py  Pet-layout dataset generator
  downstream/  unet.py      U-Net, noise-aware loss, training;  pipeline.py  end-task comparison
  eval/        synthetic.py, metrics.py, benchmark.py
  experiment.py  build_dataset / fit_and_label / run_experiment (config-driven)
  cli.py       qws demo | qws run <config.yaml>
  io.py        JSON result store
```

**Optional extras.**
- `anneal`: dimod, dwave-system, dwave-neal, minorminer, dwave-networkx.
- `gate`: qiskit, qiskit-aer, qiskit-ibm-runtime.
- `downstream`: torch.
- `dev`: pytest.

The classical core needs none of them, and all SDK imports are lazy.

## 2. Structured label model

**Variables.**
- Sites i = 0..n−1, with y_i ∈ {−1, +1} (+1 is foreground).
- Unary votes Λ ∈ {−1, 0, +1}^{n×m}.
- Pairwise votes Π ∈ {−1, 0, +1}^{|E|×q}, defined on the site graph's own edges (E_r = E; decision D4).

**Posterior** (paper Proposition 1, extended in §3): `posterior_ising` builds

- h_i = b + Σ_j θ_j Λ_ij + Σ_j ω_j 1[Λ_ij ≠ 0]
- J_ik = β w_ik + Σ_r θᴾ_r Π_ikr   on the site edges.

**Prior** (Proposition 2): `prior_ising` builds h_i = b + F, where F = Σ_c ½ log(Z_c(+1)/Z_c(−1)), and
J_ik = β w_ik. F = 0 for symmetric potentials (Proposition 3).

**Parameter vector layout** (`LabelModelParams.to_vector`):
[α (m), θ (m), αᴾ (q), θᴾ (q), γ (#cliques), ω (m), ζ (m), b, β].
Optimizers operate on this vector. Gradients are returned in the same structure.

**Cliques.** Disjoint groups of ≥ 2 correlated unary LFs. The feature is
φ_c(λ) = #{j < l ∈ c : λ_j = λ_l ≠ 0}, which is flip-invariant. `clique_logweights` enumerates all 3^|c| vote
patterns and returns their log-weights given y = +1 and y = −1. Every model moment is an exact finite sum over
this table.

## 3. Class-conditional unary potentials (decision R9)

**Parametrization.**

ψ_j(λ, y) = α_j 1[λ≠0] + θ_j λy + ω_j 1[λ≠0] y + ζ_j λ.

Over (λ, y) ∈ {±1} × {±1}, the four features take the values (1,1,1,1), (1,−1,1,−1), (1,−1,−1,1) and
(1,1,−1,−1). These are orthogonal, so they span every table of ψ restricted to λ ≠ 0. The remaining row
ψ(0, y) only adds a y-dependent constant, which the class bias b absorbs. The parametrization is therefore
fully general, and ω = ζ = 0 recovers the symmetric model.

**Meaning.**
- ω: class-dependent coverage. The LF votes more often on one class.
- ζ: vote bias. The LF prefers to vote +1 (or −1) whatever the label.

**Gradient** (`model.gradient`). By the tower property, with p_i = P_prior(y_i = +1) = (1 + ⟨y_i⟩_prior)/2,
P₁ = Σ_i p_i and P₀ = n − P₁, the per-image model terms are:

| parameter | data term | model term |
|-----------|-----------|------------|
| θ_j | Σ_i Λ_ij ⟨y_i⟩_post | P₁ E[λ_j \| +1] − P₀ E[λ_j \| −1] |
| α_j | Σ_i 1[Λ_ij ≠ 0] | P₁ E[c_j \| +1] + P₀ E[c_j \| −1] |
| ω_j | Σ_i 1[Λ_ij ≠ 0] ⟨y_i⟩_post | P₁ E[c_j \| +1] − P₀ E[c_j \| −1] |
| ζ_j | Σ_i Λ_ij | P₁ E[λ_j \| +1] + P₀ E[λ_j \| −1] |
| γ_c | Σ_i φ_c(Λ_ic) | P₁ E[φ_c \| +1] + P₀ E[φ_c \| −1] |
| b | Σ_i ⟨y_i⟩_post | Σ_i ⟨y_i⟩_prior |
| β | Σ_E w ⟨y_i y_k⟩_post | Σ_E w ⟨y_i y_k⟩_prior |
| θᴾ_r, αᴾ_r | as in the symmetric case (closed form) | |

Here c_j = 1[λ_j ≠ 0]. For symmetric potentials, E[λ | −1] = −E[λ | +1] and E[c | −1] = E[c | +1], so the
model terms do not depend on p_i and no prior sampling is needed (Theorem 1). The gradient refuses to run
without prior moments when the potentials are not symmetric, or when b or β is learned.

**Verified.** `tests/test_model.py` checks the posterior, the prior marginal and every gradient coordinate
against brute-force enumeration of the full joint distribution, including the class-conditional terms, a
correlation clique and a pairwise LF.

## 4. D-Wave QPU path (`qws.solvers.qpu`, decisions R6, D7–D10)

`QPUAnnealingSolver` drives a **structured** child sampler: `DWaveSampler` in production,
`MockDWaveSampler` offline.

1. **Sign convention.** Ocean minimizes Σ h′s + Σ J′ss, so h′ = −h and J′ = −J (`to_dimod_bqm`).
2. **Scaling.**
   κ = max(1, max|h|/h_max, max|J|/J_max, csf · max|J|/|J_min|),
   with the ranges read from `child.properties` (never hard-coded).
   - The problem is divided by κ.
   - Chains are ferromagnetic couplers of strength csf · max|J|/κ, which is ≤ |J_min| by construction.
   - `auto_scale=False`, so κ is the *only* rescaling between our problem and the hardware.
   - κ ≥ 1: problems that already fit are not rescaled. Otherwise calibration could never change the
     effective temperature (D8).
   - Default csf = 1.5.
3. **Embedding.**
   - `minorminer.find_embedding` with a fixed seed, up to 10 attempts.
   - Isolated variables get free qubits.
   - Every embedding is verified with `is_valid_embedding` against the device graph before use (D24).
   - Cached by the tuple of graph keys (sha256 of n and the sorted edges). Problems with the same graph but
     different coefficients, such as every training step for a given image, reuse the embedding.
4. **Parallel embedding.** Problems in one call are packed greedily. A problem is added to the current
   submission while the disjoint union still embeds; otherwise a new submission starts. Logical variables are
   relabelled (k, i).
   - A union is not attempted if its variable count exceeds 60% of the qubits (`max_fill`).
   - A packing attempt uses one short minorminer run (`pack_timeout` = 5 s).
   - Failed unions are cached and never retried.
   - The grouping for a given tuple of graph keys is cached, so repeated training steps cost no embedding time.
   - Without these rules, failing minorminer runs were repeated on every training step, which made
     training through the QPU path impractically slow (decision D22).
5. **Unembedding.** Majority vote on broken chains (`unembed_sampleset`). The chain-break fraction is
   reported, and calls above 10% are flagged `low_fidelity`.
6. **Accounting.** QPU access time and wall time of a shared submission are attributed to each problem in
   proportion to its number of variables. κ, chain strength, embedding statistics, packing and topology are
   stored in `SampleSet.info`.
7. **Mock.** `QPUAnnealingSolver.mock(topology_shape)` uses `MockDWaveSampler` (Pegasus P_shape; P16 is
   Advantage-sized). Its physics substitute is Ocean's simulated annealing with a **fixed** β range (0.1, 4),
   so it behaves like a device at fixed physical temperature (D9).

Production factory: `QPUAnnealingSolver.dwave(solver=None)`, which defaults to a Pegasus (Advantage) solver,
or `solver: {name: anneal_qpu}` in a config.

## 5. Classical solvers

- **Exact.** Configuration z ↦ y_i = 1 − 2·bit_i(z), the same convention as the QAOA basis. Enumeration is
  limited to n ≤ 20. Sampling is multinomial over the exact probabilities.
- **Gibbs** (`mcmc.py`).
  - Greedy graph colouring; each colour class is updated in parallel. Heat bath:
    P(y_i = +1) = (1 + tanh(β m_i))/2, with m_i = h_i + Σ_k J_ik y_k.
  - Chains are vectorized across samples.
  - Persistent chains are keyed by `problem.meta["chain_key"]`.
  - `gibbs_polish` runs k sweeps from given seeds; it never increases TV to the target (paper Proposition 10).
- **Simulated annealing.**
  - The same heat-bath kernel on a geometric β schedule from 0.05/s to 20/s, with s = max|coef|.
  - `minimize` ends with greedy single-flip descent.
  - `sample` anneals to the target β and stops.
- **Graph cut.**
  - `solve_submodular_qubo` uses PyMaxflow: x = 1 means the node is on the sink side.
  - A pair term b x_i x_k with b ≤ 0 is rewritten as b x_i + (−b) x_i(1 − x_k): an edge k→i of capacity −b.
  - `GraphCutSolver` refuses any J < 0.
- **QPBO / roof duality** (`roof_dual`).
  - Doubled energy over (x, y), where y is meant to equal 1 − x:
    E2 = c + ½ Σ a_i(x_i + 1 − y_i) + ½ Σ_{b<0} b(x_i x_j + (1−y_i)(1−y_j)) + ½ Σ_{b>0} b(x_i(1−y_j) + (1−y_i)x_j).
  - E2 is submodular in (x, y), so one max-flow minimizes it exactly.
  - Variables with y_i = 1 − x_i are weakly persistent. min E2 is a lower bound on min E.
  - `QPBOSolver` completes the unlabeled remainder with an exact solver (≤ 20 variables) or simulated annealing.

## 6. Real images (decision R5)

**Oxford-IIIT Pet** (`qws.data.pet`).
- Official layout: `images/*.jpg`, `annotations/trimaps/*.png`, `annotations/{trainval,test}.txt`.
- Trimap 1 → +1 (pet), 2 → −1, 3 → 0 (ignore).
- Images are resized so the longer side is 128 px: bilinear for images, nearest-neighbour for trimaps, so
  labels never blend.
- `download_pet` fetches the two archives from `www.robots.ox.ac.uk`. That host is currently refused by the
  environment's network policy.

**Sites.** SLIC superpixels: about 150–200 requested, compactness 10.

**Site ground truth (evaluation only).** The sign of the mean of the pixel labels, where ignore pixels count as 0.

**Generic LFs** (`qws.lf.natural`). None uses ground truth.

| LF | rule |
|----|------|
| `border_bg` | pixels in a border band (6% of the shorter side) vote −1 |
| `center_fg` | pixels in a central ellipse (semi-axes 35% of each dimension) vote +1 |
| `color_model` | 8³-bin RGB histograms (Laplace-smoothed), background from the border band and foreground from the centre ellipse; vote ±1 where the log-likelihood ratio exceeds ±1 |
| `saliency` | spectral-residual saliency (Hou & Zhang 2007) at 64×64, smoothed with σ = 2.5; top 20% vote +1, bottom 40% vote −1 |
| `color_contrast` (pairwise) | edges above the 90% quantile of the image's colour-distance distribution vote "different", below the 30% quantile vote "same" |

**Metrics.** `iou`, `dice`, `accuracy` and `ece` take `ignore_zero=True` to drop trimap-boundary pixels.

**Pet-layout generator** (`qws.data.petlike`).
- Content: a textured, wobbly ellipse ("pet") at a random position near the centre, on a textured background.
  Foreground and background colours differ by a random direction of norm 0.35, and a low-frequency
  illumination field is added.
- Files: JPEG (quality 92) images, PNG trimaps with a 2-pixel boundary band, and split lists.
- Purpose: it exists only to exercise the real-image pipeline while Pet is unreachable (decision D12).

## 7. Learning

- **Warm start.**
  - Fit the independent model (β = 0, no pairwise LFs). Its posterior is exact: ⟨y_i⟩ = tanh(h_i).
  - Initial θ = 0.5; α from observed coverage, via the closed form cov = 2e^α cosh θ/(1 + 2e^α cosh θ).
  - 300 Adam steps.
  - Pairwise LFs start at θᴾ = 0.5. ω and ζ start at 0.
- **Trainer** (`learn/trainer.py`).
  - Each step: posterior Ising per image → `solver.sample` → optional polishing → moments → gradient.
  - The gradient is divided by the number of sites in the batch, then an Adam step is taken.
  - Projection: β ≥ 0 and mean θ > 0 (label-flip symmetry breaking).
  - Gradients of b, β, ω and ζ are zeroed unless `learn_b`, `learn_beta` or `class_conditional` is set.
  - Prior moments are needed whenever b or β is learned, or the potentials are not symmetric. They are sampled
    once per distinct site graph (signature: n, edges, rounded weights) and shared by all images with that graph.
- **EM mode.** Closed-form M-step θ = artanh(agree/cover), α from coverage. Symmetric singleton model with fixed
  b and β only.
- **Temperature calibration** (`CalibratedSampler`).
  - Each call submits `p.scaled(s)` and estimates β̂ by maximum pseudo-likelihood relative to the unscaled p.
    The estimate is concave in β′ and computed by bounded 1-D optimization on [0, 1000].
  - Update: s ← s / clip(β̂^damping, 1/4, 4), clipped to [10⁻³, 10³].
  - The step bound (D11) prevents an overshoot of about 1000× when every spin is aligned with its local field,
    which makes β̂ hit its cap.

## 8. Downstream U-Net (decision R10)

- **Network.** Depth 3, 16 base channels. Each block is (conv3×3 → BN → ReLU) × 2. 2×2 max-pooling;
  transposed-conv upsampling with skip concatenation; 1×1 logit head.
- **Loss.** Noise-aware expected BCE, −mean_valid[μ log σ(z) + (1 − μ) log(1 − σ(z))]. For μ ∈ {0, 1} it equals
  the BCE, and for soft μ its minimizer is z = logit(μ) (tested).
- **Training.**
  - Inputs resized to 64×64; Adam with lr 3·10⁻³, cosine schedule, batch size 8.
  - Horizontal-flip augmentation. Predictions are resized back to the original resolution.
- **Pipeline** (`downstream/pipeline.py`).
  - The label model is trained on the `trainval` images.
  - U-Nets are trained on four signals: structured soft masks, independent soft masks, majority-vote hard
    masks, and ground truth (upper bound).
  - All are evaluated on the `test` images. For reference, the label model is also applied directly to the
    test images.

## 9. Multi-class (`qws.multiclass`)

- **Potts energy.** E(y) = offset − Σ_i u[i, y_i] − Σ Jt_ik 1[y_i = y_k].
- **K-class posterior.**
  - Votes Λ ∈ {0..K}, where 0 abstains and c means class c − 1.
  - u_ic = b_c + Σ_j θ_j 1[Λ_ij = c+1].
  - Jt_ik = β w_ik + 2 Σ_r θᴾ_r Π_ikr. The constant −θᴾΠ from (2·1[=] − 1) is dropped.
- **Symmetric normalizer.** Z_j = 1 + e^{α+θ} + (K−1)e^α does not depend on y. So the model moments are
  constants: E[1(Λ = y+1)] = e^{α+θ}/Z_j and E[1(Λ ≠ 0)] = (e^{α+θ} + (K−1)e^α)/Z_j.
- **One-hot QUBO.**
  - x_ic = 1[y_i = c], with E_K = offset − Σ u x − Σ Jt Σ_c x_ic x_kc + A Σ_i (1 − Σ_c x_ic)².
  - The penalty expands, using x² = x, to A(1 − Σ x + 2 Σ_{c<d} x_c x_d).
  - Default A = max_{i,c}(|u_ic| + Σ_k |Jt_ik|) + 10⁻³, the sufficient bound of paper Proposition 11.
- **Decoding.** An infeasible site takes the active class with the largest u_ic. If no class is active, it
  takes the argmax over all classes.
- **Samplers.** Potts heat-bath Gibbs over colour classes; Potts simulated annealing.
- **Learning.** Adam on (α, θ, αᴾ, θᴾ) with b and β fixed, using posterior class probabilities and edge
  agreement probabilities from persistent Potts Gibbs chains.

## 10. QAOA

- **Basis.** Qubit i is variable i, and measured bit 0 means y_i = +1. Qiskit count strings are little-endian.
- **Cost.** Normalized by max|coef|.
- **Gates.** RZ(−2γh_i), RZZ(−2γJ_ik), RX(2β).
- **Angles.** Linear-ramp initialization, then COBYLA on the exact statevector expectation (n ≤ 16).
- **SamplerV2 path.**
  - Circuits are transpiled to the backend's ISA if a backend is given, then run with `sampler.run([qc], shots)`.
  - Counts come from the `meas` register created by `measure_all`.
  - `QAOAQiskit.ibm()` builds `SamplerV2(mode=backend)` from `QiskitRuntimeService`.

## 11. Tests (96, plus 2 opt-in hardware tests)

| File | What it guarantees |
|------|--------------------|
| `test_ising.py` | Energy identities Ising ↔ QUBO ↔ Ocean ↔ Pauli (exhaustive); doc 02 §8 fixture; conditioning exactness; strong-field fixing preserves minimizers |
| `test_graph_lf.py` | Graph construction, lifting rule, LF value checks |
| `test_model.py` | Posterior, prior and gradient against brute-force joint enumeration (symmetric, cliques, class-conditional, learned b and β); closed forms; warm-start recovery |
| `test_solvers.py` | Graph cut = exact (ferromagnetic); refusal on frustration; Gibbs and polishing vs exact; SA optimality; Ocean sign convention; QAOA statevector = Qiskit; SamplerV2 path (statevector and Aer); roof-dual persistency and bound |
| `test_qpu_mock.py` | Range reading, κ rule, exact optima through mock Pegasus, packing, embedding cache, serial mode, calibration through the mock |
| `test_learn.py` | Pseudo-likelihood consistency; calibration convergence and bounded steps; trainer recovery (gradient, EM); learning b and β; inference helpers |
| `test_decompose_eval.py` | Tiles partition; block-CD monotone and block-locally optimal; block Gibbs vs exact; metrics; benchmarks; M1 criterion; CLI; config merge (solver specs are replaced, not merged, when the solver changes) |
| `test_data.py` | Pet layout, splits, resizing, trimap/ignore handling, LF informativeness, end-to-end Pet-format pipeline |
| `test_downstream.py` | Noise-aware loss properties, U-Net shapes, learning a toy task, pipeline smoke test |
| `test_multiclass.py` | One-hot energy identity, Proposition 11 by exhaustive enumeration, decoding repair, Potts Gibbs vs exact, MAP routes, brute-force K-class gradient, learning recovery |
| `test_qpu.py` (opt-in) | Real D-Wave and IBM smoke tests |

## 12. Results generation

- `scripts/generate_results.py` writes reports 01–04, and `scripts/generate_results_extensions.py` writes
  reports 05–11. Both take `--quick`, and the extensions script takes `--only ...` to run a subset.
- Shared helpers are in `scripts/_report.py`. `results/raw/headlines.json` feeds the index `results/README.md`.
- All seeds are fixed in the scripts. Raw numbers are written to `results/raw/*.json`.
