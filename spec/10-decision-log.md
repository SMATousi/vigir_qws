# 10 — Decision Log

Every decision that shaped the project, in two groups:

- **R-entries** were made by the project owner. They are also listed in [open-questions](open-questions.md).
- **D-entries** are implementation decisions made while building `qws`.

Each entry records the alternatives considered and the rationale, so later changes can revisit it knowingly.
Implementation details are in [doc 09](09-implementation-notes.md).

## Owner decisions

| ID | Date | Decision | Consequence in the code / spec |
|----|------|----------|--------------------------------|
| R1 | 2026-10-20 | Hardware-agnostic Ising/QUBO core, with annealing and gate-based (QAOA) backends | `IsingProblem` is the only solver input; backends in `qws.solvers` |
| R2 | 2026-10-20 | Binary segmentation first, multi-class later | Binary model throughout; `qws.multiclass` added in Phase 4 |
| R3 | 2026-10-20 | Optimize learning and inference jointly | `Trainer` (learning) + `infer_map` / `infer_marginals` |
| R4 | 2026-10-20 | Generic LF interface (unary and pairwise) | `qws.lf` interfaces; generic LF libraries only |
| R5 | 2026-11-03 | First real dataset: **Oxford-IIIT Pet** (trimaps binarized) | `qws.data.pet`, `qws.lf.natural`, `dataset.kind: pet` |
| R6 | 2026-11-03 | D-Wave Leap with QPU time; primary target **Advantage (Pegasus)**; hybrid solvers only as a labelled baseline | `QPUAnnealingSolver.dwave()` defaults to a Pegasus solver; mock tests use Pegasus P16 |
| R7 | 2026-11-03 | Gate-based provider: **IBM Quantum** through Qiskit | `QAOAQiskit` SamplerV2 path; `QAOAQiskit.ibm()` |
| R8 | 2026-11-03 | **Learn** β and b | `TrainConfig.learn_b`, `learn_beta`; report 06 |
| R9 | 2026-11-03 | Support **class-conditional** LFs | ω, ζ parameters; `TrainConfig.class_conditional`; report 07 |
| R10 | 2026-11-03 | Add a **small downstream trainer** | `qws.downstream` (U-Net, noise-aware loss); report 09 |
| R11 | 2026-11-03 | Open a pull request into `main` | `main` created at the first commit; PR SMATousi/vigir_qws#1 |
| R13 | 2026-11-03 | License: **MIT** | `LICENSE`, `pyproject.toml` |

## Implementation decisions

### D1. Python package `qws` with optional extras and lazy SDK imports
- **Alternatives:** one monolithic dependency set; separate packages per backend.
- **Rationale:** the classical core must run without quantum SDKs (doc 05 §1). Lazy imports keep `import qws`
  fast and let tests skip cleanly when an extra is missing.
- **Where:** `pyproject.toml` (`anneal`, `gate`, `downstream`, `dev`); imports inside functions in
  `solvers/anneal.py`, `solvers/qpu.py`, `solvers/qaoa.py`, `downstream/`.

### D2. One energy convention inside the code; conversions only at backend boundaries
- **Convention:** E(y) = offset − Σ h y − Σ J y y, with p ∝ e^{−E} and J > 0 ferromagnetic.
- **Alternatives:** adopt Ocean's or Qiskit's convention internally.
- **Rationale:** SDK conventions differ, so a single internal convention plus explicit, tested converters avoids
  silent sign errors. `SampleSet` energies are always recomputed in our convention.
- **Where:** `ising/convert.py`; round-trip tests in `test_ising.py`.

### D3. Measured bit 0 ↔ y = +1
- **Rationale:** y_i is the Z_i eigenvalue (doc 02 §4). The same convention is used by the exact enumerator,
  so statevector and Qiskit probabilities can be compared index by index.
- **Where:** `solvers/exact.py`, `solvers/qaoa.py`.

### D4. Pairwise LFs vote only on site-graph edges (E_r ⊆ E)
- **Alternatives:** arbitrary site pairs.
- **Rationale:** the posterior's interaction graph stays equal to the sparse site graph, which keeps
  embeddings short (report 05). Non-local votes would densify the problem and lengthen chains.
- **Where:** `LFOutputs.P` is indexed by `sites.edges`. Recorded as a deviation in doc 08.

### D5. Warm start from the independent model; positive initial accuracies; projection mean θ > 0
- **Rationale:** the marginal likelihood is non-concave and has a global label-flip symmetry. Starting from the
  exactly solvable β = 0 model with θ > 0 selects the "LFs better than random" branch (doc 01 §6).
- **Where:** `model.warm_start`, `model.project`.

### D6. Gradient normalized by the number of sites; Adam
- **Rationale:** it makes learning rates comparable across image sizes and batch sizes.
- **Where:** `learn/trainer.py`.

### D7. D-Wave: explicit embedding instead of `EmbeddingComposite`
- **Alternatives:** Ocean's `EmbeddingComposite` (re-embeds every call) or `LazyFixedEmbeddingComposite`.
- **Rationale:** training submits the same graph with new coefficients thousands of times. Explicit
  `minorminer` embeddings cached by graph shape avoid re-embedding. They also enable packing several problems
  into one submission, and give direct control of chain strength and of the scaling that calibration depends on.
- **Where:** `solvers/qpu.py`.

### D8. Range-aware scaling: κ ≥ 1, chains inside J_min, `auto_scale` off
- **Alternatives:** Ocean's `auto_scale`; always normalizing to the full range.
- **Rationale:** both alternatives rescale whatever the calibrator submits back to the full range, which makes
  the effective temperature uncontrollable. With κ = max(1, …), problems are shrunk only when they would
  exceed the device ranges, and the calibrated scale reaches the hardware unchanged. Found during
  implementation: the first version normalized every problem, and that would have broken calibration on the
  real QPU.
- **Where:** `QPUAnnealingSolver.scale_factor`; test `test_scale_factor_only_shrinks` and the calibration test
  through the mock.

### D9. The offline QPU stand-in is `MockDWaveSampler` with a fixed-temperature physics substitute
- **Alternatives:** the mock's default steepest-descent substitute; an unstructured simulated annealer.
- **Rationale:** the real topology, ranges and timing fields test the full code path. A fixed β range makes the
  mock's temperature depend on the submitted scale, like a physical device, so calibration can be validated
  offline (report 05 recovers the mock's β_eff ≈ 4).
- **Where:** `QPUAnnealingSolver.mock`.

### D10. Chain strength 1.5 × max|J| (after scaling); majority-vote unembedding; 10% chain-break flag
- **Rationale:** these are common starting points. They are exposed as parameters and every call reports its
  chain-break fraction, so they can be tuned on hardware. They are not tuned yet (roadmap 3.1).

### D11. Temperature calibration with bounded steps (factor ≤ 4 per update) and a β̂ cap of 1000
- **Rationale:** when every sampled spin is aligned with its local field, the pseudo-likelihood grows without
  bound in β, and the estimator returns its cap. An unbounded update then shrinks the scale about 1000×. This
  was observed in report 03 before the fix; bounded steps make calibration converge (regression test added).
- **Where:** `learn/temperature.py`.

### D12. A generated dataset in the exact Oxford-IIIT Pet layout stands in for Pet offline
- **Context:** the hosts `www.robots.ox.ac.uk`, `thor.robots.ox.ac.uk` and `huggingface.co` are refused by the
  environment's network policy.
- **Alternatives:** wait for network access; fabricate "Pet-like" numbers (rejected).
- **Rationale:** it exercises every real-image component (JPEG, PNG trimaps with an ignore band, split files,
  resizing, superpixels, generic LFs, ignore-masked metrics) end to end. Reports 08–09 say explicitly that these
  are not Pet results.
- **Where:** `qws/data/petlike.py`.

### D13. Class-conditional parametrization (α, θ, ω, ζ); pairwise LFs stay symmetric
- **Alternatives:** raw 3×2 log-potential tables.
- **Rationale:** the four orthogonal features span every table (doc 09 §3), keep the symmetric model as the
  special case ω = ζ = 0, and give interpretable parameters (coverage asymmetry and vote bias). Pairwise
  "same/different" votes have no natural class asymmetry.

### D14. Generic natural-image LFs: border, centre, colour model, saliency, colour contrast
- **Rationale:** these are weak, dataset-agnostic priors that need no labels. Thresholds are chosen a priori,
  not tuned on ground truth. Saliency is kept even though it is weak, because recognizing a useless LF is
  exactly what the label model must do (report 08).
- **Where:** `qws/lf/natural.py`.

### D15. Small U-Net at 64×64 with the noise-aware loss; four training signals compared
- **Rationale:** the goal is a controlled comparison of training signals, not state-of-the-art segmentation. The
  ground-truth-trained network is the upper bound and shows whether the network is the bottleneck (it is, on
  the small generated set: report 09).
- **Where:** `qws/downstream/`.

### D16. QPBO implemented in-house (doubling construction + PyMaxflow)
- **Alternatives:** an external QPBO library.
- **Rationale:** there is no maintained pip-installable QPBO for this Python version, and the doubling
  construction is short and testable. Persistency and the bound are verified against exhaustive enumeration.
- **Where:** `solvers/graphcut.py`.

### D17. Multi-class uses one-hot encoding only, with the Proposition-11 penalty plus a 10⁻³ margin, and repair decoding
- **Alternatives:** domain-wall encoding (K − 1 qubits per site); a binary (log) encoding.
- **Rationale:** one-hot is the encoding the paper analyzes, and its sufficient penalty is proven. Report 11
  quantifies its cost: more reads are needed, because class changes pass through infeasible states. Domain-wall
  is the natural next step (doc 08 deviations).

### D18. IBM access through SamplerV2, with ISA transpilation
- **Rationale:** IBM Runtime no longer supports `backend.run` for hardware, and SamplerV2 is the supported
  primitive. The same code path runs offline with `StatevectorSampler` and the Aer `SamplerV2`.

### D19. `graphcut` refuses frustrated problems
- **Rationale:** returning a wrong answer silently is worse than failing (doc 05 §3). Callers fall back to SA,
  QPBO or quantum solvers.

### D20. Prior-phase moments shared per site-graph signature
- **Rationale:** the prior Ising model depends only on the graph and the parameters (doc 03 §2.2). Sharing one
  sampling call between images with identical graphs (for example, equal-size pixel grids) cuts sampling cost.

### D21. Results are committed (`results/`) and regenerated by scripts
- **Alternatives:** ignore run outputs.
- **Rationale:** the owner asked for detailed results in the repository. The scripts make every number
  reproducible, and the raw JSON is kept alongside the Markdown reports.

### D22. Packing heuristics and failure caching in the QPU path
- **Context:** found when running `qws demo --solver anneal_mock`. Failed minorminer attempts for unions that
  do not fit were repeated on every training step and dominated run time.
- **Decision:** cache failed unions; skip unions above 60% of the qubit count; one 5-second minorminer attempt
  per packing trial (single problems keep 10 attempts and a 60-second timeout); cache the grouping per tuple of
  graph keys.
- **Where:** `QPUAnnealingSolver._groups`, `embed`.

### D23. A solver spec that names a different solver replaces the base spec in config merges
- **Context:** `solver: {name: anneal_mock}` inherited the default Gibbs option `num_sweeps`, which the mock
  QPU rejects. Ocean's `neal` silently accepted it, changing its schedule.
- **Decision:** `experiment.merge` replaces the whole spec when `name` changes, and merges otherwise.
- **Where:** `qws/experiment.py`; `test_config_merge_replaces_solver_spec_on_name_change`.

### D24. Every embedding is verified before use
- **Context:** an Advantage-size (P16) mock run failed because an embedding returned by minorminer under a
  timeout used a coupler the device does not have.
- **Decision:** reject any embedding that fails `dwave.embedding.is_valid_embedding` against the device graph,
  and treat it as a failed attempt. Real-QPU runs would otherwise fail at submission, after spending
  programming time.
- **Where:** `QPUAnnealingSolver.embed`.
