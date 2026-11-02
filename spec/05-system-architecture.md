# 05 — System Architecture

This is the layout as built in v0.1. The module boundaries and the solver interface
are normative.

## 1. Language and packaging

- Python ≥ 3.10, with the package name `qws`.
- Core dependencies: `numpy`, `scipy`, `scikit-image` (superpixels), and a
  max-flow library for `graphcut`.
- Optional extras:
  - `qws[anneal]` for the annealing SDK;
  - `qws[gate]` for the gate-based SDK and simulator;
  - `qws[dev]` for tests and linting.
- The classical core MUST install and pass all tests without the quantum
  extras.

## 2. Modules

```
qws/
  ising/       # problem.py (IsingProblem), convert.py (QUBO / Ocean / Pauli), reduce.py (conditioning, strong-field fixing)
  graph/       # sites.py: pixel grid, SLIC superpixels, region-adjacency graph, contrast weights
  lf/          # base.py (interfaces, apply_lfs, lift_votes), library.py (generic example LFs)
  model/       # params.py (LabelModelParams), label_model.py (Ising builders, moments, gradient, warm start)
  solvers/     # base.py, exact.py, mcmc.py (gibbs, sa, polish), graphcut.py, anneal.py (Ocean), qaoa.py
  decompose/   # tiling.py: grid/BFS tiles, block-coordinate-descent MAP, block Gibbs
  learn/       # optim.py (Adam), temperature.py (PL estimator, CalibratedSampler), trainer.py
  eval/        # synthetic.py, metrics.py, benchmark.py
  io.py        # JSON result store
  experiment.py, cli.py   # config-driven runs: `qws demo`, `qws run <config.yaml>`
```

**Dependency direction** (the trainer additionally imports `solvers` for polishing):

```
lf → graph → model → ising → solvers → decompose → learn → eval
```

`ising` MUST NOT import from `solvers`, and solvers MUST NOT import from
`model` or `learn`. Solvers only ever see an `IsingProblem`.

## 3. Solver interface (normative)

```python
class Solver(Protocol):
    name: str
    capabilities: set[str]   # subset of {"minimize", "sample", "marginals", "log_partition"}

    def minimize(self, problems: Sequence[IsingProblem], **params) -> list[SampleSet]: ...
    def sample(self, problems: Sequence[IsingProblem], num_samples: int, **params) -> list[SampleSet]: ...
```

```python
@dataclass(frozen=True)
class SampleSet:
    samples: np.ndarray          # (S, N) int8 in {-1,+1}, columns ordered as problem.sites
    energies: np.ndarray         # (S,) under OUR convention (doc 02 §1), recomputed classically
    counts: np.ndarray           # (S,) multiplicities
    info: dict                   # timing breakdown, κ, β̂, chain breaks, device, seeds, hashes
```

Interface rules:
- Every call takes a **sequence** of problems, so backends can batch them
  (doc 04 §6).
- `energies` MUST be recomputed from `samples` using the `IsingProblem`.
  Device-reported energies MAY be stored in `info`, but are never trusted
  directly.
- Backends MUST raise `UnsupportedProblem` instead of silently returning
  wrong answers, for example `graphcut` on a frustrated problem.

## 4. Core data types

| Type | Contents |
|------|----------|
| `Sites` | Site ids, pixel↔site map, adjacency `E` with weights `w` |
| `LFOutputs` | Sparse `Λ` (N×m, int8), sparse `Π` per pairwise LF, LF names |
| `LabelModelParams` | $\alpha, \theta, \gamma, \alpha^P, \theta^P, b, \beta$, and symmetric/class-conditional flags |
| `IsingProblem` | doc 02 §1 |
| `SampleSet` | §3 |
| `RunRecord` | Config, problem hashes, sample sets, metrics; serialized for reproducibility |

## 5. Configuration

All experiments are driven by a declarative config: YAML, validated into
typed objects. It covers dataset, site construction, LFs, model flags,
solver choice and parameters, decomposition, learning hyperparameters, and
seeds. A run is fully determined by its config together with the results
store.

## 6. Testing requirements

- **Energy-identity tests:** Ising ↔ QUBO ↔ SDK formats, by exhaustive
  enumeration (doc 02 §3, §5).
- **Exactness tests:** `exact` against analytic results (doc 01 §3.4 closed
  forms), and `graphcut` against `exact` on random ferromagnetic problems.
- **Fixtures:** the worked example (doc 02 §8).
- **Learning test:** on synthetic data, using the `exact` or `gibbs` solver,
  learned $\theta$ is within tolerance of the generating $\theta$.
- Quantum backends are tested against **local simulators or mock samplers**
  in CI. Real-hardware tests are opt-in only, because they need credentials
  and cost money.
