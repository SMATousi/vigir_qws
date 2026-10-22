# vigir_qws — Specification

**Status:** Draft v0.1, written before any code exists.

This folder defines **quantum weak supervision (QWS)**: data programming
for image segmentation, written as Ising/QUBO problems. That form lets the
problem run on quantum annealers and on gate-based quantum hardware.

## Reading order

| # | Document | What it covers |
|---|----------|----------------|
| 00 | [Vision and scope](00-vision-and-scope.md) | Goal, motivation, where quantum computing can and cannot help, non-goals, milestones |
| 01 | [Data programming for segmentation](01-data-programming-segmentation.md) | The classical probabilistic model, the three computational tasks, and their complexity |
| 02 | [Ising / QUBO formulation](02-ising-qubo-formulation.md) | Exact mapping from the label model to Ising and QUBO, sign conventions, scaling |
| 03 | [Joint learning and inference](03-learning-and-inference.md) | The hybrid classical–quantum algorithm for learning LF parameters and inferring masks |
| 04 | [Quantum backends](04-quantum-backends.md) | Annealing (QUBO-native) and gate-based (QAOA) backends, embedding, decomposition for large images |
| 05 | [System architecture](05-system-architecture.md) | Software modules, core data types, and the solver interface |
| 06 | [Evaluation](06-evaluation.md) | Datasets, baselines, metrics, and acceptance criteria |
| 07 | [Multi-class extension](07-multiclass-extension.md) | Planned extension from binary to K-class segmentation |
| — | [Open questions](open-questions.md) | Decisions that are still open |
| — | [Paper draft](paper-draft.md) | Theory-only short paper: Ising reductions, learning theorem, sampler-fidelity analysis |

## Decisions so far

| Topic | Decision |
|-------|----------|
| Quantum paradigm | A hardware-agnostic **Ising/QUBO core**, with two backend families: quantum annealing (primary fit) and gate-based QAOA |
| Segmentation setting | **Binary** foreground/background first; multi-class is a planned extension (doc 07) |
| Optimization targets | **Both** label-model learning (parameter estimation) **and** inference (MAP mask and posterior marginals), done jointly |
| Labeling functions | **Generic**: any function that returns a per-site vote or abstains; no domain chosen yet |

## Conventions used in every document

- **Spins** $y_i \in \{-1, +1\}$, where $+1$ means foreground and $-1$ means background.
  **Binary variables** $x_i \in \{0, 1\}$, with $y_i = 2x_i - 1$.
- **Energies are minimized.** Probabilities follow $p(y) \propto \exp(-E(y))$.
- **Our Ising sign convention**: $E(y) = -\sum_i h_i y_i - \sum_{(i,k)} J_{ik} y_i y_k$,
  so a positive $J$ is ferromagnetic (it favours equal labels). Hardware SDKs
  may use the opposite sign; each backend converts at its boundary (doc 02 §5).
- A **site** is the unit that gets a label: a pixel or a superpixel.
- "LF" means labeling function. $\Lambda$ is the LF output matrix.
- Key words such as MUST, SHOULD, and MAY follow RFC 2119.
