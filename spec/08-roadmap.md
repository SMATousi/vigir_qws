# 08 — Implementation Roadmap

This document turns docs 00–07 into ordered work packages. Each package lists its
deliverables and the tests that close it. Status markers:

- ✅ done
- 🟡 implemented, but needs real hardware or user input to validate
- ⛔ blocked on QPU access or on a user decision

## Phase 1 — Classical reference implementation (milestone M1)

| WP | Package | Deliverables | Closing tests | Status |
|----|---------|--------------|---------------|--------|
| 1.0 | Scaffolding | `pyproject.toml` with optional extras `anneal`, `gate`, `dev`; `qws` package skeleton; pytest config | `pip install -e .[dev]` works and `pytest` runs | ✅ |
| 1.1 | `qws.ising` | `IsingProblem`; vectorized energies; QUBO and Pauli conversion; conditioning (doc 02 §6.2); strong-field fixing (§6.1); scaling and dynamic-range report (§7) | Energy identity Ising ↔ QUBO by enumeration (N ≤ 12); doc 02 §8 fixture; exactness of conditioning | ✅ |
| 1.2 | `qws.graph` | Pixel-grid sites (4/8-neighbour) and SLIC superpixel sites; contrast-sensitive affinities; site ↔ pixel maps | Edge counts; weights in (0, 1]; round trip between pixels and sites | ✅ |
| 1.3 | `qws.lf` | Unary and pairwise LF interfaces; `apply_lfs` producing Λ and Π; majority lifting to superpixels (Q3 default); generic example LFs | Vote shapes and value ranges; lifting rule | ✅ |
| 1.4 | `qws.model` | `LabelModelParams`; posterior and prior Ising builders (doc 01 §3.2, §3.4); closed-form and clique-enumeration moments; independent-model warm start | Closed forms vs. brute force; posterior builder vs. joint-model enumeration | ✅ |
| 1.5 | `qws.solvers` (classical) | `exact`, `sa`, `gibbs` (graph-coloured, persistent chains, polishing), `graphcut` (refuses negative couplings) | `exact` vs. analytic results; `graphcut` = `exact` on random ferromagnetic problems; Gibbs marginals ≈ exact | ✅ |
| 1.6 | `qws.learn` | Gradient (Theorem 1 of the paper); Adam trainer with projection; EM mode; pseudo-likelihood temperature estimator with scale calibration; polishing | Parameter recovery on synthetic data with `exact` / `gibbs`; PL estimator recovers a known β | ✅ |
| 1.7 | `qws.decompose` | Tiling; block-coordinate-descent MAP; block Gibbs | Block-CD energy never increases; block-Gibbs marginals ≈ exact | ✅ |
| 1.8 | `qws.eval` | Synthetic generator (doc 06 §1); metrics (doc 06 §3); benchmark runner; JSON result store | Metric unit tests; the structured model beats majority vote and β = 0 on coherent synthetic data (M1 criterion) | ✅ |
| 1.9 | CLI and config | `qws demo` (synthetic end-to-end); `qws run <config.yaml>` | Smoke test of the CLI | ✅ |

## Phase 2 — Quantum simulation (milestone M2)

| WP | Package | Deliverables | Status |
|----|---------|--------------|--------|
| 2.1 | `qws.solvers.qaoa` | `QAOASimulator`, an exact statevector QAOA for N ≲ 16 with linear-ramp warm start | ✅ |
| 2.2 | `qws.solvers.qaoa` | `QAOAQiskit`, which builds a Qiskit circuit with the basis convention of doc 02 §4 and runs on any `BackendSampler`; tested with the Aer simulator | ✅ |
| 2.3 | `qws.solvers.anneal` | `AnnealingSolver`, the Ocean-SDK path: sign conversion, auto-scaling, chain strength, cached embeddings, metadata; tested end-to-end with a **simulated** Ocean sampler standing in for the QPU | ✅ |

## Phase 3 — Real hardware (milestones M3 and M4): needs access

Hardware smoke tests live in `tests/test_qpu.py` (marker `qpu`, deselected by default; run them with `pytest -m qpu`).

These items cannot be completed without QPU credentials and a budget (open question Q2).
The code paths exist, but they are untested against real devices.

| WP | Item | Needs | Status |
|----|------|-------|--------|
| 3.1 | Run `AnnealingSolver` against a real D-Wave QPU (`DWaveSampler`): verify the h/J range read from solver properties, chain-break rates, and the timing fields | D-Wave Leap token (`DWAVE_API_TOKEN`) and QPU minutes | ⛔ |
| 3.2 | Calibrate the effective temperature on real hardware: the curve of β̂ against the scale κ, per graph shape | 3.1 | ⛔ |
| 3.3 | Parallel embedding of several problems per QPU call (doc 04 §6) | 3.1; topology only available from the live solver | 🟡 designed, not implemented |
| 3.4 | Run `QAOAQiskit` on IBM hardware: transpilation to heavy-hex, readout mitigation. IBM Runtime requires `SamplerV2` rather than `backend.run`, so a small adapter is needed (and `qiskit-ibm-runtime` added to the `gate` extra) | IBM Quantum account and token | ⛔ |
| 3.5 | M3/M4 benchmark campaign (doc 06) | 3.1–3.4 and a dataset choice (Q1) | ⛔ |

## Phase 4 — Extensions (milestone M5)

| WP | Item | Status |
|----|------|--------|
| 4.1 | Multi-class one-hot encoding and the penalty bound of the paper's Proposition 11 | planned |
| 4.2 | Class-conditional LF potentials (Q5) | planned |
| 4.3 | QPBO / roof-duality persistency baseline | planned (needs a QPBO library or our own implementation) |
| 4.4 | Learning β and b (prior-phase sampling) | the code path exists (`learn_b`, `learn_beta`); validation planned |

## Simplifications made in v1 (documented deviations)

- **Pairwise LFs vote only on graph edges:** $E_r \subseteq E$. Votes on non-adjacent
  pairs would add couplings outside the site graph. The model supports them in
  principle, but the LF interface does not yet.
- **Symmetric potentials only** (Definition 1 of the paper). Correlation cliques are
  supported with the agreement feature
  $\phi_c(\lambda) = \#\{j<l \in c : \lambda_j = \lambda_l \neq 0\}$, which is invariant
  under a vote flip.
- **Strong-field fixing is applied to MAP only.** It is never applied to sampling
  (doc 02 §6.1).
