# 08 — Implementation Roadmap

This document turns docs 00–07 into ordered work packages, with each package's deliverables, the tests or
reports that close it, and its status.

- Implementation details: [doc 09](09-implementation-notes.md).
- Decisions and their rationale: [doc 10](10-decision-log.md).
- Measured results: [`results/`](../results/README.md).

Status markers:

- ✅ done (implemented, tested, and reported where applicable)
- 🟡 implemented and validated offline; needs real hardware or real data for the final run
- ⛔ blocked on access (credentials, network) that the user must provide

**Current state:** everything that can be done without a D-Wave Leap token, an IBM Quantum account, or
network access to the Oxford-IIIT Pet hosts is done. The test suite has 96 tests; 2 more real-hardware tests
are opt-in.

## Phase 1 — Classical reference implementation (milestone M1) ✅

| WP | Package | Deliverables | Closed by | Status |
|----|---------|--------------|-----------|--------|
| 1.0 | Scaffolding | `pyproject.toml` with extras `anneal`, `gate`, `downstream`, `dev`; `qws` package; pytest config | `pip install -e .[dev]`; `pytest` | ✅ |
| 1.1 | `qws.ising` | `IsingProblem`; QUBO / Ocean / Pauli conversions; conditioning; strong-field fixing; scaling and dynamic range | `tests/test_ising.py` | ✅ |
| 1.2 | `qws.graph` | Pixel-grid and SLIC superpixel sites; contrast-sensitive affinities | `tests/test_graph_lf.py` | ✅ |
| 1.3 | `qws.lf` | Unary and pairwise LF interfaces; `apply_lfs`; majority lifting (R3 default); example LFs | `tests/test_graph_lf.py` | ✅ |
| 1.4 | `qws.model` | Parameters; posterior and prior builders; moments; Theorem-1 gradient; warm start | `tests/test_model.py` (brute-force joint, finite differences) | ✅ |
| 1.5 | `qws.solvers` (classical) | `exact`, `gibbs`, `sa`, `graphcut` | `tests/test_solvers.py` | ✅ |
| 1.6 | `qws.learn` | Adam trainer, EM mode, pseudo-likelihood temperature calibration (bounded steps), polishing | `tests/test_learn.py`; reports 01, 03, 04 | ✅ |
| 1.7 | `qws.decompose` | Tiling; block coordinate descent; block Gibbs | `tests/test_decompose_eval.py`; report 02 | ✅ |
| 1.8 | `qws.eval` | Synthetic generator (blob and Ising-prior masks); metrics (with ignore mask); benchmarks | `tests/test_decompose_eval.py` | ✅ |
| 1.9 | CLI / config | `qws demo`, `qws run <config.yaml>` | CLI smoke test | ✅ |

## Phase 2 — Quantum simulation (milestone M2) ✅

| WP | Deliverable | Closed by | Status |
|----|-------------|-----------|--------|
| 2.1 | `QAOASimulator`: exact statevector QAOA, linear-ramp warm start | `tests/test_solvers.py`; report 02 | ✅ |
| 2.2 | `QAOAQiskit`: Qiskit circuits on Aer (`backend.run`) | Qiskit statevector equals our statevector | ✅ |
| 2.3 | `AnnealingSolver`: any dimod sampler; Ocean simulated-annealing stand-in | `tests/test_solvers.py`; reports 01, 03 | ✅ |

## Phase 3 — Real hardware (milestones M3 and M4)

Everything up to the device itself is implemented and tested offline. Hardware smoke tests are in
`tests/test_qpu.py` (marker `qpu`; run with `pytest -m qpu`).

| WP | Item | Offline validation | Needs | Status |
|----|------|--------------------|-------|--------|
| 3.1 | `QPUAnnealingSolver` (`qws.solvers.qpu`) on D-Wave **Advantage (Pegasus)**: explicit minorminer embedding, embedding cache, range-aware scaling including chains, `auto_scale` off, chain-break reporting, QPU timing attribution | Mock Pegasus P16 (`MockDWaveSampler`): `tests/test_qpu_mock.py`, report 05 | `DWAVE_API_TOKEN`; network access to D-Wave Leap | 🟡 |
| 3.2 | Effective-temperature calibration on the device (β̂ against submit scale, per graph shape) | Recovered the mock's β_eff ≈ 4 from samples (report 05) | 3.1 | 🟡 |
| 3.3 | Parallel embedding: several problems per QPU call | Implemented; capacity measured on mock P16 (report 05) | 3.1 | 🟡 |
| 3.4 | IBM Quantum via **SamplerV2** (`QAOAQiskit(sampler=..., backend=...)`, `QAOAQiskit.ibm()`), with ISA transpilation | `StatevectorSampler` and Aer `SamplerV2` tests | IBM Quantum account and token | 🟡 |
| 3.5 | M3/M4 benchmark campaign (doc 06) on hardware | Reports 02, 03, 05 define the protocol | 3.1–3.4 | ⛔ |

## Phase 4 — Extensions (milestone M5) ✅

| WP | Item | Closed by | Status |
|----|------|-----------|--------|
| 4.1 | **Multi-class** (`qws.multiclass`): Potts label model, one-hot QUBO with the Proposition-11 penalty, decoding with repair, Potts Gibbs and annealing, gradient, learning | `tests/test_multiclass.py` (exhaustive Prop. 11 check, brute-force gradient); report 11 | ✅ |
| 4.2 | **Class-conditional LF potentials** (ω, ζ) | `tests/test_model.py` (brute-force posterior, prior and gradient); report 07 | ✅ |
| 4.3 | **QPBO / roof duality** (`QPBOSolver`, `roof_dual`) | `tests/test_solvers.py` (persistency, bound); report 10 | ✅ |
| 4.4 | **Learning β and b** with prior-phase sampling shared per graph shape | Report 06 | ✅ |

## Phase 5 — Real data and the end task

| WP | Item | Offline validation | Needs | Status |
|----|------|--------------------|-------|--------|
| 5.1 | **Oxford-IIIT Pet** (`qws.data.pet`): loader for the official layout, trimap binarization with an ignore band, resizing; generic natural-image LFs (`qws.lf.natural`); `dataset.kind: pet` | Generated dataset in the exact Pet layout (`qws.data.petlike`): `tests/test_data.py`, report 08 | Network access to `www.robots.ox.ac.uk` (currently refused by the environment's network policy), or the dataset placed on disk | 🟡 |
| 5.2 | Learn β and b | Report 06 | — | ✅ |
| 5.3 | Class-conditional LFs | Report 07 | — | ✅ |
| 5.4 | **Downstream U-Net** (`qws.downstream`): noise-aware loss; end-task pipeline comparing training signals | `tests/test_downstream.py`; report 09 (Pet-format data) | 5.1 for real Pet results | 🟡 |

## How to finish the remaining items

1. **D-Wave:**
   - Add `DWAVE_API_TOKEN` as an environment secret, and allow the Leap hosts in the environment's network settings.
   - Run `pytest -m qpu`, then `qws run` with `solver: {name: anneal_qpu}` (or `QPUAnnealingSolver.dwave()`).
   - Repeat reports 03 and 05 with the real sampler.
2. **IBM:**
   - Save the account (`QiskitRuntimeService.save_account`) or set `QISKIT_IBM_TOKEN`, plus `QWS_IBM_BACKEND`.
   - Run `pytest -m qpu`.
3. **Oxford-IIIT Pet:**
   - Allow `www.robots.ox.ac.uk`, then call `qws.data.download_pet("data/oxford-iiit-pet")`. Alternatively, copy
     the extracted `images/` and `annotations/` folders there.
   - Run reports 08–09 with `dataset: {kind: pet, root: data/oxford-iiit-pet}`.

## Simplifications in the current version (documented deviations)

- **Pairwise LFs vote only on site-graph edges** (E_r ⊆ E). This keeps the posterior graph equal to the site
  graph, which matters for embedding (doc 09 §2).
- **Pairwise LFs are symmetric** (no class-conditional pairwise potentials). Only unary LFs can be class-conditional.
- **EM mode** supports the symmetric singleton model with fixed b and β only. Everything else uses gradient mode.
- **Strong-field fixing applies to MAP only**, never to sampling (doc 02 §6.1).
- **Multi-class learning** keeps b and β fixed and uses symmetric potentials.
- **Domain-wall and binary (log) encodings** for multi-class (doc 07 §3) are not implemented. One-hot is the only
  encoding.
