# vigir_qws: Quantum Weak Supervision for Segmentation

This repo implements data programming for image segmentation as a *structured* label
model. The model's posterior over masks is an Ising model. That lets the same
problem be solved by:
- classical solvers (exact enumeration, graph cuts, Gibbs sampling, simulated annealing);
- quantum annealers (D-Wave, through the Ocean SDK);
- gate-based QAOA (a statevector simulator, or Qiskit backends).

The design and the math are in [`spec/`](spec/README.md). The implementation plan and
its status are in [`spec/08-roadmap.md`](spec/08-roadmap.md).

## Install

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"              # classical core and tests
pip install -e ".[dev,anneal,gate]"  # also D-Wave Ocean and Qiskit/Aer
```

## Quick start

```bash
qws demo                                    # synthetic end-to-end run with Gibbs sampling
qws demo --solver anneal_sim --calibrate    # Ocean annealing interface (simulated sampler) + temperature calibration
qws run examples/synthetic.yaml             # config-driven run
pytest                                      # full test suite (skips QPU tests)
```

`qws demo` prints IoU for three label models:
- majority vote;
- the independent (Snorkel-style, β = 0) label model;
- the structured model's MAP mask.

It also reports how well the learned LF accuracies match the generating ones.

## Package layout

| Module | Contents |
|--------|----------|
| `qws.ising` | `IsingProblem`; QUBO, Ocean and Pauli conversions; conditioning; strong-field fixing |
| `qws.graph` | Pixel-grid and superpixel sites with contrast-sensitive affinities |
| `qws.lf` | Labeling-function interfaces, `apply_lfs`, example LFs |
| `qws.model` | Label-model parameters; posterior and prior Ising builders; closed-form moments; gradient; warm start |
| `qws.solvers` | `exact`, `graphcut`, `gibbs`, `sa`, `anneal` (Ocean), `qaoa_sim`, `qaoa_qiskit` |
| `qws.learn` | Trainer (gradient / EM), pseudo-likelihood temperature estimation, calibrated sampler, inference |
| `qws.decompose` | Tiling, block-coordinate-descent MAP, block Gibbs |
| `qws.eval` | Synthetic data generator, metrics, MAP and sampling benchmarks |

## Real quantum hardware

The QPU code paths exist, but they have not been validated on real devices (roadmap
Phase 3). Running them needs:

- **D-Wave:** a Leap API token in `DWAVE_API_TOKEN`. Then use
  `qws.solvers.anneal.AnnealingSolver.dwave_qpu()`, or `solver: {name: anneal_qpu}` in a config.
- **IBM / gate-based:** pass an IBM backend to `qws.solvers.qaoa.QAOAQiskit(backend=...)`.
