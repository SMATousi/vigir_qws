# vigir_qws: Quantum Weak Supervision for Segmentation

This repo implements data programming for image segmentation as a *structured* label model. The model's
posterior over masks is an Ising model (a Potts model for K classes). That lets the same problem be solved by:
- classical solvers (exact enumeration, graph cuts, QPBO, Gibbs sampling, simulated annealing);
- quantum annealers (D-Wave Advantage, through the Ocean SDK);
- gate-based QAOA (a statevector simulator, Qiskit Aer, or IBM Quantum through SamplerV2).

| Where | What |
|-------|------|
| [`spec/`](spec/README.md) | The design and the math |
| [`spec/08-roadmap.md`](spec/08-roadmap.md) | Status of every work package |
| [`spec/09-implementation-notes.md`](spec/09-implementation-notes.md) | How each part is implemented |
| [`spec/10-decision-log.md`](spec/10-decision-log.md) | Why each choice was made |
| [`results/`](results/README.md) | Measured results |

## Install

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"                          # classical core and tests
pip install -e ".[dev,anneal,gate,downstream]"   # + D-Wave Ocean, Qiskit/IBM Runtime, PyTorch
```

## Quick start

```bash
qws demo                                    # synthetic end-to-end run with Gibbs sampling
qws demo --solver anneal_mock --calibrate   # full D-Wave QPU path on a mock Pegasus device + calibration
qws run examples/synthetic.yaml             # config-driven run
pytest                                      # 96 tests (real-hardware tests are opt-in: pytest -m qpu)
python scripts/generate_results.py          # regenerate results/ reports 01–04
python scripts/generate_results_extensions.py  # regenerate results/ reports 05–11
```

## Package layout

| Module | Contents |
|--------|----------|
| `qws.ising` | `IsingProblem`; QUBO, Ocean and Pauli conversions; conditioning; strong-field fixing |
| `qws.graph` | Pixel-grid and superpixel sites with contrast-sensitive affinities |
| `qws.lf` | LF interfaces, `apply_lfs`, example LFs, generic natural-image LFs |
| `qws.model` | Label model (symmetric and class-conditional): posterior and prior builders, moments, gradient, warm start |
| `qws.solvers` | `exact`, `graphcut`, `qpbo`, `gibbs`, `sa`, `anneal_sim`, `anneal_qpu` / `anneal_mock` (`qpu.py`), `qaoa_sim`, `qaoa_qiskit` |
| `qws.learn` | Trainer (gradient / EM; learns b, β and class-conditional terms), temperature calibration, inference |
| `qws.decompose` | Tiling, block-coordinate-descent MAP, block Gibbs |
| `qws.multiclass` | K-class Potts label model, one-hot QUBO (Proposition-11 penalty), Potts samplers |
| `qws.data` | Oxford-IIIT Pet loader and downloader; Pet-layout dataset generator |
| `qws.downstream` | U-Net with noise-aware loss; end-task pipeline |
| `qws.eval` | Synthetic data, metrics (with an ignore mask), benchmarks |

## Running on real hardware and real data

The code is complete up to the devices themselves. Each needs a credential or network access in the cloud
environment:

- **D-Wave Advantage:**
  - Set `DWAVE_API_TOKEN` and allow the Leap hosts.
  - Run `pytest -m qpu`, then use `solver: {name: anneal_qpu}` in a config or
    `qws.solvers.qpu.QPUAnnealingSolver.dwave()`.
- **IBM Quantum:**
  - Save the account (or set `QISKIT_IBM_TOKEN`) and `QWS_IBM_BACKEND`.
  - Use `qws.solvers.qaoa.QAOAQiskit.ibm()`.
- **Oxford-IIIT Pet:**
  - Allow `www.robots.ox.ac.uk` and `thor.robots.ox.ac.uk` (the download redirects there), then run `python -c "import qws.data as d; d.download_pet('data/oxford-iiit-pet')"`.
    Alternatively, place `images/` and `annotations/` there.
  - Use `dataset: {kind: pet, root: data/oxford-iiit-pet}`.

## License

MIT. See [LICENSE](LICENSE).
