# 04 — Quantum (and Classical) Backends

Every solver implements one interface (doc 05 §3) and consumes the
`IsingProblem` defined in doc 02. This document specifies how each backend
family works and how problems larger than the hardware are handled.

## 1. Backend catalogue

| Backend | Family | Operations | Use |
|---------|--------|-----------|-----|
| `exact` | Classical | minimize, sample (exact Boltzmann), marginals, log Z | Ground truth for N ≲ 25; unit tests |
| `graphcut` | Classical | minimize (submodular only; MUST refuse when any $J<0$) | Exact MAP reference for ferromagnetic problems |
| `qpbo` | Classical | minimize (partial) | Persistency preprocessing; frustrated MAP baseline |
| `sa` | Classical | minimize, sample | Simulated-annealing baseline |
| `gibbs` | Classical | sample (persistent chains), polish | MCMC baseline; quantum-seeded polishing (doc 03 §4) |
| `anneal_qpu` | Quantum annealing | minimize, sample | Primary quantum backend |
| `anneal_hybrid` | Vendor hybrid solver | minimize | Large-problem reference; reported separately, never counted as a "pure QPU" result |
| `qaoa_sim` | Gate-based, simulated | minimize, seed sampling | Development and small-N studies |
| `qaoa_qpu` | Gate-based hardware | minimize, seed sampling | Small subproblems on real devices |

Quantum SDK dependencies MUST be optional extras, so the classical core
installs without them.

## 2. Quantum annealing backend (`anneal_qpu`)

The Ising problem is native to annealing hardware, so the work here is
mapping and calibration.

1. **Conversion.** Convert to the SDK's sign convention (doc 02 §5).
2. **Minor-embedding.** Map the logical graph onto the hardware graph, with
   chains of physical qubits representing one logical site.
   - Pixel grids (degree 4, or degree 8 with diagonals) and superpixel
     adjacency graphs (typical degree about 5–7) are sparse. They SHOULD embed
     with short chains on current topologies.
   - Embeddings for fixed graph shapes, such as an $L\times L$ grid tile,
     MUST be **computed once and cached**. Every image tile of that shape
     reuses the embedding.
3. **Chain strength.** Default to a relative rule (a multiple of
   $\max|J|$), exposed as a parameter. Log the chain-break fraction for every
   call, and unembed with majority vote. If chain breaks exceed a threshold,
   the call MUST be flagged as low-fidelity.
4. **Scaling.** Apply the scale factor $\kappa$ from doc 02 §7, using the
   ranges reported by the device.
5. **Run parameters.** Exposed parameters: `num_reads`, annealing time,
   spin-reversal (gauge) transforms (on by default, to average out bias), and
   annealing schedule (advanced).
6. **Returned metadata.** Raw and unembedded samples, energies under *our*
   convention, chain-break fraction, $\kappa$, QPU access time, and embedding
   statistics (qubits used, maximum and mean chain length).

**Capacity rule of thumb.** Logical sites per call = (available qubits) ÷
(mean chain length). Today that is on the order of $10^3$ sites for sparse
graphs. The exact number MUST be measured, not assumed.

## 3. Gate-based backend (`qaoa_sim`, `qaoa_qpu`)

**Circuit.** QAOA with depth $p$, built from:
- the cost Hamiltonian $H_C$ (doc 02 §4);
- the transverse-field mixer $H_M = \sum_i X_i$;
- the initial state $|+\rangle^{\otimes N}$.

**Per-term gates.**
- A local field $h_i$ becomes $R_Z$.
- A coupling $J_{ik}$ becomes $R_{ZZ}$ on the pair. On hardware with limited
  connectivity, this needs SWAP routing.
  - Grid-structured problems route well.
  - Pairwise-LF edges that are not local add depth.

**Parameters.** $(\gamma, \beta)$ angles optimized classically.
- Warm start from linear-ramp schedules, or from parameters transferred
  between problems of the same graph shape.
- The optimizer is pluggable (COBYLA, SPSA).

**Outputs.**
- MAP: the best measured bitstring, optionally improved by a classical local
  search, with the improvement recorded separately.
- Seed samples for Gibbs polishing (doc 03 §4).

**Scale.** Realistic sizes on current gate hardware are roughly 10–100 sites,
because of noise and depth. In this project, gate-based runs therefore:
1. benchmark small problems against `exact`;
2. solve **subproblems** from the decomposition in §5.

**Noise handling.** Readout-error mitigation MAY be enabled. Results are
reported with and without it.

## 4. Choosing a backend per task

| Task | Small N (≤ 25) | Medium (10²–10³, superpixels) | Large (pixel-level, ≥ 10⁴) |
|------|----------------|-------------------------------|----------------------------|
| T1 MAP, ferromagnetic | `exact`/`graphcut` (reference) | `graphcut` reference vs `anneal_qpu` | Decomposed `anneal_qpu`; `graphcut` reference |
| T1 MAP, frustrated | `exact` vs `qaoa_*` vs `anneal_qpu` | `anneal_qpu` vs `sa`/`qpbo` | Decomposed `anneal_qpu` vs `sa` |
| T2/T3 sampling | `exact` (true distribution) | `anneal_qpu` (± polish) vs `gibbs` | Decomposed sampling (§5.3) vs `gibbs` |

## 5. Decomposition for large images

Full-resolution pixel problems exceed every current device. The supported
strategies are below, and they can be combined.

### 5.1 Superpixel coarsening (default)

Segment the image into $N_s$ superpixels, with $N_s$ configurable, typically
200–2000. Build the region adjacency graph, and lift LFs to the superpixels
(doc 01 §2.1). This reduces sites by 1–3 orders of magnitude, and it is the
primary path to M3.

### 5.2 Tiling with boundary conditioning (MAP)

Partition the graph into tiles that fit the device, with an overlap margin.
1. Solve every tile, conditioning on the current labels of the sites outside
   it (doc 02 §6.2).
2. Sweep over tiles (block-coordinate descent), in a checkerboard order for
   parallelism, until no tile improves.

The global energy never increases between sweeps. Report the number of
sweeps and the final energy against the reference.

### 5.3 Blocked sampling

The sampling analogue of §5.2 is **block Gibbs**: resample one tile at a
time from its conditional distribution, given its boundary. The conditional
is exact, so the result is a valid MCMC scheme, provided each tile sample
follows the conditional, which needs the temperature correction of doc 03 §4.
Tiles that are not adjacent can be sampled in the same device call.

### 5.4 Multiscale (optional)

Solve a coarse superpixel problem first. Use its result to fix confident
regions and initialize the fine level, then refine only uncertain bands
near boundaries with §5.2.

## 6. Batching

Several independent problems, such as tiles or the posterior problems of a
minibatch, MAY be packed into one device call by embedding them on disjoint
qubit regions. This *parallel embedding* amortizes per-call overhead. The
backend MUST unpack the results and attribute timing per problem
proportionally, documenting how it does so.

## 7. Reproducibility and accounting

Every call records:
- the backend name and version, device name, and SDK version;
- the full run parameters and seeds;
- a hash of the embedding;
- the `IsingProblem` hash.

Results are stored so any figure can be regenerated without re-running
hardware. Time is reported in separate categories: QPU access time,
classical pre- and post-processing, and network or queue time. These are
never merged.
