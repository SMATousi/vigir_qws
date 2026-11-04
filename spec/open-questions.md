# Open Questions

Items are resolved by editing the relevant spec document and moving the
entry to *Resolved*.

## Open

| # | Question | Affects | Current default |
|---|----------|---------|-----------------|
| Q3 | How are pixel LFs lifted to superpixels: majority, a coverage threshold, or a soft fraction (which would need a non-ternary vote model)? | 01 §2.1 | Majority of non-abstaining votes; abstain under 50% coverage |
| Q6 | Is QAOA useful as a *sampler* (for example via parameter tuning towards a Boltzmann target), or only for MAP and seeding? | 03 §4, 04 §3 | MAP and seeding only |
| Q7 | Should higher-order (region-level) LFs, such as "this box contains foreground", be supported through quadratization with ancilla qubits? | 01, 02 | Deferred to v2 |
| Q14 | Network access: allow `www.robots.ox.ac.uk` (Oxford-IIIT Pet download) and the D-Wave Leap / IBM Quantum endpoints in the environment's network policy, or place the Pet data on disk? | 08 Phase 3/5 | Pet and hardware runs wait for this |
| Q15 | Multi-class encoding beyond one-hot: implement the domain-wall encoding (K − 1 qubits per site, fewer infeasible states)? | 07 §3, report 11 | One-hot only |
| Q16 | Chain strength and annealing time on the real Advantage: tune per graph shape, or keep the defaults (csf 1.5, default anneal)? | 09 §4, D10 | Defaults; tune in roadmap 3.1 |
| Q17 | Gully: can your lab annotate about 30–50 positive test locations (QGIS lines per year), so pixel IoU can be measured? Export: `scripts/gully_export_annotation.py` | 11 §8 | Location-level evaluation only |
| Q18 | Gully: the README mentions separate validation and test JSONs from the WACV paper. Should the paper's split replace our spatially blocked split for comparability? | 11 §1 | Our split (G2) |
| Q19 | Gully: should we add learned LFs (for example a small CNN trained on stage-1 soft labels, then fed back as an LF) or more terrain LFs (TWI, stream-power index)? | 11 §10 | Threshold LFs only |
| Q9 | What is the publication plan, and which comparison results must be reproducible for the paper? (License resolved: R13) | 06 §4 | — |

## Resolved

| # | Question | Decision | Date |
|---|----------|----------|------|
| R1 | Primary quantum paradigm | Hardware-agnostic Ising/QUBO core; annealing and gate-based (QAOA) backends | 2026-10-20 |
| R2 | Segmentation setting | Binary first; multi-class as a planned extension | 2026-10-20 |
| R3 | Which optimization is quantized | Both learning and inference, jointly | 2026-10-20 |
| R4 | LF type | Generic interface; unary and pairwise LFs | 2026-10-20 |
| R5 | Q1 — first real dataset | **Oxford-IIIT Pet**, with trimaps binarized to foreground/background | 2026-11-03 |
| R6 | Q2/Q11 — annealing hardware | **D-Wave Leap with QPU time**, primary target **Advantage (Pegasus)**; hybrid solvers only as a separately labelled baseline | 2026-11-03 |
| R7 | Q10 — gate-based provider | **IBM Quantum** through Qiskit (needs the SamplerV2 adapter, roadmap 3.4) | 2026-11-03 |
| R8 | Q4 — learn β | **Yes**: learn β and b with prior-phase sampling (roadmap 4.4 moves up) | 2026-11-03 |
| R9 | Q5 — class-conditional LFs | **Yes**: support class-conditional LF potentials (roadmap 4.2 moves up) | 2026-11-03 |
| R10 | Q8 — downstream network | **Add a small trainer**: a minimal U-Net with a noise-aware loss, to report end-task IoU | 2026-11-03 |
| R11 | Q12 — branching | Open a pull request into `main` | 2026-11-03 |
| R13 | License | **MIT** | 2026-11-03 |
