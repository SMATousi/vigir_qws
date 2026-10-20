# Open Questions

Items are resolved by editing the relevant spec document and moving the
entry to *Resolved*.

## Open

| # | Question | Affects | Current default |
|---|----------|---------|-----------------|
| Q1 | Which application domain and dataset will serve as the final demonstration? | 06 §1 | Generic; synthetic plus small public binary datasets |
| Q2 | Which specific quantum hardware and providers are available, and under what budget (QPU minutes, shots)? | 04, 06 | Design is device-agnostic; capacities measured at runtime |
| Q3 | How are pixel LFs lifted to superpixels: majority, a coverage threshold, or a soft fraction (which would need a non-ternary vote model)? | 01 §2.1 | Majority of non-abstaining votes; abstain under 50% coverage |
| Q4 | Should β (smoothness) be learned in v1, or fixed via validation? | 01 §6, 03 §2.1 | Fixed in v1; learning it is a stretch goal |
| Q5 | Should v1 include class-conditional LF potentials, or only symmetric ones? | 01 §3.1, 03 §2 | Symmetric first; class-conditional behind a flag |
| Q6 | Is QAOA useful as a *sampler* (for example via parameter tuning towards a Boltzmann target), or only for MAP and seeding? | 03 §4, 04 §3 | MAP and seeding only |
| Q7 | Should higher-order (region-level) LFs, such as "this box contains foreground", be supported through quadratization with ancilla qubits? | 01, 02 | Deferred to v2 |
| Q8 | Is the downstream segmentation network part of this repo, or only an interface? | 00 §4, 03 §7 | Interface only |
| Q9 | What is the license and the publication plan, and which comparison results must be reproducible for a paper? | 06 §4 | — |

## Resolved

| # | Question | Decision | Date |
|---|----------|----------|------|
| R1 | Primary quantum paradigm | Hardware-agnostic Ising/QUBO core; annealing and gate-based (QAOA) backends | 2026-10-20 |
| R2 | Segmentation setting | Binary first; multi-class as a planned extension | 2026-10-20 |
| R3 | Which optimization is quantized | Both learning and inference, jointly | 2026-10-20 |
| R4 | LF type | Generic interface; unary and pairwise LFs | 2026-10-20 |
