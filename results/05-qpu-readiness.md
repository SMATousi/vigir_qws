# 05 — D-Wave QPU readiness (offline, mock Pegasus)

Everything a real D-Wave run needs, except the device: the full `QPUAnnealingSolver` path runs against
`MockDWaveSampler`. The mock has the real Pegasus P16 topology (the size of an Advantage QPU),
the real coefficient ranges and QPU timing fields, and a fixed-temperature simulated-annealing substitute
for the physics. See [spec/09 §4](../spec/09-implementation-notes.md) for the design.

## Minor-embedding of representative problem graphs (minorminer, cached by graph shape)

| Problem graph | variables | couplers | physical qubits | max chain | mean chain | embed time (s) |
|---|---|---|---|---|---|---|
| grid 4×4 | 16 | 24 | 16 | 1 | 1.000 | 0.054 |
| grid 6×6 | 36 | 60 | 38 | 2 | 1.056 | 0.063 |
| grid 8×8 | 64 | 112 | 75 | 2 | 1.172 | 0.141 |
| grid 12×12 | 144 | 264 | 171 | 2 | 1.188 | 0.751 |
| superpixel RAG (SLIC, 100 requested) | 32 | 77 | 41 | 2 | 1.281 | 0.120 |
| superpixel RAG (SLIC, 300 requested) | 263 | 714 | 508 | 7 | 1.932 | 2.131 |
| one-hot K=3 on 4×4 (48 vars) | 48 | 120 | 78 | 3 | 1.625 | 0.159 |

## Parallel embedding

≥ 64 (all tested) independent 4×4 posterior problems (16 variables each) fit into one P16 submission. They share
one programming and readout cycle; QPU access time is attributed to each problem in proportion to its size.

## MAP through the mock QPU (8 posterior problems on 4×4 grids, packed, 100 reads)

| problem | n | κ | range-limited | chain strength | chain-break fraction | packed with | best − exact | hit rate |
|---|---|---|---|---|---|---|---|---|
| 0 | 16 | 2.247 | yes | 1.000 | 0 | 8 | -7.11e-15 | 0.990 |
| 1 | 16 | 2.249 | yes | 1.000 | 0 | 8 | 0 | 0.990 |
| 2 | 16 | 2.249 | yes | 1.000 | 0 | 8 | -7.11e-15 | 0.750 |
| 3 | 16 | 2.250 | yes | 1.000 | 0 | 8 | 0 | 1.000 |
| 4 | 16 | 2.239 | yes | 1.000 | 0 | 8 | 0 | 0.920 |
| 5 | 16 | 2.250 | yes | 1.000 | 0 | 8 | -1.42e-14 | 1.000 |
| 6 | 16 | 2.244 | yes | 1.000 | 0 | 8 | 0 | 1.000 |
| 7 | 16 | 2.229 | yes | 1.000 | 0 | 8 | 7.11e-15 | 0.380 |

All problems were solved in 0.22 s of wall time, in 8 problem(s) per submission.

## Temperature calibration through the mock QPU

The mock's substitute anneals to an inverse temperature of 4 in hardware units. Pseudo-likelihood calibration
should therefore converge to a submit scale of about 1/4. History of (scale, β̂):
(1.000, 1000.00), (0.250, 1.02), (0.246, 1.01), (0.243, 0.94), (0.258, 1.06), (0.243, 0.95), (0.256, 1.03), (0.249, 1.00), (0.248, 1.00), (0.249, 1.00).

Final implied β_eff = 1/scale = **4.01**.

## Observations

- Grid graphs embed with chains of at most 2 qubits. Superpixel adjacency graphs are less regular: the densest
  one (263 sites, 714 couplers) needs a mean chain of about 2 and a maximum of 7. The one-hot multi-class
  encoding is denser too (every site is a K-clique) and needs longer chains than the plain grid.
- Range-aware scaling (κ) keeps logical couplers, fields and chains inside the device ranges with
  `auto_scale` off. Problems that already fit are not rescaled, which is what makes calibration possible.
- The calibration loop recovers the mock device's temperature from samples alone. This is the first
  experiment to repeat on the real Advantage system (roadmap 3.2).
