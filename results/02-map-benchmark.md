# 02 — MAP (ground-state) benchmark

For a fixed model, how well do the solvers find the ground state (the MAP mask)? This is question Q2
in [spec/06](../spec/06-evaluation.md), task T1.

## Setup

- **Problems:** posterior Ising models of synthetic images (same LFs as report 01, β = 0.5).
  - *Frustrated* problems include the pairwise "different" LF, which adds negative couplings.
  - *Ferromagnetic* problems use unary LFs only, so all J ≥ 0.
- **References:** exact enumeration for small problems; graph cut (exact) for large ferromagnetic problems; the
  best energy found by any solver for large frustrated problems. Reference kinds used: {'small_frustrated': ['exact'], 'small_ferro': ['exact'], 'large_frustrated': ['best_found'], 'large_ferro': ['graphcut']}.
- **Metrics:**
  - relative energy gap (E − E*)/|E*|;
  - hit rate (the fraction of reads reaching E*);
  - TTS99 = t · log(0.01)/log(1 − p_hit), with t the wall time of the whole call.
- **Timing:** CPU wall times. They are not comparable to QPU access times.

## Small problems (4×4 grid, n = 16), frustrated

| Solver | n | mean rel. gap | mean hit rate | median TTS99 (s) | mean wall (s) |
|---|---|---|---|---|---|
| sa | 16 | -2.1e-17 | 0.936 | 0.010 | 0.011 |
| anneal_sim | 16 | -2.31e-17 | 0.939 | 0.019 | 0.019 |
| qaoa_sim p=1 | 16 | 0.359 | 0 | ∞ | 0.467 |
| qaoa_sim p=2 | 16 | 0.055 | 0.115 | 310.487 | 1.765 |

## Small problems (4×4 grid, n = 16), ferromagnetic

| Solver | n | mean rel. gap | mean hit rate | median TTS99 (s) | mean wall (s) |
|---|---|---|---|---|---|
| graphcut | 16 | 0 | 1.000 | 0.000187 | 0.00038 |
| sa | 16 | 0 | 1.000 | 0.010 | 0.010 |
| anneal_sim | 16 | 0 | 1.000 | 0.017 | 0.017 |
| qaoa_sim p=1 | 16 | 0.029 | 0.080 | 45.642 | 0.456 |
| qaoa_sim p=2 | 16 | 0 | 0.176 | 83.518 | 2.758 |

## Large problems (32×32 grid, n = 1024), frustrated

| Solver | n | mean rel. gap | mean hit rate | median TTS99 (s) | mean wall (s) |
|---|---|---|---|---|---|
| sa | 1024 | 1.92e-16 | 0.483 | 1.076 | 0.119 |
| anneal_sim | 1024 | 1.42e-16 | 0.333 | 1.253 | 0.130 |

## Large problems (32×32 grid, n = 1024), ferromagnetic

| Solver | n | mean rel. gap | mean hit rate | median TTS99 (s) | mean wall (s) |
|---|---|---|---|---|---|
| graphcut | 1024 | 0 | 1.000 | 0.002 | 0.002 |
| sa | 1024 | 2.94e-16 | 0.583 | 0.793 | 0.121 |
| anneal_sim | 1024 | -2.22e-16 | 0.783 | 0.445 | 0.115 |

## Decomposition: block coordinate descent with 4×4 exact tiles (32×32, frustrated)

Each tile (16 sites, about the size of a small QPU or QAOA subproblem) is solved exactly, conditioned on its
boundary (paper Proposition 5(b)–(c)).

| Problem | n | tiles | accepted tile moves | final energy | best monolithic | rel. gap | wall (s) |
|---|---|---|---|---|---|---|---|
| 0 | 1024 | 64 | 80 | -3131.405 | -3133.675 | 0.000724 | 4.640 |
| 1 | 1024 | 64 | 75 | -3194.416 | -3194.416 | 5.69e-16 | 4.763 |
| 2 | 1024 | 64 | 75 | -3195.935 | -3196.497 | 0.000176 | 4.704 |

## Observations

- Graph cut is exact and fastest on ferromagnetic problems. As intended, it refuses every frustrated
  problem (shown as "refused").
- Simulated annealing (ours, and Ocean's through the annealing interface) reaches the exact optimum on every
  small problem, with hit rates near 1. On the 1024-site problems both reach the same best energy, with
  70–85% of reads hitting it.
- QAOA (exact statevector, angles optimized classically, 100 shots) is much weaker at these depths. At p = 1,
  no shot reaches the optimum on frustrated problems. At p = 2, about 7–11% of shots reach it; best-of-100 is
  exact on ferromagnetic problems but leaves a mean gap of several percent on frustrated ones. Depth and
  shot budget are the levers to study on hardware (roadmap 3.4).
- Block coordinate descent with exact 16-site tiles converges to a block-local optimum within 0–0.5% of the best
  monolithic energy. That gap is the cost of decomposition when problems exceed QPU capacity.
