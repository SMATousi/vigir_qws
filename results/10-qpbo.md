# 10 — QPBO / roof-duality baseline

Roof duality / QPBO (roadmap 4.3; doc 02 §6.3). The doubled energy over (x, y ≈ 1 − x) is submodular, so it is
minimized exactly by one max-flow ([spec/09 §5](../spec/09-implementation-notes.md)). Variables with y = 1 − x
are weakly persistent: some global optimum agrees with all of them at once. Frustration comes from the pairwise
"different" LF (strength θᴾ). To find where roof duality stops deciding variables, we also use weak unary LFs
and a field-free ±J spin glass.

## Small problems (4×4, exact reference by enumeration)

| problem family | θᴾ | labeled fraction | bound gap (E* − bound)/|E*| | persistency verified |
|---|---|---|---|---|
| posterior, realistic LFs | 0 | 1.000 | 2.27e-17 | 10/10 |
| posterior, realistic LFs | 1.000 | 1.000 | 2.45e-17 | 10/10 |
| posterior, realistic LFs | 2.000 | 1.000 | 2.68e-17 | 10/10 |
| posterior, weak LFs (θ × 0.25) | 0 | 1.000 | -3.51e-17 | 10/10 |
| posterior, weak LFs (θ × 0.25) | 1.000 | 0.400 | 0.059 | 10/10 |
| posterior, weak LFs (θ × 0.25) | 2.000 | 0.600 | 0.014 | 10/10 |
| ±J spin glass, no fields | – | 0 | 0.358 | 10/10 |

## Large problems (32×32): QPBO (+ SA completion of the unlabeled part) vs. simulated annealing alone

| problem family | θᴾ | n | labeled fraction | E(QPBO+SA) − E(SA) | QPBO time (s) | SA time (s) |
|---|---|---|---|---|---|---|
| posterior, realistic LFs | 0 | 1024 | 1.000 | 6.82e-13 | 0.008 | 0.119 |
| posterior, realistic LFs | 2.000 | 1024 | 1.000 | 0 | 0.008 | 0.116 |
| posterior, weak LFs (θ × 0.25) | 0 | 1024 | 1.000 | -1.575 | 0.008 | 0.119 |
| posterior, weak LFs (θ × 0.25) | 2.000 | 1024 | 0 | -9.09e-13 | 0.210 | 0.119 |
| ±J spin glass, no fields | – | 1024 | 0 | – | – | – |

## Observations

- With realistic LFs, the unary fields anchor the posterior so strongly that roof duality labels every variable,
  even with strong pairwise "different" votes. The MAP problem is then solved exactly by max-flow, and no quantum
  hardware is needed for it.
- Weak LFs leave room for frustration. At θᴾ = 2 roof duality labels 60% of the variables. For the
  field-free ±J spin glass it labels 0%: the regime that remains genuinely hard.
- Persistency held in every verified case: the labels always agree with an exact optimum. The bound is tight
  whenever all variables are labeled.
- In practice, run QPBO first and send only the unlabeled remainder to the QPU (doc 02 §6.3). On posteriors with
  informative LFs that remainder is often empty, which is an honest limit on where quantum MAP helps.
