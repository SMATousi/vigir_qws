# 11 — Multi-class extension

The K-class extension (spec doc 07; paper Section 8 and Proposition 11). It consists of:
- a Potts label model with symmetric LF potentials;
- the one-hot QUBO with the Proposition-11 penalty, which makes multi-class MAP solvable by every binary backend,
  including the QPU path;
- decoding with repair of infeasible sites;
- a Potts Gibbs sampler and Potts simulated annealing.

## Learning (K = 3, synthetic 12×12 grids, 10 images, 16 s)

| parameter | true | learned |
|---|---|---|
| θ_0 | 1.200 | 1.265 |
| θ_1 | 0.700 | 0.771 |
| θ_2 | 1.500 | 1.462 |
| θᴾ_0 | 0.800 | 0.864 |

## Segmentation accuracy (pixel accuracy, K = 3)

| method | accuracy |
|---|---|
| majority vote | 0.718 |
| independent (β = 0) | 0.708 |
| structured Potts MAP (this work) | 0.927 |

## Multi-class MAP routes on small Potts problems (2×3 grid, K = 3, exact reference by enumeration)

| route | exact optimum found | infeasible-read fraction |
|---|---|---|
| SA on one-hot QUBO, 20 reads | 17/20 | 0 |
| SA on one-hot QUBO, 100 reads | 20/20 | 0 |
| mock QPU on one-hot QUBO, 100 reads | 20/20 | 0.061 |
| Potts simulated annealing (native) | 20/20 | 0 |

## Observations

- LF accuracies are learned without labels in the K-class model, as in the binary case.
- The structured Potts model clearly beats majority vote and the independent model.
- **The one-hot encoding costs search effort.** Moving a site between classes passes through infeasible
  states, which cost about A (the Proposition-11 penalty), so local search needs more reads to find the optimum.
  This is the known weakness of one-hot encodings on annealers, and the reason domain-wall encodings exist
  (spec doc 07 §3). Budget QPU reads accordingly.
- With the sufficient penalty of Proposition 11, the minimizers are always feasible (verified exhaustively in
  `tests/test_multiclass.py`). The infeasible reads seen here are non-optimal samples, which the decoder repairs.
