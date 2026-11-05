# 17 — Frustrated posteriors on real images

The full label model (unary LFs, the pairwise colour-contrast LF and β = 0.5) learned on 100 Oxford-IIIT Pet
images (seed 0; the setting of report 13). Its pairwise LF votes "different" across strong colour edges. With a
positive learned θᴾ = 1.17, each such vote adds a negative coupling, so the posteriors are
frustrated and graph cut does not apply. This report asks what these real frustrated posteriors require from a MAP
solver. Besides the learned parameters, five scaled settings show the effect of a stronger pairwise LF or weaker
unary LFs.

## Frustration, roof duality and solver gaps (means over images)

- **neg. couplings:** share of couplings J < 0.
- **odd cycles:** share of the cycles in a cycle basis whose coupling signs multiply to −1 (frustrated cycles).
- **labeled:** share of variables fixed by roof duality (QPBO). These labels are exact (persistent).
- **all labeled / certified:** images where QPBO labels every variable / where the MAP is certified optimal, because
  QPBO labels everything or the remainder (≤ 20 variables) is finished by enumeration.
- **gaps:** (E − E_ref) / |E_ref|, against the best of QPBO + long annealing and three long annealing runs
  (2,000 sweeps × 20 reads each). SA: 300 sweeps × 20 reads. Greedy: single-flip descent from 20 random starts.
- **IoU:** of the reference MAP mask against the trimaps (boundary excluded).

| setting | sites | neg. couplings | odd cycles | labeled | all labeled | certified | remainder | gap QPBO+SA | gap SA | gap greedy | IoU |
|---|---|---|---|---|---|---|---|---|---|---|---|
| learned | 120.150 | 0.101 | 0.139 | 0.990 | 97/100 | 98/100 | 1.200 | 1.22e-16 | 8.56e-17 | 0.004 | 0.585 |
| pairwise ×3 | 120.150 | 0.101 | 0.139 | 0.983 | 92/100 | 97/100 | 2.110 | 1.36e-16 | 0.000121 | 0.046 | 0.531 |
| unary ×0.5 | 120.150 | 0.101 | 0.139 | 0.977 | 91/100 | 94/100 | 2.860 | 1.35e-16 | 0.000195 | 0.017 | 0.555 |
| unary ×0.5, pairwise ×3 | 120.150 | 0.101 | 0.139 | 0.944 | 79/100 | 86/100 | 6.990 | 1.35e-16 | 0.000374 | 0.076 | 0.517 |
| unary ×0.25, pairwise ×10 | 120.150 | 0.101 | 0.139 | 0.722 | 51/100 | 58/100 | 33.640 | 0.000115 | 0.007 | 0.050 | 0.465 |
| no unary LFs, pairwise ×10 | 120.150 | 0.101 | 0.139 | 0 | 0/100 | 0/100 | 120.150 | 0.002 | 0.004 | 0.046 | 0.316 |

## D-Wave resources (first 10 images per setting; mock Advantage, Pegasus P16)

Left: the QPBO remainder, which is what would be sent to the QPU. Right: the whole image's posterior, which is
the relevant size once QPBO stops deciding variables.

| setting | images | remainder (vars) | remainder qubits | remainder longest chain | full problem (vars) | full problem qubits | full longest chain | full embedded |
|---|---|---|---|---|---|---|---|---|
| learned | 10 | 4.100 | 5.700 | 2 | 122.300 | 198.200 | 4 | 10/10 |
| pairwise ×3 | 10 | 5.900 | 7.500 | 2 | 122.300 | 198.200 | 4 | 10/10 |
| unary ×0.5 | 10 | 1.800 | 2.100 | 2 | 122.300 | 198.200 | 4 | 10/10 |
| unary ×0.5, pairwise ×3 | 10 | 11.300 | 14.400 | 2 | 122.300 | 198.200 | 4 | 10/10 |
| unary ×0.25, pairwise ×10 | 10 | 30.200 | 44.200 | 4 | 122.300 | 198.200 | 4 | 10/10 |
| no unary LFs, pairwise ×10 | 10 | 122.300 | 198.200 | 4 | 122.300 | 198.200 | 4 | 10/10 |

## Observations

- **Real frustration is easy.** With the learned parameters, 10% of couplings are negative and 14% of basis
  cycles are frustrated. Yet roof duality labels 99% of the variables, every variable in 97 of 100 images, and
  the MAP is certified optimal in 98 of 100 (QPBO alone, or QPBO plus enumeration of a remainder of at most 20
  variables). Short annealing (300 sweeps) reaches the reference energy on all 100 images. The unary evidence
  anchors the posterior, as in report 10.
- **Hardness needs weak unaries and strong pairwise votes together.** Certified images drop to 86/100 at unary
  ×0.5 with pairwise ×3, to 58/100 at unary ×0.25 with pairwise ×10, and to 0 with no unary LFs (a field-free
  frustrated model on the superpixel graph). Short annealing reaches the reference on 86%, 12% and 14% of these
  images, though its mean gap stays below 1%. Greedy descent is 0.4–8% above the reference on average.
- **These hard settings are also worse segmentations.** IoU falls from 0.585 (learned) to 0.465 and 0.316 in the
  hardest settings. That fits report 16: the generic pairwise LF lowers IoU on Pet. On these real images, a
  stronger "different" LF does not create hard problems worth solving.
- **Hardware size is not the obstacle.** A whole image's posterior (about 122 variables) embeds in about 198
  physical qubits of the mock Advantage graph, with chains of at most 4, which is about 3.5% of a Pegasus P16
  chip. With the learned parameters, the QPBO remainder sent to the QPU would have only a few variables.
- **Caveat.** Where an image is not certified, the reference is the best energy found by long annealing and
  QPBO + annealing, not a proven optimum. The gaps are therefore lower bounds on the true gaps.
- **Consequence.** For binary segmentation with a realistic pairwise LF, MAP is solved classically: QPBO, and
  annealing for a small remainder. Quantum resources remain relevant for **sampling** under strong coupling
  (report 03), for multi-class (Potts) MAP, which is NP-hard even without frustration (report 11), and for
  models whose LFs are genuinely weak relative to their pairwise terms.
