# 06 — Evaluation

The evaluation answers two separate questions. They MUST NOT be conflated:

- **(Q1) Quality.** Does the structured label model give better segmentation
  labels than unstructured weak-supervision baselines?
- **(Q2) Quantum value.** For a fixed model, do the quantum solvers match or
  beat classical solvers on optimization quality, sample quality, and time?

## 1. Datasets

| Tier | Data | Purpose |
|------|------|---------|
| Synthetic | Generated masks (random blobs, or Ising samples at a known β) with synthetic LFs whose accuracy, coverage, correlations, and pairwise "edge" LFs are known | Parameter recovery, controlled frustration, exact comparisons at small N |
| Real, binary | Small public foreground/background datasets. Candidates, to be confirmed: Weizmann Horse, Oxford-IIIT Pet (trimaps binarized), ISIC skin lesions | End-to-end segmentation quality |
| Domain | To be chosen later (open questions) | Final demonstration |

The **synthetic generator** is a first-class deliverable (`qws.eval`). It
MUST be able to sweep:
- problem size $N$;
- coupling strength $\beta$;
- the fraction of frustrated edges (from pairwise LFs);
- number of LFs, their accuracies, and their coverage.

## 2. Baselines

**For Q1 (label quality):**
1. Majority vote over LFs, per site.
2. The independent data-programming label model ($\beta = 0$), which is
   Snorkel-style.
3. The structured model of this spec solved classically: `graphcut`/`sa`
   for MAP, and `gibbs` for learning and marginals.

**For Q2 (solver value), all on identical `IsingProblem`s:**
- **MAP:** `exact` (small N); `graphcut` (ferromagnetic); `qpbo`; `sa`
  tuned to an equal time budget.
- **Sampling:** `exact` (small N), and persistent `gibbs` at an equal
  wall-clock budget and an equal sample count.

## 3. Metrics

### 3.1 Segmentation (Q1)

- IoU and Dice of the hard mask $\hat y$ against ground truth; pixel accuracy.
- Calibration of the soft mask $\mu$: expected calibration error and Brier score.
- On synthetic data: parameter-recovery error $\lVert\hat\theta - \theta^\*\rVert$.
- Optionally: IoU of a downstream network trained on $\mu$.

### 3.2 Optimization (Q2, T1)

- **Relative energy gap** $(E_{\text{found}} - E^\*)/|E^\*|$. Here $E^\*$ is
  the exact or graph-cut optimum, or the best known when no exact method
  applies.
- **Ground-state hit rate**: the fraction of reads that reach $E^\*$.
- **Time-to-solution at 99%**:
  $\mathrm{TTS}_{99} = t_{\text{run}}\cdot \frac{\log(1-0.99)}{\log(1-p_{\text{hit}})}$,
  reported separately for QPU-only time and end-to-end time.
- Agreement between the MAP mask and the reference mask (Hamming / IoU),
  because states with nearly equal energy can still differ as masks.

### 3.3 Sampling (Q2, T2/T3)

- **Small N:**
  - total-variation distance and KL divergence between the empirical and the
    exact Boltzmann distributions;
  - error in the marginals $\lvert\hat\mu_i - \mu_i\rvert$ and the edge
    correlations.
- **Any N:** the estimated $\hat\beta$ and its stability, and the gradient
  error against a long-run `gibbs` reference.
- **Downstream:** learned $\theta$, and final IoU, when the learning loop
  uses the sampler under test versus `gibbs`.

## 4. Reporting rules

- Every quantum result is reported together with:
  - the best classical baseline at an equal budget;
  - raw and polished variants (doc 03 §4);
  - embedding statistics (doc 04 §2);
  - separate timing categories (doc 04 §7).
- Results from vendor hybrid solvers are labelled as such, and never
  presented as pure-QPU results.
- Scaling plots, energy gap and TTS against $N$ and against the frustration
  fraction, are the primary Q2 deliverable.

## 5. Acceptance criteria per milestone

| Milestone | Criterion |
|-----------|-----------|
| M1 | All doc 05 §6 tests pass; on synthetic data the structured model beats both the β=0 model and majority vote on IoU, in the regime where the ground truth is spatially coherent |
| M2 | Simulated QAOA and the simulated annealer run through the same pipeline; small-N QAOA energy gap reported against `exact` |
| M3 | Real-annealer MAP on superpixel problems matches the `graphcut` optimum on ferromagnetic problems at a reported hit rate; frustrated-problem results reported against `sa`/`qpbo`; quantum-seeded learning converges on synthetic data |
| M4 | A full-image pipeline via decomposition, with an end-to-end IoU report on at least one real dataset |
