# 11 — Ephemeral-Gully Weak Segmentation (ViGIR EG dataset)

**Status:** implemented (`qws/gully/`, `qws/data/gully.py`, `qws/data/dem3dep.py`, `qws/data/hydro.py`); first
results in [report 12](../results/12-gully.md). Pixel-level evaluation waits for annotations (G9).

The branch applies the structured, quantum-ready label model of docs 01–10 to locating **ephemeral gullies (EGs)**
in agricultural fields from NAIP aerial imagery. Owner decisions are G1–G10, recorded in
[doc 10](10-decision-log.md) together with the implementation decisions D25–D31.

## 1. Data

**Source.** <https://vigir.missouri.edu/Research/GullyDetection/GD_Dataset_ViGIR_WACV_2025.zip>
(125 MB; © ViGIR, Tousi et al., WACV 2025).

| Fact | Value |
|------|-------|
| Locations (tiles) | 621, each with **6** co-registered NAIP acquisitions: `GD_Images/rgb_{year}_tile_{tile}.tif` |
| Image | **128 × 128** RGB uint8 GeoTIFF, WGS84, pixel ≈ 9.77·10⁻⁶° ≈ 1.08 m, so a tile is about 138 m × 138 m |
| Labels | `GT_L6_Tol.json`: **location-level** only, '1' = at least one EG visible at the location in some year. 354 positive, 267 negative |
| Regions | Central Iowa (467 locations, about 42.25°N 93.8°W) and the Missouri/Illinois border (154 locations, about 39.1°N 90.9°W) |
| Pixel ground truth | none |

**Notes.**
- The images are 128 × 128 (the request mentioned 64 × 64). Decision G10 keeps the native resolution.
- The README mentions separate test and validation JSONs, but only `GT_L6_Tol.json` (all 621 locations) is in
  the archive. The split is therefore our own (G2).
- **Labels are spatially clustered.** Long runs of equal labels occur along tile numbers, and the 621 locations
  fall into about 123 one-kilometre cells. A random split would leak.
- **Visual properties.** EGs are thin (about 1–4 px), sinuous linear marks inside fields, visible in only some
  years. Confounders are roads, field boundaries, ditches, tree lines, crop rows and tillage marks.

**Storage (G7).** The dataset is never committed. `qws.data.gully.download_gully()` fetches it into `data/gully`,
which is git-ignored. Reports may contain a few small example crops, with citation.

**Split (G2).**
- Tiles are grouped into geographic cells of 0.02° (about 2 km); 87 clusters result.
- Seeded random search over 5,000 cluster assignments minimizes |test size − ½| + |positive-rate differences|.
- Result: dev 310 / test 311 locations, 177 positives each (57%). No cluster appears on both sides.
- `make_split()` caches the split in `data/gully/split_seed0_cell0.02.json`.

## 2. Decisions at a glance

| ID | Decision |
|----|----------|
| G1 | Output per-year masks **and** a location-level union; the 6 years are coupled by temporal edges |
| G2 | Spatially blocked dev/test split (geographic clusters, stratified) |
| G3 | Location labels as **constraints**: negative locations clamped to background; positive locations satisfy "at least one EG pixel over the 6 years" (multiple-instance learning, MIL) |
| G4 | Use USGS 3DEP 1 m LiDAR elevation (fetched from the public USGS bucket) |
| G5 | Hydrology implemented in-house (numba), because `pysheds` is incompatible with NumPy 2 |
| G6 | Orientation-aware spatial couplings |
| G7 | Never commit the dataset; small cited figures are allowed |
| G8 | Two-stage learning: an exact independent MIL label model, then the structured posterior |
| G9 | Evaluate location-level now; pixel metrics once a small annotated set exists |
| G10 | Native 128 × 128 pixel grid (one site per pixel per year) |

## 3. Terrain from USGS 3DEP (G4, G5)

1. **Index.** `qws.data.dem3dep.build_index` lists the 1 m DEM projects for IA, MO and IL in
   `prd-tnm.s3.amazonaws.com` (5,098 file cells), keyed by `zone_x_y`. Files are named
   `USGS_1M_{zone}_x{⌊E/10 km⌋}y{⌈N/10 km⌉}_{project}.tif`. When projects overlap, the most recent wins.
2. **Window.** The tile bounds plus a **300 m margin** (upslope area outside the tile matters) are converted to
   UTM (NAD83). Every overlapping 10 km file is read by HTTP range requests (`/vsicurl/`) and mosaicked at 1 m
   with `rasterio.merge`. This takes about 1–3 s per tile.
3. **Hydrology** (`qws.data.hydro`, numba):
   - priority-flood depression filling with an ε gradient (Barnes et al. 2014), so flats drain;
   - D8 steepest-descent receivers;
   - flow accumulation in decreasing filled elevation (contributing area in m²).

   Verified on an inclined plane (accumulation 1…n), a V-valley (the outlet collects every cell) and a filled pit.
4. **Products:**
   - slope (degrees, from the σ = 1 smoothed DEM);
   - log₁₀ contributing area;
   - TWI = ln(a / tan β);
   - convergence = −∇² of the σ = 3 smoothed DEM.

   All are reprojected (bilinear) onto each tile's own WGS84 128 × 128 grid and cached in
   `data/gully/terrain/<tile>.npz`.
5. **Coverage.** All 621 locations have 1 m terrain. A visual check shows the DEM flow paths
   (area ≥ 1,000 m²) tracing the visible gullies in positive tiles, while negative tiles lack concentrated
   channels.

## 4. Labeling functions (`qws.gully.lfs`)

Every LF votes per pixel and per year, in {−1, 0, +1}, and none uses labels. The thresholds were set a priori,
not tuned on labels.

| LF | Rule | Rationale |
|----|------|-----------|
| `flow_path` | +1 if contributing area ≥ 10³ m² (dilated 1 px); −1 if no pixel within 3 px has area > 10² m² | EGs form where overland flow concentrates |
| `convergence` | +1 in the top 8% of terrain convergence; −1 below the median | Valleys and hollows vs. ridges and divergent slopes |
| `line_on_flow` | +1 if a thin line (tubeness top 5%) lies within 3 px of a flow path with area ≥ 10^2.7 m²; −1 if a thin line lies more than 8 px from any path with area ≥ 10^2.3 m² | Physics consistency: a visible line *on* a flow path is an EG; a line off every flow path is tillage or crop rows |
| `thin_line` | Per year: +1 in the top 3% of multi-scale Sato tubeness (σ = 1, 1.5, 2; dark and bright ridges); −1 below the 60% quantile | EGs are thin curvilinear structures |
| `straight_line` | −1 within 2 px of probabilistic-Hough segments ≥ 40 px (Canny σ = 1.2) | Roads, field edges, ditches and crop rows are straight; EGs are sinuous |
| `non_crop` | −1 for bright, unsaturated pixels (roads, buildings) and dark, highly textured pixels (trees) | Non-field surfaces |
| `persistent_vegetation` | −1 where excess green stays above the tile median in ≥ 5 of 6 years (only if some year is bare) | Field-edge strips, riparian corridors and grassed waterways stay green |
| `permanent_line` | −1 where a thin line appears at the same place (±1 px) in ≥ 5 of 6 years | Premise: EGs are ephemeral, so permanent lines are not EGs |

**Caveat on the last two.** EGs often **re-form in the same flow path** after tillage, so "permanent" can also
describe a recurring gully. Whether these two LFs help is therefore decided by dev cross-validation (D28), not
assumed. Report 12 gives the outcome.

Terrain LFs vote identically in every year. Whether a gully is *visible* in a given year comes from the imagery
LFs and the temporal coupling.

## 5. Model (G1, G6, G10)

**Sites.** All pixels of all six years, in year-major order: site = t·H·W + r·W + c, which is 98,304 sites per
location.

**Spatial edges** (8-neighbour, within each year), with weight

w_ik = exp(−‖rgb_i − rgb_k‖² / 2s²) · ½[(1 − coh_i) + coh_i cos²(φ − d_i) + (1 − coh_k) + coh_k cos²(φ − d_k)] · ρ,

where:
- φ is the edge direction;
- d and coh are the line direction and coherence from the luminance structure tensor (σ = 1, ρ = 2.5);
- s is the RMS colour difference over that image's edges;
- ρ = 1 for axial and 1/√2 for diagonal neighbours.

**Temporal edges.** The same pixel in consecutive years, with weight τ.

**Properties of this design.**
- In coherent linear structures, couplings **along** the line stay strong and couplings **across** it vanish.
  The smoothness prior therefore connects thin gullies instead of eroding them, as isotropic smoothing would.
- All weights are ≥ 0, so with no pairwise LFs the posterior is **ferromagnetic**, and MAP is exact by graph
  cut. A vectorized PyMaxflow construction takes 0.08 s per 98k-site location.
- The edge structure (470,540 edges) is identical for every location. It is built once and shared (int32);
  only the float32 weights differ per location.

**Potentials.** Class-conditional unary potentials (α, θ, ω, ζ; doc 09 §3), with class bias b. EG pixels are
rare, so b is strongly negative.

## 6. Learning (G3, G8)

**Why two stages.** In a first attempt, the structured model was trained by Gibbs sampling, with the MIL
constraint enforced by rejection. It learned a strongly negative b, its MAP masks came out empty, and up to a
third of the constraint checks rejected every chain. The symmetric potentials also could not represent LFs that
vote −1 on half of all pixels. Stage 1 avoids all three problems, exactly.

**Stage 1: an exact independent MIL label model** (`warm_start`, `stage1_gradient`). At β = 0 the sites are
independent, so every quantity is exact:

- Free sites: P(y_i = +1) = σ(2h_i), with h_i = b + Σ_j θ_j L_ij + Σ_j ω_j 1[L_ij ≠ 0].
- **Negative locations:** ⟨y_i⟩ = −1 (clamped).
- **Positive locations, MIL constraint:** with p_i = σ(2h_i) and P(none) = Π_j(1 − p_j), the exact conditional
  marginal is P(y_i = +1 | at least one) = p_i / (1 − P(none)). It is computed in log space, and verified
  against brute-force enumeration in `tests/test_gully.py`.
- The prior moment is tanh(b + F), with F the class-conditional prior field (Proposition 2).
- The gradient is Lemma 1 with these moments. It is vectorized over all locations' votes (about 30M
  site-years for dev) in float32, and matches `qws.model.gradient` to float32 precision (tested).
- Optimizer: Adam, lr 0.05, at most 400 steps, stopping early when no parameter moves more than 2·10⁻⁴.

**Stage 2: structured posterior.** The stage-1 parameters, with smoothness β and temporal weight τ. Optional
sampling-based training of the symmetric parameters (`GullyConfig.epochs`) is implemented. It uses clamps and
rejection for MIL (`Trainer.fit(clamps=..., min_positive=...)`), but it is off by default.

**Model selection (D28).** The LF set, β ∈ {0, 0.5, 1, 2} and τ ∈ {0.25, 1} are chosen by **3-fold
cross-validation within dev, grouped by geographic cluster**. The criterion is the mean validation AUC of the
largest-MAP-component score. Test labels are used only for the final evaluation.

## 7. Inference and location scores

- **MAP:** graph cut, giving per-year masks. The location union is the OR over years.
- **Marginals:** Gibbs sampling (16 chains × 20 sweeps) started at the MAP state.
- **Location scores** (`location_scores`):
  - largest 8-connected EG component in any year (MAP, and marginal > 0.5);
  - EG pixel fraction;
  - marginal mass;
  - mean of the top-100 marginals;
  - the same for the independent model (stage 1 at β = 0) and LF majority vote;
  - per-LF fraction of +1 votes;
  - terrain alone (maximum log contributing area).

## 8. Evaluation (G9)

- **Location level** (now): AUC (Mann–Whitney) and best F1 per score on the test locations, overall and per
  region.
- **Pixel level** (once annotated): `qws.gully.annotate`.
  - **Formats:** `<tile>.geojson` (QGIS lines on the GeoTIFFs, a `year` property per feature, buffered to 3 m)
    or `<tile>_y<year>.png` masks (0 = background, 1 = EG, 255 = not annotated).
  - **Metrics:** strict IoU, precision, recall and F1, plus tolerance-based precision, recall and F1 within
    ±2 px.
  - **Export:** `scripts/gully_export_annotation.py` exports a seeded random set of positive test locations,
    with images and optional prediction overlays, for annotators.

## 9. Relevance to the quantum backends

Each location's posterior is a 98,304-variable Ising model on a sparse 3-D lattice:
- spatial degree 8 plus temporal degree 2, so degree ≤ 10;
- 470k couplers.

That is far beyond current QPUs, so the decomposition of doc 04 §5 applies. Hardware-sized tiles (for example
12 × 12 pixels × 6 years ≈ 864 variables) are solved by block coordinate descent, or sampled by block Gibbs.

**Honest scope.**
- With no pairwise LFs, the posterior is ferromagnetic, so **MAP is solved exactly and fast classically**
  (graph cut).
- The quantum role in this branch is **sampling**: marginals, calibrated soft labels and the structured-learning
  stage. It would also cover MAP if frustrated couplings are introduced, for example a pairwise "these two
  pixels differ" LF across field boundaries.

The D-Wave path (`QPUAnnealingSolver`, report 05) consumes these tile problems unchanged.

## 10. Limitations and next steps

- **No pixel ground truth.** Location-level detection is a proxy for segmentation quality. An annotated subset
  (G9) is the most valuable next step.
- **Terrain and imagery dates differ.** The DEM is one LiDAR snapshot (2018–2021 projects), while the NAIP years
  span several acquisitions. Flow paths are stable, but field boundaries can change.
- **Threshold-based LFs** are set a priori. Learned LFs (for example a classifier trained on the label model's
  output) are the natural next iteration.
- **Regions differ strongly** (report 12, by region). More regions would be needed to claim generalization.
