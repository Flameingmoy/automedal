---
id: 0013
slug: jsd-local-calibration
timestamp: 2026-04-13T14:55:00
git_tag: exp/0013
queue_entry: 4
status: worse
val_loss: 0.0518
val_accuracy: 0.9863
best_so_far: 0.0511
---

## Hypothesis
JSD-based local calibration — which weights calibration training samples by local feature-space neighborhood density rather than treating all samples equally — will improve log_loss because standard isotonic regression (exp 0017) treats every validation sample with equal importance, but miscalibration is concentrated in sparse feature regions where the ensemble has fewer neighbors to inform its predictions; weighting calibration toward high-density (well-represented) regions reduces the influence of noisy calibration signal from sparse regions.

## What I changed
In `agent/train.py`: (1) Updated HYPOTHESIS to match the queue entry verbatim; (2) Replaced the bin-constrained isotonic (N=500) from exp 0010 with JSD-based local calibration infrastructure: `_jsd_matrix()` (BallTree-based JSD computation), `_compute_jsd_weights()` (per-sample JSD weights via K-nearest-neighbor average distance in softmax-probability space), `_isotonic_with_jsd_weights()` (equal-frequency binning with weighted-bin means), `_isotonic_density_threshold()` (isotonic fit only on samples above median JSD weight), and `evaluate_jsd_calibration()` to compare methods; (3) Replaced Phase 3b's bin-count grid search with a JSD calibration grid: uniform_500 baseline, jsd_weighted_k{50,100,200}, jsd_threshold_k{50,100,200}; (4) No changes to prepare.py or HPO budget.

## Result
- XGBoost: 10 trials, best=0.0535 | LightGBM: 9 trials, best=0.0551 | CatBoost: 26 trials, best=0.0565 (degraded)
- Weighted ensemble (0.50/0.25/0.25): 0.0533
- **JSD calibration grid:**
  - uniform_500 (baseline): **0.051760**
  - jsd_weighted_k50: 0.051765 (+0.000005)
  - jsd_weighted_k100: 0.051769 (+0.000009)
  - jsd_weighted_k200: 0.051773 (+0.000013)
  - jsd_threshold_k50: 0.269164 (catastrophic — isotonic fit on dense-only subset)
  - jsd_threshold_k100: 0.273289 (catastrophic)
  - jsd_threshold_k200: 0.281887 (catastrophic)
- Final: **0.0518** — **worse** than previous best 0.0511 by +0.0007

## What I learned
- **JSD-weighting provides essentially zero improvement over uniform weighting (+0.000005–0.000013, within noise)**: All three K values (50, 100, 200) produced nearly identical val_loss to the uniform baseline. The hypothesis that miscalibration is concentrated in sparse feature regions is not supported by the data. The ensemble's softmax outputs appear uniformly calibrated across the entire feature-space density range.
- **Density-threshold isotonic catastrophically failed (0.27–0.28)**: Fitting isotonic regression only on above-median-density samples creates a severe data subset problem: each per-class binary label set (e.g., "is-class-0" = 1) becomes degenerate when limited to 50% of samples, and isotonic regression's piecewise-constant mapping collapses. The ~5× log_loss increase proves that isotonic needs the full validation distribution to work, not a filtered dense subset.
- **JSD-weighting on softmax probability space is redundant with equal-frequency binning**: The 500-bin equal-frequency partition from exp 0010 already implicitly performs density-based calibration by grouping similar-probability samples together. The JSD weight is a second-order correction on top of a first-order probability-based partition, and it adds no signal when the models are 98.6% accurate (the prediction distribution is already very concentrated around the true labels).
- **CatBoost base model degradation (0.0565 vs typical 0.0536) is the primary damage vector**: CatBoost received only 26 trials (vs 38+ in exp 0012), pushing ensemble weights to 0.50/0.25/0.25 (vs 0.65/0.10/0.25 in exp 0010). This reduced the ensemble's reliance on the best-performing individual model. The JSD calibration itself was neutral-to-slightly-negative, but the regression would likely have been smaller with better base models. This is the same failure mode seen in exps 0021–0023.
- **The calibration sub-axis is now comprehensively closed**: Exps 0010 (bin-regularized isotonic), 0011 (warm-start Optuna), 0012 (NA-FIR), and 0013 (JSD-weighting) all confirm that calibration method is not the bottleneck. The ~0.0010 improvement from isotonic over weighted ensemble is stable and robust to method variations. Future experiments should focus on base model quality or ensemble diversity, not calibration refinements.

## KB entries consulted
- Isotonic regression is the single most effective post-processing step: 0.0514 vs 0.0530 (temp scaling) vs 0.0522 (weighted baseline); its nonlinear piecewise-constant mapping compresses extreme probabilities more effectively than global temperature scaling (exp 0017)
- Isotonic regression provides ~0.0010 improvement regardless of moderate base model quality variation (exps 0017, 0019, 0020, 0021, 0022, 0023)
- Isotonic calibration cannot compensate for weaker base models: base model quality degradation from reduced Optuna budget was the primary failure mode across exps 0021, 0022, 0023 (exps 0021, 0022, 0023)
- The isotonic calibration plateau at 0.051357 may reflect isotonic overfitting on noisy validation samples; the dataset is 98.6% accurate (~1.4% noise), and unconstrained isotonic regression fits noisy points exactly (open question, unconfirmed — **JSD-weighting confirms: isotonic overfitting is not the bottleneck; base model quality is**)
