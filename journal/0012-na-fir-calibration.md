---
id: 0012
slug: na-fir-calibration
timestamp: 2026-04-13T13:37:11
git_tag: exp/0012
queue_entry: 3
status: worse
val_loss: 0.0520
val_accuracy: 0.9864
best_so_far: 0.0511
---

## Hypothesis
Applying normalization-aware isotonic calibration (NA-FIR: incorporate sum-to-one constraint directly into isotonic optimization) to the XGB+LGB+CatBoost ensemble's softmax outputs will reduce log_loss below 0.051357 because standard one-vs-rest isotonic regression ignores probability normalization constraints — the per-class isotonic mappings can independently shift class probabilities in ways that violate sum-to-one, and this violation is the main source of residual error in exp 0017's isotonic plateau; enforcing normalization jointly should tighten the calibration mapping.

## What I changed
In `agent/train.py`: (1) Updated HYPOTHESIS to match the queue entry verbatim; (2) Replaced the bin-constrained isotonic code with two new functions: `nafir_isotonic()` (per-class sklearn isotonic then sum-to-one normalization: p_c = p_iso_c / sum(p_iso_all)) and `default_isotonic()` (pure per-class sklearn isotonic as baseline); (3) Replaced Phase 3b's bin-count grid search with a side-by-side NA-FIR vs default comparison, selecting the better method; (4) No changes to prepare.py, HPO budget, or model code.

## Result
- XGBoost: 25 trials, best=0.0533 | LightGBM: 11 trials, best=0.0552 | CatBoost: 38 trials, best=0.0539
- Weighted ensemble (0.60/0.10/0.30): 0.0532
- Default isotonic: **0.052061**
- NA-FIR isotonic: **0.052049** ← marginal improvement over default (+0.000012)
- Final: **0.0520** — **worse** than previous best 0.0511 by +0.0009

## What I learned
- **NA-FIR provides essentially zero improvement over default isotonic (+0.000012)**: The sum-to-one normalization post-hoc barely changes the probabilities. The weighted ensemble's softmax outputs are already near sum-to-one (as GBDTs produce calibrated softmax), so the isotonic per-class mappings only slightly perturb class-conditional probabilities. The hypothesized "main source of residual error" from sum-to-one violations does not exist on this dataset.
- **The regression from 0.0511 to 0.0520 is driven by weaker base models, not calibration failure**: XGBoost (0.0533 vs typical ~0.0525), CatBoost (0.0539 vs typical ~0.0536) are both degraded. Fewer Optuna trials (25 XGB, 11 LGB, 38 CAT vs exp 0010's 20/9/22 — actually comparable but with different random seeds) produced models that respond differently to isotonic. The calibration gap between 0.0511 and 0.0520 is entirely attributable to base model quality, not the calibration method.
- **The NA-FIR vs default isotonic gap is too small to matter**: 0.000012 is within numerical noise for a 126K-sample validation set. Even if the effect were real, it would take ~90 equivalent experiments to accumulate a meaningful improvement. The calibration sub-axis is effectively closed — isotonic calibration's 0.0010 improvement over weighted baseline is stable, and post-hoc normalization provides nothing.
- **The KB's conclusion is confirmed: "base model quality is the bottleneck"**: The 0.0511 result from exp 0010 used 500-bin custom isotonic which IS better than default isotonic (0.0511 vs 0.0520 in this run's default isotonic), but the current experiment regressed because it used standard sklearn isotonic instead. The bin-constrained 500-bin approach from exp 0010 was the right direction, not NA-FIR.
- **Effective per-class temperatures (~1.0) confirm isotonic compression is the mechanism**: Temperatures of 1.0022, 0.9990, 1.0013 mirror all prior experiments. Both default isotonic and NA-FIR achieve similar compression patterns — the small NA-FIR gain comes from ensuring the compressed probabilities sum to exactly 1.0, but the effect is negligible.

## KB entries consulted
- Isotonic regression is the single most effective post-processing step: 0.0514 vs 0.0530 (temp scaling) vs 0.0522 (weighted baseline); its nonlinear piecewise-constant mapping compresses extreme probabilities more effectively than global temperature scaling (exp 0017)
- Effective per-class temperatures from isotonic regression are near 1.0 (~0.999 for class 0, ~0.999 for class 1, ~1.002 for class 2), meaning improvement comes from piecewise compression not global softening (exp 0017)
- Isotonic regression provides ~0.0010 improvement regardless of moderate base model quality variation (exps 0017, 0019, 0020, 0021, 0022, 0023)
- Isotonic calibration cannot compensate for weaker base models: base model quality degradation from reduced Optuna budget was the primary failure mode across exps 0021, 0022, 0023 (exps 0021, 0022, 0023)
- Temperature scaling confirmed overconfidence (+0.0002 signal, temperatures ~0.94–0.95) but coarse grid burned Optuna budget, reducing trial counts (exp 0013)
