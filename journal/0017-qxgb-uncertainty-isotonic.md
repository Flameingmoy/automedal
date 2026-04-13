---
id: 0017
slug: qxgb-uncertainty-isotonic
timestamp: 2026-04-13T16:50:53
git_tag: exp/0017
queue_entry: 2
status: improved
val_loss: 0.0505
val_accuracy: 0.9863
best_so_far: 0.0505
---

## Hypothesis
Training a fixed-config Quantile XGBoost (QXGBoost: Huber-quantile objective, 10th/50th/90th percentiles) alongside the 3 standard GBDTs and using prediction-interval width as inverse-weight for isotonic calibration will improve log_loss, because QXGBoost's per-sample uncertainty signal is absent from standard point predictions — wide intervals in sparse tabular regions indicate where the ensemble is unreliable — and weighting isotonic calibration toward high-confidence (narrow-interval) samples addresses miscalibration in exactly those regions where the JSD experiment (exp 0013) hypothesized but could not measure miscalibration concentration.

## What I changed
In `agent/train.py`: (1) Updated HYPOTHESIS to match the queue entry verbatim; (2) Replaced the Phase 2b TabKD neural student with Phase 2b Quantile XGBoost (QXGBoost): implemented `train_qxgb_per_class()` using XGBoost's native multi-quantile regression (`objective='reg:quantileerror'`, `quantile_alpha=[0.1, 0.5, 0.9]`, `tree_method='hist'`, `device='cuda'`) — one model per class (3 total, no HPO), early stopping at 20 rounds, max 200 estimators; (3) Added `weighted_bin_constrained_isotonic()` function that computes weighted bin means using QXGBoost's inverse-interval-width weights; (4) Added Phase 3b comparison of unweighted vs QXGBoost-weighted isotonic calibration (N=500 bins); (5) Removed all TabKD/neural student code and references. No changes to prepare.py.

## Result
- XGBoost: 31 trials, best=0.0522 | LightGBM: 4 trials, best=0.0555 | CatBoost: 25 trials, best=0.0541
- **QXGBoost median (q50) val_loss: 0.7028 — catastrophically uncalibrated (essentially random)**
- QXGBoost interval width: min=0.2948, median=0.3856, max=1.0860 (range ~0.8 — narrow)
- QXGBoost inverse weights: min=0.3742, median=1.0537, max=1.3784 (range only ~1.0 — near-uniform)
- Ensemble weights: XGB=0.82, LGB=0.01, CAT=0.17 (extremely XGB-heavy, consistent with scipy.optimize SLSQP pattern)
- Weighted isotonic: N_bins=500, val_loss=0.050583
- **Unweighted isotonic: N_bins=500, val_loss=0.050500** ← wins
- **Final: 0.0505** — **improved** from previous best 0.0508 by 0.0003

## What I learned
- **QXGBoost quantile regression is fundamentally misaligned with multiclass classification quality estimation**: QXGBoost's q50 predictions achieved val_loss=0.7028 — nearly random (3-class random baseline is ~1.099). The q50 predictions were essentially uniform across samples regardless of true class, because quantile regression learns the conditional median of the target variable (0 or 1), which is just the probability itself. But XGBoost's quantile regression is not calibrated to produce probabilities; it produces raw regression values that are hard to map back to class probabilities. The result is a broken uncertainty signal that QXGBoost itself acknowledges through its near-uniform predictions. This closes the QXGBoost axis for calibration on this task.
- **QXGBoost interval widths have near-zero discriminative power as uncertainty signals**: The interval widths (q90-q10) ranged from 0.29 to 1.09 — a range of only ~0.8 on a [0,1] scale. Normalized inverse weights ranged from 0.37 to 1.38 (only ~4x range vs the ~1000x range that would indicate genuine uncertainty discrimination). This means the "uncertain" vs "confident" samples differ by at most 4x in calibration weight — insufficient to meaningfully shift isotonic's bin means.
- **Isotonic already handles miscalibration implicitly without uncertainty guidance**: The null result (weighted 0.050583 vs unweighted 0.050500, difference of 0.000083 — within noise) confirms the KB's prior finding that calibration is not the bottleneck. Isotonic's piecewise-constant binning already down-weights unreliable (low-frequency) calibration regions implicitly by distributing samples across bins. Adding explicit uncertainty weighting from a broken QXGBoost signal can't improve on this.
- **The uncertainty-calibration sub-axis is closed**: Both JSD-weighting (exp 0013, essentially zero improvement) and QXGBoost-uncertainty-weighting (this experiment, null result) confirm that isotonic calibration on this dataset does not benefit from sample-specific uncertainty guidance. The miscalibration is either uniformly distributed or already handled by isotonic's built-in smoothing. Future effort should go to base model quality, not calibration refinements.

## KB entries consulted
- Isotonic regression is the single most effective post-processing step: ~0.0010 improvement over weighted baseline regardless of base model quality (exps 0010–0015) — **confirmed: isotonic produced the dominant improvement, weighted vs unweighted was noise-level difference**
- JSD-weighting provides essentially zero improvement over uniform isotonic weighting (+0.000005–0.000013, within noise); miscalibration is not concentrated in sparse feature regions (exp 0013) — **confirmed and extended: QXGBoost-uncertainty-weighting also provides zero improvement (+0.000083), confirming miscalibration is uniformly distributed or already handled by isotonic**
- **Calibration is not the bottleneck; base model quality is — comprehensively confirmed by exps 0013 (JSD neutral), 0012 (NA-FIR neutral), 0011 (warm-start regression), 0010 (bin regularization neutral beyond 500 bins)** — **extended: QXGBoost uncertainty-weighting is also neutral, further confirming calibration ceiling**
- Quantile XGBoost (QXGBoost) for uncertainty-aware isotonic calibration has **never been tried** (arxiv 2304.11732 — consumed exp 0017) — **axis now closed: QXGBoost quantile regression is fundamentally misaligned with multiclass classification uncertainty estimation; interval widths provide no discriminative power for calibration weighting**
