---
id: 0006
slug: covar-pseudo-label-selection
timestamp: 2026-04-13T11:24:34
git_tag: exp/0006
queue_entry: 2
status: worse
val_loss: 0.0519
val_accuracy: 0.9863
best_so_far: 0.0514
---

## Hypothesis
CoVar-based pseudo-label selection (high Maximum Confidence AND low Residual-Class Variance across non-maximum classes) will produce cleaner pseudo-labels than fixed 0.95 confidence thresholds because the ensemble's 270K test predictions include high-confidence-but-wrong predictions near class boundaries that a raw max-probability threshold would include as reliable, and CoVar's variance criterion filters these out.

## What I changed
In `agent/train.py`: (1) Updated HYPOTHESIS to match the queue entry; (2) Added PHASE 3b between weighted ensemble grid search and isotonic calibration — CoVar pseudo-label selection: computed per-sample MC (max probability) and RCV (sum of squared non-max probabilities) on 270K test rows; selected rows where MC > 0.88 AND RCV < median_RCV; augmented training set from 504K to 639K rows; (3) Added PHASE 2b — trained final GBDT models on augmented data using Phase 1 best params (no re-HPO to stay within budget); (4) Added augmented ensemble weight grid search; (5) Added fallback logic — if augmented ensemble is worse than original, use original ensemble for isotonic calibration. The key deviation from the queue sketch: Phase 2b trained final models only (no fresh Optuna HPO) because the full 10-minute budget couldn't accommodate two full HPO passes.

## Result
- Phase 1 Optuna: XGBoost 11 trials (best=0.0528), LightGBM 7 trials (best=0.0552), CatBoost 31 trials (best=0.0550) — fewer trials than usual due to the model_budget allocation
- CoVar selection: 135K/270K test rows (50%) selected at MC > 0.88 AND RCV < 0.00000 (median RCV)
- Pseudo-label distribution: {class 0: 1860, class 1: 90572, class 2: 42568} — **severe class imbalance**
- Augmented ensemble (w=0.65/0.10/0.25): **0.0532** vs original **0.0526** → augmented is worse
- Script correctly fell back to original ensemble
- Final ISO-calibrated: **0.0519** (vs previous best 0.0514)
- Result: **worse** by +0.0005 relative to best_so_far

## What I learned
- **RCV ≈ 0.00000 for 50% of test rows** means the CoVar variance criterion is essentially degenerate — most high-confidence predictions have near-zero non-max probabilities, so the RCV threshold provides no additional filtering beyond MC alone. The dual criterion MC > 0.88 AND RCV < 0.00000 reduces to just MC > 0.88 effectively.
- **Pseudo-label distribution is severely imbalanced** (90K class 1 vs 1.8K class 0): the ensemble is highly biased toward class 1 at high confidence, which corrupts the augmented training set with class 1 majority pseudo-labels. This class imbalance in pseudo-labels likely explains why the augmented models degraded.
- **Pseudo-label augmentation hurts when base models are weak**: With only 11/7/31 Phase 1 trials (vs typical 40-100+), base model quality was below optimal. Training on augmented data with suboptimal hyperparameters produces models that overfit to the pseudo-label distribution.
- **The RCV metric is not discriminative on this dataset**: with 98.6% accuracy, the two non-max probabilities are almost always near-zero. A different variance metric (e.g., entropy of non-max classes, or a more calibrated confidence measure) would be needed for this criterion to be useful.
- **Budget conflict is fundamental**: Phase 1 HPO + Phase 2b augmented training + isotonic calibration exceeds 10 minutes even without Phase 2b HPO. A valid pseudo-label experiment would need persisted Optuna configs or a smaller Phase 1 budget.

## KB entries consulted
- 3-model GPU ensemble (XGBoost + LightGBM + CatBoost) with LR meta-learner stacking is the dominant architecture; best val_loss 0.052421 at w=(0.65, 0.10, 0.25) (exp 0005)
- Isotonic regression dramatically outperforms temperature scaling for this ensemble: 0.0514 vs 0.0530 (exp 0002) and vs the weighted baseline 0.0522. Temperature scaling's +0.0002 signal was real but too coarse with the grid search approach; isotonic regression's nonlinear piecewise-constant mapping is far more expressive. (exp 0005)
- 60–100 Optuna trials per model within a 3-minute budget is the stable operating point (exps 0002, 0005, 0006, 0011)
- All Optuna-based improvements over 0.052421 have failed; base model quality is not the bottleneck at this point
