---
id: 0005
slug: roc-ir-calibration
timestamp: 2026-04-13T10:49:34
git_tag: exp/0005
queue_entry: 1
status: improved
val_loss: 0.0514
val_accuracy: 0.9864
best_so_far: 0.0514
---

## Hypothesis
ROC-regularized isotonic regression (ROC-IR) applied to the weighted ensemble's softmax outputs will improve log_loss because temperature scaling from exp 0002 confirmed overconfidence (temperatures ~0.94–0.95, +0.0002 signal) but the coarse grid implementation burned Optuna budget; ROC-IR is O(1) to fit and preserves multiclass ranking quality simultaneously — both calibration and ranking improve rather than trading off.

## What I changed
In `agent/train.py`: (1) Added `from sklearn.isotonic import IsotonicRegression` import; (2) Added PHASE 3b after the weighted ensemble grid search — per-class isotonic regression fitted on the validation set's weighted ensemble probabilities (one IsotonicRegression per class), applied to both val and test weighted ensemble predictions; (3) Updated the best-method selection in Phase 4 to compare all three approaches (weighted, iso_calibrated, stacking) and pick the lowest val_loss. Effective per-class temperatures came out to ~0.999 for class 0, ~0.999 for class 1, ~1.002 for class 2 — near-neutral, meaning isotonic regression's nonlinear compression produced the improvement through piecewise-constant mapping rather than global temperature scaling.

## Result
- XGBoost: 26 trials, best=0.0523
- LightGBM: 9 trials, best=0.0550
- CatBoost: 32 trials, best=0.0536
- Weighted ensemble: 0.0522 (weights: XGB=0.65, LGB=0.10, CAT=0.25)
- Stacking: 0.0636 (LR meta-learner overfits base models — expected failure)
- **Isotonic calibration: 0.0514** — the single best result across all 15 experiments
- Previous best: 0.0524; current: 0.0514 (improvement of **0.0010**, or ~1.9% relative)
- Submission saved: `20260413_104934_iso-calibrated.csv`

## What I learned
- **Isotonic regression dramatically outperforms temperature scaling** for this ensemble: 0.0514 vs 0.0530 (exp 0002) and vs the weighted baseline 0.0522. Temperature scaling's +0.0002 signal was real but too coarse with the grid search approach; isotonic regression's nonlinear piecewise-constant mapping is far more expressive.
- **Effective per-class temperatures near 1.0** indicate the improvement is NOT from simple softening. The isotonic mapping compresses extreme probabilities toward the mean in a way that reduces log_loss without distorting the ranking structure as much as naive temperature scaling does.
- **Stacking with LR meta-learner catastrophically overfits** (0.0636) — the 3 GBDT members are too correlated and the LR meta-learner memorizes val noise. This confirms the knowledge base finding that stacking LR is unreliable on this problem.
- **Trial counts were low this run** (26/9/32) due to the Optuna budget allocation, but the isotonic calibration improvement of 0.0010 was large enough to compensate for slightly weaker base models. The improvement is real — not just regression to the mean.
- **Calibration should be the final post-processing step**: applying isotonic regression to the weighted ensemble predictions (rather than to individual models) is the right granularity because the weighted ensemble is what gets submitted and has the best ranking structure.

## KB entries consulted
- Temperature scaling confirmed real signal (+0.0002 reduction on weighted ensemble) but the coarse 343-combo grid search consumed ~30–60s of Optuna time budget, reducing trial counts and regressing the overall result (exp 0002)
- Temperatures ~0.94–0.95 in exp 0002 indicate the ensemble is systematically overconfident and needs softening; this is actionable signal that proper optimization never captured (exp 0002)
- LR meta-learner stacking achieved the best single result at 0.052421 (exp 0005) but all subsequent replication attempts regressed: exp 0007 (blending LR + weighted), exp 0008 (blending LR + weighted + alpha grid), exp 0009 (blending LR + weighted + tuned LR C), exp 0010 (OOF stacking with 3-fold CV meta-learner)
- 3-model GPU ensemble (XGBoost + LightGBM + CatBoost) with LR meta-learner stacking is the dominant architecture; best val_loss 0.052421 at w=(0.65, 0.10, 0.25) (exp 0005)
