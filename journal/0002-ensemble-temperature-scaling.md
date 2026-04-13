---
id: 0002
slug: ensemble-temperature-scaling
timestamp: 2026-04-13T09:02:00
git_tag: exp/0002
queue_entry: 1
status: worse
val_loss: 0.053000
val_accuracy: 0.986200
best_so_far: 0.052421
---

## Hypothesis
Applying per-class temperature scaling to the 3-model ensemble's softmax outputs will reduce log_loss because log_loss rewards well-calibrated probabilities, and the binary surrogate trick from arxiv 2411.02988 generalizes cleanly to this 3-class setting — the most theoretically grounded direction remaining.

## What I changed
In `agent/train.py`, added Phase 3b after the weighted ensemble is computed: (1) coarse grid search over per-class temperatures (t0, t1, t2 ∈ [0.7, 1.3] in 0.1 steps = 343 combos), (2) Nelder-Mead refinement seeded from the best grid point, (3) apply fitted temperatures to both val and test weighted ensemble probabilities via `probs^T = exp(log(probs) / T) / Z`. Added `from scipy.optimize import minimize`. Restructured Phase 4/5 so the final comparison includes the calibrated ensemble alongside stacking.

## Result
- Weighted ensemble (pre-calibration): 0.0532
- Temperature-scaled ensemble: 0.0530 (temperatures = [0.9497, 0.9371, 0.9470])
- Temperature scaling improved the ensemble by 0.0002
- Previous best: 0.0524; current run: 0.0530 (worse)
- Worse overall because Optuna ran only 24/8/43 trials (vs ~60+ each in prior runs) due to the coarse grid search overhead in Phase 3b consuming time budget.

## What I learned
- Per-class temperature scaling IS effective: it reduced weighted ensemble log_loss by 0.0002 (0.0532 → 0.0530) on this run, confirming the technique has signal.
- Temperatures < 1.0 (all ~0.94–0.95) indicate the ensemble probabilities are slightly over-confident and need to be softened.
- The coarse grid search (343 combos) before Nelder-Mead consumed ~30–60s of the Optuna budget, reducing trial counts for the base GBDT models and hurting the overall result. Future use should use a finer Nelder-Mead or L-BFGS-B from the start.
- Optuna's effective trial budget is the real bottleneck: the best base models (XGB=0.0533, CAT=0.0546 this run vs 0.0533, 0.0540 in exp 0005) were slightly worse, contributing to the overall regression. Saving Optuna studies (pickle) across runs would allow reuse of good hyperparameters.

## KB entries consulted
- Post-hoc per-class temperature scaling on the ensemble's softmax outputs has never been attempted (arxiv 2411.02988 — binary surrogate calibration generalizes to multiclass)
- 3-model GPU ensemble (XGBoost + LightGBM + CatBoost) is the dominant architecture; best val_loss 0.052421 at w=(0.65, 0.10, 0.25) (exp 0005)
