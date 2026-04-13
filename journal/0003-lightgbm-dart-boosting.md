---
id: 0003
slug: lightgbm-dart-boosting
timestamp: 2026-04-13T09:52:25
git_tag: exp/0003
queue_entry: 1
status: worse
val_loss: 0.053700
val_accuracy: 0.986100
best_so_far: 0.052421
---

## Hypothesis
LightGBM in DART boosting mode will structurally decorrelate its predictions from XGBoost/CatBoost, breaking the persistent XGB-heavy weight pattern (0.65/0.10/0.25) that signals correlated ensemble errors across 8+ experiments.

## What I changed
In `agent/train.py`: (1) Added `build_lgb_dart()` model builder with `boosting_type="dart"`, `drop_rate`, `max_drop`, `skip_drop` as Optuna parameters; (2) Added PHASE 2b Optuna HPO loop for DART (60→40 trials, model_budget=120s each); (3) Updated Phase 3 ensemble to 4-model grid search (coarse grid with DART weight ∈ {0.0, 0.10, 0.20, 0.30} + fine-tune); (4) Updated Phase 4 stacking to 3-model and 4-model variants; (5) Updated model_budget allocation from `// 3` to `// 4`; (6) Reduced trial counts (XGB: 80, LGB: 80, DART: 40, CAT: 30).

## Result
- XGBoost: 14 trials, best=0.0539
- LightGBM: 4 trials, best=0.0555
- LightGBM DART: 3 trials, best=0.0566
- CatBoost: 21 trials, best=0.0567
- 4-model weighted ensemble: 0.0537 (weights: XGB=0.65, LGB=0.25, DART=0.00, CAT=0.10)
- Previous best: 0.0524; current: 0.0537 (+0.0013 regression)

DART received 0.00 weight in the ensemble — the coarse grid found zero contribution. The overall result regressed because the 4-model pipeline reduced effective Optuna trials (42 total vs ~100+ in prior runs), degrading base model quality.

## What I learned
- **DART boosting produces structurally weaker LGB models** (0.0566 vs 0.0555 GBDT-LGB in this run) and the ensemble grid-search assigns it zero weight, confirming the hypothesis was wrong for this dataset.
- **4-model pipeline halves effective Optuna trials per model**: 42 total trials split across 4 models vs ~100+ in 3-model runs. This is a fundamental budget conflict — adding a 4th model to a fixed-time budget degrades all base models.
- **DART does not support early stopping** in LightGBM, causing each trial to run the full n_estimators (300-800), which dramatically slows the Optuna loop and reduces trial count.
- **Coarse DART grid (4 values × 3 XGB values × ~5 LGB values)** was sufficient to detect zero contribution but consumed time budget that would have been better spent on more base model trials.
- **Fixes for future 4-model runs**: persist Optuna studies across runs (pickle), or reduce per-model budgets further (//5), or skip the 4th model's Optuna phase entirely and use fixed hyperparameters.

## KB entries consulted
- LightGBM DART boosting mode — applies dropout to trees, creating a pseudo-ensemble-of-subnetworks that structurally decorrelates LGB from XGBoost/CatBoost, breaking the persistent XGB-heavy weight pattern (arxiv 2512.05469)
- 3-model GPU ensemble (XGBoost + LightGBM + CatBoost) is the dominant architecture; best val_loss 0.052421 at w=(0.65, 0.10, 0.25) (exp 0005)
- XGB-heavy weights (0.55–0.65 XGB, 0.10 LGB, 0.25–0.35 CatBoost) are the most reliable pattern across 8+ experiments (exps 0002, 0003, 0005, 0008, 0011, 0012)
- 60–100 Optuna trials per model within a 3-minute budget is the stable operating point (exps 0002, 0005, 0006, 0011)
