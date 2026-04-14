---
id: 0024
slug: irredundant-kfold-hpo
timestamp: 2026-04-13T21:30:00
git_tag: exp/0024
queue_entry: 1
status: worse
val_loss: 0.0529
val_accuracy: 0.9863
best_so_far: 0.0505
---

## Hypothesis
Running Optuna HPO with irredundant k-fold CV (arxiv 2507.20048, non-overlapping training partitions per fold) instead of standard overlapping k-fold will improve base model quality, because overlapping folds inflate validation correlation and produce optimistically biased variance estimates that cause Optuna to select hyperparameters overfit to the specific fold pattern — irredundant k-fold's honest variance estimates give sharper comparative distinctions between configurations, enabling Optuna to select more robust hyperparameter settings that generalize better to the held-out test set and shift the base model quality ceiling above the current 0.0534 XGB individual ceiling.

## What I changed
In `agent/train.py`: (1) Updated HYPOTHESIS to match the queue entry verbatim; (2) Added RANDOM_SEED=42 and N_K_FOLDS=3 constants for irredundant k-fold; (3) Split the full training set (504K rows) into an 80% HPO set (403K rows) and a 20% calibration set (101K rows) using StratifiedShuffleSplit; (4) Replaced Optuna objectives for all 3 models (XGB, LGB, CatBoost) with irredundant 3-fold CV: each fold uses non-overlapping training blocks (67% of HPO set per fold), reporting mean log_loss across folds as the Optuna objective; (5) Retrained final models on the full HPO set with calibration-set early stopping, evaluated on the held-out validation set; (6) Replaced Phase 3/3b/4 with a clean pipeline: SLSQP weight optimization (50 restarts) + 500-bin isotonic calibration; (7) Removed QXGBoost, weighted isotonic, and LR stacking (budget cleanup per sketch). No changes to `agent/prepare.py`.

## Result
- XGBoost: 11 irredundant 3-fold trials, best CV=0.0582; final=0.0547
- LightGBM: 6 trials, best CV=0.0601; final=0.0553
- CatBoost: 19 trials, best CV=0.0589; final=0.0548
- Ensemble weights: XGB=0.264, LGB=0.395, CAT=0.341 (much more balanced than typical XGB-heavy pattern)
- Pre-calibration loss: 0.0538 | ISO-500 loss: 0.0529
- **Final: val_loss=0.0529** — worse than best 0.0505 by +0.0024
- The irredundant k-fold approach regressed significantly vs the standard holdout approach

## What I learned
- **Irredundant k-fold CV reduces effective training data per fold by ~40% vs standard k-fold**: Standard k-fold with 80/20 train/val split gives 80% data per fold for training; irredundant 3-fold with a 20% calibration holdout gives only 67% × 80% = 53.6% of data per fold for training. This smaller training set directly degrades model quality — the 0.0547 XGB final vs typical 0.0523-0.0534 confirms this.
- **Fewer Optuna trials are the dominant regression cause**: LightGBM completed only 6 trials (vs typical 30+), XGB completed 11 (vs typical 30+). The 3-fold irredundant approach uses 3× more compute per trial than standard holdout, consuming the budget before Optuna can explore the hyperparameter space adequately. This confirms the established trial-count correlation (exps 0016–0023).
- **The calibration set holdout reduces HPO set size further**: The 20% calibration holdout (101K rows) was necessary for isotonic calibration, but it reduced the HPO set from 504K to 403K rows — an 80% reduction that compounds with the 3-fold training block reduction (67% per fold). The combined reduction (503K → 269K per fold) is the root cause of model quality degradation.
- **Balanced ensemble weights (0.26/0.40/0.34) suggest more model diversity but lower individual quality**: The SLSQP optimizer found a more balanced weight distribution, implying the irredundant k-fold selected hyperparams that produce more diverse but individually weaker models. However, the isotonic calibration absorbed all diversity gains, consistent with prior experiments.
- **Budget constraints make irredundant k-fold infeasible for this dataset/HPO configuration**: At 504K rows and GPU training speed (~5-6s per fold-model), irredundant k-fold requires ~3× the compute of standard holdout per trial, making it impossible to complete the required 30+ trials within the 10-minute budget. The approach could work if (a) the dataset were smaller, (b) GPU training were faster, or (c) fewer folds (3 vs 4) were used with more aggressive early stopping.

## KB entries consulted
- **60–100 Optuna trials per model within a 3-minute budget is the stable operating point; reduced trial counts are the dominant regression cause** — CONFIRMED and EXTENDED: fewer trials (XGB=11, LGB=6 vs typical 30+) are the primary regression cause from irredundant k-fold; the 3× compute per trial (3 folds × ~5s vs 1 holdout × ~4s) consumed the budget before Optuna could explore adequately, consistent with the established trial-count → val_loss correlation
- **The sole remaining ensembling lever is variance reduction at the individual-model level via multi-seed averaging** — CONSISTENT: irredundant k-fold did not improve individual model quality; the more balanced weights (0.26/0.40/0.34 vs typical 0.65/0.10/0.25) suggest more model diversity but lower individual quality; isotonic absorbed all diversity gains
