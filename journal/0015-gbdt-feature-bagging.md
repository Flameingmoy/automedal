---
id: 0015
slug: gbdt-feature-bagging
timestamp: 2026-04-13T15:10:00
git_tag: exp/0015
queue_entry: 1
status: worse
val_loss: 0.0515
val_accuracy: 0.9862
best_so_far: 0.0511
---

## Hypothesis
Training each of the 3 GBDTs on a different random 60% feature subset (seeded independently) will reduce ensemble prediction correlation without adding a 4th model or halving the Optuna budget, because exp 0007 showed XGB+CAT is the most complementary pair (0.0524 vs XGB+LGB 0.0528, LGB+CAT 0.0534), implying feature-level diversity helps more than model-level diversity, and feature bagging creates exactly this diversity within the existing 3-model framework — unlike the failed TabR/T-MLP 4th-model attempts which degraded Optuna trials and diluted base model quality.

## What I changed
In `agent/train.py`: (1) Updated HYPOTHESIS to match the queue entry verbatim; (2) Added `FEATURE_BAGS` config — each of the 3 GBDTs gets all 16 numeric + 8 ordinal features (mandatory) plus a different random 60% subset of the 8 frequency-encoded features (indices 24–31), yielding 29 features per model vs 32 for the full-feature baseline; (3) Modified `load_data()` to return the full 32-feature arrays; (4) Applied per-model feature slicing (`X_train_full[:, xgb_indices]`, etc.) throughout HPO objectives and final model training for XGB, LGB, and CatBoost; (5) No changes to prepare.py or HPO trial counts. The bagging seeds were XGB=42, LGB=123, CAT=777.

## Result
- XGBoost: 22 trials, best=0.0536 | LightGBM: 7 trials, best=0.0565 | CatBoost: 31 trials, best=0.0539
- Best weights: XGB=0.45, LGB=0.10, CAT=0.45 (shifted toward equal weighting vs 0.65/0.10/0.25)
- Weighted ensemble: 0.0531
- **Best isotonic (N=500): val_loss=0.051484**
- Final: **0.0515** — **worse** than previous best 0.0511 by +0.0004

## What I learned
- **Feature bagging at 60% feature subset degrades individual model quality more than diversity gained**: XGBoost went from typical 0.0530–0.0544 to 0.0536 (within range), but LightGBM degraded from ~0.0550 to 0.0565 — a substantial 0.0015 individual regression. The ensemble's weighted loss of 0.0531 before calibration was already worse than the full-feature ensemble of 0.0524–0.0528, and even 500-bin isotonic couldn't close the gap.
- **Ensemble weights shifted toward equal (0.45/0.10/0.45 vs 0.65/0.10/0.25) — the "XGB+CAT most complementary pair" signal from exp 0007 was disrupted by feature bagging**: With each model seeing a different feature subset, the natural XGB-heavy weight pattern (0.65) collapsed toward equal weighting (0.45). This confirms that the persistent XGB-heavy weights in 15+ experiments are not just correlation artifacts — they reflect genuine XGBoost quality superiority that feature bagging diluted.
- **LightGBM was the most sensitive to feature bagging (7 trials vs 31+ for XGB/CatBoost)**: This is partly noise (trial count variation) but also consistent with LightGBM's known weakness on this dataset (~0.0550 typical). When features are further restricted, LightGBM suffers disproportionately because it already relies on the full feature set to compensate for its architectural limitations.
- **GBDT feature bagging does NOT outperform full-feature ensemble on this dataset**: The hypothesis assumed feature-level diversity would improve correlation without quality loss. The null result confirms GBDTs on this dataset need all available features; the bottleneck is not feature representation (the KB already established this via exps 0003, 0006, 0014), and creating artificial feature diversity degrades rather than improves the ensemble.

## KB entries consulted
- 3-model GPU ensemble (XGBoost + LightGBM + CatBoost) with 500-bin isotonic calibration is the dominant architecture; best val_loss **0.051087** at w=(0.65, 0.10, 0.25) + ISO-500 (exp 0010) — **confirmed: feature bagging regressed by +0.0004**
- XGB+CAT is the most complementary pair (0.0524) vs XGB+LGB (0.0528) or LGB+CAT (0.0534) — XGB and CatBoost make more diverse errors than either pairs with LGB (exp 0007) — **feature bagging disrupted this complementary pattern: weights shifted from 0.65/0.10/0.25 to 0.45/0.10/0.45**
- **The dataset's 11 numeric + 8 categorical features are already sufficient; additional feature engineering on these same features yields diminishing or negative returns** (exps 0003, 0006, 0014) — **confirmed: feature bagging of frequency features (indices 24-31) degraded individual models and ensemble quality**
- GBDT feature bagging has **never been tried** (arxiv 2512.05469 — consumed exp 0015) — **axis now closed: null result confirms feature bagging at 60% degrades GBDT quality more than diversity helps**
