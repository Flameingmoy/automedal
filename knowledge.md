# AutoMedal Knowledge Base
_Last curated: migration from flat results.tsv (pre-exp 13)_

## Models
- 3-model ensemble (XGBoost + LightGBM + CatBoost) beats any single GBDT; weighted ensemble at val_loss 0.0524 is the current best (exp 0005)
- LightGBM GPU hard-caps `max_bin` at 255 — do not raise it (exp 0002)
- XGBoost baseline alone landed at val_loss 0.0544; ensemble improved to 0.0532 in one step (exps 0001, 0002)

## Ensembling
- LR meta-learner (stacking on val predictions) consistently loses to weighted ensemble by +0.001 to +0.011 log loss across exps 0005, 0007, 0008, 0009, 0010 — stop trying LR stacking on top of 3 GBDTs
- XGB-heavy weights (w=(0.65, 0.10, 0.25)) are the most reliable ensemble pattern; exps 0002, 0003, 0005, 0008, 0012 all landed here or near
- Wider weight grid search (0.01 step vs 0.05 step) did not find meaningfully better weights (exp 0011)
- OOF stacking with 3-fold CV meta-learner did not outperform weighted ensemble (exp 0010)
- Blending stacking with weighted ensemble via grid-searched alpha did not beat either one alone (exps 0008, 0009)

## Features
- Target encoding with `smoothing=0.3` alongside ordinal encoding produced noise, no signal — val_loss worsened by +0.00002 (exp 0003)
- 168 categorical aggregation features (mean/std/median of 7 numerics across 8 categories) plus polynomial/ratio/binned features did not improve val_loss (exp 0006)
- 5-fold stratified CV blend increased val_loss compared to single holdout (exp 0004) — likely fold noise or insufficient trials per fold

## HPO
- Narrowing Optuna search ranges to known-good regions based on prior experiments did not beat wider ranges (exp 0012)
- 60-100 Optuna trials per model within a 3-minute-per-model budget is the stable operating point (exps 0002, 0005, 0006, 0011)

## Open questions
- CatBoost with native categorical handling (`cat_features=...`) instead of pre-encoding — never attempted on this competition
- PyTorch TabNet as a fourth ensemble member, diversifying from GBDT-only
- Dart boosting mode in XGBoost or LightGBM to reduce overfitting on validation
- Pseudo-labeling high-confidence test predictions as extra training data
- FLAML AutoML within the 10-minute budget as an orthogonal baseline
