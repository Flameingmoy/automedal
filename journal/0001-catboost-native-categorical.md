---
id: 0001
slug: catboost-native-categorical
timestamp: 2026-04-13T08:48:00
git_tag: exp/0001
queue_entry: 1
status: worse
val_loss: 0.053300
val_accuracy: 0.986200
best_so_far: 0.052421
---

## Hypothesis
Passing categorical columns directly to CatBoost via `cat_features=...` (using its ordered target statistics encoding) will extract more signal than ordinal encoding, because CatBoost's internal algorithm is specifically designed for categoricals and avoids the information loss of manual pre-encoding.

## What I changed
Modified `agent/prepare.py` to produce two separate data versions: (1) ordinal-encoded arrays for XGBoost/LightGBM (16 numeric + 8 OrdinalEncoder-coded categoricals = 24 features), and (2) a pickle DataFrame for CatBoost with 16 numeric + 8 raw LabelEncoder-coded integer categoricals = 24 features, with `cat_feature_indices=[16..23]`. Modified `agent/train.py` to pass `cat_features=[16,17,18,19,20,21,22,23]` to CatBoost and set `border_count` search range to [256, 512]. Removed the stacking phase (per queue sketch: retain weighted ensemble only). XGBoost and LightGBM used unchanged ordinal-encoded arrays.

## Result
- XGBoost final: 0.0535
- LightGBM final: 0.0558
- CatBoost (native cats) final: 0.0560
- Ensemble (w=0.65/0.10/0.25): 0.0533
- Previous best (ordinal CatBoost): 0.0524

CatBoost with native `cat_features` performed *worse* than CatBoost with ordinal encoding in the ensemble. The val_loss regressed by +0.0009, confirming a null result.

## What I learned
- CatBoost's ordered target statistics encoding (via `cat_features`) did not outperform ordinal encoding on this dataset — the signal gain hypothesis was wrong.
- The `y_prob does not sum to one` warnings from LightGBM during Optuna HPO suggest that using a DataFrame for CatBoost with raw integer cat codes (rather than ordinal-encoded floats) changed how CatBoost normalizes its internal probabilities, which may have affected ensemble calibration.
- Implementing native categorical handling required complex data preparation (separate pickle DataFrames with correct int dtypes), which is not worth the complexity for a null result.
- The best CatBoost approach remains ordinal encoding + weighted ensemble (the approach that produced the 0.052421 baseline).

## KB entries consulted
- 3-model GPU ensemble (XGBoost + LightGBM + CatBoost) is the dominant architecture; best val_loss 0.052421 at w=(0.65, 0.10, 0.25) (exp 0005)
- CatBoost with native categorical handling (`cat_features=...`) instead of pre-encoding — never attempted; CatBoost's ordered TS encoding is a different algorithm from `category_encoders.TargetEncoder` — **null result confirmed; this axis is now closed**
