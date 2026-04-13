---
id: 0014
slug: frequency-encoding
timestamp: 2026-04-13T14:48:46
git_tag: exp/0014
queue_entry: 1
status: worse
val_loss: 0.0515
val_accuracy: 0.9861
best_so_far: 0.0511
---

## Hypothesis
Replacing ordinal encoding with frequency encoding (category count / total count) for the 8 categorical features will reduce log_loss because ordinal encoding imposes an arbitrary numerical ordering on categorical levels that distorts GBDT split quality — ordinal '1 < 2 < 3' forces the model to learn that category-1 and category-2 are more similar than category-1 and category-3, regardless of whether this reflects actual predictive relationships — while frequency encoding preserves the distributional information (how common each category is) that GBDTs can exploit for more informed splits, and this distributional signal is complementary to the existing numeric features rather than being artificially ordinal; this is especially relevant for high-cardinality categoricals in multi-class problems, both conditions present here (arxiv 2403.19405, unconsumed).

## What I changed
In `agent/prepare.py`: (1) Replaced the 2-return-value `encode_categoricals()` with a 4-return-version that computes both ordinal encoding (unchanged) and frequency encoding (category_count / total_count) for each of the 8 categorical features; (2) Updated `prepare_data()` to concatenate frequency-encoded features as 8 new numeric columns alongside ordinal-encoded categoricals, yielding 32 total features (16 numerics + 8 ordinal + 8 frequency) vs the previous 24 (16 numerics + 8 ordinal). No changes to train.py HPO budget, calibration code, or model architectures.

## Result
- XGBoost: 27 trials, best=0.0536 | LightGBM: 8 trials, best=0.0553 | CatBoost: 36 trials, best=0.0555
- Weighted ensemble (0.55/0.20/0.25): 0.0534
- Best isotonic: N_bins=500, val_loss=0.051531
- **Final: 0.0515** — **worse** than previous best 0.0511 by +0.0004

## What I learned
- **Frequency encoding as additional features degraded CatBoost most (0.0555 vs typical 0.0536–0.0540)**: The 0.002 individual quality drop is substantial and consistent with CatBoost being most sensitive to redundant/partial-information features. LightGBM stayed near its typical ~0.0550 range, while XGBoost was 0.0536 (within its normal range of 0.0525–0.0544). The overall ensemble regressed because CatBoost — historically the second-best individual model receiving ~0.25 weight — was weakened.
- **Frequency encoding is redundant with ordinal encoding on this dataset**: The hypothesis assumed ordinal encoding's arbitrary ordering was harmful and frequency encoding's distributional signal was complementary. Instead, the ordinal encoding apparently already captures the signal needed, and frequency encoding adds noise or splits the feature-importance budget. GBDTs can already learn frequency-like information from ordinal-encoded features via their split conditions (e.g., ordinal ≥ 2 captures "categories seen in ≥X% of data"). The extra 8 features reduced the effective information per feature.
- **The distributional signal (category prevalence) does not add predictive value beyond the ordinal ordering**: In a well-calibrated 3-class irrigation prediction problem with 98.6% accuracy, the ordinal encoding of categorical features already correlates with the target. A "Soil_Type=3" (ordinal) being more common than "Soil_Type=7" is already implicit in the GBDT's split decisions — adding "Soil_Type_freq=0.15" as a separate feature is redundant.
- **Feature count increased from 24 to 32 without improving base models**: This confirms the KB's prior observations (exp 0006: 168 aggregation features also did not help). The dataset's 11 numeric + 8 categorical features are already sufficient; additional feature engineering on these same features yields diminishing or negative returns. The bottleneck remains GBDT hyperparameter tuning and ensemble diversity, not feature representation.
- **CatBoost's degradation is the primary regression driver**: Ensemble weights shifted from 0.65/0.10/0.25 (XGB-heavy) to 0.55/0.20/0.25, reflecting CatBoost's weaker individual quality. Since CatBoost and XGBoost are the most complementary pair (KB: "XGB+CAT is the most complementary pair at 0.0524"), degrading CatBoost's quality reduces ensemble diversity and hurts overall log_loss.

## KB entries consulted
- Frequency encoding (category count/total count) and similarity encoding have **never been tried** (arxiv 2403.19405, unconsumed) — **axis now closed: frequency encoding provided no improvement, degraded CatBoost individual quality**
- CatBoost achieves ~0.0536–0.0540 individually, receiving ~0.25 ensemble weight (exps 0001–0023) — CatBoost's 0.0555 individual here is the worst observed in recent experiments, confirming frequency features degraded its quality
- Target encoding with `smoothing=0.3` alongside ordinal encoding produced noise; val_loss worsened by +0.00002 (exp 0003) — **confirms that ordinal encoding is near-optimal for categoricals on this dataset; alternative encodings add noise**
