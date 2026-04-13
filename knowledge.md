# AutoMedal Knowledge Base
_Last curated: exp 0015_

## Models
- 3-model GPU ensemble (XGBoost + LightGBM + CatBoost) with 500-bin isotonic calibration is the dominant architecture; best val_loss **0.051087** at w=(0.65, 0.10, 0.25) + ISO-500 (exp 0010)
- XGB-heavy weights (0.55–0.65 XGB, 0.10 LGB, 0.25–0.35 CatBoost) are the most reliable pattern across 15+ experiments; the persistent pattern reflects genuine XGBoost quality superiority, not grid-search overfitting (exps 0001–0015)
- XGB+CAT is the most complementary pair (0.0524) vs XGB+LGB (0.0528) or LGB+CAT (0.0534) — XGB and CatBoost make more diverse errors than either pair with LGB (exp 0007)
- LightGBM GPU hard-caps `max_bin` at 255; achieves ~0.0550 individually, receiving only ~0.10 ensemble weight (exps 0001–0015)
- CatBoost achieves ~0.0536–0.0540 individually, receiving ~0.25 ensemble weight; sensitive to redundant features — frequency encoding degraded it to 0.0555 in exp 0014 (exps 0001–0015)
- CatBoost with native `cat_features=...` (ordered target statistics) performed worse than ordinal encoding by +0.0009 (exp 0011)
- LightGBM DART boosting produced structurally weaker models (0.0566 vs 0.0555 GBDT-LGB) and received 0.00 ensemble weight (exp 0006)
- Focal loss XGB with sample weights catastrophically regressed (+0.018 log_loss); **axis closed** (exps 0009, 0010)
- Focal loss XGB with Optuna HPO still regressed by +0.0025 (exp 0010); fixed γ and HPO-tuned γ both harmful; **axis closed** (exp 0010)
- T-MLP individually achieved 0.0658 (vs TabR's 0.0688); both are fundamentally uncompetitive (~5% max ensemble weight); **axis closed** (exps 0008, 0009)
- LR meta-learner stacking failed across 10+ replications (0.0636–0.0651) due to multicollinearity; **axis closed** (exps 0005–0011)
- GBDT feature bagging at 60% feature subsets degrades individual model quality more than diversity helps; val_loss regressed +0.0004 to 0.051484 vs best 0.051087 (exp 0015)
- LightGBM was most sensitive to feature bagging (degraded 0.0015 individually); weights shifted toward equal (0.45/0.10/0.45 vs 0.65/0.10/0.25), confirming XGB-heavy pattern reflects genuine quality superiority; **axis closed** (exp 0015)

## Features
- 168 categorical aggregation features (mean/std/median of 7 numerics across 8 categories) plus polynomial/ratio/binned features did not improve val_loss (exp 0006)
- Target encoding with `smoothing=0.3` alongside ordinal encoding produced noise; val_loss worsened by +0.00002 (exp 0003)
- Frequency encoding as additional features (16→24 features) degraded CatBoost to 0.0555 vs typical 0.0536–0.0540, the worst individual CatBoost in recent experiments; primary regression driver (exp 0014)
- Frequency encoding is redundant with ordinal encoding on this dataset: GBDTs already learn frequency-like information from ordinal features via split conditions (exp 0014)
- **The dataset's 11 numeric + 8 categorical features are already sufficient; additional feature engineering on these same features yields diminishing or negative returns** (exps 0003, 0006, 0014, 0015)
- **Frequency encoding as additional features: axis closed** (exp 0014)
- **GBDT feature bagging: axis closed** (exp 0015)

## Ensembling
- Isotonic regression is the single most effective post-processing step: ~0.0010 improvement over weighted baseline regardless of base model quality (exps 0010–0015)
- 500-bin custom isotonic calibration: 0.051087 vs sklearn default isotonic 0.051716 — a +0.0006 gap confirming sklearn auto-binning is suboptimal for 126K-sample val set (exp 0010)
- U-shaped bin-count curve: val_loss increases monotonically as N decreases (500→200→100→50→30); fewer bins = underfitting (exp 0010)
- Grid-searched unequal weights (0.65/0.10/0.25) consistently beat equal-weight Caruana subsets; the XGB-heavy pattern is real (exp 0007)
- JSD-weighting provides essentially zero improvement over uniform isotonic weighting (+0.000005–0.000013, within noise); miscalibration is not concentrated in sparse feature regions (exp 0013)
- Density-threshold isotonic catastrophically failed (0.27–0.28) because isotonic needs the full validation distribution (exp 0013)
- NA-FIR sum-to-one post-hoc normalization: only +0.000012 over default isotonic; sum-to-one violations are negligible (exp 0012)
- Warm-start Optuna via enqueue_trial regressed base models (+0.0004); Optuna's random sampler is already near-optimal (exp 0011)
- ROC-regularized isotonic did not outperform plain per-class isotonic (exp 0005)
- **Calibration is not the bottleneck; base model quality is — comprehensively confirmed by exps 0013 (JSD neutral), 0012 (NA-FIR neutral), 0011 (warm-start regression), 0010 (bin regularization neutral beyond 500 bins). Future effort should go to base model quality and feature engineering, not calibration refinements.** (exps 0010–0013)
- Adding a 4th model to a fixed-time budget degrades ALL base models; **axis closed** (exps 0008, 0009)
- CoVar pseudo-labeling with MC > 0.88 AND RCV < median_RCV produced severe class imbalance; RCV was ≈0.000 for 50% of test rows; **axis closed** (exp 0006)
- Pseudo-labeling with fewer than typical Optuna trials exceeds the 10-minute budget; **axis closed** (exp 0006)
- SEA (Self-Error Adjustment) ensembling has **never been run** (arxiv 2508.04948)
- CAST density-aware pseudo-labeling has **never been tried** (arxiv 2310.06380)
- TabKD interaction-diversity knowledge distillation has **never been tried** (arxiv 2603.15481)
- Quantile XGBoost (QXGBoost) for uncertainty-aware isotonic calibration has **never been tried** (arxiv 2304.11732)
- NACP noise-aware conformal prediction for isotonic calibration has **never been tried** (arxiv 2501.12749)
- Unified Theory of Diversity in Ensemble Learning has **never been empirically tested** (arxiv 2301.03962)

## HPO
- 60–100 Optuna trials per model within a 3-minute budget is the stable operating point; reduced trial counts are the dominant regression cause (exps 0001–0015)
- Narrowing search ranges to known-good regions did not beat wider ranges (exp 0012)
- Focal loss Optuna HPO for XGBoost produces models 0.018 worse than standard objective (exp 0009)
- 5-fold stratified CV blend (37 trials) was worse than single holdout (0.0550 vs 0.0544); CV overhead exceeds benefit for this dataset size (exp 0004)
- Temperature scaling temperatures ~0.94–0.95 confirmed systematic overconfidence; isotonic's nonlinear approach handles this where temperature scaling's linear approach fails (exp 0005)
- Warm-start Optuna with prior trial histories regressed base model quality (+0.0004); Optuna's random sampler already efficiently explores the hyperparameter landscape (exp 0011)

## Calibration
- Per-class isotonic regression temperatures are consistently near 1.0 (~0.999–1.002); improvement comes from piecewise probability compression, not global softening (exps 0010–0015)
- **Calibration is not the bottleneck; base model quality is — comprehensively confirmed** (exps 0010–0013, 0015)
- The gap between current best (0.051087) and the theoretical isotonic ceiling is ~0.0003 (calibration refinements exhausted)
- The next improvement opportunity must come from base model quality, not post-hoc calibration

## Open questions
- **TabKD interaction-diversity knowledge distillation** — standard KL-divergence KD fails on tabular because tabular models encode knowledge through feature interactions, not just logits; TabKD generates synthetic queries maximizing pairwise interaction coverage, producing a genuinely diverse neural student without a 4th-model HPO budget penalty; the TabR/T-MLP neural axis is closed (both uncompetitive individually), but TabKD's student is trained via interaction-diversity matching on 3 GBDT teachers — fundamentally different training signal; risk: if the neural student remains uncompetitive (~0.065+), the axis closes permanently (arxiv 2603.15481 — unconsumed)
- **Quantile XGBoost (QXGBoost) for uncertainty-weighted isotonic** — QXGBoost with Huber-quantile loss produces prediction intervals whose width is a direct per-sample uncertainty measure absent from point predictions; isotonic calibration can be weighted by inverse-interval-width (confident samples get higher calibration weight), addressing miscalibration in high-uncertainty regions; risk: QXGBoost adds a 4th model whose HPO budget would reduce the 3 GBDT trial counts — use a single fixed QXGBoost config (no HPO) to avoid budget penalty (arxiv 2304.11732 — unconsumed)
- **NACP noise-aware conformal prediction for isotonic calibration** — isotonic treats every validation label as ground truth, but the dataset's ~1.4% noisy labels distort the calibration mapping; NACP estimates a noise transition matrix from calibration data and adjusts validation sample weights accordingly, down-weighting samples whose labels are likely mislabeled; risk: isotonic already handles noise implicitly via piecewise-constant smoothing, so NACP's correction may be redundant with isotonic's built-in regularization (arxiv 2501.12749 — unconsumed)
- **Unified Theory of Diversity — empirical validation** — the persistent XGB-heavy weights (0.65/0.10/0.25) across 15+ experiments may represent the exact optimal bias-variance-diversity tradeoff for this dataset's label distribution; arxiv 2301.03962 proves that for certain label skews, promoting diversity hurts accuracy; empirically testing equal-weight Caruana (λ→0) vs current unequal weights confirms whether diversity-forcing methods are fundamentally misaligned with this data's structure; risk: already tested in exp 0007 (Caruana equal-weight was worse), but the theory suggests computing the exact optimal diversity level rather than grid-searching weights (arxiv 2301.03962 — unconsumed)
- **XGBoost with Huber loss instead of logistic** — all 3 GBDTs use standard log-loss (logistic for multiclass), which is sensitive to mislabeled samples and miscalibrated extreme probabilities; Huber loss is more robust to label noise because it transitions from L2 to L1 past a threshold, down-weighting ambiguous samples without collapsing gradient quality like focal loss did; risk: Huber loss for multiclass GBDT is less common and may require custom implementation; if individual quality degrades, the axis closes (arxiv 2310.05067 — partially consumed for CatBoost RFL, not for XGB Huber)
