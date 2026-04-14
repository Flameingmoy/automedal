# AutoMedal Knowledge Base
_Last curated: exp 0023_

## Models
- 3-model GPU ensemble (XGBoost + LightGBM + CatBoost) with 500-bin isotonic calibration is the dominant architecture; best val_loss **0.050500** at w=(0.82, 0.01, 0.17) + ISO-500 (exp 0017); the improvement over prior runs came from XGB base quality (0.0522 individually)
- XGB-heavy weights (0.55–0.92 XGB, 0.01–0.10 LGB, 0.06–0.35 CatBoost) are the most reliable pattern across 23+ experiments; scipy.optimize SLSQP continuously finds more XGB-heavy corners than the 0.05-increment grid; confirmed as theoretically optimal by UTD (exp 0020)
- QXGBoost quantile regression is fundamentally misaligned with multiclass classification uncertainty — q50 val_loss=0.7028 (near-random); interval widths provide near-zero discriminative power; **QXGBoost axis permanently closed** (exp 0017)
- Huber XGB individually at 0.0537 vs standard XGB 0.0534 (+0.0003); provided genuine diversity in 4-model ensemble but isotonic absorbed all diversity gains; **noise-robust-loss sub-axis permanently closed** (exps 0017, 0019)
- NACP sample weighting is near-uniform (95.7% of samples weight >0.5); cannot meaningfully shift isotonic's bin means; **calibration-weighting sub-axis permanently closed** (exp 0018)
- TabKD neural student catastrophically collapsed (val_loss=8.4962, exp 0016); binary categoricals yield only 112 base interaction patterns, insufficient; **neural-student axis permanently closed** (exps 0008, 0009, 0016)
- MorphBoost library not available on PyPI (no matching distribution) and GitHub returns 404 Not Found; **new-model GBDT-variant axis permanently closed** (exp 0022)
- CatBoost with native `cat_features=...` performed worse than ordinal encoding by +0.0009 (exp 0011)
- LightGBM DART boosting produced structurally weaker models (0.0566 vs 0.0555 GBDT-LGB) and received 0.00 ensemble weight (exp 0006)
- Focal loss XGB with Optuna HPO regressed catastrophically (+0.018, exp 0010); **axis permanently closed**
- T-MLP individually achieved 0.0658 (vs TabR's 0.0688); fundamentally uncompetitive; **axis permanently closed** (exps 0008, 0009)

## Features
- 168 categorical aggregation features plus polynomial/ratio/binned features did not improve val_loss (exp 0006)
- Target encoding with `smoothing=0.3` alongside ordinal encoding produced noise; val_loss worsened by +0.00002 (exp 0003)
- Frequency encoding as additional features (16→24 features) degraded CatBoost to 0.0555 vs typical 0.0536–0.0540 (exp 0014)
- GBDT feature bagging at 60% feature subsets degrades individual model quality more than diversity helps; val_loss regressed +0.0004 to 0.051484 vs best 0.050500 (exp 0015)
- **The dataset's 11 numeric + 8 categorical features are already sufficient; additional feature engineering yields diminishing or negative returns** (exps 0003, 0006, 0014, 0015)

## Ensembling
- Isotonic regression with 500 bins is the single most effective post-processing step: ~0.0017 improvement over weighted baseline regardless of base model quality; it absorbs nearly all diversity gains from any ensemble variant (exps 0010–0023)
- **Calibration is not the bottleneck; base model quality is — confirmed across 8 independent calibration approaches** (JSD, NA-FIR, warm-start, bin regularization, QXGBoost-uncertainty, NACP, UTD, QP-theory) all produced noise-level or null improvements over unweighted isotonic (exps 0011–0023)
- 3-GBDT with isotonic (0.050500) has never been beaten by any 4th-model addition — TabR, T-MLP, TabKD, Huber-XGB, QXGBoost all failed (exps 0008, 0009, 0016–0019)
- scipy.optimize SLSQP continuously finds XGB-heavy corners; the theoretical ceiling for weight optimization is essentially exhausted (exp 0016)
- **UTD confirms XGB-heavy pattern is theoretically optimal**: Bias²=0.012280 is fixed across all weight configurations, confirming base model quality is the only remaining lever (exp 0020)
- LR meta-learner stacking failed across 10+ replications due to multicollinearity; **axis permanently closed** (exps 0005–0011)
- Multi-seed XGB averaging was budget-inefficient: Optuna HPO consumed 553s of 600s budget (20 XGB, 8 LGB, 23 CatBoost trials), leaving only 64s for multi-seed retraining; the experiment regressed to 0.050806 vs 0.0505 due to fewer trials, not the multi-seed idea; **the core hypothesis (5-seed XGB averaging → smoother OOF → tighter isotonic) remains UNTESTED** (exp 0023)
- SEA (Self-Error Adjustment, arxiv 2508.04948) has never been empirically tested; QP-based ensemble weights (arxiv 2512.22286) has never been empirically tested
- **The sole remaining ensembling lever is variance reduction at the individual-model level via multi-seed averaging** — UTD identified variance as the sole weight-dependent term; multi-seed averaging on a fixed config (without competing with Optuna for budget) could test this

## HPO
- **60–100 Optuna trials per model within a 3-minute budget is the stable operating point; reduced trial counts are the dominant regression cause across all failed experiments** — confirmed: 31 trials → 0.0505; 21 → 0.0515; 14 → 0.0514; 10 → 0.0515; 6 → 0.0518; fewer trials regress every time (exps 0016–0023)
- **SMAC unavailable on this system**: pyrfr C++ extension build failure; cannot be installed via pip; requires `conda install pyrfr` or system-level gcc setup (exp 0021)
- **No-nEst Optuna (fixed 2000 estimators + early stopping) is budget-inefficient**: Each trial takes ~16s (full 2000 trees before early stopping), producing only 10 XGB trials in 160s vs typical 30+ at ~5s/trial with n_est in search space (exp 0021)
- Warm-start Optuna with prior trial histories regressed base model quality (+0.0004); **axis closed** (exp 0011)
- Narrowing search ranges to known-good regions did not beat wider ranges (exp 0012)
- Focal loss Optuna HPO for XGBoost produces models 0.018 worse than standard objective (exp 0010); **axis permanently closed**
- **StratifiedKFold(n_splits=5) for Optuna HPO validation has never been tested**; standard holdout may leave minority classes with unreliable validation signals for LGB and CatBoost
- **Irredundant k-fold CV (non-overlapping training partitions, arxiv 2507.20048) has never been tested**; standard overlapping folds produce optimistically biased variance estimates
- **Early-stopping k-fold CV during Optuna HPO (arxiv 2405.03389) has never been tested**; the paper shows +167% more configurations within the same time budget by stopping individual fold training when val_loss flattens

## Calibration
- Per-class isotonic regression temperatures are consistently near 1.0 (~0.999–1.002); improvement comes from piecewise probability compression, not global softening (exps 0010–0020)
- The gap between current best (0.050500) and the theoretical isotonic ceiling is ~0.0003; calibration refinements are exhausted across 8 independent approaches
- **The next improvement opportunity must come from base model quality, not post-hoc calibration** — confirmed across 8 calibration variants, 4 neural additions, 2 loss-function changes, and UTD first-principles analysis

## Open questions
- **Irredundant k-fold HPO** — arxiv 2507.20048 proves non-overlapping training partitions give less optimistic variance estimates and reduce compute by eliminating redundant instance usage; if Optuna selects more robust hyperparameters from honest variance comparisons, base model quality improves and isotonic gains ~0.0002–0.0005; risk: the benefit requires enough data per fold to avoid training on too-small partitions; also requires more folds or larger dataset than standard k-fold
- **Early-stopping k-fold HPO** — arxiv 2405.03389 shows +167% more configurations within the same time budget by stopping fold training when val_loss plateaus; this could restore the ~30+ trial count that exp 0023 lost (only 20 XGB trials due to budget exhaustion); risk: premature stopping per fold may select low-quality early-stopping rounds that hurt overall model quality
- **StratifiedKFold Optuna HPO** — standard random 5-fold may leave minority classes with unreliable validation folds, degrading HPO reliability for LGB and CatBoost which receive only 0.01–0.35 ensemble weight; class-stratified fold assignment ensures each fold faithfully represents the class distribution; risk: if minority class weakness is a genuine quality problem (not a fold reliability problem), stratified folds don't help
- **SEA (Self-Error Adjustment) for accuracy-diversity tradeoff** — arxiv 2508.04948 decomposes ensemble error into self-error + diversity terms with adjustable λ; unlike NCL which failed, SEA provides tighter theoretical bounds and broader λ range; grid-searching λ may find a better operating point than scipy-optimized fixed weights; risk: UTD showed variance reduction dominates on this dataset, so promoting diversity via high λ may hurt
- **QP-based ensemble weight optimization** — arxiv 2512.22286 proves optimal ensemble weights are solutions to constrained quadratic programs rather than gradient-based SLSQP search; replacing SLSQP with QP may find better optima; risk: SLSQP already finds XGB-heavy corners robustly across 23+ experiments; QP may not improve on an already-converged optimum
