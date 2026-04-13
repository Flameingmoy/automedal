# AutoMedal Knowledge Base
_Last curated: exp 0023_

## Models
- 3-model GPU ensemble (XGBoost + LightGBM + CatBoost) with isotonic calibration is the dominant architecture; best val_loss 0.051357 at w=(0.65, 0.10, 0.25) + ISO-reg (exp 0017, 0025)
- XGB-heavy weights (0.55–0.65 XGB, 0.10 LGB, 0.25–0.35 CatBoost) are the most reliable pattern across 23 experiments; the pattern is real signal, not grid-search overfitting (exps 0001–0023)
- LightGBM GPU hard-caps `max_bin` at 255 — do not raise it (exp 0002)
- CatBoost with native `cat_features=...` (ordered target statistics) performed worse than ordinal encoding by +0.0009 (exp 0012)
- LightGBM DART boosting produced structurally weaker models (0.0566 vs 0.0555 GBDT-LGB) and received 0.00 ensemble weight (exp 0014)
- Focal loss XGB with sample weights catastrophically regressed (+0.018 log_loss, avg weight 0.0015, collapsed XGB to 0.10 contribution); this axis is **closed** (exps 0017, 0018)
- TabR individually achieved 0.0688 val_loss (vs XGB 0.0533), far too weak to contribute meaningfully (5% weight); architectural diversity does not help when the diverse model is uncompetitive (exp 0021)
- T-MLP individually achieved 0.0658 (vs TabR's 0.0688), confirming the GBDT-feature gate helps vs plain MLP, but still far too weak for ensemble contribution (5% max weight); this confirms MLPs are fundamentally uncompetitive on this dataset regardless of inductive bias (exps 0022, 0023)
- XGBoost single-model baseline landed at 0.0544 (exp 0001)
- LGB achieves ~0.0550 individually, receiving only ~0.10 ensemble weight consistently (exps 0001–0023)
- CatBoost achieves ~0.0536–0.0540 individually, receiving ~0.25 ensemble weight (exps 0001–0023)

## Features
- 168 categorical aggregation features (mean/std/median of 7 numerics across 8 categories) plus polynomial/ratio/binned features did not improve val_loss (exp 0013)
- Target encoding with `smoothing=0.3` alongside ordinal encoding produced noise; val_loss worsened by +0.00002 (exp 0003)
- No frequency encoding, embedding features, or categorical interaction features have been tried

## Ensembling
- Isotonic regression is the single most effective post-processing step: 0.0514 vs 0.0530 (temp scaling) vs 0.0522 (weighted baseline); its nonlinear piecewise-constant mapping compresses extreme probabilities more effectively than global temperature scaling (exp 0017)
- Effective per-class temperatures from isotonic regression are near 1.0 (~0.999 for class 0, ~0.999 for class 1, ~1.002 for class 2), meaning improvement comes from piecewise compression not global softening (exp 0017)
- Isotonic regression provides ~0.0010 improvement regardless of moderate base model quality variation (exps 0017, 0019, 0020, 0021, 0022, 0023)
- Isotonic calibration cannot compensate for weaker base models: base model quality degradation from reduced Optuna budget was the primary failure mode across exps 0021, 0022, 0023 (exps 0021, 0022, 0023)
- LR meta-learner stacking overfits catastrophically (0.0636–0.0651) due to multicollinearity between the 3 GBDTs; all stacking replications failed (exps 0005–0018)
- XGB+CAT is the most complementary pair (0.0524) vs XGB+LGB (0.0528) or LGB+CAT (0.0534), but equal-weight Caruana subsets never beat grid-searched unequal weights (exp 0020)
- Grid-searched unequal weights (0.65/0.10/0.25) consistently beat equal-weight Caruana subsets; the persistent XGB-heavy pattern is real, not grid-search overfitting (exp 0020)
- Blending LR stacking with weighted ensemble via alpha grid-search failed across 3 experiments (exps 0007–0009)
- OOF stacking with 3-fold CV meta-learner did not outperform weighted ensemble (exp 0010)
- Finer weight grid search (0.01 step) found w=(0.53, 0.14, 0.33) but did not beat 0.65/0.10/0.25 (exp 0012)
- 4-model pipeline halved Optuna trials (42 total vs ~100+ in 3-model runs), degrading ALL base models; even with persisted configs, the chicken-and-egg problem (need strong configs to persist, but getting them requires full budget) prevents escape (exps 0021, 0022, 0023)
- Model diversity is the bottleneck: even the best equal-weight Caruana subset (XGB+CAT) loses to well-tuned weighted average (exp 0020)
- CoVar pseudo-labeling with MC > 0.88 AND RCV < median_RCV produced severe class imbalance (90K class 1 vs 1.8K class 0), corrupting augmented training; RCV was degenerate (≈0.000 for 50% of test rows), reducing dual criterion to MC alone (exp 0019)
- RCV ≈ 0.000 for 50% of test rows; the CoVar variance criterion is degenerate on this high-accuracy dataset — this closes the CoVar pseudo-labeling sub-axis (exp 0019)
- Pseudo-label augmentation hurts when base models have fewer than typical Optuna trials; Phase 1 HPO + augmented training + ISO calibration exceeds the 10-minute budget (exp 0019)

## HPO
- 60–100 Optuna trials per model within a 3-minute budget is the stable operating point (exps 0001–0023)
- Narrowing search ranges to known-good regions based on prior experiments did not beat wider ranges (exp 0012)
- All Optuna-based improvements over 0.051357 have failed; base model quality is not the bottleneck at this point (exps 0005–0023)
- Focal loss Optuna HPO for XGBoost with sample weights produces models 0.018 worse than standard objective (0.0726 vs 0.0539); the optimization landscape is fundamentally different and HPO cannot compensate (exp 0017)
- 5-fold stratified CV blend (37 trials) was worse than single holdout (0.0550 vs 0.0544) — CV overhead exceeds benefit for this dataset size (exp 0004)

## Calibration
- Temperature scaling confirmed overconfidence (+0.0002 signal, temperatures ~0.94–0.95) but coarse grid burned Optuna budget, reducing trial counts (exp 0013)
- Temperatures ~0.94–0.95 indicate systematic overconfidence; isotonic regression addresses this nonlinearly where temperature scaling's linear approach fails (exp 0013)
- ROC-regularized isotonic regression theoretically preserves multiclass ranking quality but plain per-class isotonic regression performed better; the ensemble's ranking structure is already near-optimal (exp 0017)
- Focal loss conclusively disproves the "XGBoost overconfidence causes XGB-heavy weights" hypothesis: weights collapsed from 0.65 to 0.10, confirming the pattern reflects complementary error patterns, not calibration bias (exps 0017, 0018)
- Per-class isotonic regression outperformed stacking + blending + Caruana + pseudo-labeling as the sole improvement axis (exp 0017 vs 0019, 0020)
- Temperature scaling + isotonic stacking cannot compensate for weaker base models; base model quality degradation is the primary failure mode in recent experiments (exps 0021, 0022, 0023)
- The isotonic calibration plateau at 0.051357 may reflect isotonic overfitting on noisy validation samples; the dataset is 98.6% accurate (~1.4% noise), and unconstrained isotonic regression fits noisy points exactly (open question, unconfirmed)

## Open questions
- **Isotonic bin regularization via Fisher ratio** — exp 0017's best (0.051357) may reflect isotonic overfitting on ~1.4% noisy validation samples; constraining the mapping to 50–200 bins informed by Fisher ratio and noise estimate would regularize against fitting noise without sacrificing the ~0.0010 improvement (arxiv 2512.05469 — still unconsumed)
- **JSD-based local calibration** — calibration miscalibration may be concentrated in sparse feature-space regions; JSD distance weights calibration training by local neighborhood density rather than treating all samples equally, addressing proximity bias that global temperature scaling and isotonic regression cannot fix (arxiv 2510.26566 — still unconsumed)
- **Warm-start Optuna from 20+ trial histories** — prior experiments span diverse hyperparameter configurations and their val_losses; a meta-learner on trial histories (dataset features + trial metadata → val_loss) could bias Optuna toward regions that produced strong results on this dataset, improving base model quality without adding model time budget (arxiv 2507.12604 — still unconsumed)
- **GBDT feature bagging** — train each of the 3 GBDTs on a different random 60% feature subset; reduces ensemble member correlation without adding a 4th model or halving Optuna budget; XGB+CAT complementary pair suggests feature-level diversity helps more than model-level diversity (arxiv 2512.05469 — still unconsumed)
- **Density-aware pseudo-labeling with CatBoost as 4th member** — CoVar's fixed dual-criterion failed due to RCV degeneracy; density-aware weighting uses local density in feature space to weight pseudo-labels, addressing the core issue of noisy pseudo-labels in low-density tabular regions (arxiv 2302.14013 — still unconsumed, separate from the closed CoVar axis)
- **Normalization-aware isotonic calibration (NA-FIR/SCIR)** — standard one-vs-rest isotonic ignores sum-to-one constraints; NA-FIR and SCIR enforce normalization directly in the optimization objective, potentially squeezing additional log-loss below 0.051357 (arxiv 2512.09054 — still unconsumed)
