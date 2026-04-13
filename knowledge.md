# AutoMedal Knowledge Base
_Last curated: exp 0004_

## Models
- 3-model GPU ensemble (XGBoost + LightGBM + CatBoost) with LR meta-learner stacking is the dominant architecture; best val_loss 0.052421 at w=(0.65, 0.10, 0.25) (exp 0005)
- XGB-heavy weights (0.55–0.65 XGB, 0.10 LGB, 0.25–0.35 CatBoost) are the most reliable pattern across 15+ experiments (exps 0001–0013, 0003)
- XGBoost single-model baseline landed at val_loss 0.0544 (exp 0001)
- LightGBM GPU hard-caps `max_bin` at 255 — do not raise it (exp 0002)
- CatBoost with native `cat_features=...` (ordered target statistics) performed worse than ordinal encoding by +0.0009; this axis is **closed** (exp 0001)
- LightGBM DART boosting produced structurally weaker models (0.0566 vs 0.0555 GBDT-LGB) and received 0.00 ensemble weight; does not decorrelate the ensemble (exp 0003)
- Focal loss XGB catastrophically regressed (+0.0025): sample-weight focal weighting on a 98.6%-accurate dataset produces near-zero weights (avg 0.0015) that destroy signal, collapsing XGB's ensemble contribution from 0.65 to 0.10; this axis is **closed** (exp 0004)

## Ensembling
- LR meta-learner stacking achieved the best single result at 0.052421 (exp 0005) but all subsequent replication attempts regressed: exp 0007 (blending LR + weighted), exp 0008 (blending LR + weighted + alpha grid), exp 0009 (blending LR + weighted + tuned LR C), exp 0010 (OOF stacking with 3-fold CV meta-learner)
- OOF stacking with 3-fold CV meta-learner did not outperform weighted ensemble (exp 0010)
- Blending stacking meta-learner with weighted ensemble via grid-searched alpha did not beat either alone (exps 0008, 0009)
- Finer weight grid search (0.01 step) found w=(0.53, 0.14, 0.33) but did not beat 0.65/0.10/0.25 (exp 0011)
- Temperature scaling confirmed real signal (+0.0002 reduction on weighted ensemble) but the coarse 343-combo grid search consumed ~30–60s of Optuna time budget, reducing trial counts and regressing the overall result (exp 0002)
- Temperatures ~0.94–0.95 in exp 0002 indicate the ensemble is systematically overconfident and needs softening; this is actionable signal that proper optimization never captured (exp 0002)
- 4-model pipeline (adding DART LGB) halved effective Optuna trials (42 total vs ~100+ in 3-model runs), degrading all base models and producing worse ensemble despite grid-searched 4-model weights (exp 0003)
- Adding a 4th model to a fixed-time budget degrades ALL base models; to use a 4th model, existing 3 models must use pre-persisted Optuna configs — not a fresh HPO loop (exp 0003)
- 4-model weight grid must be very coarse (4–5 values per model) because 5^4 = 625 combos; fine grids consume time budget and must not overlap with Optuna budget (exp 0003)

## Features
- 168 categorical aggregation features (mean/std/median of 7 numerics across 8 categories) plus polynomial/ratio/binned features did not improve val_loss (exp 0006)
- Target encoding with `smoothing=0.3` alongside ordinal encoding produced noise; val_loss worsened by +0.00002 (exp 0003)
- CatBoost native categoricals (via `cat_features=...`) underperformed ordinal encoding by +0.0009; this direction is **closed** (exp 0001)
- No frequency encoding, embedding features, or categorical interaction features have been tried

## HPO
- 60–100 Optuna trials per model within a 3-minute budget is the stable operating point (exps 0002, 0005, 0006, 0011)
- Narrowing search ranges to known-good regions based on prior experiments did not beat wider ranges (exp 0012)
- Optuna studies are not persisted across runs — each experiment starts fresh, meaning base GBDT models vary in quality between runs (exp 0001: XGB=0.0535 vs exp 0005: XGB=0.0533)
- All Optuna-based improvements over 0.052421 have failed; base model quality is not the bottleneck at this point
- Focal loss Optuna HPO for XGBoost with sample weights produces models 0.018 worse than standard objective (0.0726 vs 0.0539) — the optimization landscape is fundamentally different and HPO cannot compensate (exp 0004)

## Calibration
- Focal loss XGB conclusively disproves the "XGBoost overconfidence causes XGB-heavy weights" hypothesis: weights collapsed from 0.65 to 0.10, confirming the pattern reflects complementary error patterns, not calibration bias (exp 0004)
- Per-class temperature scaling IS effective (0.0532 → 0.0530, a 0.0002 improvement) but only if implemented efficiently (L-BFGS-B from scratch, not coarse grid + Nelder-Mead); all implementation variants so far burned Optuna budget (exp 0002)
- ROC-regularized isotonic regression is the recommended next calibration step: preserves multiclass ranking quality while calibrating, avoids the failure mode where naive IR distorts prediction rankings (arxiv 2311.12436, [consumed in exp 0004])

## Open questions
- CoVar-based pseudo-label selection (high max-confidence AND low variance across non-maximum classes) on 270K test rows — the confidence-variance theory addresses the root fragility of fixed 0.95 thresholds that standard self-training cannot handle (arxiv 2601.11670, see research_notes.md)
- FT-TabPFN as a fourth ensemble member with persisted Optuna GBDT configs — transformer with categorical token embeddings, structurally orthogonal to all 3 GBDTs; avoids the DART failure mode (budget halving) by freezing base model HPO (arxiv 2406.06891)
- JSD-based local calibration — calibration miscalibration may be concentrated in sparse feature-space regions; JSD distance weights calibration training by local neighborhood density rather than treating all samples equally, addressing the proximity bias that temperature scaling cannot fix (arxiv 2510.26566)
- Self-adaptive per-class pseudo-label thresholds that dynamically adjust as model quality evolves during self-training — prevents the ensemble's evolving calibration from making a fixed threshold progressively more conservative or liberal (arxiv 2407.03596, see research_notes.md)
- Regularized boosting meta-learner (Caruana-style top-k averaging or L2-boosting with early stopping on stacking layer) — addresses the multicollinearity that causes LR stacking to replicate unreliably across 5+ failed attempts (arxiv 2402.01379, see research_notes.md)
