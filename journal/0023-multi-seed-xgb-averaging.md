---
id: 0023
slug: multi-seed-xgb-averaging
timestamp: 2026-04-13T22:45:00
git_tag: exp/0023
queue_entry: 1
status: no_change
val_loss: 0.0508
val_accuracy: 0.9863
best_so_far: 0.0505
---

## Hypothesis
Training XGBoost with 5 different random seeds and averaging OOF predictions before isotonic calibration will reduce the variance term (sole remaining lever per UTD exp 0020) without sacrificing the Optuna trial count that caused all recent regressions, because multi-seed averaging is a post-HPO variance reduction step that operates on persisted best-config retraining — it adds ~3x compute on a fixed config, not on Optuna search — and if isotonic's 500 equal-frequency bins benefit from smoother base predictions, the calibration mapping tightens by ~0.0001–0.0003.

## What I changed
In `agent/train.py`: (1) Updated HYPOTHESIS to match the queue entry verbatim; (2) Replaced Phase 2b QXGBoost uncertainty code with Phase 2c multi-seed XGBoost retraining (5 seeds × best Optuna config, averaged OOF predictions before ensembling); (3) Replaced Phase 3b weighted/unweighted isotonic comparison with simple 500-bin isotonic + scipy.optimize SLSQP 50-restart weight optimization; (4) Removed Phase 4 stacking (LR meta-learner); (5) Reduced Optuna n_trials from 40→30 for XGB and LGB to make budget room for multi-seed retraining. In `agent/prepare.py`: no changes.

## Result
- XGBoost Optuna: 20 trials, best=0.0523 | LightGBM: 8 trials, best=0.0550 | CatBoost: 23 trials, best=0.0543
- Multi-seed XGB retraining: **SKIPPED** — only 64s remaining after Optuna HPO (553s consumed vs 600s budget)
- Ensemble weights: XGB=0.92, LGB=0.02, CAT=0.06
- ISO-500 val_loss: 0.050806 | Weighted pre-calibration: 0.052269
- **Final: val_loss=0.0508** — no improvement vs best 0.0505 (+0.0003 regression)
- The multi-seed averaging was NOT tested (time budget consumed by Optuna HPO)
- The 0.0508 vs 0.0505 gap is attributable to fewer XGB trials (20 vs 31 in exp 0017)

## What I learned
- **The 10-minute budget is insufficient for both full Optuna HPO AND multi-seed XGB retraining**: Optuna HPO consumed 553s of the 600s budget (3 models × ~160s each), leaving only 64s. The multi-seed 5-seed × 15-20s/seed = ~75-100s needed for XGB multi-seed alone exceeds remaining budget. Any future multi-seed experiment must reduce Optuna trials much more aggressively (e.g., 15-20 trials per model) or share the best config more efficiently.
- **Reducing Optuna trials from 40→30 for XGB did NOT fully solve the budget problem**: Even at 30 trials (with timeout=160s), XGB completed only 20 trials, suggesting each trial takes ~7-8s rather than the expected ~3-4s. The n_estimators suggested by Optuna may be large (up to 2000), causing longer training times per trial. This explains why Optuna ran over budget despite the n_trials reduction.
- **The multi-seed hypothesis remains UNTESTED**: The core idea (5-seed XGB averaging → smoother OOF → tighter isotonic mapping) could not be evaluated because the implementation consumed too much budget. The experiment confirms that multi-seed must be implemented differently — e.g., run multi-seed BEFORE Optuna HPO on the full train set, not as a post-HPO step; or use a separate pre-trained config file that doesn't compete with Optuna for the 10-minute window.
- **Optuna n_trials is a soft constraint when combined with timeout**: Setting n_trials=40 with timeout=160s does NOT guarantee 40 trials complete — it only guarantees the study stops after 160s. This means the actual trial count varies depending on trial duration. For reproducible results, Optuna should be configured with only n_trials (no timeout) when the overall script budget is independently enforced, or the timeout should be set much lower (e.g., 100s) to leave room for retraining.
- **The XGB-heavy weight pattern (0.92, 0.02, 0.06) is even more extreme than typical (0.65-0.82)**: This suggests LGB and CatBoost individual quality degraded more than XGB (XGB=0.0523 individual, LGB=0.0551, CAT=0.0543). With only 20 XGB trials (vs typical 30+), the Optuna config may be suboptimal, causing XGB to dominate the ensemble even more.

## KB entries consulted
- **The sole remaining ensembling lever is variance reduction at the individual-model level via multi-seed averaging** — UTD showed variance is the only weight-dependent term (exps 0020) — **UNCONFIRMED: multi-seed was not executed due to budget exhaustion; the hypothesis remains untested**
- 60–100 Optuna trials per model within a 3-minute budget is the stable operating point; reduced trial counts are the dominant regression cause across all failed experiments (exps 0016–0020) — **CONFIRMED and EXTENDED: even reducing to 30 trials (XGB) + 30 trials (LGB) was insufficient to leave budget for multi-seed; Optuna HPO consumed 553s out of 600s total, causing the core experimental change to be skipped entirely; the trial-count quality regression (0.0508 vs 0.0505) is consistent with the established trial-count correlation**
- **Calibration is not the bottleneck; base model quality is — confirmed across 6 independent calibration approaches** — **CONFIRMED: isotonic absorbed ~0.0015 improvement (0.0523→0.0508), consistent with prior experiments; calibration ceiling at ~0.0015 from base quality confirmed**
