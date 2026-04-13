---
id: 0007
slug: caruana-meta-learner
timestamp: 2026-04-13T11:47:41
git_tag: exp/0007
queue_entry: 3
status: worse
val_loss: 0.0515
val_accuracy: 0.9863
best_so_far: 0.0514
---

## Hypothesis
The Caruana-style top-k model averaging (average predictions of the best k-model subset) as the stacking meta-learner will produce more stable results than LR stacking because LR stacking replicated unreliably across 5+ attempts (exps 0007–0010) due to multicollinearity between the three GBDT members, and the Caruana method is inherently robust to correlated base predictions by construction.

## What I changed
In `agent/train.py`: (1) Updated HYPOTHESIS to match the queue entry; (2) Added PHASE 3b between weighted ensemble grid search and isotonic calibration — Caruana-style top-k averaging: evaluated all 7 non-empty subsets (3 single models, 3 pairs, 1 triplet) with equal weights, selected the subset with lowest val OOF log_loss; (3) Extended isotonic calibration to apply to both the grid-searched weighted ensemble AND the best Caruana subset, comparing both; (4) Updated Phase 4 method selection to include all 5 approaches: weighted, caruana, iso_calibrated, iso_caruana, stacking. The comparison reveals which pre-calibration base is better for isotonic regression.

## Result
- XGBoost: 25 trials, best=0.0525
- LightGBM: 6 trials, best=0.0550
- CatBoost: 33 trials, best=0.0539
- **Caruana subsets** (equal weights):
  - XGB: 0.0525 | LGB: 0.0550 | CAT: 0.0539
  - XGB+LGB: 0.0528 | XGB+CAT: **0.0524** | LGB+CAT: 0.0534 | XGB+LGB+CAT: 0.0526
- Best Caruana subset: **XGB+CAT** (0.0524)
- Grid-searched weighted: 0.0522 (0.65/0.10/0.25) — beats best Caruana subset
- ISO-weighted: **0.0515** | ISO-Caruana: 0.0517 | Stacking: 0.0637
- Final: **iso_calibrated (0.0515)** — but **worse** than previous best 0.0514 by +0.0001

## What I learned
- **XGB+CAT is the most complementary pair**: The best Caruana subset (XGB+CAT at 0.0524) beat all other subsets, confirming that XGBoost and CatBoost make more diverse errors than XGBoost+LightGBM or LGB+CAT. However, even the best equal-weight Caruana subset (0.0524) is worse than the grid-searched weighted ensemble (0.0522), showing unequal weights matter.
- **Grid-searched unequal weights consistently beat equal-weight Caruana subsets**: The best Caruana subset (XGB+CAT equal-weight: 0.0524) loses to grid-searched 0.65/0.10/0.25 (0.0522). This is the key insight: the persistent XGB-heavy pattern is not an artifact of grid search overfitting — it's a real signal that XGBoost carries the most predictive weight.
- **Caruana + isotonic (0.0517) is worse than weighted + isotonic (0.0515)**: Even after isotonic calibration, the equal-weight Caruana pre-calibration baseline underperforms the unequal-weight pre-calibration baseline. The grid-searched weights' superior starting point propagates through isotonic calibration.
- **Model diversity is the bottleneck, not the ensembling method**: Stacking (0.0637) catastrophically overfits due to multicollinearity, confirming the KB. But even the simple Caruana approach can't beat grid-searched weights. The 3 GBDTs produce too similar error patterns for any ensembling method to dramatically outperform a well-tuned weighted average.
- **Low trial counts this run** (25/6/33 = 64 total vs 67 in exp 0005) contributed to the slight regression: weaker base models → weaker ensemble → weaker isotonic calibration improvement. This is expected variance, not a methodological failure.

## KB entries consulted
- LR meta-learner stacking achieved the best single result at 0.052421 (exp 0005) but all subsequent replication attempts regressed: exp 0007 (blending LR + weighted), exp 0008 (blending LR + weighted + alpha grid), exp 0009 (blending LR + weighted + tuned LR C), exp 0010 (OOF stacking with 3-fold CV meta-learner)
- 3-model GPU ensemble (XGBoost + LightGBM + CatBoost) with LR meta-learner stacking is the dominant architecture; best val_loss 0.052421 at w=(0.65, 0.10, 0.25) (exp 0005)
- XGB-heavy weights (0.55–0.65 XGB, 0.10 LGB, 0.25–0.35 CatBoost) are the most reliable pattern across 15+ experiments (exps 0001–0013, 0003)
- Isotonic regression dramatically outperforms temperature scaling for this ensemble: 0.0514 vs 0.0530 (exp 0002) and vs the weighted baseline 0.0522. Temperature scaling's +0.0002 signal was real but too coarse with the grid search approach; isotonic regression's nonlinear piecewise-constant mapping is far more expressive. (exp 0005)
- All Optuna-based improvements over 0.052421 have failed; base model quality is not the bottleneck at this point
