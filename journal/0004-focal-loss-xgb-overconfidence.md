---
id: 0004
slug: focal-loss-xgb-overconfidence
timestamp: 2026-04-13T10:32:30
git_tag: exp/0004
queue_entry: 1
status: worse
val_loss: 0.0549
val_accuracy: 0.9862
best_so_far: 0.0524
---

## Hypothesis
XGBoost trained with focal loss (gamma=1.0–3.0) will produce better-calibrated class probabilities, breaking the persistent XGB-heavy weight pattern (0.65/0.10/0.25) that indicates XGBoost is the most overconfident ensemble member across 12+ experiments. Focal loss's down-weighting of well-classified examples specifically targets the root cause of the log_loss plateau at 0.052421.

## What I changed
In `agent/train.py`: Replaced XGBoost's standard `multi:softprob` objective with focal-weighted sample weights. Computed focal weights `w_n = (1 - max_c p_c)^gamma` from a fast baseline XGBoost model (val_loss=0.0539), then used these as `sample_weight` in the Optuna HPO loop for 72 trials. Added `compute_focal_weights()` helper function. Focal gamma and alpha params were Optuna-tuned (gamma: 1.0–3.0, alpha_0/alpha_2: 0.5–3.0). LGB and CatBoost used default objectives unchanged.

**Failed attempt**: First tried full custom objective (manual gradient/hessian) — timed out at 660s with only 2 XGBoost trials completed because XGBoost's Python API forces CPU computation for custom objectives on 504K rows (~80s/trial).

## Result
- XGBoost focal-weighted: 72 trials, best=0.0726 (vs standard XGB baseline 0.0539 — **much worse**)
- LightGBM: 11 trials, best=0.0557
- CatBoost: 34 trials, best=0.0546
- Ensemble (w=0.10/0.25/0.65): 0.0549 (weighted wins over stacking)
- Previous best: 0.0524; current: 0.0549 (+0.0025 regression)

Focal weighting catastrophically hurt XGBoost. Ensemble weights shifted to XGB=0.10, LGB=0.25, CAT=0.65 — XGBoost's weight collapsed from 0.65 to 0.10.

## What I learned
- **Focal weighting is too aggressive on this dataset**: With max_p=0.9999 (98.6% accuracy), `(1-0.9999)^2 = 1e-8`. Focal weights averaged 0.0015 — the optimizer essentially ignored most samples. This is a dataset where models are near-perfect, so focal weighting destroys signal rather than focusing on hard examples.
- **Full custom objective is infeasible**: XGBoost with custom objective forces CPU computation per boosting round (~80s/trial on 504K rows) vs ~2s with GPU-optimized built-in objectives. Only 2 trials fit in 160s vs 72 with sample-weight approach.
- **Sample-weight focal loss ≠ true focal loss**: True focal loss modifies the loss gradient/hessian per class (reducing gradient magnitude for confident correct predictions). Sample-weight focal loss uniformly scales the gradient for ALL classes of a sample, which changes the optimization dynamics in ways that are harmful when the model is already very accurate.
- **The persistent XGB-heavy weight pattern (0.65/0.10/0.25) is NOT caused by XGBoost being overconfident**: If it were, focal loss should reduce XGBoost's loss and increase its ensemble weight. Instead, XGBoost's weight collapsed from 0.65 to 0.10, confirming the weight pattern reflects something else (complementary error patterns, not calibration).

## KB entries consulted
- XGB-heavy weights (0.55–0.65 XGB, 0.10 LGB, 0.25–0.35 CatBoost) are the most reliable pattern across 12+ experiments (exps 0001–0013, 0003)
- 3-model GPU ensemble (XGBoost + LightGBM + CatBoost) with LR meta-learner stacking is the dominant architecture; best val_loss 0.052421 at w=(0.65, 0.10, 0.25) (exp 0005)
- Temperature scaling confirmed real signal (+0.0002 reduction on weighted ensemble) but the coarse 343-combo grid search consumed ~30–60s of Optuna time budget, reducing trial counts and regressing the overall result (exp 0002)
- Temperatures ~0.94–0.95 in exp 0002 indicate the ensemble is systematically overconfident and needs softening; this is actionable signal that proper optimization never captured (exp 0002)
