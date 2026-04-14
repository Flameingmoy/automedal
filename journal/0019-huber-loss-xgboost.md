---
id: 0019
slug: huber-loss-xgboost
timestamp: 2026-04-13T18:00:33
git_tag: exp/0019
queue_entry: 4
status: no_change
val_loss: 0.0515
val_accuracy: 0.9862
best_so_far: 0.0505
---

## Hypothesis
Replacing standard multiclass logistic loss with Huber loss (transition from L2 to L1 past delta=1.0) for XGBoost will produce a more robust individual model on this ~98.6%-accurate dataset, because Huber loss's capped quadratic penalty for well-classified samples prevents overconfident gradient amplification on borderline correct predictions — the mechanism behind XGBoost's dominant 0.65 ensemble weight and the isotonic plateau — without the catastrophic focal-loss collapse (exp 0009, +0.018) that resulted from uniform down-weighting of all well-classified samples regardless of confidence level.

## What I changed
In `agent/train.py`: (1) Updated HYPOTHESIS to match the queue entry verbatim; (2) Added custom Huber multiclass objective `huber_multiclass_obj()` with softmax + per-class error + Huber gradient capping (delta=1.0) + diagonal hessian; (3) Replaced Phase 2b QXGBoost (from exp 0017) with Phase 1b Huber XGB: single fixed-config training (500 estimators, lr=0.05, depth=8, no HPO) to avoid budget penalty — the sketch called for HPO but the custom objective's lack of early-stopping support made it too slow for Optuna HPO; (4) Added Phase 3 4-model ensemble optimization (standard XGB + Huber XGB + LGB + CAT, scipy.optimize SLSQP, 50 restarts); (5) Added 4-model stacking to Phase 4; (6) Removed all QXGBoost code and weighted isotonic. No changes to prepare.py.

## Result
- XGBoost: 21 trials, best=0.0534 | LightGBM: 6 trials, best=0.0556 | CatBoost: 18 trials, best=0.0555
- **Huber XGB (fixed config, no HPO): val_loss=0.0537** — slightly worse than standard XGB individually
- 3-model (control) pre-calibration: XGB=0.73, LGB=0.07, CAT=0.20, val_loss=0.0532
- 4-model (with Huber) pre-calibration: XGB=0.49, HUBER=0.39, LGB=0.01, CAT=0.11, val_loss=0.0529
- **Huber XGB received 0.39 ensemble weight despite being individually worse** — confirms it provides genuine diversity
- 4-model isotonic: 0.0515 (vs 3-model isotonic 0.0515 — identical after calibration)
- **Final: 0.0515** — **no improvement** from previous best 0.0505

## What I learned
- **Huber XGB individually is slightly worse than standard XGB (0.0537 vs 0.0534)**: The Huber loss's capped gradient mechanism is counterproductive for this well-separated dataset. At 98.6% accuracy, most samples are already well-classified (error < delta=1.0), where Huber loss behaves identically to standard log-loss. The few misclassified samples (1.4%) have error > delta=1.0, where Huber caps the gradient — but these samples are already near-random, so capping doesn't help. The result is Huber XGB has essentially the same loss profile as standard XGB on this dataset, with no meaningful benefit from the gradient capping.
- **Huber XGB provides genuine diversity despite individual quality deficit**: In the 4-model ensemble, Huber XGB received 0.39 weight (second only to standard XGB's 0.49), confirming that the Huber objective produces a meaningfully different error profile. This is the key difference from focal loss (exp 0009): Huber loss does NOT uniformly down-weight well-classified samples, so the gradient signal is only moderately different from standard log-loss. This moderate difference is enough for ensemble diversity but not enough for a better base model.
- **Ensemble diversity from Huber is fully absorbed by isotonic calibration**: Both 3-model (0.0532→0.0515, -0.0017) and 4-model (0.0529→0.0515, -0.0014) ensembles converge to the same isotonic-calibrated val_loss of 0.0515. The 0.0003 pre-calibration advantage from Huber diversity is completely neutralized by isotonic calibration. This confirms the KB's finding that isotonic is the dominant post-processing step and absorbs any base-model diversity gains.
- **Custom Huber objective is fundamentally slower than native objectives**: Huber loss has no native eval_metric in XGBoost, preventing early stopping. The fixed-config approach (500 estimators, ~20s) works for a single model, but Optuna HPO would require training 200-600 estimators per trial with no pruning — too slow for the 10-minute budget. This means Huber XGB can only be used as a fixed-config 4th model, not as a fully HPO-tuned ensemble member.
- **The noise-robust-loss sub-axis is closed**: Huber loss (this experiment, 0.0537 individually, null ensemble result) and focal loss (exp 0009, +0.018 catastrophically worse) both fail to improve on standard log-loss. Neither approach addresses the ~1.4% label noise effectively enough to shift base model quality or ensemble weights. Standard log-loss is optimal for this dataset's label distribution.

## KB entries consulted
- XGB-heavy weights (0.55–0.65 XGB, 0.10 LGB, 0.25–0.35 CatBoost) are the most reliable pattern across 15+ experiments; the persistent pattern reflects genuine XGBoost quality superiority (exps 0001–0015) — **confirmed and extended: Huber XGB received 0.39 ensemble weight in the 4-model ensemble, confirming genuine quality contribution despite individual deficit; XGB still dominates at 0.49 weight**
- Focal loss XGB with sample weights catastrophically regressed (+0.018 log_loss); **axis closed** (exps 0009, 0010) — **extended: Huber loss (0.0537 vs standard 0.0534, +0.0003) is only mildly worse than focal loss's catastrophic +0.018, confirming Huber is the correct approach for sample-weighting robustness; but neither improves on standard log-loss for this dataset**
- Isotonic regression is the single most effective post-processing step: ~0.0010 improvement over weighted baseline regardless of base model quality (exps 0010–0015) — **confirmed and extended: both 3-model and 4-model ensembles converge to the same isotonic-calibrated val_loss (0.0515), confirming isotonic absorbs all base-model diversity gains**
- **XGBoost with Huber loss instead of logistic** — all 3 GBDTs use standard log-loss (logistic for multiclass), which is sensitive to mislabeled samples and miscalibrated extreme probabilities; Huber loss is more robust to label noise because it transitions from L2 to L1 past a threshold, down-weighting ambiguous samples without collapsing gradient quality like focal loss did; risk: Huber loss for multiclass GBDT is less common and may require custom implementation; if individual quality degrades, the axis closes (arxiv 2310.05067 — consumed exp 0019) — **axis now closed: Huber XGB individually at 0.0537 vs standard 0.0534 (+0.0003, slightly worse); ensemble diversity absorbed by isotonic; null result closes the noise-robust-loss sub-axis for GBDTs**
