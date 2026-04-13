---
id: 0008
slug: tabr-fourth-member-persisted
timestamp: 2026-04-13T12:07:41
git_tag: exp/0008
queue_entry: 4
status: worse
val_loss: 0.0527
val_accuracy: 0.9862
best_so_far: 0.0514
---

## Hypothesis
TabR (feed-forward network with k-NN retrieval in middle layers) as a fourth ensemble member will reduce log_loss because DART LGB's failure (exp 0003) was structural (same error patterns, zero weight), while TabR's k-NN retrieval architecture is fundamentally different from all 3 GBDTs, providing genuine ensemble diversity that the 3-model ensemble has been unable to achieve despite 12+ experiments. Using persisted Optuna configs for the 3 GBDTs avoids the budget-halving trap.

## What I changed
In `agent/train.py`: (1) Updated HYPOTHESIS to match the queue entry; (2) Added PyTorch/TabR imports (`torch`, `nn`, `F`, `DataLoader`, `TensorDataset`, `MinMaxScaler`); (3) Implemented `TabRModel` class — 2-layer MLP encoder (input→256→128), concatenated with original features, 2-layer MLP decoder → 3-class softmax; (4) Implemented `train_tabr()` function with GPU training, batch_size=8192, lr=1e-3, 30 epochs, patience=5, 90s budget; (5) Added Phase 1b: persisted GBDT Optuna configs to `agent/gbdt_configs.json`; (6) Added Phase 2b: TabR training on GPU (completed all 30 epochs, best val_loss=0.0688); (7) Replaced 3-model weight grid with 4-model coarse grid (4 values per model = ~64 combos), restricting TabR weight ≤0.5; (8) Added 3-model baseline computation for comparison; (9) Kept isotonic calibration on 4-model ensemble. TabR ran to full 30 epochs (no early stopping triggered) but val_loss plateaued around 0.0688.

## Result
- XGBoost: 12 trials, best=0.0533
- LightGBM: 3 trials, best=0.0561
- CatBoost: 21 trials, best=0.0561
- **TabR: 0.0688** (best individual, but far weaker than any GBDT)
- 4-model weights: XGB=0.50, LGB=0.15, CAT=0.30, TABR=0.05
- 3-model baseline: **0.0534** (better than 4-model 0.0538)
- ISO-calibrated: **0.0527**
- Previous best: 0.0514; current: 0.0527 (**worse by +0.0013**)

## What I learned
- **TabR is too weak individually** (0.0688 vs XGB=0.0533) to contribute meaningfully to the ensemble: it only received 5% weight in the 4-model grid search, and the 4-model ensemble (0.0538) was actually worse than the 3-model baseline (0.0534) before isotonic calibration. Even after isotonic calibration, the result regressed vs previous best.
- **MLPs on tabular data are fundamentally weaker than GBDTs on this problem**: TabR achieved 98.2% accuracy (vs GBDT ~98.6%) but log_loss of 0.0688 (vs XGB 0.0533). The gap in log_loss is much larger than the gap in accuracy because TabR produces less well-calibrated probabilities. This confirms that gradient-boosted decision trees remain the dominant architecture for this task.
- **Reduced Optuna budget (120s/model) severely hurt GBDT quality**: XGBoost had only 12 trials (vs typical 40-100+), LightGBM had 3 trials, and CatBoost had 21. These weaker base models (XGB=0.0533 vs typical 0.0523-0.0525) contributed to the regression. Even though we persisted configs, the persisted configs came from this same (weakened) HPO run — we didn't escape the budget-halving trap.
- **Persisting configs only helps if the persisted configs come from a full-quality HPO run**: The queue sketch envisioned persisting configs from a strong HPO run and reusing them. But in this experiment, the persisted configs came from the same weakened HPO run, so TabR had to "pay" for the HPO budget reduction with no compensating benefit from persisted configs.
- **The architecture diversity hypothesis was wrong for this dataset**: TabR's MLP architecture, despite being "fundamentally different" from GBDTs, produced error patterns that are correlated with GBDTs (both learn similar feature-importance relationships from tabular data). Model diversity is only useful if the diverse model is competitive.
- **Isotonic calibration cannot compensate for weaker base models**: Even with isotonic calibration (which has been the key to past improvements), the 0.0527 result is worse than previous best 0.0514. The base model quality degradation from reduced Optuna budget was the primary failure mode.

## KB entries consulted
- 4-model pipeline (adding DART LGB) halved effective Optuna trials (42 total vs ~100+ in 3-model runs), degrading all base models and producing worse ensemble despite grid-searched 4-model weights (exp 0003)
- Adding a 4th model to a fixed-time budget degrades ALL base models; to use a 4th model, existing 3 models must use pre-persisted Optuna configs — not a fresh HPO loop (exp 0003)
- 4-model weight grid must be very coarse (4–5 values per model) because 5^4 = 625 combos; fine grids consume time budget and must not overlap with Optuna budget (exp 0003)
- XGB-heavy weights (0.55–0.65 XGB, 0.10 LGB, 0.25–0.35 CatBoost) are the most reliable pattern across 15+ experiments (exps 0001–0013, 0003)
- All Optuna-based improvements over 0.052421 have failed; base model quality is not the bottleneck at this point
- Isotonic regression dramatically outperforms temperature scaling for this ensemble: 0.0514 vs 0.0530 (exp 0002) and vs the weighted baseline 0.0522. Temperature scaling's +0.0002 signal was real but too coarse with the grid search approach; isotonic regression's nonlinear piecewise-constant mapping is far more expressive. (exp 0005)
