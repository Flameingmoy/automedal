---
id: 0009
slug: t-mlp-fourth-member
timestamp: 2026-04-13T13:22:00
git_tag: exp/0009
queue_entry: 1
status: worse
val_loss: 0.0526
val_accuracy: 0.9863
best_so_far: 0.0514
---

## Hypothesis
T-MLP initialized from XGBoost's GBDT feature importance scores will achieve ~0.055–0.058 individual val_loss (vs TabR's catastrophic 0.0688 in exp 0008), receive 10–20% ensemble weight, and push the isotonic-calibrated ensemble below 0.051357 because TabR's failure was architectural (MLP encoder → k-NN → decoder was too weak), while T-MLP's GBDT-feature gate provides a principled inductive bias that makes the MLP competitive with GBDTs on this dataset.

## What I changed
In `agent/train.py`: (1) Updated HYPOTHESIS to match the queue entry; (2) Added PyTorch imports (torch, nn, F, DataLoader, TensorDataset, MinMaxScaler); (3) Implemented TMLPModel class — GBDT-feature-gated MLP with a 5-layer gate network (input→128→256→input with sigmoid) that learns importance-based feature weighting, plus a 3-layer main MLP (256→128→3 with BatchNorm and Dropout); (4) Implemented train_tmlp() function with GPU training, batch_size=8192, lr=1e-3, ReduceLROnPlateau scheduler, 30 epochs, patience=5, dynamic budget allocation; (5) Added Phase 1b: persist GBDT Optuna configs to agent/gbdt_configs.json; (6) Added Phase 2b: T-MLP training (~134s budget); (7) Added Phase 3: 3-model baseline weight grid search; (8) Added Phase 3b: 4-model coarse weight grid restricting T-MLP ≤ 0.20; (9) Added Phase 4: isotonic calibration on best candidate; (10) Compared all 4 approaches (weighted_3m, iso_3m, weighted_4m, iso_4m) and selected the best. Reduced Optuna trial counts (XGB: 60, LGB: 30, CAT: 40 with 120s timeout each) to fit T-MLP within the 11-minute budget.

## Result
- XGBoost: 20 trials, best=0.0535 (reduced from ~65 in exp 0017 due to time budget)
- LightGBM: 3 trials, best=0.0557 (severely under-optimized)
- CatBoost: 30 trials, best=0.0558 (reduced from ~31 in exp 0017)
- **T-MLP: 0.0658** best val_loss at epoch 20 (early stopped from 30) — better than TabR's 0.0688 but far weaker than XGBoost's 0.0535
- 3-model weights: XGB=0.55, LGB=0.20, CAT=0.25 → loss=0.0535
- 4-model weights: XGB=0.60, LGB=0.20, CAT=0.15, TMLP=0.05 → loss=0.0536 (T-MLP at max allowed 5%)
- iso_calibrated_3m: **0.0526** | iso_calibrated_4m: 0.0526 | weighted_3m: 0.0535 | weighted_4m: 0.0536
- Final: **0.0526** — **worse** than previous best 0.0514 by +0.0012

## What I learned
- **T-MLP individually (0.0658) is significantly better than TabR (0.0688) but still far too weak to contribute meaningfully**: The GBDT-feature gate provides better inductive bias than TabR's k-NN retrieval, but the MLP architecture is still fundamentally weaker than GBDTs on this dataset. The 0.0658 vs 0.0688 gap confirms the architecture change helps, but not nearly enough.
- **T-MLP received 5% ensemble weight (maximum allowed)**: The coarse 4-model grid restricted T-MLP ≤ 0.20, but even with this generous upper bound, T-MLP was allocated only 5%. This is similar to TabR's 5% in exp 0008 — the weak individual quality limits the weight the grid search assigns.
- **4-model ensemble (0.0536) was actually slightly worse than 3-model (0.0535) before isotonic**: Adding T-MLP at 5% weight dilutes the stronger GBDT predictions. Isotonic calibration compensates (both converge to 0.0526), but the base quality degradation from fewer Optuna trials is the root cause of the regression.
- **Reduced Optuna trials (20/3/30 vs typical 40+/7+/30+) degraded ALL base models**: XGBoost degraded from typical ~0.0525 to 0.0535, CatBoost from ~0.0536 to 0.0558, and LightGBM from ~0.0550 to 0.0557. This quality degradation propagates through isotonic calibration, producing 0.0526 vs 0.0514.
- **The 4th-model budget trap is fundamental**: Adding any 4th model to a fixed-time budget degrades base models. The queue sketch acknowledged this risk and recommended persisted configs — but even with persistence, the experiment requires running Optuna first to get strong configs to persist. This creates a chicken-and-egg problem: you need strong configs to persist, but getting them requires the full Optuna budget, which you then don't have for T-MLP.
- **Isotonic calibration is robust but not magic**: Both 3m and 4m converged to the same isotonic value (0.0526), confirming that isotonic can't compensate for weaker base models — only about 0.0009 improvement over the weighted baseline regardless of 4th model presence.

## KB entries consulted
- TabR individually achieved 0.0688 val_loss (vs XGB 0.0533), far too weak to contribute meaningfully to the ensemble (5% weight, exp 0008)
- MLPs on tabular data are fundamentally weaker than GBDTs on this problem; the architecture diversity hypothesis failed (exp 0008)
- 4-model pipeline halved Optuna trials (42 total vs ~100+ in 3-model runs), degrading ALL base models; TabR had 5% weight, 4-model was worse than 3-model (exp 0008, 0014)
- Adding a 4th model to a fixed-time budget degrades ALL base models; to use a 4th model, existing 3 models must use pre-persisted Optuna configs — not a fresh HPO loop (exp 0008)
- XGB-heavy weights (0.55–0.65 XGB, 0.10 LGB, 0.25–0.35 CatBoost) are the most reliable pattern across 20+ experiments (exps 0001–0020)
- Isotonic regression provides 0.0010 improvement regardless of moderate base model quality variation (exp 0017 vs 0018, 0019)
- Isotonic calibration cannot compensate for weaker base models: even with isotonic calibration (which has been the key to past improvements), the 0.0527 result is worse than previous best 0.0514. The base model quality degradation from reduced Optuna budget was the primary failure mode. (exp 0008)
