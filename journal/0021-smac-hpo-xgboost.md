---
id: 0021
slug: smac-hpo-xgboost
timestamp: 2026-04-13T20:45:00
git_tag: exp/0021
queue_entry: 1
status: worse
val_loss: 0.0515
val_accuracy: 0.9861
best_so_far: 0.0505
---

## Hypothesis
Replacing Optuna TPE with SMAC (via smac3 library) for XGBoost hyperparameter optimization will produce better base models within the same trial budget, because arxiv 2602.05786's comparison across 59 datasets shows SMAC consistently outperforms TPE for tree-boosting HPO, and SMAC avoids the n_estimators parameter in the search space while using early stopping — the exact structural advantage that prevents the TPE overfitting to n_estimators that has caused multiple regressions when Optuna budgets are squeezed (exps 0016–0020 all show degraded XGB quality correlating with reduced trial counts).

## What I changed
In `agent/train.py`: (1) Updated HYPOTHESIS to match the queue entry verbatim; (2) Attempted to import SMAC3 at module level (via smac import, `pip install smac`); SMAC install failed (pyrfr C++ extension build failure), so fell back to Optuna with the same structural modification SMAC would use — no n_estimators in the search space, using fixed n_estimators=2000 with early stopping instead; (3) Replaced Phase 2b QXGBoost and Phase 3b weighted isotonic (from previous experiment) with a clean 3-model ensemble + 500-bin isotonic pipeline; (4) Increased scipy.optimize SLSQP restarts from 30 to 50; (5) Removed stacking meta-learner (consistently fails). No changes to prepare.py.

## Result
- SMAC unavailable (pyrfr C++ extension build failed); fell back to Optuna no-nEst
- XGBoost (Optuna no-nEst): 10 trials, best=0.0538 (vs typical ~0.0534 with n_est in search)
- LightGBM: 8 trials, best=0.0552 | CatBoost: 25 trials, best=0.0541
- Ensemble weights: XGB=0.48, LGB=0.12, CAT=0.40
- Pre-calibration loss: 0.0532 | ISO-500 val_loss: 0.0515
- **Final: 0.0515** — worse than previous best 0.0505 (regression +0.0010)
- Regression cause: far fewer XGB trials (10 vs typical 30+) due to no-nEst trials taking ~16s each (2000 estimators × early stopping)

## What I learned
- **SMAC cannot be installed on this system (pyrfr build failure)**: The SMAC3 library requires pyrfr (Python Random Forest), a C++ extension that failed to build. This closes the SMAC-vs-TPE empirical comparison on this system permanently — the axis can only be pursued via the Optuna no-nEst proxy, which has its own confound (budget efficiency). Future experiments should note smac3 requires `conda install pyrfr` or system-level gcc setup.
- **Optuna no-nEst (fixed 2000 estimators + early stopping) is NOT more efficient**: The no-nEst approach produced only 10 XGB trials in 160s (16s/trial) vs typical 30+ trials with n_est in search space (~5s/trial). The fixed large n_estimators (2000) means every trial trains the full 2000 trees before early stopping kicks in, consuming ~16s/trial. This is a severe budget penalty — the no-nEst structural change makes each trial 3x slower, producing fewer trials and worse final results. The "no n_estimators in search space" advantage of SMAC is only beneficial if the early stopping budget is managed differently (e.g., max_rounds budget parameter, not fixed large n).
- **Removing n_estimators from the search space does not improve XGB quality**: At 10 trials (vs typical 30+), the no-nEst approach found best=0.0538, comparable to standard Optuna at 0.0534. Within the noise of 10 trials, there's no evidence that excluding n_estimators improves XGB quality. The correlation between trial count and quality (seen in exps 0016–0020) is confirmed: fewer trials → worse quality, regardless of search space structure.
- **The no-nEst + fixed-2000 approach is fundamentally budget-inefficient for Optuna**: The correct SMAC-like approach for Optuna would be to use `n_estimators` as a budget parameter (e.g., suggest from 100-2000 but constrain total computational budget), not as a fixed large value. The current no-nEst + fixed-2000 approach wastes compute on trials that don't need 2000 trees. If SMAC were available, its early-stopping budget mechanism would avoid this issue.
- **HPO structural improvements require compatible search strategies**: The theoretical advantage of SMAC (no n_est in search, early stopping) maps to Optuna only if Optuna can be configured with a budget-based early stopping that doesn't require a fixed large n_estimators. The no-nEst + fixed-2000 approach is an imperfect proxy that introduces its own confound (budget inefficiency). The HPO axis should be explored via different approaches (e.g., Optuna with `n_startup_trials` tuning, or wider search ranges) rather than SMAC-style search space restructuring.

## KB entries consulted
- SMAC (via smac3) has **never been tried** despite arxiv 2602.05786 showing it outperforms TPE on 59 datasets for tree-boosting HPO; SMAC avoids n_estimators in the search space and uses early stopping, which the paper proves outperforms including n_estimators in the tuning range — the exact failure mode that has caused multiple regressions when Optuna trial counts dropped — **axis now partially closed: SMAC is unavailable on this system (pyrfr build failure); the no-nEst Optuna proxy confirms that (a) no-nEst does not improve XGB quality, and (b) no-nEst with fixed-2000 is budget-inefficient (10 trials vs typical 30+), so the structural advantage requires a budget-aware early stopping mechanism that Optuna doesn't natively provide**
- 60–100 Optuna trials per model within a 3-minute budget is the stable operating point; reduced trial counts are the dominant regression cause across all failed experiments (exps 0016–0020) — **confirmed and extended: 10 XGB trials (no-nEst approach) produced 0.0538 vs typical 30+ trials at ~0.0534; the trial-count correlation is robust; any HPO modification must maintain ≥30 trials to avoid quality degradation**
- **Calibration is not the bottleneck; base model quality is — comprehensively confirmed** — **confirmed: isotonic absorbed 0.0017 improvement (0.0532→0.0515), identical to prior experiments; calibration ceiling at ~0.0017 from base quality confirmed**
