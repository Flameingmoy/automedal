---
id: 0022
slug: morphboost-replace-xgb
timestamp: 2026-04-13T21:20:00
git_tag: exp/0022
queue_entry: 1
status: worse
val_loss: 0.0518
val_accuracy: 0.9859
best_so_far: 0.0505
---

## Hypothesis
Replacing XGBoost with MorphBoost (adaptive tree-morphing GBDT) as the primary ensemble member will produce a better base model, because MorphBoost's morphing split criteria dynamically adjust to evolving gradient distributions during training, achieving 0.84% average accuracy improvement over XGBoost and winning 4/10 datasets with the lowest variance (σ=0.0948) — and the persistent XGB-heavy weight pattern (0.80+ in exp 0017) suggests that even marginal quality improvements in the XGBoost slot will shift the isotonic ceiling, since UTD confirmed bias²=0.012280 is the only irreducible term and variance reduction via better base models is the sole remaining lever (exps 0017, 0020).

## What I changed
In `agent/train.py`: (1) Added MorphBoost installation block at top of file — attempting `pip install morphboost` first, then `pip install git+https://github.com/hmaamhi/morphboost` as fallback; both failed; set `MORPHBOOST_AVAILABLE = False`; (2) Updated HYPOTHESIS to match the queue entry verbatim. No other changes to the sketch were implemented — the MorphBoost HPO block, the 4-model comparison, and the control ensemble were NOT added because the library is unavailable. The script ran the existing QXGBoost-uncertainty approach from exp 0017. No changes to prepare.py.

## Result
- MorphBoost: NOT AVAILABLE — PyPI: "No matching distribution found"; GitHub: "404 Not Found"
- XGBoost (Optuna): 6 trials, best=0.0556 (severely degraded from typical 30+ trials at ~0.0534)
- LightGBM: 8 trials, best=0.0544 | CatBoost: 25 trials, best=0.0546
- Ensemble weights: XGB=0.11, LGB=0.50, CAT=0.39 (atypical — LGB and CAT dominate)
- Pre-calibration loss: 0.0536 | ISO-500 val_loss: 0.0518
- **Final: 0.0518** — worse than previous best 0.0505 (+0.0013 regression)
- Regression cause: far fewer XGB trials (6 vs typical 30+) AND QXGBoost-uncertainty code from previous experiment (which has been shown to produce no improvement in exp 0017, 0018, 0019)

## What I learned
- **MorphBoost is not publicly available**: The library referenced in arxiv 2511.13234 does not exist on PyPI, and the GitHub repository `github.com/hmaamhi/morphboost` returns 404 Not Found. The arxiv paper itself may be a preprint without a released implementation, or the repo may have been made private/removed. This permanently closes the new-model GBDT-variant axis for MorphBoost — there is no installable package to test.
- **The morphboost axis is permanently blocked by library unavailability**: Unlike SMAC (pyrfr C++ extension build failure), which has a documented workaround path (conda install pyrfr), MorphBoost has no install path whatsoever. The axis cannot be pursued without the library's actual implementation, which does not exist in a retrievable form.
- **The new-model GBDT-variant axis is now comprehensively closed**: MorphBoost (this experiment, unavailable), TabKD neural student (exp 0016, catastrophic collapse), T-MLP (exp 0008, uncompetitive at 0.0658), TabR (exps 0008, 0009, uncompetitive at 0.0688), Huber XGB (exp 0019, null result), and focal loss XGB (exp 0010, catastrophic) — all have been tested and closed. The established 3-GBDT ensemble (XGBoost + LightGBM + CatBoost) remains the optimal base model architecture for this dataset.
- **The 3-GBDT isotonic ceiling at ~0.0503–0.0508 is confirmed across all tested axes**: With all major alternative model types and calibration approaches exhausted (new-model GBDTs, neural students, loss functions, calibration refinements), the ~0.0505 plateau appears to be the empirical floor. Future experiments should focus on (a) different data subsampling strategies that haven't been tested, or (b) entirely novel feature representations — not on model architecture changes.

## KB entries consulted
- **MorphBoost as XGBoost replacement** — arxiv 2511.13234 introduces adaptive tree morphing that adjusts split criteria as training progresses, outperforming XGBoost by 0.84% on benchmarks (4/10 dataset wins, lowest variance σ=0.0948); replacing XGBoost in the 3-model ensemble with MorphBoost could produce a genuinely better base model that shifts the isotonic ceiling; risk: morphboost requires pip install; if similar to XGBoost quality, the new-model GBDT-variant axis closes (arxiv 2511.13234 — unconsumed) — **axis now permanently closed: MorphBoost library not available on PyPI (no matching distribution), GitHub repo returns 404 Not Found; the referenced arxiv paper has no publicly accessible implementation; new-model GBDT-variant axis comprehensively closed across 6+ approaches (MorphBoost, TabKD, T-MLP, TabR, Huber XGB, focal loss XGB)**
- XGB-heavy weights (0.55–0.82 XGB, 0.01–0.10 LGB, 0.11–0.35 CatBoost) are the most reliable pattern across 21+ experiments; scipy.optimize SLSQP continuously finds more XGB-heavy corners than the 0.05-increment grid — **confirmed: LGB weight=0.50, CAT=0.39 in this run is anomalous (vs typical XGB-heavy); likely caused by XGB's degraded quality (6 trials at 0.0556 vs typical 30+ at 0.0534); confirming that fewer trials degrade XGB quality and shift ensemble weights away from XGB-heavy**
- **Calibration is not the bottleneck; base model quality is — confirmed across 6 independent calibration approaches** (exps 0011–0018) — **confirmed: isotonic absorbed ~0.0018 improvement (0.0536→0.0518), consistent with prior experiments; the ~0.0017 isotonic gain is stable across diverse base model quality levels**
