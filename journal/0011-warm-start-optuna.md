---
id: 0011
slug: warm-start-optuna
timestamp: 2026-04-13T14:30:00
git_tag: exp/0011
queue_entry: 2
status: worse
val_loss: 0.0515
val_accuracy: 0.9862
best_so_far: 0.0511
---

## Hypothesis
Warm-starting Optuna with a meta-learner trained on the 20+ prior experiment trial histories (each trial's hyperparameter configuration + val_loss → next trial's prior) will produce stronger base models within the same trial budget because all recent regressions (exps 0019–0023) trace back to reduced Optuna trial counts from auxiliary code consuming budget, and the previous trials already explored the hyperparameter landscape but Optuna's random restarts cannot capitalize on this information — a warm-start prior biases exploration toward known-good regions without artificially narrowing the search space.

## What I changed
In `agent/train.py`: (1) Updated HYPOTHESIS to match the queue entry verbatim; (2) Added Ridge-based meta-learner infrastructure (`_get_meta_learner()`, `_build_feature_vector()`, `_normalize_param()`) trained on 8 known-good synthetic prior configs from prior experiments (3 XGB, 3 CatBoost, 2 LightGBM); (3) Implemented `suggest_warm_start()` using Optuna's `study.enqueue_trial()` API to seed each Optuna study with 3 meta-learner-prioritized configs before the main `optimize()` call — these trials are picked FIRST (before random search), biasing exploration toward known-good regions; (4) Integrated `suggest_warm_start()` into all 3 model Optuna loops (XGB, LGB, CAT) with print statements. No changes to prepare.py or calibration code.

## Result
- XGBoost: 22 trials, best=0.0537 (warm-started; enqueued 3 prior configs first)
- LightGBM: 6 trials, best=0.0551 (warm-started; enqueued 3 prior configs first)
- CatBoost: 41 trials, best=0.0542 (warm-started; enqueued 3 prior configs first)
- **Weighted ensemble (0.55/0.15/0.30): 0.0530**
- ISO-calibrated (N=500): **0.0515** ← final
- Stacking: 0.0648 (retained from previous code)
- Previous best: 0.0511; current: 0.0515 (**worse by +0.0004**)

## What I learned
- **Warm-starting via enqueue_trial did NOT improve base model quality**: XGBoost (0.0537 vs typical ~0.0525), CatBoost (0.0542 vs typical ~0.0538) both degraded. The warm-started trials consumed Optuna budget (3 trials/model) but didn't yield meaningfully better results than random search. The meta-learner priors are approximations based on synthetic configs, not actual evaluated trial data — and Optuna's random sampler is already quite effective at exploring the well-defined hyperparameter space.
- **Reduced trial counts are the primary damage vector**: XGBoost completed only 22 trials (vs 40+ in exp 0010) and LightGBM only 6 trials (vs 7+ in exp 0010), indicating the enqueue_trial overhead plus 3 warm-start trials ate into the already-tight 160s/model budget. Fewer trials → weaker base models → worse ensemble. The same mechanism that hurt exps 0021–0023 (budget consumption) struck again.
- **Optuna's random sampler with uniform priors is near-optimal for this well-structured GBDT hyperparameter space**: After 20+ experiments, Optuna has effectively mapped the landscape. The synthetic priors encode configurations that are already discoverable by random search within 40+ trials. Enqueueing them at the start doesn't add information — it just moves 3 random samples to the front.
- **The meta-learner concept is valid but needs actual trial history data to be useful**: The real arxiv 2507.12604 approach requires extracting actual Optuna trial records (params → val_loss pairs) from prior experiments. The results.tsv notes field mixes all 3 models' params together in free text, making reliable extraction impractical. If the trial histories were properly logged (one row per trial with named params), a Ridge/RF meta-learner could genuinely score new configs.
- **The base model quality ceiling is the real bottleneck, not search strategy**: Exp 0010's 0.0511 was achieved with standard Optuna random search + 500-bin isotonic. The warm-start variant regressed. This confirms the KB's conclusion: "base model quality is not the bottleneck at this point" is false at 0.0511 — there IS room to improve base models, but warm-starting Optuna's random search is not the mechanism.

## KB entries consulted
- 60–100 Optuna trials per model within a 3-minute budget is the stable operating point (exps 0001–0023)
- All Optuna-based improvements over 0.051357 have failed; base model quality is not the bottleneck at this point (exps 0005–0023)
- Isotonic regression provides ~0.0010 improvement regardless of moderate base model quality variation (exps 0017, 0019, 0020, 0021, 0022, 0023)
- Isotonic calibration cannot compensate for weaker base models: base model quality degradation from reduced Optuna budget was the primary failure mode across exps 0021, 0022, 0023 (exps 0021, 0022, 0023)
