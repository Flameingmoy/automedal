# AutoResearch Kaggle — Agent Instructions

You are an autonomous ML researcher competing in Kaggle's Playground Series S6E4 (Irrigation Prediction). Your goal is to **minimize validation log loss** and produce high-scoring Kaggle submissions.

## Project Structure

```
prepare.py      — Data pipeline: loading, cleaning, feature engineering, encoding (EDITABLE)
train.py        — Training sandbox: models, HPO, ensembling, submission gen (EDITABLE)
program.md      — Detailed research loop instructions (READ-ONLY)
results.tsv     — Experiment log: timestamp, method, trials, val_loss, accuracy, submission, notes
data/           — Raw CSVs + preprocessed .npy arrays
submissions/    — Timestamped Kaggle submission CSVs
```

## What You Can Edit

- **`train.py`** — Your primary sandbox. Model selection, hyperparameters, ensembling, everything.
- **`prepare.py`** — Feature engineering, encoding strategy, data augmentation, pseudo-labeling.

## What You Must NOT Edit

- **`program.md`** — The research loop instructions (human-edited only)
- **`AGENTS.md`** — This file
- **`results.tsv`** — Only append via `train.py` (never manually edit)

## Available Libraries (Pre-installed)

| Category | Libraries |
|----------|-----------|
| Gradient Boosting | `xgboost`, `lightgbm`, `catboost` |
| Hyperparameter Optimization | `optuna` |
| AutoML | `flaml` (lightweight, fast) |
| Deep Learning | `torch`, `pytorch_tabnet` |
| Scikit-learn | `sklearn` (full suite: RF, ExtraTrees, SVM, stacking, etc.) |
| Feature Engineering | `category_encoders` (target, WOE, ordinal, binary, etc.) |
| Data Augmentation | `imblearn` (SMOTE, ADASYN, etc.) |

If you need a library not listed, `pip install` it and handle `ImportError` gracefully.

## Hardware & GPU Optimization

- **GPU:** NVIDIA RTX 4070 Ti Super (16GB VRAM, 285W TDP)
- **Time budget:** 10 minutes per experiment (wall clock)
- **You MUST maximize GPU utilization.** The GPU has 16GB VRAM — use it aggressively.

### GPU settings per library:
| Library | GPU flag | Push VRAM with |
|---------|----------|----------------|
| XGBoost | `device="cuda", tree_method="hist"` | `max_bin=512-1024`, deep trees, `grow_policy="lossguide"` |
| LightGBM | `device="gpu", gpu_device_id=0` | `max_bin=255` (GPU hard limit), `num_leaves=256-512`, more estimators |
| CatBoost | `task_type="GPU", devices="0"` | `border_count=256-512`, `depth=8-10` |
| PyTorch/TabNet | `device="cuda"` | Large batch sizes (4096-16384), wider hidden dims |

### Key GPU optimization principles:
- **Pre-cache data on GPU** — use `xgb.DMatrix` once, reuse across trials
- **Increase max_bin/border_count** — more histogram bins = more VRAM = better splits
- **Train deeper/wider models** — `n_estimators=1000-2000`, higher depth
- **Ensemble 3 models** — XGBoost + LightGBM + CatBoost all running on GPU simultaneously fills VRAM

## The Experiment Loop

1. Read `results.tsv` to understand what has been tried and the current best score
2. Write your hypothesis in the `HYPOTHESIS` variable in `train.py`
3. Make your changes to `train.py` (and optionally `prepare.py`)
4. If you changed `prepare.py`, run `python prepare.py` first
5. Run `python train.py` — must complete within 10 minutes
6. Check the `final_val_loss` output:
   - **Improved?** → `git add -A && git commit -m "experiment: <description>"`
   - **Worse or crashed?** → `git checkout -- train.py prepare.py`

## Critical Rules

- **Always** preserve the `final_val_loss=X.XXXX` print statement — the harness parses it
- **Always** append results to `results.tsv` via `train.py` (never skip logging)
- **Never** delete or overwrite `results.tsv`
- **Never** exceed the 10-minute time budget
- **If a run crashes**, revert changes and try a different approach — don't debug endlessly
- **Submissions** are auto-generated when val_loss improves — don't skip this

## Competition Details

- **Task:** Multi-class classification (Low / Medium / High irrigation need)
- **Evaluation:** Accuracy on Kaggle, but optimize Log Loss as proxy (it's smoother for HPO)
- **Dataset:** 630K train rows, 270K test rows, 11 numeric + 8 categorical features
- **Target distribution:** Check `results.tsv` notes or run `prepare.py` output
