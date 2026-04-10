# AutoResearch for Kaggle

Autonomous ML research for Kaggle competitions, powered by [OpenCode](https://opencode.ai) + GLM-5.1. Adapted from [Karpathy's autoresearch](https://github.com/karpathy/autoresearch) — same philosophy (edit, train, check, keep/revert), but targeting tabular ML competitions instead of LLM pretraining.

Give an AI agent a Kaggle dataset and a flexible training pipeline. It experiments autonomously — trying different models, feature engineering, hyperparameters, and ensembles — keeping only what improves the score. You wake up to a log of experiments and (hopefully) a leaderboard-climbing submission.

## Current Competition

**Playground Series S6E4 — Irrigation Prediction**
- Task: Multi-class classification (Low / Medium / High)
- Dataset: 630K train rows, 270K test rows, 11 numeric + 8 categorical features
- Baseline: **98.62% accuracy** / 0.0532 log loss (XGBoost+LightGBM+CatBoost ensemble)

## How It Works

The agent follows the loop defined in `program.md`:

```
Read results.tsv → Form hypothesis → Edit train.py/prepare.py → Run training
→ Score improved? → git commit : git revert → Repeat
```

Each experiment runs within a **10-minute wall clock budget** on an **RTX 4070 Ti Super** (16GB VRAM). All three gradient boosting libraries run GPU-accelerated.

## Project Structure

```
prepare.py        — Data pipeline: loading, cleaning, feature engineering, encoding (agent-editable)
train.py          — Training sandbox: models, HPO, ensembling, submissions (agent-editable)
program.md        — Research loop instructions for the agent (human-editable)
AGENTS.md         — OpenCode project context: rules, libraries, hardware (human-editable)
run.sh            — Headless automation loop (invokes opencode repeatedly)
analysis.ipynb    — Experiment tracking and visualization
results.tsv       — Experiment log (appended by train.py)
pyproject.toml    — Dependencies (managed by uv)
data/             — Raw CSVs + preprocessed .npy arrays
submissions/      — Timestamped Kaggle submission CSVs
```

## Quick Start

**Requirements:** NVIDIA GPU, Python 3.10+, [uv](https://docs.astral.sh/uv/), [OpenCode](https://opencode.ai)

```bash
# 1. Install dependencies
uv sync

# 2. Prepare data (one-time — encodes features, splits train/val, saves .npy)
uv run python prepare.py

# 3. Run a single training experiment (~10 min)
uv run python train.py
```

## Running the Agent

### Interactive (single session)

```bash
opencode -m opencode-go/glm-5.1
```

Then prompt:

```
Read program.md and execute one full experiment cycle.
```

### Automated (overnight)

```bash
# Run 50 experiment iterations headlessly
bash run.sh 50

# Or fewer for a quick test
bash run.sh 5
```

Each iteration invokes `opencode run -m opencode-go/glm-5.1 --dangerously-skip-permissions`, which reads `program.md`, decides what to try, edits files, trains, and commits or reverts.

## Available Libraries

The agent can freely use any of these (all pre-installed):

| Category | Libraries |
|----------|-----------|
| Gradient Boosting | XGBoost, LightGBM, CatBoost (all GPU-accelerated) |
| Hyperparameter Optimization | Optuna |
| AutoML | FLAML |
| Deep Learning | PyTorch, TabNet |
| Feature Engineering | category_encoders, scikit-learn |
| Data Augmentation | imbalanced-learn (SMOTE, ADASYN) |

AutoGluon is available as an optional install: `uv sync --extra automl`

## Design Choices

- **Two files to modify.** The agent edits both `train.py` (models, HPO, ensembling) and `prepare.py` (features, encoding, augmentation). This gives it full control over the ML pipeline while keeping scope manageable.
- **Fixed time budget.** Each experiment runs for at most 10 minutes, making results directly comparable regardless of what the agent changes.
- **GPU-first.** XGBoost uses `device="cuda"` with high `max_bin`, LightGBM uses `device="gpu"`, CatBoost uses `task_type="GPU"`. The RTX 4070 Ti Super's 16GB VRAM is utilized aggressively.
- **Ensemble by default.** The baseline trains XGBoost + LightGBM + CatBoost in parallel, then finds optimal blending weights via grid search.
- **Automatic submissions.** Every time `val_loss` improves, a Kaggle-ready submission CSV is generated in `submissions/`.

## Adapting to a New Competition

1. Replace `data/train.csv` and `data/test.csv` with the new competition data
2. Update `prepare.py`: change `TARGET_COL`, `NUMERIC_FEATURES`, `CATEGORICAL_FEATURES`
3. Update `train.py`: adjust `num_classes`, metric, and submission format
4. Update `program.md` and `AGENTS.md` with competition context
5. Run `uv run python prepare.py` to regenerate `.npy` files
6. Clear `results.tsv` and start fresh

## Acknowledgements

Based on [autoresearch](https://github.com/karpathy/autoresearch) by Andrej Karpathy. The core idea — programming research via Markdown instructions for AI agents — is his.

## License

MIT
