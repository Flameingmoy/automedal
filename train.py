"""
AutoResearch Kaggle — Training Sandbox
=======================================
This is the agent's primary sandbox. Everything here is fair game:
- Model selection (XGBoost, LightGBM, CatBoost, PyTorch, AutoML, ensembles)
- Hyperparameter search (Optuna)
- Cross-validation strategy
- Feature selection
- Ensembling / stacking / blending

Competition: Playground Series S6E4 — Irrigation Prediction
Target: Irrigation_Need (Low / Medium / High) — multi-class classification
Metric: Accuracy (Kaggle leaderboard) / Log Loss (optimization proxy)
Hardware: RTX 4070 Ti Super (16GB VRAM)

RULES:
  - Must complete within TIME_BUDGET_MINUTES wall clock
  - Must print 'final_val_loss=X.XXXX' at the end (the harness parses this)
  - Must generate a submission CSV if val_loss improves
  - Append results to results.tsv
"""

import time
import os
import json
import datetime
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
import optuna
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import StratifiedKFold

# ─── HARD CONSTRAINTS ───────────────────────────────────────────────────
TIME_BUDGET_MINUTES = 10
DATA_DIR = "data"
SUBMISSION_DIR = "submissions"
# ────────────────────────────────────────────────────────────────────────

# ─── GPU CONFIG ──────────────────────────────────────────────────────────
# RTX 4070 Ti Super: 16GB VRAM, 285W TDP — push it hard
# XGBoost/LightGBM use GPU for histogram building; CatBoost is fully GPU-native
# For maximum GPU utilization, use large max_bin, deep trees, and DMatrix caching
GPU_ID = 0
# ────────────────────────────────────────────────────────────────────────

# ─── HYPOTHESIS ──────────────────────────────────────────────────────────
# Agent: write your hypothesis for this experiment here before running.
HYPOTHESIS = "GPU-optimized 3-model ensemble (XGBoost+LightGBM+CatBoost) with aggressive GPU params"
# ────────────────────────────────────────────────────────────────────────


def load_data():
    """Load preprocessed numpy arrays from prepare.py."""
    X_train = np.load(os.path.join(DATA_DIR, "X_train.npy"))
    y_train = np.load(os.path.join(DATA_DIR, "y_train.npy"))
    X_val = np.load(os.path.join(DATA_DIR, "X_val.npy"))
    y_val = np.load(os.path.join(DATA_DIR, "y_val.npy"))
    X_test = np.load(os.path.join(DATA_DIR, "X_test.npy"))
    return X_train, y_train, X_val, y_val, X_test


def get_best_score():
    """Read the current best val_loss from results.tsv (if it exists)."""
    results_file = "results.tsv"
    if not os.path.exists(results_file):
        return float("inf")
    try:
        with open(results_file) as f:
            lines = f.readlines()
        if len(lines) <= 1:  # header only
            return float("inf")
        losses = []
        for line in lines[1:]:
            parts = line.strip().split("\t")
            if len(parts) >= 4:
                losses.append(float(parts[3]))
        return min(losses) if losses else float("inf")
    except Exception:
        return float("inf")


def generate_submission(model, X_test, class_names, tag):
    """Generate a Kaggle submission CSV from the trained model."""
    test_ids = np.load(os.path.join(DATA_DIR, "test_ids.npy"))
    preds = model.predict(X_test)

    # Map numeric predictions back to class names
    pred_labels = [class_names[int(p)] for p in preds]

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_{tag}.csv"
    filepath = os.path.join(SUBMISSION_DIR, filename)

    import pandas as pd
    sub_df = pd.DataFrame({"id": test_ids, "Irrigation_Need": pred_labels})
    sub_df.to_csv(filepath, index=False)
    print(f"  Submission saved: {filepath}")
    return filename


def main():
    print("=" * 60)
    print("AutoResearch Kaggle — Training")
    print(f"Hypothesis: {HYPOTHESIS}")
    print("=" * 60)

    # Load data
    print("\nLoading data...")
    X_train, y_train, X_val, y_val, X_test = load_data()
    num_classes = len(np.unique(y_train))
    print(f"  Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    print(f"  Classes: {num_classes}")

    # Load metadata
    with open(os.path.join(DATA_DIR, "class_names.json")) as f:
        class_names = json.load(f)
    with open(os.path.join(DATA_DIR, "feature_names.json")) as f:
        feature_names = json.load(f)

    start_time = time.time()
    budget_seconds = TIME_BUDGET_MINUTES * 60
    previous_best = get_best_score()
    print(f"  Previous best val_loss: {previous_best:.4f}")

    # ─── OPTUNA OBJECTIVE ────────────────────────────────────────────────
    # Agent: modify this function freely. Swap XGBoost for LightGBM, CatBoost,
    # PyTorch TabNet, a sklearn ensemble, or anything else.
    def objective(trial):
        elapsed = time.time() - start_time
        if elapsed > budget_seconds - 90:  # leave 90s for final train + submission
            raise optuna.exceptions.TrialPruned("Time budget nearing end.")

        params = {
            "objective": "multi:softprob",
            "num_class": num_classes,
            "eval_metric": "mlogloss",
            "tree_method": "hist",
            "device": "cuda",
            "verbosity": 0,
            # Core hyperparameters
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
            # Regularization
            "gamma": trial.suggest_float("gamma", 0, 5.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        }

        model = xgb.XGBClassifier(**params, early_stopping_rounds=50)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        preds_proba = model.predict_proba(X_val)
        return log_loss(y_val, preds_proba)

    # ─── RUN OPTUNA SEARCH ───────────────────────────────────────────────
    print("\nStarting Optuna study...")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize")
    study.optimize(
        objective,
        n_trials=200,
        timeout=budget_seconds - 90,
        catch=(optuna.exceptions.TrialPruned,),
    )

    completed = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    print(f"\n  Completed {completed} trials (best loss: {study.best_value:.4f})")

    # ─── TRAIN FINAL MODEL ───────────────────────────────────────────────
    print("\nTraining final model with best params...")
    best_params = study.best_params
    best_params.update({
        "objective": "multi:softprob",
        "num_class": num_classes,
        "eval_metric": "mlogloss",
        "tree_method": "hist",
        "device": "cuda",
        "verbosity": 0,
    })

    final_model = xgb.XGBClassifier(**best_params, early_stopping_rounds=50)
    final_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    # ─── EVALUATE ────────────────────────────────────────────────────────
    preds_proba = final_model.predict_proba(X_val)
    preds = final_model.predict(X_val)
    val_loss = log_loss(y_val, preds_proba)
    val_accuracy = accuracy_score(y_val, preds)
    elapsed = time.time() - start_time

    print(f"\n  Validation Loss:     {val_loss:.4f}")
    print(f"  Validation Accuracy: {val_accuracy:.4f}")
    print(f"  Time elapsed:        {elapsed:.0f}s")

    # ─── GENERATE SUBMISSION ─────────────────────────────────────────────
    submission_file = ""
    improved = val_loss < previous_best
    if improved:
        print(f"\n  IMPROVED! {previous_best:.4f} -> {val_loss:.4f}")
        submission_file = generate_submission(final_model, X_test, class_names, "xgboost")
    else:
        print(f"\n  No improvement ({val_loss:.4f} >= {previous_best:.4f})")

    # ─── LOG RESULTS ─────────────────────────────────────────────────────
    results_file = "results.tsv"
    write_header = not os.path.exists(results_file)
    with open(results_file, "a") as f:
        if write_header:
            f.write("timestamp\tmethod\ttrials\tval_loss\tval_accuracy\tsubmission\tnotes\n")
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"{timestamp}\txgboost\t{completed}\t{val_loss:.6f}\t{val_accuracy:.6f}\t{submission_file}\t{HYPOTHESIS}\n")

    # ─── AUTORESEARCH OUTPUT (DO NOT CHANGE THIS LINE) ───────────────────
    print(f"final_val_loss={val_loss:.4f}")


if __name__ == "__main__":
    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    main()
