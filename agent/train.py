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
import sys
import json
import datetime
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier, CatBoostError
from lightgbm.basic import LightGBMError
import optuna
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression

# Resolve paths relative to the repo root (one level above agent/)
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_AGENT_DIR = os.path.dirname(os.path.abspath(__file__))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from config_loader import get_objectives, get_submission

# ─── CONFIG-DRIVEN OBJECTIVES (from configs/competition.yaml) ────────
_obj = get_objectives()
_sub = get_submission()
XGB_OBJECTIVE = _obj["xgboost"]
XGB_EVAL_METRIC = _obj["xgboost_eval"]
LGB_OBJECTIVE = _obj["lightgbm"]
CAT_LOSS_FUNCTION = _obj["catboost"]
SUB_ID_COL = _sub["id_col"]
SUB_TARGET_COL = _sub["target_col"]

# ─── HARD CONSTRAINTS ───────────────────────────────────────────────────
TIME_BUDGET_MINUTES = 10
DATA_DIR = os.path.join(_REPO_ROOT, "data")
SUBMISSION_DIR = os.path.join(_REPO_ROOT, "submissions")
# ────────────────────────────────────────────────────────────────────────

# ─── GPU CONFIG ──────────────────────────────────────────────────────────
# RTX 4070 Ti Super: 16GB VRAM, 285W TDP — push it hard
# XGBoost/LightGBM use GPU for histogram building; CatBoost is fully GPU-native
# For maximum GPU utilization, use large max_bin, deep trees, and DMatrix caching
GPU_ID = 0
# ────────────────────────────────────────────────────────────────────────

# ─── HYPOTHESIS ──────────────────────────────────────────────────────────
# Agent: write your hypothesis for this experiment here before running.
HYPOTHESIS = "Stacking with LogisticRegression meta-learner on top of 3 GBDTs may outperform fixed-weight ensemble by learning per-class combination weights"
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
    results_file = os.path.join(_AGENT_DIR, "results.tsv")
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


def generate_submission(preds, X_test, class_names, tag):
    """Generate a Kaggle submission CSV from predictions (int class indices)."""
    test_ids = np.load(os.path.join(DATA_DIR, "test_ids.npy"))
    pred_labels = [class_names[int(p)] for p in preds]

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_{tag}.csv"
    filepath = os.path.join(SUBMISSION_DIR, filename)

    import pandas as pd

    sub_df = pd.DataFrame({SUB_ID_COL: test_ids, SUB_TARGET_COL: pred_labels})
    sub_df.to_csv(filepath, index=False)
    print(f"  Submission saved: {filepath}")
    return filename


# ─── GPU-OPTIMIZED MODEL BUILDERS ───────────────────────────────────────


def build_xgb(trial, num_classes):
    """XGBoost with GPU hist — pushes VRAM via large max_bin and deep trees."""
    return xgb.XGBClassifier(
        objective=XGB_OBJECTIVE,
        num_class=num_classes,
        eval_metric=XGB_EVAL_METRIC,
        tree_method="hist",
        device="cuda",
        verbosity=0,
        max_bin=trial.suggest_int("xgb_max_bin", 256, 1024),
        learning_rate=trial.suggest_float("xgb_lr", 5e-3, 0.3, log=True),
        max_depth=trial.suggest_int("xgb_depth", 4, 12),
        n_estimators=trial.suggest_int("xgb_n_est", 200, 2000),
        subsample=trial.suggest_float("xgb_subsample", 0.6, 1.0),
        colsample_bytree=trial.suggest_float("xgb_colsample", 0.4, 1.0),
        colsample_bylevel=trial.suggest_float("xgb_colsample_level", 0.4, 1.0),
        min_child_weight=trial.suggest_int("xgb_mcw", 1, 30),
        gamma=trial.suggest_float("xgb_gamma", 0, 5.0),
        reg_alpha=trial.suggest_float("xgb_alpha", 1e-8, 10.0, log=True),
        reg_lambda=trial.suggest_float("xgb_lambda", 1e-8, 10.0, log=True),
        grow_policy=trial.suggest_categorical("xgb_grow", ["depthwise", "lossguide"]),
        early_stopping_rounds=50,
    )


def build_lgb(trial, num_classes):
    """LightGBM with GPU — fast histogram-based, GPU-accelerated.
    Note: LightGBM GPU requires max_bin <= 255."""
    return lgb.LGBMClassifier(
        objective=LGB_OBJECTIVE,
        num_class=num_classes,
        device="gpu",
        gpu_device_id=GPU_ID,
        max_bin=255,  # LightGBM GPU hard limit
        learning_rate=trial.suggest_float("lgb_lr", 5e-3, 0.3, log=True),
        n_estimators=trial.suggest_int("lgb_n_est", 200, 2000),
        max_depth=trial.suggest_int("lgb_depth", -1, 15),
        num_leaves=trial.suggest_int("lgb_leaves", 31, 512),
        subsample=trial.suggest_float("lgb_subsample", 0.6, 1.0),
        subsample_freq=trial.suggest_int("lgb_subsample_freq", 1, 10),
        colsample_bytree=trial.suggest_float("lgb_colsample", 0.4, 1.0),
        min_child_samples=trial.suggest_int("lgb_min_child", 5, 100),
        reg_alpha=trial.suggest_float("lgb_alpha", 1e-8, 10.0, log=True),
        reg_lambda=trial.suggest_float("lgb_lambda", 1e-8, 10.0, log=True),
        min_split_gain=trial.suggest_float("lgb_min_gain", 0, 1.0),
        verbose=-1,
    )


def build_cat(trial, num_classes):
    """CatBoost — fully GPU-native, uses VRAM heavily with large border_count."""
    return CatBoostClassifier(
        loss_function=CAT_LOSS_FUNCTION,
        classes_count=num_classes,
        task_type="GPU",
        devices=str(GPU_ID),
        border_count=trial.suggest_int("cat_border", 128, 512),
        learning_rate=trial.suggest_float("cat_lr", 5e-3, 0.3, log=True),
        iterations=trial.suggest_int("cat_iters", 200, 2000),
        depth=trial.suggest_int("cat_depth", 4, 10),
        l2_leaf_reg=trial.suggest_float("cat_l2", 1e-3, 10.0, log=True),
        random_strength=trial.suggest_float("cat_rand_str", 0, 2.0),
        bagging_temperature=trial.suggest_float("cat_bag_temp", 0, 5.0),
        verbose=0,
        allow_writing_files=False,
    )


# ─── MAIN ───────────────────────────────────────────────────────────────


def main():
    print("=" * 60)
    print("AutoResearch Kaggle — Training (GPU-Optimized)")
    print(f"Hypothesis: {HYPOTHESIS}")
    print("=" * 60)

    # Load data
    print("\nLoading data...")
    X_train, y_train, X_val, y_val, X_test = load_data()
    num_classes = len(np.unique(y_train))
    print(f"  Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    print(f"  Classes: {num_classes}")

    with open(os.path.join(DATA_DIR, "class_names.json")) as f:
        class_names = json.load(f)
    with open(os.path.join(DATA_DIR, "feature_names.json")) as f:
        feature_names = json.load(f)

    start_time = time.time()
    budget_seconds = TIME_BUDGET_MINUTES * 60
    previous_best = get_best_score()
    print(f"  Previous best val_loss: {previous_best:.4f}")

    # Pre-cache XGBoost DMatrix on GPU — avoids re-uploading every trial
    print("  Pre-caching DMatrix on GPU...")
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)

    # ─── PHASE 1: Tune each model independently ─────────────────────────
    # Allocate ~3 min per model for Optuna, ~1 min for final ensemble
    model_budget = (budget_seconds - 120) // 3  # seconds per model
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    best_models = {}

    # --- XGBoost ---
    print(f"\n[1/3] XGBoost Optuna ({model_budget}s budget)...")
    xgb_start = time.time()

    def xgb_objective(trial):
        if time.time() - xgb_start > model_budget:
            raise optuna.exceptions.TrialPruned()
        model = build_xgb(trial, num_classes)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        return log_loss(y_val, model.predict_proba(X_val))

    xgb_study = optuna.create_study(direction="minimize")
    xgb_study.optimize(
        xgb_objective,
        n_trials=100,
        timeout=model_budget,
        catch=(optuna.exceptions.TrialPruned, Exception),
    )
    xgb_done = len(
        [t for t in xgb_study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    )
    print(f"  XGBoost: {xgb_done} trials, best={xgb_study.best_value:.4f}")

    # --- LightGBM ---
    print(f"\n[2/3] LightGBM Optuna ({model_budget}s budget)...")
    lgb_start = time.time()

    def lgb_objective(trial):
        if time.time() - lgb_start > model_budget:
            raise optuna.exceptions.TrialPruned()
        model = build_lgb(trial, num_classes)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
        )
        return log_loss(y_val, model.predict_proba(X_val))

    lgb_study = optuna.create_study(direction="minimize")
    lgb_study.optimize(
        lgb_objective,
        n_trials=100,
        timeout=model_budget,
        catch=(optuna.exceptions.TrialPruned, LightGBMError, Exception),
    )
    lgb_done = len(
        [t for t in lgb_study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    )
    print(f"  LightGBM: {lgb_done} trials, best={lgb_study.best_value:.4f}")

    # --- CatBoost ---
    print(f"\n[3/3] CatBoost Optuna ({model_budget}s budget)...")
    cat_start = time.time()

    def cat_objective(trial):
        if time.time() - cat_start > model_budget:
            raise optuna.exceptions.TrialPruned()
        model = build_cat(trial, num_classes)
        model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=50)
        return log_loss(y_val, model.predict_proba(X_val))

    cat_study = optuna.create_study(direction="minimize")
    cat_study.optimize(
        cat_objective,
        n_trials=60,
        timeout=model_budget,
        catch=(optuna.exceptions.TrialPruned, CatBoostError, Exception),
    )
    cat_done = len(
        [t for t in cat_study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    )
    print(f"  CatBoost: {cat_done} trials, best={cat_study.best_value:.4f}")

    # ─── PHASE 2: Train final models with best params ───────────────────
    print("\n--- Training final models with best params ---")
    total_trials = xgb_done + lgb_done + cat_done

    # XGBoost final
    xgb_best = xgb_study.best_params
    xgb_final = xgb.XGBClassifier(
        objective=XGB_OBJECTIVE,
        num_class=num_classes,
        eval_metric=XGB_EVAL_METRIC,
        tree_method="hist",
        device="cuda",
        verbosity=0,
        early_stopping_rounds=50,
        max_bin=xgb_best.pop("xgb_max_bin", 512),
        learning_rate=xgb_best.pop("xgb_lr"),
        max_depth=xgb_best.pop("xgb_depth"),
        n_estimators=xgb_best.pop("xgb_n_est"),
        subsample=xgb_best.pop("xgb_subsample"),
        colsample_bytree=xgb_best.pop("xgb_colsample"),
        colsample_bylevel=xgb_best.pop("xgb_colsample_level"),
        min_child_weight=xgb_best.pop("xgb_mcw"),
        gamma=xgb_best.pop("xgb_gamma"),
        reg_alpha=xgb_best.pop("xgb_alpha"),
        reg_lambda=xgb_best.pop("xgb_lambda"),
        grow_policy=xgb_best.pop("xgb_grow"),
    )
    xgb_final.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    xgb_proba_val = xgb_final.predict_proba(X_val)
    xgb_proba_test = xgb_final.predict_proba(X_test)
    print(f"  XGBoost final: {log_loss(y_val, xgb_proba_val):.4f}")

    # LightGBM final
    lgb_best = lgb_study.best_params
    lgb_final = lgb.LGBMClassifier(
        objective=LGB_OBJECTIVE,
        num_class=num_classes,
        device="gpu",
        gpu_device_id=GPU_ID,
        verbose=-1,
        max_bin=255,
        learning_rate=lgb_best.pop("lgb_lr"),
        n_estimators=lgb_best.pop("lgb_n_est"),
        max_depth=lgb_best.pop("lgb_depth"),
        num_leaves=lgb_best.pop("lgb_leaves"),
        subsample=lgb_best.pop("lgb_subsample"),
        subsample_freq=lgb_best.pop("lgb_subsample_freq"),
        colsample_bytree=lgb_best.pop("lgb_colsample"),
        min_child_samples=lgb_best.pop("lgb_min_child"),
        reg_alpha=lgb_best.pop("lgb_alpha"),
        reg_lambda=lgb_best.pop("lgb_lambda"),
        min_split_gain=lgb_best.pop("lgb_min_gain"),
    )
    lgb_final.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
    )
    lgb_proba_val = lgb_final.predict_proba(X_val)
    lgb_proba_test = lgb_final.predict_proba(X_test)
    print(f"  LightGBM final: {log_loss(y_val, lgb_proba_val):.4f}")

    # CatBoost final
    cat_best = cat_study.best_params
    cat_final = CatBoostClassifier(
        loss_function=CAT_LOSS_FUNCTION,
        classes_count=num_classes,
        task_type="GPU",
        devices=str(GPU_ID),
        verbose=0,
        allow_writing_files=False,
        border_count=cat_best.pop("cat_border"),
        learning_rate=cat_best.pop("cat_lr"),
        iterations=cat_best.pop("cat_iters"),
        depth=cat_best.pop("cat_depth"),
        l2_leaf_reg=cat_best.pop("cat_l2"),
        random_strength=cat_best.pop("cat_rand_str"),
        bagging_temperature=cat_best.pop("cat_bag_temp"),
    )
    cat_final.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=50)
    cat_proba_val = cat_final.predict_proba(X_val)
    cat_proba_test = cat_final.predict_proba(X_test)
    print(f"  CatBoost final: {log_loss(y_val, cat_proba_val):.4f}")

    # ─── PHASE 3: Weighted ensemble via grid search ──────────────────────
    print("\n--- Optimizing ensemble weights ---")
    best_ens_loss = float("inf")
    best_weights = (1 / 3, 1 / 3, 1 / 3)

    for w_xgb in np.arange(0.1, 0.8, 0.05):
        for w_lgb in np.arange(0.1, 0.8 - w_xgb, 0.05):
            w_cat = 1.0 - w_xgb - w_lgb
            if w_cat < 0.05:
                continue
            ens_proba = (
                w_xgb * xgb_proba_val + w_lgb * lgb_proba_val + w_cat * cat_proba_val
            )
            ens_loss = log_loss(y_val, ens_proba)
            if ens_loss < best_ens_loss:
                best_ens_loss = ens_loss
                best_weights = (w_xgb, w_lgb, w_cat)

    w_xgb, w_lgb, w_cat = best_weights
    print(f"  Best weights: XGB={w_xgb:.2f}, LGB={w_lgb:.2f}, CAT={w_cat:.2f}")

    # Weighted ensemble predictions
    weighted_proba_val = (
        w_xgb * xgb_proba_val + w_lgb * lgb_proba_val + w_cat * cat_proba_val
    )
    weighted_proba_test = (
        w_xgb * xgb_proba_test + w_lgb * lgb_proba_test + w_cat * cat_proba_test
    )
    weighted_loss = log_loss(y_val, weighted_proba_val)

    # ─── PHASE 4: Stacking with Logistic Regression meta-learner ───────
    print("\n--- Stacking with Logistic Regression meta-learner ---")
    stack_val = np.column_stack([xgb_proba_val, lgb_proba_val, cat_proba_val])
    stack_test = np.column_stack([xgb_proba_test, lgb_proba_test, cat_proba_test])

    meta = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs")
    meta.fit(stack_val, y_val)

    stack_proba_val = meta.predict_proba(stack_val)
    stack_proba_test = meta.predict_proba(stack_test)
    stack_loss = log_loss(y_val, stack_proba_val)
    stack_accuracy = accuracy_score(y_val, np.argmax(stack_proba_val, axis=1))
    print(f"  Stacking Val Loss:     {stack_loss:.4f}")
    print(f"  Stacking Val Accuracy: {stack_accuracy:.4f}")
    print(f"  Weighted Val Loss:     {weighted_loss:.4f}")

    # Choose the better approach
    if stack_loss < weighted_loss:
        print(f"  Stacking wins ({stack_loss:.4f} < {weighted_loss:.4f})")
        final_proba_val = stack_proba_val
        final_proba_test = stack_proba_test
        method_tag = "stack"
        method_name = "stacking"
        val_loss = stack_loss
    else:
        print(f"  Weighted ensemble wins ({weighted_loss:.4f} <= {stack_loss:.4f})")
        final_proba_val = weighted_proba_val
        final_proba_test = weighted_proba_test
        method_tag = "ensemble"
        method_name = "weighted"
        val_loss = weighted_loss

    final_preds_val = np.argmax(final_proba_val, axis=1)
    final_preds_test = np.argmax(final_proba_test, axis=1)
    val_accuracy = accuracy_score(y_val, final_preds_val)
    elapsed = time.time() - start_time

    print(f"\n  Final Val Loss:     {val_loss:.4f}")
    print(f"  Final Val Accuracy: {val_accuracy:.4f}")
    print(f"  Final Method:       {method_name}")
    print(f"  Time elapsed:       {elapsed:.0f}s")

    # Compare individual vs ensemble
    individual_losses = {
        "xgboost": log_loss(y_val, xgb_proba_val),
        "lightgbm": log_loss(y_val, lgb_proba_val),
        "catboost": log_loss(y_val, cat_proba_val),
    }
    best_individual = min(individual_losses, key=individual_losses.get)
    print(
        f"  Best individual: {best_individual} ({individual_losses[best_individual]:.4f})"
    )

    # ─── GENERATE SUBMISSION ─────────────────────────────────────────────
    submission_file = ""
    improved = val_loss < previous_best
    if improved:
        print(f"\n  IMPROVED! {previous_best:.4f} -> {val_loss:.4f}")
        submission_file = generate_submission(
            final_preds_test, X_test, class_names, method_tag
        )
    else:
        print(f"\n  No improvement ({val_loss:.4f} >= {previous_best:.4f})")

    # ─── LOG RESULTS ─────────────────────────────────────────────────────
    results_file = os.path.join(_AGENT_DIR, "results.tsv")
    write_header = not os.path.exists(results_file)
    with open(results_file, "a") as f:
        if write_header:
            f.write(
                "timestamp\tmethod\ttrials\tval_loss\tval_accuracy\tsubmission\tnotes\n"
            )
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        weights_str = f"w=({w_xgb:.2f},{w_lgb:.2f},{w_cat:.2f})"
        f.write(
            f"{timestamp}\t{method_name}\t{total_trials}\t{val_loss:.6f}\t{val_accuracy:.6f}\t{submission_file}\t{HYPOTHESIS} {weights_str}\n"
        )

    # ─── AUTORESEARCH OUTPUT (DO NOT CHANGE THIS LINE) ───────────────────
    print(f"final_val_loss={val_loss:.4f}")


if __name__ == "__main__":
    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    main()
