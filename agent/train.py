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
from sklearn.isotonic import IsotonicRegression

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
TIME_BUDGET_MINUTES = int(os.environ.get("TRAIN_BUDGET_MINUTES", "10"))
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
HYPOTHESIS = "Training a fixed-config Quantile XGBoost (QXGBoost: Huber-quantile objective, 10th/50th/90th percentiles) alongside the 3 standard GBDTs and using prediction-interval width as inverse-weight for isotonic calibration will improve log_loss, because QXGBoost's per-sample uncertainty signal is absent from standard point predictions — wide intervals in sparse tabular regions indicate where the ensemble is unreliable — and weighting isotonic calibration toward high-confidence (narrow-interval) samples addresses miscalibration in exactly those regions where the JSD experiment (exp 0013) hypothesized but could not measure miscalibration concentration."
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


# ─── BIN-CONSTRAINED ISOTONIC CALIBRATION ───────────────────────────────
# Custom isotonic regression with explicit bin control to prevent overfitting
# on noisy validation samples (~1.4% estimated noise from 98.6% accuracy).

def bin_constrained_isotonic(proba_val, y_val, proba_test, n_bins):
    """
    Apply bin-constrained isotonic calibration per class.
    
    For each class c:
    1. Sort val samples by predicted probability of class c.
    2. Partition into n_bins equal-frequency bins.
    3. Assign each bin the mean of true labels (binary: 1 if class c, 0 otherwise).
    4. Apply piecewise-constant mapping to test predictions.
    
    This regularizes the piecewise-constant mapping against fitting noise,
    unlike sklearn's IsotonicRegression which can create an arbitrary number
    of bins equal to unique input values.
    """
    num_classes = proba_val.shape[1]
    calibrated_val = np.zeros_like(proba_val)
    calibrated_test = np.zeros_like(proba_test)
    
    for c in range(num_classes):
        p_val = proba_val[:, c]
        y_binary = (y_val == c).astype(float)
        p_test = proba_test[:, c]
        
        # Sort by predicted probability
        sort_idx = np.argsort(p_val)
        sorted_p = p_val[sort_idx]
        sorted_y = y_binary[sort_idx]
        n = len(p_val)
        
        if n_bins == 'default':
            # Use sklearn default (unconstrained)
            iso = IsotonicRegression(out_of_bounds='clip')
            iso.fit(p_val, y_binary)
            calibrated_val[:, c] = iso.predict(p_val)
            calibrated_test[:, c] = iso.predict(p_test)
        else:
            # Custom bin-constrained: equal-frequency bins
            bin_size = n // n_bins
            bin_means = []
            bin_edges = []  # right edge of each bin
            
            for b in range(n_bins):
                start = b * bin_size
                if b == n_bins - 1:
                    end = n  # last bin takes remainder
                else:
                    end = (b + 1) * bin_size
                bin_y = sorted_y[start:end]
                bin_p = sorted_p[start:end]
                mean_y = np.mean(bin_y)
                mean_p = np.mean(bin_p) if len(bin_p) > 0 else bin_p[0] if len(bin_p) > 0 else 0.5
                bin_means.append(mean_y)
                bin_edges.append(bin_p[-1] if len(bin_p) > 0 else mean_p)
            
            # Apply mapping to validation: for each sample, find its bin and assign bin mean
            # Use digitize to find which bin each probability falls into
            bin_edges_arr = np.array(bin_edges)
            bin_indices = np.digitize(p_val, bin_edges_arr, right=False)
            bin_indices = np.clip(bin_indices, 0, n_bins - 1)
            calibrated_val[:, c] = np.array(bin_means)[bin_indices]
            
            # Apply mapping to test: same approach
            bin_indices_test = np.digitize(p_test, bin_edges_arr, right=False)
            bin_indices_test = np.clip(bin_indices_test, 0, n_bins - 1)
            calibrated_test[:, c] = np.array(bin_means)[bin_indices_test]
    
    return calibrated_val, calibrated_test


def weighted_bin_constrained_isotonic(proba_val, y_val, proba_test, weights, n_bins):
    """
    Apply weighted bin-constrained isotonic calibration per class.
    
    For each class c:
    1. Sort val samples by predicted probability of class c.
    2. Partition into n_bins equal-frequency bins.
    3. Assign each bin the weighted mean of true labels (binary: 1 if class c, 0 otherwise).
    4. Apply piecewise-constant mapping to test predictions.
    
    Sample weights are used to compute weighted bin means.
    """
    num_classes = proba_val.shape[1]
    calibrated_val = np.zeros_like(proba_val)
    calibrated_test = np.zeros_like(proba_test)
    
    for c in range(num_classes):
        p_val = proba_val[:, c]
        y_binary = (y_val == c).astype(float)
        p_test = proba_test[:, c]
        
        # Sort by predicted probability
        sort_idx = np.argsort(p_val)
        sorted_p = p_val[sort_idx]
        sorted_y = y_binary[sort_idx]
        sorted_weights = weights[sort_idx]
        n = len(p_val)
        
        if n_bins == 'default':
            # Use sklearn with sample_weight
            iso = IsotonicRegression(out_of_bounds='clip')
            iso.fit(p_val, y_binary, sample_weight=weights)
            calibrated_val[:, c] = iso.predict(p_val)
            calibrated_test[:, c] = iso.predict(p_test)
        else:
            # Weighted equal-frequency bins
            bin_size = n // n_bins
            bin_means = []
            bin_edges = []
            
            for b in range(n_bins):
                start = b * bin_size
                if b == n_bins - 1:
                    end = n
                else:
                    end = (b + 1) * bin_size
                bin_y = sorted_y[start:end]
                bin_w = sorted_weights[start:end]
                bin_p = sorted_p[start:end]
                total_w = bin_w.sum()
                if total_w > 0:
                    mean_y = np.sum(bin_y * bin_w) / total_w
                else:
                    mean_y = np.mean(bin_y)
                mean_p = np.mean(bin_p) if len(bin_p) > 0 else 0.5
                bin_means.append(mean_y)
                bin_edges.append(bin_p[-1] if len(bin_p) > 0 else mean_p)
            
            # Apply mapping
            bin_edges_arr = np.array(bin_edges)
            bin_indices = np.digitize(p_val, bin_edges_arr, right=False)
            bin_indices = np.clip(bin_indices, 0, n_bins - 1)
            calibrated_val[:, c] = np.array(bin_means)[bin_indices]
            
            bin_indices_test = np.digitize(p_test, bin_edges_arr, right=False)
            bin_indices_test = np.clip(bin_indices_test, 0, n_bins - 1)
            calibrated_test[:, c] = np.array(bin_means)[bin_indices_test]
    
    return calibrated_val, calibrated_test


def evaluate_bin_counts(proba_val, y_val, proba_test, bin_counts):
    """Evaluate all bin counts and return the best one + all results."""
    results = {}
    for bc in bin_counts:
        cal_val, cal_test = bin_constrained_isotonic(proba_val, y_val, proba_test, bc)
        loss = log_loss(y_val, cal_val)
        results[bc] = {'val_loss': loss, 'cal_val': cal_val, 'cal_test': cal_test}
    return results


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
        n_trials=40,
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
        n_trials=40,
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
        n_trials=25,
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

    # ─── PHASE 2b: Quantile XGBoost (QXGBoost) for uncertainty estimation ─
    # QXGBoost produces prediction intervals whose width is a direct uncertainty signal
    # absent from standard point predictions. This is used to weight isotonic calibration.
    print("\n--- Quantile XGBoost (QXGBoost) for uncertainty estimation ---")
    
    def train_qxgb_per_class(X_tr, y_tr_all, X_v, X_te, quantile_alphas, seed=42):
        """
        Train QXGBoost for a single class with multiple quantiles in one model call.
        Returns predictions for all quantiles for this class.
        """
        y_binary = (y_tr_all == 0).astype(float)  # placeholder, will loop over classes
        n_val = X_v.shape[0]
        n_test = X_te.shape[0]
        
        # DMatrix for faster training
        dtrain_qxgb = xgb.DMatrix(X_tr)
        dval_qxgb = xgb.DMatrix(X_v)
        dtest_qxgb = xgb.DMatrix(X_te)
        
        results = {}
        for c in range(num_classes):
            y_binary = (y_tr_all == c).astype(float)
            dtrain_qxgb.set_label(y_binary)
            dval_qxgb.set_label((y_val == c).astype(float))
            
            params = {
                'objective': 'reg:quantileerror',
                'quantile_alpha': quantile_alphas,
                'tree_method': 'hist',
                'device': 'cuda',
                'max_depth': 6,  # reduced for speed
                'learning_rate': 0.08,
                'subsample': 0.7,
                'colsample_bytree': 0.6,
                'min_child_weight': 20,
                'reg_alpha': 0.5,
                'reg_lambda': 2.0,
                'seed': seed + c,
            }
            
            # Train with early stopping
            evallist = [(dtrain_qxgb, 'train'), (dval_qxgb, 'eval')]
            bst = xgb.train(
                params, dtrain_qxgb,
                num_boost_round=200,
                evals=evallist,
                early_stopping_rounds=20,
                verbose_eval=False
            )
            
            # Predict all quantiles at once
            pred_val = bst.predict(dval_qxgb)
            pred_test = bst.predict(dtest_qxgb)
            
            # pred_val is (n_val, n_quantiles) — reshape
            n_q = len(quantile_alphas)
            results[c] = {'val': pred_val.reshape(n_val, n_q), 
                          'test': pred_test.reshape(n_test, n_q)}
        
        return results
    
    print("  Training QXGBoost for quantiles [0.1, 0.5, 0.9] (3-class, single model per class)...")
    
    # Train QXGBoost — single model per class, all 3 quantiles together
    # Fixed config (no HPO) to avoid budget penalty — per the experiment sketch
    qxgb_results = train_qxgb_per_class(X_train, y_train, X_val, X_test, [0.1, 0.5, 0.9], seed=42)
    
    # Collect quantile predictions
    n_val = X_val.shape[0]
    n_test = X_test.shape[0]
    q10_val = np.zeros((n_val, num_classes))
    q50_val = np.zeros((n_val, num_classes))
    q90_val = np.zeros((n_val, num_classes))
    q10_test = np.zeros((n_test, num_classes))
    q50_test = np.zeros((n_test, num_classes))
    q90_test = np.zeros((n_test, num_classes))
    
    for c in range(num_classes):
        q10_val[:, c] = qxgb_results[c]['val'][:, 0]
        q50_val[:, c] = qxgb_results[c]['val'][:, 1]
        q90_val[:, c] = qxgb_results[c]['val'][:, 2]
        q10_test[:, c] = qxgb_results[c]['test'][:, 0]
        q50_test[:, c] = qxgb_results[c]['test'][:, 1]
        q90_test[:, c] = qxgb_results[c]['test'][:, 2]
    
    # Compute per-sample prediction interval width as uncertainty measure
    # For each sample, interval_width = mean(q90 - q10) across classes
    interval_width_val = np.mean(q90_val - q10_val, axis=1)
    interval_width_test = np.mean(q90_test - q10_test, axis=1)
    
    # Convert to inverse weights: confident (narrow interval) → higher weight
    # Clip to avoid extreme weights
    eps = 1e-6
    inverse_weights = 1.0 / (interval_width_val + eps)
    inv_weights_test = 1.0 / (interval_width_test + eps)
    
    # Normalize weights to sum to N (so average weight = 1.0)
    n_val_samples = len(inverse_weights)
    weights_val = inverse_weights * (n_val_samples / inverse_weights.sum())
    weights_test = inv_weights_test * (n_val_samples / inv_weights_test.sum())
    
    # Stats on uncertainty distribution
    print(f"  Interval width stats: min={interval_width_val.min():.4f}, "
          f"median={np.median(interval_width_val):.4f}, "
          f"max={interval_width_val.max():.4f}")
    print(f"  Weight stats: min={weights_val.min():.4f}, "
          f"median={np.median(weights_val):.4f}, "
          f"max={weights_val.max():.4f}")
    
    # QXGBoost median predictions quality check
    # For multiclass, we treat q50 predictions as class probabilities (normalized)
    qxgb_proba_val = np.exp(q50_val - q50_val.max(axis=1, keepdims=True))
    qxgb_proba_val = qxgb_proba_val / qxgb_proba_val.sum(axis=1, keepdims=True)
    qxgb_median_loss = log_loss(y_val, qxgb_proba_val)
    print(f"  QXGBoost median (q50) val_loss: {qxgb_median_loss:.4f}")
    
    # ─── PHASE 3: Weighted ensemble via grid search ──────────────────────
    print("\n--- Optimizing ensemble weights ---")
    
    # 3-model ensemble (control baseline)
    xgb_val = xgb_proba_val
    lgb_val = lgb_proba_val
    cat_val = cat_proba_val
    
    # Use scipy minimize for faster optimization
    from scipy.optimize import minimize
    
    def neg_log_loss_3(weights):
        w1, w2 = weights
        w3 = 1.0 - w1 - w2
        if w3 < 0.01 or w3 > 0.9:
            return 10.0
        ens = w1 * xgb_val + w2 * lgb_val + w3 * cat_val
        return log_loss(y_val, ens)
    
    # 3-model optimization
    print("  Optimizing 3-model weights...")
    best_3_loss = float("inf")
    best_weights = (1/3, 1/3, 1/3)
    for _ in range(30):  # Multiple random starts
        w0 = [np.random.uniform(0.1, 0.75), np.random.uniform(0.05, 0.5)]
        result = minimize(neg_log_loss_3, w0, method='SLSQP',
                         bounds=[(0.05, 0.90), (0.01, 0.90)],
                         constraints={'type': 'ineq', 'fun': lambda w: 1.0 - w[0] - w[1] - 0.01})
        if result.fun < best_3_loss:
            best_3_loss = result.fun
            w1, w2 = result.x
            w3 = 1.0 - w1 - w2
            best_weights = (w1, w2, w3)
    
    w_xgb, w_lgb, w_cat = best_weights
    print(f"  3-model best weights: XGB={w_xgb:.2f}, LGB={w_lgb:.2f}, CAT={w_cat:.2f}")
    print(f"  Best pre-calibration loss: {best_3_loss:.4f}")
    
    weighted_proba_val = w_xgb * xgb_proba_val + w_lgb * lgb_proba_val + w_cat * cat_proba_val
    weighted_proba_test = w_xgb * xgb_proba_test + w_lgb * lgb_proba_test + w_cat * cat_proba_test
    weighted_loss = log_loss(y_val, weighted_proba_val)

    # ─── PHASE 3b: Bin-Constrained Isotonic Calibration (weighted + unweighted) ──
    # Compare standard unweighted isotonic with QXGBoost-uncertainty-weighted isotonic
    print("\n--- Bin-Constrained Isotonic Calibration ---")
    
    # Test standard unweighted isotonic (control)
    bin_counts = [100, 200, 500]
    print(f"  Testing bin counts (unweighted): {bin_counts}")
    unweighted_results = evaluate_bin_counts(
        weighted_proba_val, y_val, weighted_proba_test, bin_counts
    )
    for bc, res in sorted(unweighted_results.items(), key=lambda x: x[1]['val_loss']):
        marker = " <-- best" if bc == min(unweighted_results, key=lambda k: unweighted_results[k]['val_loss']) else ""
        print(f"  N_bins={str(bc):>7}: val_loss={res['val_loss']:.6f}{marker}")
    
    best_unweighted_bc = min(unweighted_results, key=lambda k: unweighted_results[k]['val_loss'])
    best_unweighted_loss = unweighted_results[best_unweighted_bc]['val_loss']
    print(f"  Best unweighted: N_bins={best_unweighted_bc}, val_loss={best_unweighted_loss:.6f}")
    
    # Test weighted isotonic (using QXGBoost uncertainty weights)
    print(f"\n  Testing weighted isotonic with QXGBoost uncertainty weights...")
    print(f"  Weight stats: min={weights_val.min():.4f}, median={np.median(weights_val):.4f}, max={weights_val.max():.4f}")
    
    weighted_bin_results = {}
    for bc in [500]:  # Only test N=500 with weights (main comparison)
        cal_val, cal_test = weighted_bin_constrained_isotonic(
            weighted_proba_val, y_val, weighted_proba_test, weights_val, bc
        )
        wloss = log_loss(y_val, cal_val)
        weighted_bin_results[bc] = {'val_loss': wloss, 'cal_val': cal_val, 'cal_test': cal_test}
        print(f"  Weighted N_bins={bc}: val_loss={wloss:.6f}")
    
    best_weighted_bc = min(weighted_bin_results, key=lambda k: weighted_bin_results[k]['val_loss'])
    best_weighted_loss = weighted_bin_results[best_weighted_bc]['val_loss']
    print(f"  Best weighted: N_bins={best_weighted_bc}, val_loss={best_weighted_loss:.6f}")
    
    # Select the best overall approach
    if best_weighted_loss < best_unweighted_loss:
        print(f"\n  Weighted isotonic WINS: {best_weighted_loss:.6f} < {best_unweighted_loss:.6f}")
        best_bc = best_weighted_bc
        calibrated_val = weighted_bin_results[best_bc]['cal_val']
        calibrated_test = weighted_bin_results[best_bc]['cal_test']
        calibrated_loss = best_weighted_loss
        use_weighted_iso = True
    else:
        print(f"\n  Unweighted isotonic wins (or equal): {best_unweighted_loss:.6f}")
        best_bc = best_unweighted_bc
        calibrated_val = unweighted_results[best_bc]['cal_val']
        calibrated_test = unweighted_results[best_bc]['cal_test']
        calibrated_loss = best_unweighted_loss
        use_weighted_iso = False
    
    print(f"  Final isotonic: {'weighted' if use_weighted_iso else 'unweighted'}, N_bins={best_bc}")
    
    # Effective per-class temperatures (mean ratio of original/calibrated)
    iso_temperatures = []
    for c in range(num_classes):
        orig_mean = weighted_proba_val[:, c].mean()
        cal_mean = calibrated_val[:, c].mean()
        iso_temperatures.append(orig_mean / (cal_mean + 1e-10))
    print(f"  Effective per-class temperatures: {[f'{t:.4f}' for t in iso_temperatures]}")

    # ─── PHASE 4: Stacking with Logistic Regression meta-learner ───────
    print("\n--- Stacking with Logistic Regression meta-learner ---")
    stack_val = np.column_stack([xgb_proba_val, lgb_proba_val, cat_proba_val])
    stack_test = np.column_stack([xgb_proba_test, lgb_proba_test, cat_proba_test])
    print(f"  3-model stacking")

    meta = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs")
    meta.fit(stack_val, y_val)

    stack_proba_val = meta.predict_proba(stack_val)
    stack_proba_test = meta.predict_proba(stack_test)
    stack_loss = log_loss(y_val, stack_proba_val)
    stack_accuracy = accuracy_score(y_val, np.argmax(stack_proba_val, axis=1))
    print(f"  Stacking Val Loss:     {stack_loss:.4f}")
    print(f"  Stacking Val Accuracy: {stack_accuracy:.4f}")
    
    weighted_loss = log_loss(y_val, weighted_proba_val)
    print(f"  Weighted Val Loss:     {weighted_loss:.4f}")
    print(f"  ISO-Calibrated Val Loss: {calibrated_loss:.4f}")

    # Choose the best approach among all three
    best_method = min(
        [("weighted", weighted_loss, weighted_proba_val, weighted_proba_test),
         ("iso_calibrated", calibrated_loss, calibrated_val, calibrated_test),
         ("stacking", stack_loss, stack_proba_val, stack_proba_test)],
        key=lambda x: x[1]
    )
    method_name, val_loss, final_proba_val, final_proba_test = best_method
    print(f"  Best method: {method_name} ({val_loss:.4f})")
    method_tag = method_name.replace("_", "-")

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
        iso_type = "weighted" if use_weighted_iso else "unweighted"
        notes = f"QXGB-uncertainty-iso, {iso_type}, N_bins={best_bc} {weights_str}"
        f.write(
            f"{timestamp}\t{method_name}\t{total_trials}\t{val_loss:.6f}\t{val_accuracy:.6f}\t{submission_file}\t{HYPOTHESIS} {notes}\n"
        )

    # ─── AUTORESEARCH OUTPUT (DO NOT CHANGE THIS LINE) ───────────────────
    print(f"final_val_loss={val_loss:.4f}")


if __name__ == "__main__":
    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    main()
