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
import torch
import torch.nn as nn
import torch.optim as optim

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
HYPOTHESIS = "Training a TabKD neural student (Tabular Knowledge Distillation with interaction-diversity synthetic queries) on the 3 GBDT teachers' softmax predictions will produce a diverse 4th ensemble member without the HPO budget penalty that killed TabR (exp 0008) and T-MLP (exp 0009), because TabKD generates synthetic training data by maximizing pairwise feature interaction coverage between the teachers — creating a fundamentally different training signal than standard KL-divergence matching — and the neural student is trained on fixed teacher predictions (no HPO budget consumed) while achieving high teacher-student agreement (14/16 configurations in the paper) on tabular data."
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

    # ─── PHASE 2b: TabKD Neural Student (interaction-diversity KD) ────────
    # TabKD generates synthetic tabular queries maximizing pairwise feature
    # interaction coverage, then trains a neural student on teacher logits.
    # This is a fundamentally different training signal from standard KL-divergence.
    print("\n--- TabKD Neural Student (interaction-diversity KD) ---")
    
    # Use GPU for PyTorch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  PyTorch device: {device}")
    
    class TabKDStudent(nn.Module):
        """Simple 2-layer MLP student — no HPO, fixed architecture."""
        def __init__(self, input_dim, hidden_dim, num_classes):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim // 2),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim // 2, num_classes)
            )
        def forward(self, x):
            return self.net(x)
    
    def generate_interaction_synthetic_data(X_train, n_categories, num_numeric, 
                                            n_target=50000):
        """
        Generate synthetic tabular data for TabKD.
        Strategy:
        1. Generate samples varying each (cat_i, cat_j) pair across all combinations
           while varying numeric features WITHIN the real data distribution.
        2. Add a large pool of fully random samples (within real data ranges) for
           coverage of the full feature space beyond interaction pairs.
        
        The key insight: with binary categoricals (all 8 cats have 2 levels),
        the interaction space is only 28 pairs × 4 combinations = 112 base patterns.
        We add heavy numeric variation + random sampling to make each pattern diverse.
        """
        n_cat_features = len(n_categories)
        
        # Compute stats from training data for realistic numeric ranges
        numeric_means = X_train[:, :num_numeric].mean(axis=0)
        numeric_stds = X_train[:, :num_numeric].std(axis=0) + 1e-6
        numeric_mins = X_train[:, :num_numeric].min(axis=0)
        numeric_maxs = X_train[:, :num_numeric].max(axis=0)
        
        # All pairwise combinations of categorical feature indices
        pairs = []
        for i in range(n_cat_features):
            for j in range(i + 1, n_cat_features):
                pairs.append((i, j))
        
        # Part 1: Interaction-diversity samples (28 pairs × 4 combos × N per combo)
        # With binary cats: 112 base patterns, target ~10K samples → ~90 per combo
        n_per_combo = max(5, n_target // (len(pairs) * 4))
        
        cat_medians = []
        for i in range(n_cat_features):
            vals = X_train[:, num_numeric + i]
            cat_medians.append(np.median(vals))
        
        all_synthetic = []
        
        for (i, j) in pairs:
            cat_i_vals = np.arange(n_categories[i])
            cat_j_vals = np.arange(n_categories[j])
            
            for ci in cat_i_vals:
                for cj in cat_j_vals:
                    # Generate samples with varied numeric features: 
                    # Real-data-range noise centered on feature means
                    noise_part = np.random.randn(n_per_combo, num_numeric) * numeric_stds * 0.5
                    numeric_part = numeric_means + noise_part
                    # Clip to real data range
                    numeric_part = np.clip(numeric_part, numeric_mins, numeric_maxs)
                    
                    cat_part = np.full((n_per_combo, n_cat_features), cat_medians)
                    cat_part[:, i] = ci
                    cat_part[:, j] = cj
                    
                    synth = np.concatenate([numeric_part, cat_part], axis=1)
                    all_synthetic.append(synth)
        
        interaction_samples = np.concatenate(all_synthetic, axis=0)
        
        # Part 2: Random sampling for full feature-space coverage
        n_random = n_target - len(interaction_samples)
        if n_random > 0:
            random_cat = np.column_stack([
                np.random.randint(0, n_categories[i], n_random) 
                for i in range(n_cat_features)
            ])
            random_num = np.random.uniform(
                np.tile(numeric_mins, (n_random, 1)),
                np.tile(numeric_maxs, (n_random, 1))
            )
            random_samples = np.concatenate([random_num, random_cat], axis=1)
            all_synthetic = [interaction_samples, random_samples]
        else:
            all_synthetic = [interaction_samples]
        
        result = np.concatenate(all_synthetic, axis=0)
        # Remove duplicates within epsilon and clip to valid range
        result = np.clip(result, numeric_mins.min() - 1, numeric_maxs.max() + 1)
        return result.astype(np.float32)
    
    # Determine feature dimensions from data
    n_features = X_train.shape[1]
    num_numeric = n_features - 8  # 8 categorical features
    n_categories = [int(X_train[:, num_numeric + i].max()) + 2 for i in range(8)]
    
    print(f"  Generating synthetic interaction data...")
    print(f"  Feature dims: {n_features} ({num_numeric} numeric + 8 categorical)")
    print(f"  Category cardinalities: {n_categories}")
    
    # Generate synthetic data for TabKD (limit to keep GPU memory manageable)
    # Target ~50K synthetic samples covering all pairwise interactions + random coverage
    np.random.seed(42)
    n_synth_target = 30000
    
    # Generate synthetic data covering all 28 pairwise interactions + random coverage
    X_synth = generate_interaction_synthetic_data(
        X_train, n_categories, num_numeric, 
        n_target=n_synth_target
    )
    X_synth = X_synth[:n_synth_target].astype(np.float32)
    print(f"  Synthetic samples: {X_synth.shape[0]}")
    
    # Collect teacher logits on synthetic data
    print("  Collecting teacher logits on synthetic data...")
    
    # All 3 teachers produce probability outputs for consistency in KL-divergence
    # XGBoost teacher (probabilities)
    xgb_synth_proba = xgb_final.predict_proba(X_synth)
    # LightGBM teacher (probabilities)
    lgb_synth_proba = lgb_final.predict_proba(X_synth)
    # CatBoost teacher (probabilities)
    cat_synth_proba = cat_final.predict_proba(X_synth)
    
    # Average of 3 teachers' probabilities as soft training target
    teacher_avg_proba = (xgb_synth_proba + lgb_synth_proba + cat_synth_proba) / 3.0
    # Convert to logits for student training (more stable for KL-divergence)
    teacher_avg_logit = np.log(teacher_avg_proba + 1e-10)
    print(f"  Teacher avg proba shape: {teacher_avg_proba.shape}")
    
    # Average teacher logits as the soft training target

    
    # Train neural student on (synthetic_features, teacher_avg_proba)
    print("  Training TabKD neural student...")
    X_synth_tensor = torch.tensor(X_synth, dtype=torch.float32).to(device)
    # Use log-probabilities for KL-divergence loss (PyTorch KL-div expects log-probs for input)
    y_synth_logprob = np.log(teacher_avg_proba + 1e-10)
    y_synth_tensor = torch.tensor(y_synth_logprob, dtype=torch.float32).to(device)
    
    hidden_dim = 256
    student = TabKDStudent(n_features, hidden_dim, num_classes).to(device)
    optimizer = optim.AdamW(student.parameters(), lr=2e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)
    criterion = nn.KLDivLoss(reduction="batchmean")
    
    batch_size = 2048
    n_epochs = 15
    
    student.train()
    for epoch in range(n_epochs):
        perm = torch.randperm(X_synth_tensor.size(0))
        total_loss = 0
        n_batches = 0
        for i in range(0, X_synth_tensor.size(0), batch_size):
            idx = perm[i:i+batch_size]
            xb = X_synth_tensor[idx]
            yb = y_synth_tensor[idx]
            
            optimizer.zero_grad()
            # Student outputs logits; use log_softmax for KL divergence
            student_logits = student(xb)
            # KL-divergence: target is teacher proba (not logged), input is student log_proba
            log_student_proba = torch.log_softmax(student_logits, dim=1)
            loss = criterion(log_student_proba, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        scheduler.step()
        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1}/{n_epochs}: avg KL loss={total_loss/n_batches:.4f}")
    
    # Get student predictions on validation and test sets
    student.eval()
    with torch.no_grad():
        X_val_t = torch.tensor(X_val.astype(np.float32), device=device)
        X_test_t = torch.tensor(X_test.astype(np.float32), device=device)
        
        student_logits_val = student(X_val_t).cpu().numpy()
        student_logits_test = student(X_test_t).cpu().numpy()
        
        # Convert logits to probabilities
        student_proba_val = np.exp(student_logits_val - student_logits_val.max(axis=1, keepdims=True))
        student_proba_val = student_proba_val / student_proba_val.sum(axis=1, keepdims=True)
        student_proba_test = np.exp(student_logits_test - student_logits_test.max(axis=1, keepdims=True))
        student_proba_test = student_proba_test / student_proba_test.sum(axis=1, keepdims=True)
    
    tabkd_val_loss = log_loss(y_val, student_proba_val)
    print(f"  TabKD student val_loss: {tabkd_val_loss:.4f}")
    
    # Free GPU memory
    del X_synth_tensor, y_synth_tensor, student
    torch.cuda.empty_cache()
    
    # ─── PHASE 3: Weighted ensemble via grid search ──────────────────────
    print("\n--- Optimizing ensemble weights ---")
    best_ens_loss = float("inf")
    best_weights = (1 / 3, 1 / 3, 1 / 3)
    best_4_weights = None
    
    # First: 3-model ensemble (control baseline) — precompute weighted arrays
    xgb_val = xgb_proba_val
    lgb_val = lgb_proba_val
    cat_val = cat_proba_val
    tabkd_val = student_proba_val
    
    # Use scipy minimize for faster optimization
    from scipy.optimize import minimize
    
    def neg_log_loss_3(weights):
        w1, w2 = weights
        w3 = 1.0 - w1 - w2
        if w3 < 0.01 or w3 > 0.9:
            return 10.0
        ens = w1 * xgb_val + w2 * lgb_val + w3 * cat_val
        return log_loss(y_val, ens)
    
    def neg_log_loss_4(weights):
        w1, w2, w3, w4 = weights
        if w1 < 0.05 or w2 < 0.01 or w3 < 0.05 or w4 < 0.0 or w4 > 0.10:
            return 10.0
        if abs(w1 + w2 + w3 + w4 - 1.0) > 0.01:
            return 10.0
        ens = w1 * xgb_val + w2 * lgb_val + w3 * cat_val + w4 * tabkd_val
        return log_loss(y_val, ens)
    
    # 3-model optimization
    print("  Optimizing 3-model weights...")
    best_3_loss = float("inf")
    for _ in range(50):  # Multiple random starts
        w0 = [np.random.uniform(0.1, 0.75), np.random.uniform(0.05, 0.5)]
        result = minimize(neg_log_loss_3, w0, method='SLSQP',
                         bounds=[(0.05, 0.90), (0.01, 0.90)],
                         constraints={'type': 'ineq', 'fun': lambda w: 1.0 - w[0] - w[1] - 0.01})
        if result.fun < best_3_loss:
            best_3_loss = result.fun
            w1, w2 = result.x
            w3 = 1.0 - w1 - w2
            best_weights = (w1, w2, w3)
    
    best_ens_loss = best_3_loss
    
    # 4-model optimization
    print("  Optimizing 4-model weights (3 GBDT + TabKD ≤10%)...")
    best_4_loss = float("inf")
    for _ in range(100):  # Multiple random starts for 4-model
        w0 = [np.random.uniform(0.1, 0.70), np.random.uniform(0.05, 0.40),
              np.random.uniform(0.10, 0.50), np.random.uniform(0.00, 0.10)]
        result = minimize(neg_log_loss_4, w0, method='SLSQP',
                         bounds=[(0.05, 0.85), (0.01, 0.50), (0.05, 0.70), (0.00, 0.10)],
                         constraints={'type': 'eq', 'fun': lambda w: sum(w) - 1.0})
        if result.fun < best_4_loss:
            best_4_loss = result.fun
            best_4_weights = list(result.x)
    
    if best_4_weights and best_4_loss < best_ens_loss:
        best_ens_loss = best_4_loss
        print(f"  4-model wins: {best_4_loss:.4f} < 3-model: {best_3_loss:.4f}")
    else:
        best_4_weights = None
        print(f"  3-model wins: {best_3_loss:.4f}")
    
    print(f"  3-model best weights: XGB={best_weights[0]:.2f}, LGB={best_weights[1]:.2f}, CAT={best_weights[2]:.2f}")
    if best_4_weights:
        print(f"  4-model best weights: XGB={best_4_weights[0]:.2f}, LGB={best_4_weights[1]:.2f}, CAT={best_4_weights[2]:.2f}, TabKD={best_4_weights[3]:.2f}")
    print(f"  Best pre-calibration loss: {best_ens_loss:.4f}")

    # Determine best ensemble: 3-model or 4-model (TabKD)
    if best_4_weights is not None:
        w_xgb, w_lgb, w_cat, w_tabkd = best_4_weights
        print(f"\n  Using 4-model ensemble (3-GBDT + TabKD student)")
        weighted_proba_val = (
            w_xgb * xgb_proba_val + w_lgb * lgb_proba_val + 
            w_cat * cat_proba_val + w_tabkd * student_proba_val
        )
        weighted_proba_test = (
            w_xgb * xgb_proba_test + w_lgb * lgb_proba_test + 
            w_cat * cat_proba_test + w_tabkd * student_proba_test
        )
        use_4model = True
    else:
        w_xgb, w_lgb, w_cat = best_weights
        weighted_proba_val = (
            w_xgb * xgb_proba_val + w_lgb * lgb_proba_val + w_cat * cat_proba_val
        )
        weighted_proba_test = (
            w_xgb * xgb_proba_test + w_lgb * lgb_proba_test + w_cat * cat_proba_test
        )
        w_tabkd = 0.0
        use_4model = False
    
    print(f"  Final weights: XGB={w_xgb:.2f}, LGB={w_lgb:.2f}, CAT={w_cat:.2f}" + 
          (f", TabKD={w_tabkd:.2f}" if use_4model else ""))
    weighted_loss = log_loss(y_val, weighted_proba_val)

    # ─── PHASE 3b: Bin-Constrained Isotonic Calibration ───────────────────
    # Grid-search bin counts to find optimal regularization.
    # The hypothesis: default isotonic overfits to ~1.4% noisy val samples;
    # constraining bins tightens the mapping to the noise floor.
    print("\n--- Bin-Constrained Isotonic Calibration ---")
    bin_counts = [30, 50, 100, 200, 500, 'default']
    print(f"  Testing bin counts: {bin_counts}")
    bin_results = evaluate_bin_counts(
        weighted_proba_val, y_val, weighted_proba_test, bin_counts
    )
    for bc, res in sorted(bin_results.items(), key=lambda x: x[1]['val_loss']):
        marker = " <-- best" if bc == min(bin_results, key=lambda k: bin_results[k]['val_loss']) else ""
        print(f"  N_bins={str(bc):>7}: val_loss={res['val_loss']:.6f}{marker}")
    
    # Select best bin count
    best_bc = min(bin_results, key=lambda k: bin_results[k]['val_loss'])
    calibrated_val = bin_results[best_bc]['cal_val']
    calibrated_test = bin_results[best_bc]['cal_test']
    calibrated_loss = bin_results[best_bc]['val_loss']
    print(f"  Best bin count: {best_bc} (val_loss={calibrated_loss:.6f})")
    
    # Effective per-class temperatures (mean ratio of original/calibrated)
    iso_temperatures = []
    for c in range(num_classes):
        orig_mean = weighted_proba_val[:, c].mean()
        cal_mean = calibrated_val[:, c].mean()
        iso_temperatures.append(orig_mean / (cal_mean + 1e-10))
    print(f"  Effective per-class temperatures: {[f'{t:.4f}' for t in iso_temperatures]}")

    # ─── PHASE 4: Stacking with Logistic Regression meta-learner ───────
    print("\n--- Stacking with Logistic Regression meta-learner ---")
    if use_4model:
        stack_val = np.column_stack([xgb_proba_val, lgb_proba_val, cat_proba_val, student_proba_val])
        stack_test = np.column_stack([xgb_proba_test, lgb_proba_test, cat_proba_test, student_proba_test])
        print(f"  4-model stacking (3 GBDT + TabKD)")
    else:
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
        if use_4model:
            weights_str = f"w=({w_xgb:.2f},{w_lgb:.2f},{w_cat:.2f},{w_tabkd:.2f})"
            notes = f"TabKD_student={tabkd_val_loss:.4f} {weights_str}"
        else:
            weights_str = f"w=({w_xgb:.2f},{w_lgb:.2f},{w_cat:.2f})"
            notes = weights_str
        f.write(
            f"{timestamp}\t{method_name}\t{total_trials}\t{val_loss:.6f}\t{val_accuracy:.6f}\t{submission_file}\t{HYPOTHESIS} {notes}\n"
        )

    # ─── AUTORESEARCH OUTPUT (DO NOT CHANGE THIS LINE) ───────────────────
    print(f"final_val_loss={val_loss:.4f}")


if __name__ == "__main__":
    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    main()
