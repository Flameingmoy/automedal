"""
AutoMedal — CSV Schema Inference
==================================
Sniffs train.csv and test.csv to infer:
- target_col, id_col
- numeric vs categorical features
- task_type (multiclass / binary / regression)
- submission format (from sample_submission.csv)

Returns a structured dict + confidence score + warnings.
"""

import os
import sys
import builtins as _builtins

import pandas as pd
import numpy as np

# All status logging from this shim must go to stderr — Go reads stdout
# as JSON. Override the module-local `print` to default file=sys.stderr.
_orig_print = _builtins.print


def print(*args, **kwargs):  # noqa: A001 — intentional shadow
    kwargs.setdefault("file", sys.stderr)
    return _orig_print(*args, **kwargs)

# Threshold: int columns with nunique <= this are treated as categorical
CATEGORICAL_NUNIQUE_THRESHOLD = 20
# Ambiguous zone: int columns with nunique in this range get a warning
AMBIGUOUS_LOW = 20
AMBIGUOUS_HIGH = 100


def _detect_id_col(df):
    """Detect the ID column by name heuristic or monotonic integer."""
    # Name heuristic
    for col in df.columns:
        if col.lower() in ("id", "idx", "index", "row_id"):
            return col, "name_match"

    # Monotonic integer detection (first column only)
    first_col = df.columns[0]
    if df[first_col].dtype in (np.int64, np.int32, np.float64):
        vals = df[first_col].dropna().values
        if len(vals) > 10:
            diffs = np.diff(vals)
            if np.all(diffs == 1):
                return first_col, "monotonic_integer"

    return None, "not_found"


def _detect_target_col(train_df, test_df):
    """Detect target column via set difference (train cols - test cols)."""
    train_cols = set(train_df.columns)
    test_cols = set(test_df.columns)
    diff = train_cols - test_cols

    if len(diff) == 1:
        return list(diff)[0], "set_difference"
    elif len(diff) > 1:
        # Multiple columns only in train — return all, flag as ambiguous
        return list(diff), "multiple_candidates"
    else:
        return None, "no_difference"


def _classify_features(df, exclude_cols):
    """Classify columns as numeric or categorical."""
    numeric = []
    categorical = []
    ambiguous = []

    for col in df.columns:
        if col in exclude_cols:
            continue

        dtype = df[col].dtype
        nunique = df[col].nunique()

        if dtype == object or dtype.name == "category":
            categorical.append(col)
        elif dtype in (np.float64, np.float32, np.float16):
            numeric.append(col)
        elif dtype in (np.int64, np.int32, np.int16, np.int8):
            if nunique <= CATEGORICAL_NUNIQUE_THRESHOLD:
                categorical.append(col)
            elif nunique <= AMBIGUOUS_HIGH:
                ambiguous.append(col)
                # Default to numeric but flag it
                numeric.append(col)
            else:
                numeric.append(col)
        elif dtype == bool:
            categorical.append(col)
        else:
            # Unknown dtype — default to numeric
            numeric.append(col)

    return numeric, categorical, ambiguous


def _infer_task_type(train_df, target_col):
    """Infer task type from target column."""
    if target_col is None:
        return "unknown", 0

    target = train_df[target_col]
    dtype = target.dtype
    nunique = target.nunique()

    if dtype == object or dtype.name == "category":
        if nunique == 2:
            return "binary", nunique
        else:
            return "multiclass", nunique
    elif dtype in (np.int64, np.int32, np.int16, np.int8):
        if nunique <= CATEGORICAL_NUNIQUE_THRESHOLD:
            if nunique == 2:
                return "binary", nunique
            else:
                return "multiclass", nunique
        else:
            return "regression", nunique
    else:
        # Float target → regression
        return "regression", nunique


def _infer_submission_format(data_dir):
    """Infer submission format from sample_submission.csv if available."""
    sample_paths = [
        os.path.join(data_dir, "sample_submission.csv"),
        os.path.join(data_dir, "sampleSubmission.csv"),
        os.path.join(data_dir, "sample_submission.csv.zip"),
    ]

    sample_df = None
    for path in sample_paths:
        if os.path.exists(path):
            try:
                sample_df = pd.read_csv(path, nrows=5)
                break
            except Exception:
                continue

    if sample_df is None:
        return {"format": "class_label", "detected": False}

    cols = list(sample_df.columns)
    if len(cols) == 2:
        # Simple: id + prediction
        return {
            "format": "class_label",
            "id_col": cols[0],
            "target_col": cols[1],
            "detected": True,
        }
    elif len(cols) > 2:
        # Multiple columns — likely probability submissions
        return {
            "format": "probabilities",
            "id_col": cols[0],
            "probability_cols": cols[1:],
            "target_col": cols[1],  # fallback
            "detected": True,
        }
    else:
        return {"format": "class_label", "detected": False}


def sniff_schema(data_dir="data"):
    """Sniff CSV schema from train.csv and test.csv.

    Args:
        data_dir: Directory containing train.csv and test.csv.

    Returns:
        dict with keys:
            - target_col: str or None
            - id_col: str or None
            - numeric_features: list[str]
            - categorical_features: list[str]
            - task_type: "multiclass" | "binary" | "regression" | "unknown"
            - num_classes: int or None
            - class_names: list[str] or None
            - submission: dict with format info
            - train_rows: int
            - test_rows: int
            - confidence: float (0-1)
            - warnings: list[str]
    """
    warnings = []
    confidence = 1.0

    train_path = os.path.join(data_dir, "train.csv")
    test_path = os.path.join(data_dir, "test.csv")

    if not os.path.exists(train_path):
        return {"error": f"train.csv not found in {data_dir}", "confidence": 0.0, "warnings": ["train.csv missing"]}
    if not os.path.exists(test_path):
        return {"error": f"test.csv not found in {data_dir}", "confidence": 0.0, "warnings": ["test.csv missing"]}

    # Read full train for nunique, but only need nrows for dtype detection
    print("  Reading train.csv...")
    train_df = pd.read_csv(train_path)
    print("  Reading test.csv...")
    test_df = pd.read_csv(test_path)

    train_rows = len(train_df)
    test_rows = len(test_df)

    # Detect ID column
    id_col, id_method = _detect_id_col(train_df)
    if id_col is None:
        warnings.append("Could not detect ID column — defaulting to None")
        confidence -= 0.1

    # Detect target column
    target_result, target_method = _detect_target_col(train_df, test_df)

    if isinstance(target_result, list):
        # Multiple candidates
        warnings.append(
            f"Multiple target candidates: {target_result}. "
            "Using the first non-ID column."
        )
        target_candidates = [c for c in target_result if c != id_col]
        target_col = target_candidates[0] if target_candidates else target_result[0]
        confidence -= 0.2
    elif target_result is None:
        warnings.append("Could not detect target column via set difference")
        target_col = None
        confidence -= 0.4
    else:
        target_col = target_result

    # Classify features
    exclude = {id_col, target_col} - {None}
    numeric, categorical, ambiguous = _classify_features(train_df, exclude)

    if ambiguous:
        warnings.append(
            f"Ambiguous columns (int, 20 < nunique < 100): {ambiguous}. "
            "Defaulted to numeric — verify manually."
        )
        confidence -= 0.1

    # Infer task type
    if target_col:
        task_type, nunique = _infer_task_type(train_df, target_col)
    else:
        task_type = "unknown"
        nunique = 0
        confidence -= 0.3

    # Class names
    class_names = None
    num_classes = None
    if task_type in ("multiclass", "binary") and target_col:
        class_names = sorted(train_df[target_col].dropna().unique().tolist(), key=str)
        num_classes = len(class_names)
    elif task_type == "regression":
        num_classes = None
        class_names = None

    # Submission format
    submission = _infer_submission_format(data_dir)
    if not submission.get("detected"):
        warnings.append("sample_submission.csv not found — submission format guessed")
        confidence -= 0.05
    # Override submission id/target with detected values if available
    sub_id = submission.get("id_col", id_col)
    sub_target = submission.get("target_col", target_col)

    # Clamp confidence
    confidence = max(0.0, min(1.0, confidence))

    result = {
        "target_col": target_col,
        "id_col": id_col,
        "numeric_features": numeric,
        "categorical_features": categorical,
        "task_type": task_type,
        "num_classes": num_classes,
        "class_names": class_names,
        "submission": {
            "id_col": sub_id,
            "target_col": sub_target,
            "format": submission["format"],
        },
        "train_rows": train_rows,
        "test_rows": test_rows,
        "confidence": round(confidence, 2),
        "warnings": warnings,
    }

    # Summary
    print(f"\n  Schema inference results:")
    print(f"    Target:     {target_col} ({target_method})")
    print(f"    ID:         {id_col} ({id_method})")
    print(f"    Task:       {task_type}" + (f" ({num_classes} classes)" if num_classes else ""))
    print(f"    Numeric:    {len(numeric)} features")
    print(f"    Categorical:{len(categorical)} features")
    print(f"    Train:      {train_rows:,} rows")
    print(f"    Test:       {test_rows:,} rows")
    print(f"    Confidence: {confidence:.0%}")
    if warnings:
        print(f"    Warnings:")
        for w in warnings:
            print(f"      - {w}")

    return result
