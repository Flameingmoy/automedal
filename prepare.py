"""
AutoResearch Kaggle — Data Preparation Pipeline
=================================================
This file is AGENT-EDITABLE. The agent can modify feature engineering,
encoding strategies, augmentation, and preprocessing as part of its
research loop. The raw CSV loading and train/val split logic should
generally be preserved.

Competition: Playground Series S6E4 — Irrigation Prediction
Target: Irrigation_Need (Low / Medium / High)
"""

import pandas as pd
import numpy as np
import os
import json
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.model_selection import train_test_split

# ─── PATHS ───────────────────────────────────────────────────────────────
DATA_DIR = "data"
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
TEST_CSV = os.path.join(DATA_DIR, "test.csv")
RANDOM_SEED = 42

# ─── COLUMN DEFINITIONS ─────────────────────────────────────────────────
TARGET_COL = "Irrigation_Need"
ID_COL = "id"

NUMERIC_FEATURES = [
    "Soil_pH", "Soil_Moisture", "Organic_Carbon", "Electrical_Conductivity",
    "Temperature_C", "Humidity", "Rainfall_mm", "Sunlight_Hours",
    "Wind_Speed_kmh", "Field_Area_hectare", "Previous_Irrigation_mm",
]

CATEGORICAL_FEATURES = [
    "Soil_Type", "Crop_Type", "Crop_Growth_Stage", "Season",
    "Irrigation_Type", "Water_Source", "Mulching_Used", "Region",
]

# ─── FEATURE ENGINEERING ────────────────────────────────────────────────
# Agent: modify this function to add interactions, ratios, polynomial
# features, binning, or any transformations you want to try.
def engineer_features(df):
    """Apply feature engineering to a dataframe. Must work on both train and test."""
    # --- Interaction features ---
    df["Moisture_x_Humidity"] = df["Soil_Moisture"] * df["Humidity"]
    df["Temp_x_Sunlight"] = df["Temperature_C"] * df["Sunlight_Hours"]
    df["Rain_per_Area"] = df["Rainfall_mm"] / (df["Field_Area_hectare"] + 1e-6)
    df["Moisture_Deficit"] = df["Humidity"] - df["Soil_Moisture"]
    df["Irrigation_Intensity"] = df["Previous_Irrigation_mm"] / (df["Field_Area_hectare"] + 1e-6)

    return df


# ─── ENCODING ────────────────────────────────────────────────────────────
# Agent: swap OrdinalEncoder for TargetEncoder, frequency encoding, etc.
def encode_categoricals(train_df, test_df):
    """Encode categorical features. Fits on train, transforms both."""
    encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    encoder.fit(train_df[CATEGORICAL_FEATURES])

    train_encoded = pd.DataFrame(
        encoder.transform(train_df[CATEGORICAL_FEATURES]),
        columns=CATEGORICAL_FEATURES,
        index=train_df.index,
    )
    test_encoded = pd.DataFrame(
        encoder.transform(test_df[CATEGORICAL_FEATURES]),
        columns=CATEGORICAL_FEATURES,
        index=test_df.index,
    )

    # Save encoder categories for reproducibility
    categories = {col: list(cats) for col, cats in zip(CATEGORICAL_FEATURES, encoder.categories_)}
    with open(os.path.join(DATA_DIR, "encoder_categories.json"), "w") as f:
        json.dump(categories, f, indent=2)

    return train_encoded, test_encoded


# ─── MAIN PIPELINE ──────────────────────────────────────────────────────
def prepare_data():
    print("=" * 60)
    print("AutoResearch Kaggle — Data Preparation")
    print("=" * 60)

    # Load raw CSVs
    print("\nLoading CSVs...")
    train_df = pd.read_csv(TRAIN_CSV)
    test_df = pd.read_csv(TEST_CSV)

    print(f"  Train: {train_df.shape[0]:,} rows x {train_df.shape[1]} cols")
    print(f"  Test:  {test_df.shape[0]:,} rows x {test_df.shape[1]} cols")

    # Quick data quality report
    train_nulls = train_df.isnull().sum()
    if train_nulls.any():
        print(f"\n  Missing values:\n{train_nulls[train_nulls > 0]}")
    else:
        print("  No missing values in train.")

    print(f"\n  Target distribution:\n{train_df[TARGET_COL].value_counts().to_string()}")

    # Save test IDs for submission
    test_ids = test_df[ID_COL].values
    np.save(os.path.join(DATA_DIR, "test_ids.npy"), test_ids)

    # Encode target
    le = LabelEncoder()
    y_all = le.fit_transform(train_df[TARGET_COL])
    class_names = list(le.classes_)
    with open(os.path.join(DATA_DIR, "class_names.json"), "w") as f:
        json.dump(class_names, f)
    print(f"\n  Classes: {class_names} -> {list(range(len(class_names)))}")

    # Fill NAs (numeric: median, categorical: mode)
    for col in NUMERIC_FEATURES:
        median = train_df[col].median()
        train_df[col] = train_df[col].fillna(median)
        test_df[col] = test_df[col].fillna(median)

    for col in CATEGORICAL_FEATURES:
        mode = train_df[col].mode()[0]
        train_df[col] = train_df[col].fillna(mode)
        test_df[col] = test_df[col].fillna(mode)

    # Feature engineering
    print("\n  Applying feature engineering...")
    train_df = engineer_features(train_df)
    test_df = engineer_features(test_df)

    # Encode categoricals
    print("  Encoding categorical features...")
    train_cat, test_cat = encode_categoricals(train_df, test_df)

    # Build feature matrices
    # Get all numeric columns (original + engineered) excluding id and target
    all_numeric = [c for c in train_df.select_dtypes(include=[np.number]).columns
                   if c not in [ID_COL, TARGET_COL]]

    X_train_full = pd.concat([train_df[all_numeric].reset_index(drop=True),
                              train_cat.reset_index(drop=True)], axis=1)
    X_test = pd.concat([test_df[all_numeric].reset_index(drop=True),
                        test_cat.reset_index(drop=True)], axis=1)

    # Save feature names
    feature_names = list(X_train_full.columns)
    with open(os.path.join(DATA_DIR, "feature_names.json"), "w") as f:
        json.dump(feature_names, f)
    print(f"\n  Total features: {len(feature_names)}")

    # Train/Val split (stratified)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full.values, y_all,
        test_size=0.2, random_state=RANDOM_SEED, stratify=y_all
    )

    # Save as NumPy arrays
    np.save(os.path.join(DATA_DIR, "X_train.npy"), X_train)
    np.save(os.path.join(DATA_DIR, "y_train.npy"), y_train)
    np.save(os.path.join(DATA_DIR, "X_val.npy"), X_val)
    np.save(os.path.join(DATA_DIR, "y_val.npy"), y_val)
    np.save(os.path.join(DATA_DIR, "X_test.npy"), X_test.values)

    print(f"\n  Train: {X_train.shape}")
    print(f"  Val:   {X_val.shape}")
    print(f"  Test:  {X_test.shape}")
    print(f"\n  Saved to {DATA_DIR}/: X_train.npy, y_train.npy, X_val.npy, y_val.npy, X_test.npy")
    print("  Saved: test_ids.npy, class_names.json, feature_names.json, encoder_categories.json")
    print("\nDone.")


if __name__ == "__main__":
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs("submissions", exist_ok=True)
    if not os.path.exists(TRAIN_CSV):
        print(f"Error: Place train.csv in '{DATA_DIR}/' first.")
    elif not os.path.exists(TEST_CSV):
        print(f"Error: Place test.csv in '{DATA_DIR}/' first.")
    else:
        prepare_data()
