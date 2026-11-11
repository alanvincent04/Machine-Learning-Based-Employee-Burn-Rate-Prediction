#!/usr/bin/env python3
"""
BurnoutSense AI — Part A
File: ml/preprocess.py

Loads and preprocesses the Kaggle burnout dataset.
Produces three analytical populations:
  - A1: 18,590 complete cases including Mental Fatigue Score
  - A2-Matched: the same 18,590 rows, excluding Mental Fatigue Score
  - A2-Maximum-Data: 20,348 rows eligible for the non-MFS feature set

Target variable: Burn Rate (continuous 0.0 to 1.0)

Dataset: Are Your Employees Burning Out?
Source:  https://www.kaggle.com/datasets/blurredmachine/
         are-your-employees-burning-out
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ── CONSTANTS ─────────────────────────────────────────────────────────────────
KAGGLE_COLUMNS = [
    "Employee ID",
    "Date of Joining",
    "Gender",
    "Company Type",
    "WFH Setup Available",
    "Designation",
    "Resource Allocation",
    "Mental Fatigue Score",
    "Burn Rate",
]

# Features WITH Mental Fatigue Score (Experiment A1)
FEATURES_WITH_MFS = [
    "Gender",
    "Company Type",
    "WFH Setup Available",
    "Designation",
    "Resource Allocation",
    "Mental Fatigue Score",
    "Tenure Months",
]

# Features WITHOUT Mental Fatigue Score (A2-Matched and A2-Maximum-Data)
FEATURES_WITHOUT_MFS = [
    "Gender",
    "Company Type",
    "WFH Setup Available",
    "Designation",
    "Resource Allocation",
    "Tenure Months",
]

TARGET = "Burn Rate"


# ── LOADER ────────────────────────────────────────────────────────────────────
def load_dataset(filepath: str) -> pd.DataFrame:
    """
    Load the raw Kaggle CSV file.
    Raises FileNotFoundError with a helpful message if not found.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"\nDataset not found at: {filepath}\n"
            f"Download from: https://www.kaggle.com/datasets/"
            f"blurredmachine/are-your-employees-burning-out\n"
            f"Save the file as: ml/burnout_data.csv"
        )
    df = pd.read_csv(filepath)
    logger.info(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    logger.info(f"Columns found: {df.columns.tolist()}")
    return df


# ── TRANSFORMATION ────────────────────────────────────────────────────────────
def transform_raw(df: pd.DataFrame) -> pd.DataFrame:
    """Apply deterministic transformations without experiment-specific row deletion."""
    df = df.copy()
    if "Employee ID" in df.columns:
        df = df.drop(columns=["Employee ID"])
    if "Date of Joining" in df.columns:
        df["Date of Joining"] = pd.to_datetime(df["Date of Joining"], errors="coerce")
        reference_date = pd.Timestamp("2024-01-01")
        df["Tenure Months"] = ((reference_date - df["Date of Joining"]).dt.days / 30.44).round(1)
        df = df.drop(columns=["Date of Joining"])
    else:
        df["Tenure Months"] = 0.0
    if "Gender" in df.columns:
        df["Gender"] = df["Gender"].map({"Male": 1, "Female": 0})
    if "Company Type" in df.columns:
        df["Company Type"] = df["Company Type"].map({"Product": 1, "Service": 0})
    if "WFH Setup Available" in df.columns:
        df["WFH Setup Available"] = df["WFH Setup Available"].map({"Yes": 1, "No": 0})
    return df


def clean_for_features(df: pd.DataFrame, required_features: list, target: str = TARGET) -> pd.DataFrame:
    """Return rows complete for exactly the requested feature set and target."""
    required = required_features + [target]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Required columns missing: {missing}")
    return df.dropna(subset=required).reset_index(drop=True)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Backward-compatible complete-case cleaning using the A1 feature requirements."""
    transformed = transform_raw(df)
    return clean_for_features(transformed, FEATURES_WITH_MFS)


# ── SUMMARY ───────────────────────────────────────────────────────────────────
def print_summary(df: pd.DataFrame):
    """Print a statistical summary of the cleaned dataset."""
    print("\n" + "=" * 60)
    print("  DATASET SUMMARY")
    print("=" * 60)
    print(f"  Total records     : {len(df):,}")
    print(f"  Features          : {len(df.columns) - 1}")
    print(f"\n  Target — Burn Rate statistics:")
    print(f"    Min    : {df[TARGET].min():.4f}")
    print(f"    Max    : {df[TARGET].max():.4f}")
    print(f"    Mean   : {df[TARGET].mean():.4f}")
    print(f"    Median : {df[TARGET].median():.4f}")
    print(f"    Std    : {df[TARGET].std():.4f}")
    print(f"\n  Burn Rate distribution (indicative only):")
    low    = (df[TARGET] < 0.40).sum()
    medium = ((df[TARGET] >= 0.40) & (df[TARGET] < 0.70)).sum()
    high   = (df[TARGET] >= 0.70).sum()
    total  = len(df)
    print(f"    < 0.40  (Low zone)   : {low:5d} ({low/total*100:.1f}%)")
    print(f"    0.40–0.70 (Med zone) : {medium:5d} ({medium/total*100:.1f}%)")
    print(f"    >= 0.70 (High zone)  : {high:5d} ({high/total*100:.1f}%)")
    print(f"\n  NOTE: These zones are indicative only.")
    print(f"  Burn Rate is treated as continuous in all experiments.")
    print("=" * 60)

    print(f"\n  Feature correlation with Burn Rate:")
    print(f"  {'-'*50}")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if TARGET in numeric_cols:
        numeric_cols.remove(TARGET)
    corr = df[numeric_cols].corrwith(df[TARGET]).sort_values(
        key=abs, ascending=False
    )
    for col, val in corr.items():
        bar_len = int(abs(val) * 30)
        direction = "+" if val > 0 else "-"
        bar = direction * bar_len
        print(f"  {col:25s}: {val:+.4f}  {bar}")
    print("=" * 60)


# ── MAIN PREPROCESSING FUNCTION ───────────────────────────────────────────────
def preprocess(filepath: str = "ml/burnout_data.csv"):
    """Build the three independent experimental populations.

    A1 and A2-Matched use identical eligibility (all six A2 features plus MFS
    and Burn Rate must be present), while A2-Maximum-Data only requires the
    six non-MFS features and Burn Rate. This prevents A1 missingness from
    unnecessarily removing otherwise usable A2 observations.
    """
    df_raw = load_dataset(filepath)
    transformed = transform_raw(df_raw)

    df_a1 = clean_for_features(transformed, FEATURES_WITH_MFS)
    df_a2_matched = clean_for_features(transformed, FEATURES_WITH_MFS)
    df_a2_max = clean_for_features(transformed, FEATURES_WITHOUT_MFS)

    X_a1 = df_a1[FEATURES_WITH_MFS].copy()
    y_a1 = df_a1[TARGET].copy()
    X_a2_matched = df_a2_matched[FEATURES_WITHOUT_MFS].copy()
    y_a2_matched = df_a2_matched[TARGET].copy()
    X_a2_max = df_a2_max[FEATURES_WITHOUT_MFS].copy()
    y_a2_max = df_a2_max[TARGET].copy()

    logger.info(f"A1 population: {len(df_a1):,} rows")
    logger.info(f"A2-Matched population: {len(df_a2_matched):,} rows")
    logger.info(f"A2-Maximum-Data population: {len(df_a2_max):,} rows")

    # Summary is based on the A1 analytical population for compatibility.
    print_summary(df_a1)

    return {
        "df": df_a1,
        "df_a1": df_a1,
        "df_a2_matched": df_a2_matched,
        "df_a2_maximum": df_a2_max,
        "X_with_mfs": X_a1,
        "X_without_mfs": X_a2_matched,
        "y": y_a1,
        "X_a2_matched": X_a2_matched,
        "y_a2_matched": y_a2_matched,
        "X_a2_maximum": X_a2_max,
        "y_a2_maximum": y_a2_max,
        "feature_names_with_mfs": FEATURES_WITH_MFS,
        "feature_names_without_mfs": FEATURES_WITHOUT_MFS,
    }


if __name__ == "__main__":
    data = preprocess("ml/burnout_data.csv")
    print("\nPreprocessing complete.")
    print(f"X_with_mfs shape    : {data['X_with_mfs'].shape}")
    print(f"X_without_mfs shape : {data['X_without_mfs'].shape}")
    print(f"y shape             : {data['y'].shape}")
