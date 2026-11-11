#!/usr/bin/env python3
"""
BurnoutSense AI — Part A
File: ml/experiments.py

Runs three regression experiments:

  Experiment A1 — WITH Mental Fatigue Score
    Features: Gender, Company Type, WFH Setup, Designation,
              Resource Allocation, Mental Fatigue Score, Tenure
    Purpose:  Upper performance benchmark

  Experiment A2-Matched — WITHOUT Mental Fatigue Score
    Features: Gender, Company Type, WFH Setup, Designation,
              Resource Allocation, Tenure
    Purpose:  Real research finding — do workplace variables
              alone predict burnout?

Both experiments compare 4 algorithms:
  1. Random Forest Regressor
  2. Linear Regression
  3. Support Vector Regression (SVR)
  4. Decision Tree Regressor

Each experiment runs with 10 different random seeds for
sensitivity analysis. Mean and std of MAE, RMSE, R² reported.
"""

import numpy as np
import pandas as pd
import time
import warnings
import joblib
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
import logging

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ── SEEDS FOR SENSITIVITY ANALYSIS ───────────────────────────────────────────
SEEDS = [42, 7, 13, 21, 99, 55, 37, 81, 64, 28]

# ── ALGORITHM DEFINITIONS ─────────────────────────────────────────────────────
def get_algorithms():
    """
    Return the four algorithms as specified by the guide.
    Each algorithm is returned with its best general-purpose
    hyperparameters for this type of regression task.
    """
    return {
        "Random Forest": RandomForestRegressor(
            n_estimators=200,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features="sqrt",
            random_state=42,
            n_jobs=-1,
        ),
        "Linear Regression": LinearRegression(
            fit_intercept=True,
            n_jobs=-1,
        ),
        "SVR": SVR(
            kernel="rbf",
            C=1.0,
            epsilon=0.1,
            gamma="scale",
        ),
        "Decision Tree": DecisionTreeRegressor(
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
        ),
    }


# ── SINGLE SEED EVALUATION ────────────────────────────────────────────────────
def evaluate_one_seed(
    X: pd.DataFrame,
    y: pd.Series,
    algorithm_name: str,
    algorithm,
    seed: int,
    test_size: float = 0.2,
) -> dict:
    """
    Train and evaluate one algorithm on one random seed.

    Returns dict with:
      seed, algorithm, MAE, RMSE, R², runtime_seconds
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )

    # Scale features
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    # Train
    start = time.time()
    algorithm.fit(X_train_s, y_train)
    runtime = round(time.time() - start, 4)

    # Predict
    y_pred = algorithm.predict(X_test_s)

    # Clip predictions to valid burn rate range
    y_pred = np.clip(y_pred, 0.0, 1.0)

    # Metrics
    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)

    return {
        "seed"           : seed,
        "algorithm"      : algorithm_name,
        "MAE"            : round(mae,  4),
        "RMSE"           : round(rmse, 4),
        "R2"             : round(r2,   4),
        "Runtime_Seconds": runtime,
    }


# ── SENSITIVITY ANALYSIS ──────────────────────────────────────────────────────
def sensitivity_analysis(
    X: pd.DataFrame,
    y: pd.Series,
    experiment_name: str,
) -> pd.DataFrame:
    """
    Run all 4 algorithms across all 10 seeds.
    Returns a DataFrame with one row per (algorithm, seed) combination.
    """
    print(f"\n{'#'*60}")
    print(f"  {experiment_name}")
    print(f"  Features: {X.columns.tolist()}")
    print(f"  Samples : {len(X)}")
    print(f"  Seeds   : {SEEDS}")
    print(f"{'#'*60}")

    algorithms = get_algorithms()
    all_results = []

    for algo_name, algo in algorithms.items():
        print(f"\n  Running: {algo_name}")
        for seed in SEEDS:
            # Re-initialise algorithm with current seed where applicable
            if "random_state" in algo.get_params():
                algo.set_params(random_state=seed)

            result = evaluate_one_seed(X, y, algo_name, algo, seed)
            all_results.append(result)
            print(
                f"    Seed {seed:3d} | "
                f"MAE={result['MAE']:.4f}  "
                f"RMSE={result['RMSE']:.4f}  "
                f"R²={result['R2']:.4f}  "
                f"({result['Runtime_Seconds']}s)"
            )

    return pd.DataFrame(all_results)


# ── SUMMARY STATISTICS ────────────────────────────────────────────────────────
def compute_summary(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute mean ± std for each algorithm across all seeds.
    Returns a summary DataFrame.
    """
    summary = (
        results_df.groupby("algorithm")
        .agg(
            MAE_Mean=("MAE",   "mean"),
            MAE_Std =("MAE",   "std"),
            RMSE_Mean=("RMSE", "mean"),
            RMSE_Std =("RMSE", "std"),
            R2_Mean  =("R2",   "mean"),
            R2_Std   =("R2",   "std"),
            Runtime_Mean=("Runtime_Seconds", "mean"),
        )
        .round(4)
        .reset_index()
    )
    # Sort by R2 descending
    summary = summary.sort_values("R2_Mean", ascending=False).reset_index(
        drop=True
    )
    return summary


# ── FEATURE IMPORTANCE ────────────────────────────────────────────────────────
def get_feature_importance(
    X: pd.DataFrame,
    y: pd.Series,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Train a Random Forest and extract feature importances.
    Used to show which features contribute most to predicting Burn Rate.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    rf = RandomForestRegressor(
        n_estimators=200, random_state=seed, n_jobs=-1
    )
    rf.fit(X_scaled, y)

    importance_df = pd.DataFrame(
        {
            "Feature"   : X.columns.tolist(),
            "Importance": rf.feature_importances_,
        }
    ).sort_values("Importance", ascending=False).reset_index(drop=True)

    importance_df["Importance_%"] = (
        importance_df["Importance"] * 100
    ).round(2)

    return importance_df


# ── CROSS VALIDATION ─────────────────────────────────────────────────────────
def cross_validate_best(
    X: pd.DataFrame,
    y: pd.Series,
    experiment_name: str,
    cv: int = 5,
) -> pd.DataFrame:
    """
    Run 5-fold cross validation on all 4 algorithms.
    Uses R² as the scoring metric.
    Returns a DataFrame of CV results.
    """
    print(f"\n  Cross Validation ({cv}-fold) — {experiment_name}")
    algorithms = get_algorithms()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    cv_results = []
    for name, algo in algorithms.items():
        scores = cross_val_score(
            algo, X_scaled, y, cv=cv, scoring="r2", n_jobs=-1
        )
        cv_results.append(
            {
                "Algorithm"    : name,
                "CV_R2_Mean"   : round(scores.mean(), 4),
                "CV_R2_Std"    : round(scores.std(),  4),
                "CV_R2_Min"    : round(scores.min(),  4),
                "CV_R2_Max"    : round(scores.max(),  4),
            }
        )
        print(
            f"    {name:20s}: "
            f"R²={scores.mean():.4f} ± {scores.std():.4f}"
        )

    return pd.DataFrame(cv_results).sort_values(
        "CV_R2_Mean", ascending=False
    ).reset_index(drop=True)


# ── PRINT SUMMARY TABLE ───────────────────────────────────────────────────────
def print_summary_table(summary: pd.DataFrame, experiment_name: str):
    """Print a formatted summary table to the terminal."""
    print(f"\n{'='*70}")
    print(f"  RESULTS SUMMARY — {experiment_name}")
    print(f"  (Mean ± Std across {len(SEEDS)} seeds)")
    print(f"{'='*70}")
    print(
        f"  {'Algorithm':<22} "
        f"{'MAE':>12} "
        f"{'RMSE':>12} "
        f"{'R²':>12} "
        f"{'Time(s)':>10}"
    )
    print(f"  {'-'*66}")
    for _, row in summary.iterrows():
        print(
            f"  {row['algorithm']:<22} "
            f"{row['MAE_Mean']:.4f}±{row['MAE_Std']:.4f}  "
            f"{row['RMSE_Mean']:.4f}±{row['RMSE_Std']:.4f}  "
            f"{row['R2_Mean']:.4f}±{row['R2_Std']:.4f}  "
            f"{row['Runtime_Mean']:>8.3f}s"
        )
    print(f"{'='*70}")
    best = summary.iloc[0]
    print(f"\n  Best algorithm: {best['algorithm']}")
    print(f"  Best R²       : {best['R2_Mean']:.4f} ± {best['R2_Std']:.4f}")
    print(f"  Best MAE      : {best['MAE_Mean']:.4f} ± {best['MAE_Std']:.4f}")
    print(f"  Best RMSE     : {best['RMSE_Mean']:.4f} ± {best['RMSE_Std']:.4f}")


# ── SAVE BEST MODEL ───────────────────────────────────────────────────────────
def save_best_model(
    X: pd.DataFrame,
    y: pd.Series,
    experiment_name: str,
    output_dir: str = "output",
):
    """
    Train the best algorithm (Random Forest) on full data
    and save the model and scaler.
    """
    os.makedirs(output_dir, exist_ok=True)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    rf = RandomForestRegressor(
        n_estimators=200, random_state=42, n_jobs=-1
    )
    rf.fit(X_scaled, y)

    if "Mental Fatigue Score" in X.columns:
        tag = "with_mfs"
    elif "Maximum-Data" in experiment_name:
        tag = "without_mfs_maximum_data"
    else:
        tag = "without_mfs_matched"
    joblib.dump(rf,     f"{output_dir}/rf_model_{tag}.pkl")
    joblib.dump(scaler, f"{output_dir}/scaler_{tag}.pkl")
    logger.info(
        f"Saved model and scaler for {experiment_name} to {output_dir}/"
    )


# ── MAIN ──────────────────────────────────────────────────────────────────────
def run_one_experiment(X, y, experiment_name, model_label, save_tag=None):
    """Run the common evaluation suite and print all results to the terminal.

    save_tag is retained for backward compatibility but is intentionally ignored
    in terminal-only mode; no model/result files are generated.
    """
    print("\n" + "=" * 60)
    print(f"  {experiment_name}")
    print(f"  {model_label}")
    print("=" * 60)
    raw = sensitivity_analysis(X, y, experiment_name)
    summary = compute_summary(raw)
    cv = cross_validate_best(X, y, experiment_name)
    fi = get_feature_importance(X, y)
    print_summary_table(summary, experiment_name)
    print(f"\n  Feature Importance (Random Forest — {experiment_name}):")
    for _, row in fi.iterrows():
        bar = "█" * int(row["Importance_%"] / 2)
        print(f"    {row['Feature']:25s}: {row['Importance_%']:5.2f}%  {bar}")
    return {"raw": raw, "summary": summary, "cv": cv, "importance": fi, "label": experiment_name}


def run_all_experiments(data: dict) -> dict:
    """Run A1, A2-Matched, and A2-Maximum-Data.

    A2-Matched uses the same 18,590 eligible rows as A1, but removes MFS.
    A2-Maximum-Data uses the full 20,348-row population eligible for the
    non-MFS feature set.
    """
    results = {}

    X_a1, y_a1 = data["X_with_mfs"], data["y"]
    X_matched, y_matched = data["X_a2_matched"], data["y_a2_matched"]
    X_max, y_max = data["X_a2_maximum"], data["y_a2_maximum"]

    results["A1"] = run_one_experiment(
        X_a1, y_a1,
        "EXPERIMENT A1 — WITH Mental Fatigue Score",
        "Upper performance benchmark (18,590 samples)",
        None,
    )
    results["A2_Matched"] = run_one_experiment(
        X_matched, y_matched,
        "EXPERIMENT A2-Matched — WITHOUT Mental Fatigue Score",
        "Controlled comparison using the same 18,590 samples as A1",
        None,
    )
    results["A2_Maximum_Data"] = run_one_experiment(
        X_max, y_max,
        "EXPERIMENT A2-Maximum-Data — WITHOUT Mental Fatigue Score",
        "Maximum eligible population (20,348 samples)",
        None,
    )

    # Primary controlled finding: A1 versus A2-Matched.
    best_a1 = results["A1"]["summary"].iloc[0]
    best_matched = results["A2_Matched"]["summary"].iloc[0]
    best_max = results["A2_Maximum_Data"]["summary"].iloc[0]
    drop = best_a1["R2_Mean"] - best_matched["R2_Mean"]

    print("\n" + "=" * 60)
    print("  THREE-EXPERIMENT COMPARISON")
    print("=" * 60)
    print(f"  A1 best R²                 : {best_a1['R2_Mean']:.4f} ± {best_a1['R2_Std']:.4f}")
    print(f"  A2-Matched best R²         : {best_matched['R2_Mean']:.4f} ± {best_matched['R2_Std']:.4f}")
    print(f"  A2-Maximum-Data best R²    : {best_max['R2_Mean']:.4f} ± {best_max['R2_Std']:.4f}")
    print(f"  A1 → A2-Matched R² drop    : {drop:.4f}")
    print("\n  Interpretation:")
    print("  A1 vs A2-Matched isolates the effect of removing Mental Fatigue Score.")
    print("  A2-Matched vs A2-Maximum-Data tests the value of retaining more eligible data.")
    print("=" * 60)
    return results


if __name__ == "__main__":
    from preprocess import preprocess
    data    = preprocess("ml/burnout_data.csv")
    results = run_all_experiments(data)
    print("\nAll experiments complete.")
