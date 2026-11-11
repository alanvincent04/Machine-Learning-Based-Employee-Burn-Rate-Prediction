#!/usr/bin/env python3
"""
BurnoutSense AI — Part A
File: main.py

Terminal-only entry point for the full Part A ML research pipeline.
Runs preprocessing and all three experiments, printing results directly
in the terminal. No PDF/CSV report generation is performed.

Usage:
  python main.py
  python main.py --data ml/burnout_data.csv
"""

import argparse
import os
import sys
import time
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def check_dependencies():
    """Check all required packages are installed."""
    required = {
        "pandas": "pandas",
        "numpy": "numpy",
        "sklearn": "scikit-learn",
        "joblib": "joblib",
    }
    missing = []
    for module, package in required.items():
        try:
            __import__(module)
        except ImportError:
            missing.append(package)

    if missing:
        print("\nMissing required packages:")
        for pkg in missing:
            print(f"  pip install {pkg}")
        print("\nInstall all at once:")
        print("  pip install -r requirements.txt")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="BurnoutSense AI — Part A ML Research Pipeline (terminal only)"
    )
    parser.add_argument(
        "--data",
        default="ml/burnout_data.csv",
        help="Path to Kaggle burnout CSV file (default: ml/burnout_data.csv)",
    )
    args = parser.parse_args()

    print("\n" + "=" * 72)
    print("  BURNOUTSENSE AI — PART A")
    print("  Machine Learning Research Pipeline")
    print("  TERMINAL-ONLY RESULTS MODE")
    print("=" * 72)
    print(f"  Dataset      : {args.data}")
    print("  Seeds        : 10 (42, 7, 13, 21, 99, 55, 37, 81, 64, 28)")
    print("  Algorithms   : Random Forest, Linear Regression, SVR, Decision Tree")
    print("  Experiments  : A1 + A2-Matched + A2-Maximum-Data")
    print("=" * 72)

    print("\n[1/3] Checking dependencies...")
    check_dependencies()
    print("      All dependencies found.")

    print("\n[2/3] Loading and preprocessing dataset...")
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "ml"))
    from preprocess import preprocess
    try:
        data = preprocess(args.data)
    except FileNotFoundError as e:
        print(f"\nERROR: {e}")
        sys.exit(1)

    print("\n  Experimental populations")
    print("  -------------------------")
    print(f"  A1                  : {len(data['y']):,} samples")
    print(f"  A2-Matched          : {len(data['y_a2_matched']):,} samples")
    print(f"  A2-Maximum-Data     : {len(data['y_a2_maximum']):,} samples")

    print("\n[3/3] Running all experiments...")
    start_time = time.time()
    from experiments import run_all_experiments
    results = run_all_experiments(data)
    elapsed = round(time.time() - start_time, 1)

    print("\n" + "=" * 72)
    print("  TERMINAL PIPELINE COMPLETE")
    print("=" * 72)
    print(f"  Total experiment time : {elapsed} seconds")
    print("  Results printed above for:")
    print("    • A1 — With Mental Fatigue Score (18,590)")
    print("    • A2-Matched — Without Mental Fatigue Score (18,590)")
    print("    • A2-Maximum-Data — Without Mental Fatigue Score (20,348)")
    print("  No PDF report or result CSV files were generated.")
    print("=" * 72)

    return results


if __name__ == "__main__":
    main()
