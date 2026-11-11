#!/usr/bin/env python3
"""
BurnoutSense AI — Part A
File: PROJECT_OVERVIEW.py

Run this file to print a complete readable overview of
the entire project — dataset, methodology, algorithms,
experiments, and what every file does.

Usage:
  python PROJECT_OVERVIEW.py
"""

OVERVIEW = """
╔══════════════════════════════════════════════════════════════════╗
║         BURNOTSENSE AI — COMPLETE PROJECT OVERVIEW               ║
║         AI-Based Mental Burnout Detection System                 ║
║         for Remote Workers Using Cloud and Machine Learning       ║
║         MSc Dissertation — Part A: ML Research                   ║
╚══════════════════════════════════════════════════════════════════╝

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. PROJECT TITLE AND SCOPE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Title:
    AI-Based Mental Burnout Detection System for Remote Workers
    Using Cloud and Machine Learning

  Privacy stance:
    PRIVACY-AWARE (not privacy-preserving)
    Implemented through:
      - Consent-based data collection
      - Role-based access control (RBAC)
      - End-to-end TLS 1.3 encryption
      - Minimal data collection principles
    Full differential privacy is a planned future enhancement.

  Project structure:
    PART A — ML Research (this codebase)
              Real data, honest results, academic contribution.

    PART B — Technical System (separate codebase)
              Agent + API + Dashboard + Cloud infrastructure.
              Uses simulated data for pipeline demonstration ONLY.
              Makes NO accuracy claims.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
2. DATASET
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Name:     Are Your Employees Burning Out?
  Source:   Kaggle — blurredmachine
  URL:      https://www.kaggle.com/datasets/blurredmachine/
            are-your-employees-burning-out
  Size:     22,750 employee records
  Format:   CSV
  Cost:     Free (Kaggle account required to download)

  RAW COLUMNS:
  ┌─────────────────────────┬───────────┬──────────────────────────┐
  │ Column                  │ Type      │ How We Use It            │
  ├─────────────────────────┼───────────┼──────────────────────────┤
  │ Employee ID             │ ID        │ Dropped — not a feature  │
  │ Date of Joining         │ Date      │ → Tenure Months          │
  │ Gender                  │ Category  │ Encoded: Male=1, Female=0│
  │ Company Type            │ Category  │ Product=1, Service=0     │
  │ WFH Setup Available     │ Category  │ Yes=1, No=0              │
  │ Designation             │ Numeric   │ Used as-is (0–5)         │
  │ Resource Allocation     │ Numeric   │ Used as-is (1–10 hours)  │
  │ Mental Fatigue Score    │ Numeric   │ A1 only (0–10)           │
  │ Burn Rate               │ Numeric   │ TARGET (0.0–1.0)         │
  └─────────────────────────┴───────────┴──────────────────────────┘

  TARGET VARIABLE:
    Burn Rate — continuous float between 0.0 and 1.0
    0.0 = no burnout, 1.0 = complete burnout
    Treated as a continuous regression target (NOT classified)

  WHY NOT CLASSIFICATION:
    The guide specifically recommended regression to preserve
    the continuous nature of Burn Rate and avoid arbitrary
    threshold decisions. Metrics: MAE, RMSE, R²

  TARGET LEAKAGE — REMOVED:
    Task Completion Rate = (1 - Burn Rate) × 100
    This was originally a mapped feature but creates direct
    target leakage (answer inside the input). REMOVED from
    all Kaggle experiments. Acknowledged in dissertation.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
3. TWO EXPERIMENTS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  EXPERIMENT A1 — WITH Mental Fatigue Score
  ─────────────────────────────────────────
  Purpose:  Upper performance benchmark
  Features: Gender, Company Type, WFH Setup, Designation,
            Resource Allocation, Mental Fatigue Score, Tenure
  Why MFS included: Guide asked for it as benchmark.
                    MFS is self-reported and strongly correlated
                    with Burn Rate so high R² is expected.
  Claim:    This is NOT a novel finding. It shows what the
            maximum achievable accuracy looks like when a
            survey item is included.

  EXPERIMENT A2 — WITHOUT Mental Fatigue Score
  ─────────────────────────────────────────────
  Purpose:  REAL RESEARCH FINDING
  Features: Gender, Company Type, WFH Setup, Designation,
            Resource Allocation, Tenure
  Why MFS excluded: To test whether WORKPLACE STRUCTURAL
                    VARIABLES ALONE predict burnout without
                    any self-reported psychological input.
  Claim:    If A2 achieves meaningful R² it proves that
            objective workplace factors contain genuine
            independent predictive information about burnout.
            This IS a novel and academically defensible finding.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
4. FOUR ALGORITHMS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  ALGORITHM 1 — Random Forest Regressor
  ───────────────────────────────────────
  Type:       Ensemble of 200 decision trees
  Parameters: n_estimators=200, max_features=sqrt,
              min_samples_split=2, random_state=seed
  Why chosen: Best general performance on tabular data.
              Handles non-linear relationships.
              Provides feature importance rankings.
              Resistant to noise and overfitting.
              Breiman L. (2001). Random Forests. Machine Learning.
              https://link.springer.com/article/10.1023/A:1010933404324

  ALGORITHM 2 — Linear Regression
  ─────────────────────────────────
  Type:       Ordinary Least Squares regression
  Parameters: fit_intercept=True
  Why chosen: Simple interpretable baseline.
              Shows if a linear relationship is sufficient.
              Coefficients reveal direction of each feature's effect.
              If RF >> LR then non-linear patterns exist.

  ALGORITHM 3 — Support Vector Regression (SVR)
  ───────────────────────────────────────────────
  Type:       Kernel-based regression (RBF kernel)
  Parameters: kernel=rbf, C=1.0, epsilon=0.1, gamma=scale
  Why chosen: Non-linear without tree structure.
              Different inductive bias from RF and LR.
              Good comparison point on medium-sized datasets.
              Guide specifically requested SVM comparison.

  ALGORITHM 4 — Decision Tree Regressor
  ───────────────────────────────────────
  Type:       Single decision tree
  Parameters: max_depth=10, min_samples_split=5,
              min_samples_leaf=2, random_state=seed
  Why chosen: Interpretable — you can visualise the rules.
              Direct comparison to the RF ensemble.
              Shows value of ensembling over single tree.
              Guide specifically requested Decision Tree.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
5. EVALUATION METRICS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  MAE — Mean Absolute Error
    Formula:  Average of |actual Burn Rate - predicted Burn Rate|
    Range:    0.0 to 1.0 (same scale as Burn Rate)
    Example:  MAE=0.08 means predictions are off by 0.08 on average
    Goal:     Lower is better

  RMSE — Root Mean Squared Error
    Formula:  √(Average of (actual - predicted)²)
    Range:    0.0 to 1.0+
    Example:  RMSE=0.12 means large errors exist
    Note:     Penalises large errors more than MAE
    Goal:     Lower is better

  R² — R-Squared (Coefficient of Determination)
    Formula:  1 - (sum of squared residuals / total variance)
    Range:    -∞ to 1.0 (negative means worse than mean prediction)
    Example:  R²=0.65 means the model explains 65% of why
              some employees burn out more than others
    Goal:     Higher is better (1.0 = perfect prediction)

  SENSITIVITY ANALYSIS
    Each metric reported as: Mean ± Standard Deviation
    across 10 different random seeds.
    Small std = stable reliable results.
    Large std = model is sensitive to data splitting.

  SEEDS USED: 42, 7, 13, 21, 99, 55, 37, 81, 64, 28

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
6. FILE STRUCTURE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  burnoutsense_partA/
  ├── main.py                        ← RUN THIS to start everything
  ├── PROJECT_OVERVIEW.py            ← This file (print project info)
  ├── requirements.txt               ← pip install -r requirements.txt
  ├── literature_mapping_table.csv   ← Feature justification table
  │
  ├── ml/
  │   ├── preprocess.py              ← Load + clean Kaggle dataset
  │   │     Loads burnout_data.csv
  │   │     Engineers Tenure Months from Date of Joining
  │   │     Encodes Gender, Company Type, WFH Setup
  │   │     Produces X_with_mfs and X_without_mfs
  │   │     Target y = Burn Rate (continuous)
  │   │
  │   ├── experiments.py             ← Run A1 and A2 experiments
  │   │     sensitivity_analysis()  — 4 algorithms × 10 seeds
  │   │     compute_summary()       — mean ± std per algorithm
  │   │     cross_validate_best()   — 5-fold CV on all algorithms
  │   │     get_feature_importance() — RF feature rankings
  │   │     save_best_model()       — saves RF pkl files
  │   │
  │   └── burnout_data.csv          ← PUT KAGGLE CSV HERE
  │
  ├──   │         save_csvs()             — saves all result CSVs
  │         Terminal output              — prints experiment results directly
  │
  └── output/                       ← All results saved here
      ├── results_raw_a1.csv        ← All A1 seed-level results
      ├── results_raw_a2.csv        ← All A2 seed-level results
      ├── results_summary_a1.csv    ← A1 mean ± std per algorithm
      ├── results_summary_a2.csv    ← A2 mean ± std per algorithm
      ├── results_cv_a1.csv         ← A1 5-fold CV results
      ├── results_cv_a2.csv         ← A2 5-fold CV results
      ├── feature_importance_a1.csv ← A1 RF feature importances
      ├── feature_importance_a2.csv ← A2 RF feature importances
      ├── results_all_experiments.csv ← Combined all results
      ├── No PDF report is generated in terminal-only mode
      ├── rf_model_with_mfs.pkl     ← Trained RF model (A1)
      ├── rf_model_without_mfs.pkl  ← Trained RF model (A2)
      ├── scaler_with_mfs.pkl       ← Scaler for A1
      └── scaler_without_mfs.pkl    ← Scaler for A2

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
7. HOW TO RUN
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  STEP 1 — Install dependencies
    pip install -r requirements.txt

  STEP 2 — Download Kaggle dataset
    Go to: https://www.kaggle.com/datasets/blurredmachine/
           are-your-employees-burning-out
    Download the CSV file
    Save it as: ml/burnout_data.csv

  STEP 3 — Run the pipeline
    python main.py

  STEP 4 — View results
    Run: python main.py and read the terminal results
    Open: output/results_summary_a1.csv (in Excel)
    Open: output/results_summary_a2.csv (in Excel)

  OPTIONAL — Print this overview
    python PROJECT_OVERVIEW.py

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
8. WHAT THE GUIDE ASKED FOR — COMPLIANCE CHECK
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  ✅  Use real Kaggle data — NOT fabricated mapped features
  ✅  Predict Burn Rate as continuous regression (not classification)
  ✅  Report MAE, RMSE, R² (not just accuracy)
  ✅  Run with AND without Mental Fatigue Score
  ✅  Compare Random Forest, Linear Regression, SVR, Decision Tree
  ✅  Use multiple random seeds (10 seeds) for sensitivity analysis
  ✅  Report per-experiment mean ± std for all metrics
  ✅  Remove Task Completion Rate (target leakage)
  ✅  Remove cross-domain validation claims
  ✅  Remove 70% accuracy threshold claim
  ✅  Remove 100% synthetic accuracy as validity evidence
  ✅  Change privacy-preserving → privacy-aware
  ✅  Prepare literature mapping table for all 6 synthetic features
  ✅  5-fold cross validation on all algorithms
  ✅  Sensitivity analysis showing result stability across seeds

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
9. KEY RESEARCH FINDING (what to expect)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Experiment A1 (with MFS) will likely achieve R² ≈ 0.85-0.95
  Because Mental Fatigue Score strongly predicts Burn Rate.
  This is the CEILING. Not a novel finding.

  Experiment A2 (without MFS) will likely achieve R² ≈ 0.20-0.45
  This is the REAL FINDING.

  If A2 R² > 0.20:
    "Workplace structural variables — designation, resource
     allocation, WFH setup, company type, and tenure —
     contain genuine independent predictive information
     about employee burn rate, beyond what is explained
     by self-reported mental fatigue."

  The R² drop from A1 to A2 is also a finding:
    "Removing the self-reported psychological measure causes
     an R² drop of X, indicating that Y% of burnout predictability
     comes from objective workplace factors alone."

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
10. LITERATURE — KEY REFERENCES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Random Forest Algorithm:
    Breiman L. (2001). Random Forests. Machine Learning, 45, 5-32.
    https://link.springer.com/article/10.1023/A:1010933404324

  Burnout Definition (ICD-11):
    WHO (2019). Burn-out as occupational phenomenon.
    https://www.who.int/news/item/28-05-2019-burn-out-an-
    occupational-phenomenon-international-classification-of-diseases

  Long Working Hours and Health:
    Pega F et al. (2021). Global burdens attributable to long
    working hours. Environment International.
    https://www.sciencedirect.com/science/article/pii/S0160412021002208

  Social Jetlag:
    Roenneberg T et al. (2012). Social jetlag and obesity.
    Current Biology.
    https://www.cell.com/current-biology/fulltext/S0960-9822(12)00534-5

  Break Irregularity:
    Ariga A and Lleras A (2011). Brief mental breaks keep you
    focused. Cognition.
    https://www.sciencedirect.com/science/article/abs/pii/S0010027710002994

  Keystroke Dynamics:
    Bergadano et al. (2002). User authentication through keystroke
    dynamics. ACM TISSEC.
    https://dl.acm.org/doi/10.1145/581271.581272

  Burnout Measurement:
    Leiter and Maslach (2016). Burnout and engagement.
    Burnout Research.
    https://www.sciencedirect.com/science/article/pii/S2213058616300043

  Presenteeism:
    Hemp P (2004). Presenteeism: At work but out of it.
    Harvard Business Review.
    https://hbr.org/2004/10/presenteeism-at-work-but-out-of-it

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""


if __name__ == "__main__":
    print(OVERVIEW)
