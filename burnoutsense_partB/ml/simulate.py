#!/usr/bin/env python3
"""
BurnoutSense AI — Part B
File: ml/simulate.py

Generates simulated behavioral data for pipeline demonstration.

IMPORTANT ACADEMIC NOTE:
  This simulated data is used ONLY to:
    - Demonstrate the pipeline runs end to end
    - Show the dashboard updates in real time
    - Test system scalability under load
    - Show the alert system triggers correctly

  It is NOT used for any ML accuracy claims.
  All genuine ML results come from Part A (Kaggle dataset).

  The Random Forest model loaded here was trained in Part A.
  If Part A model is not available, a simple rule-based
  fallback scorer is used for demonstration purposes.
"""

import numpy as np
import random
import os
import joblib
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

np.random.seed(42)

FEATURES = [
    "typing_speed_variation",
    "idle_time_pct",
    "session_duration_hrs",
    "break_irregularity_index",
    "work_hour_deviation",
    "task_completion_rate",
]

# ── Load Part A model if available ───────────────────────────────────────────
_model  = None
_scaler = None

def _load_part_a_model():
    global _model, _scaler
    model_path  = os.path.join(
        os.path.dirname(__file__),
        "../../burnoutsense_partA/output/rf_model_without_mfs.pkl"
    )
    scaler_path = os.path.join(
        os.path.dirname(__file__),
        "../../burnoutsense_partA/output/scaler_without_mfs.pkl"
    )
    try:
        _model  = joblib.load(model_path)
        _scaler = joblib.load(scaler_path)
        logger.info("Part A model loaded for pipeline demonstration")
    except FileNotFoundError:
        logger.warning(
            "Part A model not found. Using rule-based fallback scorer.\n"
            "Run Part A first (python burnoutsense_partA/main.py) "
            "to use the trained model."
        )

_load_part_a_model()


# ── Rule-based fallback scorer ────────────────────────────────────────────────
def _rule_based_score(features: dict) -> dict:
    """
    Simple rule-based burnout scorer for demonstration when
    Part A model is not available.
    NOT used for any academic accuracy claims.
    """
    score = int(
        features["idle_time_pct"]             * 0.28 +
        features["session_duration_hrs"]      * 4.5  +
        features["break_irregularity_index"]  * 4.8  +
        features["work_hour_deviation"]       * 5.5  +
        (100 - features["task_completion_rate"]) * 0.14 +
        features["typing_speed_variation"]    * 0.18
    )
    score = max(0, min(100, score))
    level = "High" if score >= 70 else "Medium" if score >= 45 else "Low"
    return {"score": score, "level": level, "confidence": 0.75,
            "source": "rule_based_fallback"}


# ── Main prediction function ───────────────────────────────────────────────────
def predict_burnout(features: dict) -> dict:
    """
    Predict burnout risk from feature dict.
    Uses Part A Random Forest model if available,
    otherwise falls back to rule-based scorer.
    """
    import numpy as np

    if _model is not None and _scaler is not None:
        # Use Part A features (excludes Mental Fatigue Score)
        feature_order = [
            "typing_speed_variation",
            "idle_time_pct",
            "session_duration_hrs",
            "break_irregularity_index",
            "work_hour_deviation",
            "task_completion_rate",
        ]
        X = np.array([[features.get(f, 0) for f in feature_order]])
        X_scaled = _scaler.transform(X)
        burn_rate = float(_model.predict(X_scaled)[0])
        burn_rate = max(0.0, min(1.0, burn_rate))
        score = int(burn_rate * 100)
        level = "High" if burn_rate >= 0.70 else \
                "Medium" if burn_rate >= 0.40 else "Low"
        return {
            "score"        : score,
            "level"        : level,
            "burn_rate"    : round(burn_rate, 4),
            "confidence"   : 0.88,
            "source"       : "part_a_random_forest",
        }
    else:
        return _rule_based_score(features)


# ── Simulated employee generator ──────────────────────────────────────────────
def generate_simulated_employee(employee_id: str,
                                 risk_profile: str = "random") -> dict:
    """
    Generate one simulated behavioral data record.
    risk_profile: 'low', 'medium', 'high', or 'random'

    NOTE: These distributions are for pipeline demonstration only.
    They are NOT used for any academic accuracy claims.
    """
    if risk_profile == "random":
        risk_profile = random.choices(
            ["low", "medium", "high"],
            weights=[60, 27, 13]
        )[0]

    if risk_profile == "low":
        features = {
            "typing_speed_variation"  : max(0, random.gauss(38, 12)),
            "idle_time_pct"           : random.uniform(5, 28),
            "session_duration_hrs"    : random.uniform(4.5, 7.5),
            "break_irregularity_index": random.uniform(0.5, 3.0),
            "work_hour_deviation"     : random.uniform(0, 1.5),
            "task_completion_rate"    : random.uniform(68, 100),
        }
    elif risk_profile == "medium":
        features = {
            "typing_speed_variation"  : max(0, random.gauss(24, 10)),
            "idle_time_pct"           : random.uniform(28, 58),
            "session_duration_hrs"    : random.uniform(7.5, 10.5),
            "break_irregularity_index": random.uniform(3.0, 6.5),
            "work_hour_deviation"     : random.uniform(1.5, 3.5),
            "task_completion_rate"    : random.uniform(38, 68),
        }
    else:  # high
        features = {
            "typing_speed_variation"  : max(0, random.gauss(14, 8)),
            "idle_time_pct"           : random.uniform(58, 92),
            "session_duration_hrs"    : random.uniform(10.5, 15.5),
            "break_irregularity_index": random.uniform(6.5, 10.0),
            "work_hour_deviation"     : random.uniform(3.5, 6.0),
            "task_completion_rate"    : random.uniform(8, 38),
        }

    # Round all values
    features = {k: round(v, 2) for k, v in features.items()}
    features["employee_id"] = employee_id
    features["timestamp"]   = datetime.utcnow().isoformat()

    return features


def generate_team_snapshot(n_employees: int = 50) -> list:
    """Generate a snapshot of simulated data for n employees."""
    employees = []
    names = [
        "Alice Johnson", "Bob Smith", "Carol White", "David Brown",
        "Eva Martinez", "Frank Lee", "Grace Kim", "Henry Davis",
        "Iris Chen", "James Wilson", "Karen Moore", "Liam Taylor",
        "Maya Anderson", "Noah Thomas", "Olivia Jackson", "Peter Harris",
        "Quinn Martin", "Rachel Thompson", "Sam Garcia", "Tina Robinson",
    ]
    depts = ["Engineering", "Product", "Design", "Marketing",
             "Sales", "HR", "Data Analytics", "Customer Success"]

    for i in range(n_employees):
        emp_id   = f"emp_{i+1:03d}"
        name     = names[i % len(names)]
        dept     = depts[i % len(depts)]
        features = generate_simulated_employee(emp_id)
        result   = predict_burnout(features)

        employees.append({
            "employee_id"  : emp_id,
            "name"         : name,
            "department"   : dept,
            "features"     : features,
            "risk_level"   : result["level"],
            "burn_rate"    : result.get("burn_rate", result["score"] / 100),
            "score"        : result["score"],
            "confidence"   : result["confidence"],
            "timestamp"    : features["timestamp"],
        })

    return employees


if __name__ == "__main__":
    print("Generating team snapshot (50 employees)...")
    team = generate_team_snapshot(50)
    levels = {"Low": 0, "Medium": 0, "High": 0}
    for emp in team:
        levels[emp["risk_level"]] += 1
    print(f"\nTeam distribution:")
    for level, count in levels.items():
        print(f"  {level:8s}: {count} employees ({count/50*100:.1f}%)")

    print("\nSample record:")
    import json
    print(json.dumps(team[0], indent=2))
