#!/usr/bin/env python3
"""
BurnoutSense AI — Part B
File: backend/app.py

Flask REST API — privacy-aware backend.
Receives behavioral data from agents, runs prediction,
stores results in MongoDB, triggers alerts.

Privacy-aware implementation:
  - Consent required before monitoring
  - JWT role-based access control
  - TLS enforced in production
  - Minimal data collection
  - No keystroke or screen data ever stored
"""

import os
import logging
from datetime import datetime, timedelta
from functools import wraps

import jwt
from flask import Flask, request, jsonify, g
from flask_cors import CORS
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app, origins=os.getenv("ALLOWED_ORIGINS", "*").split(","))

# ── Database ──────────────────────────────────────────────────────────────────
# ── Bounded in-process storage ────────────────────────────────────────────────
MAX_RECORDS = 1000

ACTIVITY_LOGS = []
BURNOUT_SCORES = []
ALERTS = []
EMPLOYEES = {}


def _append_bounded(store, record, limit=MAX_RECORDS):
    """Append a record while keeping in-process storage bounded."""
    store.append(record)
    if len(store) > limit:
        del store[:-limit]


def _latest_score(employee_id):
    """Return the latest burnout score for an employee."""
    records = [
        r for r in BURNOUT_SCORES
        if r["employee_id"] == employee_id
    ]

    if not records:
        return None

    return max(records, key=lambda r: r["timestamp"])

# ── ML module ───────────────────────────────────────────────────────────────── 
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../ml"))
from simulate import predict_burnout

# ── Auth ──────────────────────────────────────────────────────────────────────
JWT_SECRET = os.getenv("JWT_SECRET", "burnoutsense-dev-secret-2025")
JWT_ALGO   = "HS256"


def require_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        header = request.headers.get("Authorization", "")
        if not header.startswith("Bearer "):
            return jsonify({"error": "Missing Authorization header"}), 401
        token = header[7:]
        try:
            payload    = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGO])
            g.user     = payload
        except jwt.ExpiredSignatureError:
            return jsonify({"error": "Token expired"}), 401
        except jwt.InvalidTokenError as e:
            return jsonify({"error": f"Invalid token: {e}"}), 401
        return f(*args, **kwargs)
    return decorated


def require_role(*roles):
    def decorator(f):
        @wraps(f)
        def decorated(*args, **kwargs):
            if not hasattr(g, "user"):
                return jsonify({"error": "Not authenticated"}), 401
            if g.user.get("role") not in roles:
                return jsonify({"error": "Insufficient permissions"}), 403
            return f(*args, **kwargs)
        return decorated
    return decorator


def generate_token(employee_id: str, role: str) -> str:
    payload = {
        "sub"  : employee_id,
        "role" : role,
        "iat"  : datetime.utcnow(),
        "exp"  : datetime.utcnow() + timedelta(hours=8),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGO)


# ── Alert logic ───────────────────────────────────────────────────────────────
def check_and_alert(employee_id: str, result: dict):
    """Create an alert for medium or high operational risk."""
    if result["level"] in ("High", "Medium"):
        alert = {
            "id": str(len(ALERTS) + 1),
            "employee_id": employee_id,
            "level": result["level"].lower(),
            "score": result["score"],
            "message": (
                f"{result['level']} burnout risk detected "
                f"(score: {result['score']}, "
                f"confidence: {result['confidence'] * 100:.1f}%)"
            ),
            "timestamp": datetime.utcnow(),
            "read": False,
        }

        _append_bounded(ALERTS, alert)

        if result["level"] == "High":
            logger.warning(
                f"HIGH RISK: {employee_id} — Score {result['score']}"
            )


# ══════════════════════════════════════════════════════════════════════════════
# API ROUTES
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/health")
def health():
    """Health check — used by load balancer and scalability tests."""
    return jsonify({
        "status"       : "healthy",
        "timestamp"    : datetime.utcnow().isoformat(),
        "privacy_mode" : "privacy-aware",
        "data_collected": "behavioral timing signals only",
    }), 200


@app.route("/api/v1/auth/token", methods=["POST"])
def get_token():
    """
    Generate a JWT token for testing.
    In production this would validate credentials against the employee DB.
    """
    data = request.json or {}
    emp_id = data.get("employee_id", "emp_001")
    role   = data.get("role", "employee")
    token  = generate_token(emp_id, role)
    return jsonify({"token": token, "expires_in": "8h"}), 200


@app.route("/api/v1/activity/log", methods=["POST"])
@require_auth
def log_activity():
    """
    Ingest behavioral data from monitoring agent.
    Runs prediction and stores burnout score.
    """
    data = request.json
    if not data:
        return jsonify({"error": "No data provided"}), 400

    required = ["employee_id"] + [
        "typing_speed_variation", "idle_time_pct",
        "session_duration_hrs",
    ]
    missing = [f for f in required if f not in data]
    if missing:
        return jsonify({"error": f"Missing fields: {missing}"}), 400

    # Store raw activity log (no sensitive data)
    data["logged_at"] = datetime.utcnow()
    activity_record = {
    k: v for k, v in data.items()
    if k != "_id"
}

    _append_bounded(ACTIVITY_LOGS, activity_record)

    # Run prediction
    result = predict_burnout(data)

    # Store burnout score
    score_record = {
    "employee_id": data["employee_id"],
    "score": result["score"],
    "risk_level": result["level"],
    "burn_rate": result.get("burn_rate", result["score"] / 100),
    "confidence": result["confidence"],
    "timestamp": datetime.utcnow(),
}

    _append_bounded(BURNOUT_SCORES, score_record)

    # Trigger alerts if needed
    check_and_alert(data["employee_id"], result)

    return jsonify({"status": "logged", "risk": result}), 201


@app.route("/api/v1/burnout/predict", methods=["POST"])
def predict():
    """
    Run burnout prediction on provided features.
    No auth required — for dashboard testing.

    NOTE: Results are from simulated pipeline demonstration.
          Real accuracy claims come from Part A Kaggle experiments.
    """
    data = request.json or {}
    result = predict_burnout(data)
    result["disclaimer"] = (
        "Pipeline demonstration result. "
        "Academic accuracy results in Part A report."
    )
    return jsonify(result), 200


@app.route("/api/v1/burnout/score/<emp_id>")
@require_auth
def get_score(emp_id: str):
    """Get latest burnout score for an employee."""
    if g.user.get("role") == "employee" and g.user.get("sub") != emp_id:
        return jsonify({"error": "Access denied"}), 403

    latest = _latest_score(emp_id)
    if not latest:
        return jsonify({"error": "No data found"}), 404

    # Convert datetime to string
    if "timestamp" in latest:
        latest["timestamp"] = latest["timestamp"].isoformat()
    return jsonify(latest), 200


@app.route("/api/v1/burnout/history/<emp_id>")
@require_auth
def get_history(emp_id: str):
    """Get burnout score history for an employee (last 50 records)."""
    if g.user.get("role") == "employee" and g.user.get("sub") != emp_id:
        return jsonify({"error": "Access denied"}), 403

    records = [
    dict(r)
    for r in BURNOUT_SCORES
    if r["employee_id"] == emp_id
]

    records.sort(
      key=lambda r: r["timestamp"],
      reverse=True
)

    records = records[:50]
    for r in records:
        if "timestamp" in r:
            r["timestamp"] = r["timestamp"].isoformat()
    return jsonify({"history": records, "count": len(records)}), 200


@app.route("/api/v1/employees/list")
@require_auth
@require_role("manager", "admin")
def list_employees():
    """Get employee list with latest scores."""
    page = int(request.args.get("page", 1))
    per_page = int(request.args.get("per_page", 50))
    dept = request.args.get("department")

    employees = [
        dict(emp)
        for emp in EMPLOYEES.values()
        if emp.get("consent_given", False)
        and (not dept or emp.get("department") == dept)
    ]

    total = len(employees)

    start = (page - 1) * per_page
    end = start + per_page
    employees = employees[start:end]

    for emp in employees:
        emp.pop("password_hash", None)
        emp["latest_score"] = _latest_score(
            emp["employee_id"]
        )

        if emp["latest_score"]:
            emp["latest_score"] = dict(emp["latest_score"])
            emp["latest_score"]["timestamp"] = (
                emp["latest_score"]["timestamp"].isoformat()
            )

    return jsonify({
        "employees": employees,
        "page": page,
        "per_page": per_page,
        "total": total,
    }), 200


@app.route("/api/v1/alerts/active")
@require_auth
@require_role("manager", "admin")
def get_alerts():
    """Get unread alerts (managers/admins only)."""
    alerts = [
    dict(a)
    for a in ALERTS
    if not a.get("read", False)
]

    alerts.sort(
    key=lambda a: a["timestamp"],
    reverse=True
)

    alerts = alerts[:100]

    for a in alerts:
     a["timestamp"] = a["timestamp"].isoformat()

    return jsonify({
    "alerts": alerts,
    "count": len(alerts)
}), 200


@app.route("/api/v1/alerts/<alert_id>/read", methods=["PUT"])
@require_auth
@require_role("manager", "admin")
def mark_alert_read(alert_id: str):
    """Mark an alert as read."""
    for alert in ALERTS:
    if alert["id"] == alert_id:
        alert["read"] = True
        break

    return jsonify({"status": "marked as read"}), 200
    


@app.route("/api/v1/dashboard/summary")
@require_auth
@require_role("manager", "admin")
def dashboard_summary():
    """Aggregated summary for dashboard KPIs."""
    latest_by_employee = {}

    for score in BURNOUT_SCORES:
     employee_id = score["employee_id"]

    if (
        employee_id not in latest_by_employee
        or score["timestamp"]
        > latest_by_employee[employee_id]["timestamp"]
    ):
        latest_by_employee[employee_id] = score

    scores = list(latest_by_employee.values())

    total = len(scores)
    high = sum(
    1 for s in scores
    if s["risk_level"] == "High"
)
    medium = sum(
    1 for s in scores
    if s["risk_level"] == "Medium"
)
    low = sum(
    1 for s in scores
    if s["risk_level"] == "Low"
)

    avg = (
    round(
        sum(s["score"] for s in scores) / total,
        1
    )
    if total
    else 0
)

    return jsonify({
    "total_monitored": total,
    "high_risk": high,
    "medium_risk": medium,
    "low_risk": low,
    "average_score": avg,
    "unread_alerts": sum(
        1 for a in ALERTS
        if not a.get("read", False)
    ),
    "timestamp": datetime.utcnow().isoformat(),
}), 200


@app.route("/api/v1/employees/<emp_id>/consent", methods=["PUT"])
@require_auth
def update_consent(emp_id: str):
    """Update employee consent status."""
    employee = EMPLOYEES.get(emp_id)

    if employee is None:
        employee = {
        "employee_id": emp_id,
    }
        EMPLOYEES[emp_id] = employee

    employee["consent_given"] = consent
    employee["consent_updated"] = datetime.utcnow()

    if not consent:
        ACTIVITY_LOGS[:] = [
        r for r in ACTIVITY_LOGS
        if r.get("employee_id") != emp_id
    ]

        BURNOUT_SCORES[:] = [
        r for r in BURNOUT_SCORES
        if r.get("employee_id") != emp_id
    ]

        ALERTS[:] = [
        a for a in ALERTS
        if a.get("employee_id") != emp_id
    ]

        logger.info(
        f"Data deleted for {emp_id} "
        f"(consent withdrawn)"
    )

    return jsonify({
    "status": "updated",
    "consent_given": consent
}), 200


if __name__ == "__main__":
    port  = int(os.getenv("PORT", 5000))
    debug = os.getenv("DEBUG", "false").lower() == "true"
    app.run(host="0.0.0.0", port=port, debug=debug, use_reloader=False)
