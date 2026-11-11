#!/usr/bin/env python3
"""
BurnoutSense AI — Part B
File: tests/test_api.py

Unit tests for the Flask REST API.
Run with: pytest tests/test_api.py -v
"""

import pytest
import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../backend"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../ml"))

from app import app, generate_token


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


@pytest.fixture
def employee_token():
    return generate_token("emp_test_001", "employee")


@pytest.fixture
def manager_token():
    return generate_token("mgr_test_001", "manager")


@pytest.fixture
def employee_headers(employee_token):
    return {
        "Authorization": f"Bearer {employee_token}",
        "Content-Type" : "application/json",
    }


@pytest.fixture
def manager_headers(manager_token):
    return {
        "Authorization": f"Bearer {manager_token}",
        "Content-Type" : "application/json",
    }


def make_payload(employee_id="emp_test_001"):
    return {
        "employee_id"             : employee_id,
        "typing_speed_variation"  : 28.4,
        "idle_time_pct"           : 43.2,
        "session_duration_hrs"    : 6.5,
        "break_irregularity_index": 3.2,
        "work_hour_deviation"     : 1.5,
        "task_completion_rate"    : 72.0,
    }


# ── Health ────────────────────────────────────────────────────────────────────


# ── Auth ──────────────────────────────────────────────────────────────────────
class TestAuth:
    def test_get_token_returns_200(self, client):
        resp = client.post(
            "/api/v1/auth/token",
            json={"employee_id": "emp_001", "role": "employee"},
        )
        assert resp.status_code == 200

    def test_get_token_returns_token(self, client):
        data = json.loads(client.post(
            "/api/v1/auth/token",
            json={"employee_id": "emp_001", "role": "employee"},
        ).data)
        assert "token" in data
        assert len(data["token"]) > 20

    def test_unauthenticated_request_rejected(self, client):
        resp = client.post(
            "/api/v1/activity/log",
            json=make_payload(),
            content_type="application/json",
        )
        assert resp.status_code == 401

    def test_invalid_token_rejected(self, client):
        resp = client.post(
            "/api/v1/activity/log",
            json=make_payload(),
            headers={"Authorization": "Bearer invalid.token.here"},
            content_type="application/json",
        )
        assert resp.status_code == 401


# ── Activity Logging ──────────────────────────────────────────────────────────
class TestActivityLog:
    def test_valid_payload_accepted(self, client, employee_headers):
        resp = client.post(
            "/api/v1/activity/log",
            json=make_payload(),
            headers=employee_headers,
        )
        assert resp.status_code == 201

    def test_response_contains_risk(self, client, employee_headers):
        data = json.loads(client.post(
            "/api/v1/activity/log",
            json=make_payload(),
            headers=employee_headers,
        ).data)
        assert "risk" in data
        assert "level" in data["risk"]
        assert "score" in data["risk"]
        assert data["risk"]["level"] in ("Low", "Medium", "High")

    def test_score_in_valid_range(self, client, employee_headers):
        data = json.loads(client.post(
            "/api/v1/activity/log",
            json=make_payload(),
            headers=employee_headers,
        ).data)
        assert 0 <= data["risk"]["score"] <= 100

    def test_missing_employee_id_rejected(self, client, employee_headers):
        payload = make_payload()
        del payload["employee_id"]
        resp = client.post(
            "/api/v1/activity/log",
            json=payload,
            headers=employee_headers,
        )
        assert resp.status_code == 400

    def test_missing_required_fields_rejected(self, client, employee_headers):
        resp = client.post(
            "/api/v1/activity/log",
            json={"employee_id": "emp_001"},
            headers=employee_headers,
        )
        assert resp.status_code == 400

    def test_empty_body_rejected(self, client, employee_headers):
        resp = client.post(
            "/api/v1/activity/log",
            json=None,
            headers=employee_headers,
            content_type="application/json",
        )
        assert resp.status_code in (400, 415)


# ── Prediction ────────────────────────────────────────────────────────────────
class TestPrediction:
    def test_predict_returns_200(self, client):
        resp = client.post(
            "/api/v1/burnout/predict",
            json={
                "typing_speed_variation"  : 35.0,
                "idle_time_pct"           : 72.0,
                "session_duration_hrs"    : 11.5,
                "break_irregularity_index": 8.2,
                "work_hour_deviation"     : 3.0,
                "task_completion_rate"    : 42.0,
            },
            content_type="application/json",
        )
        assert resp.status_code == 200

    def test_predict_high_risk_features(self, client):
        data = json.loads(client.post(
            "/api/v1/burnout/predict",
            json={
                "typing_speed_variation"  : 45.0,
                "idle_time_pct"           : 85.0,
                "session_duration_hrs"    : 14.0,
                "break_irregularity_index": 9.5,
                "work_hour_deviation"     : 5.5,
                "task_completion_rate"    : 12.0,
            },
            content_type="application/json",
        ).data)
        assert data["level"] in ("High", "Medium")

    def test_predict_low_risk_features(self, client):
        data = json.loads(client.post(
            "/api/v1/burnout/predict",
            json={
                "typing_speed_variation"  : 8.0,
                "idle_time_pct"           : 12.0,
                "session_duration_hrs"    : 5.5,
                "break_irregularity_index": 1.2,
                "work_hour_deviation"     : 0.5,
                "task_completion_rate"    : 92.0,
            },
            content_type="application/json",
        ).data)
        assert data["level"] in ("Low", "Medium")

    def test_predict_contains_disclaimer(self, client):
        data = json.loads(client.post(
            "/api/v1/burnout/predict",
            json={"idle_time_pct": 50},
            content_type="application/json",
        ).data)
        assert "disclaimer" in data


