#!/usr/bin/env python3
"""
BurnoutSense AI — Part B
File: tests/scalability_test.py

Scalability testing using Locust.
Tests API performance at 50, 100, 250, 500 concurrent agents.

Usage:
  pip install locust
  locust -f tests/scalability_test.py --host=http://localhost:5000

  Or headless:
  locust -f tests/scalability_test.py --host=http://localhost:5000 \
         --users 100 --spawn-rate 10 --run-time 60s --headless

Metrics collected:
  - API response time (min, max, mean, p50, p95, p99)
  - Requests per second throughput
  - Failure rate
  - MongoDB write throughput
"""

import random
import json
from locust import HttpUser, task, between, events
from datetime import datetime


# ── Simulated payload generator ───────────────────────────────────────────────
def make_payload(employee_id: str) -> dict:
    """Generate a realistic simulated behavioral payload."""
    risk = random.choices(["low", "medium", "high"], weights=[60, 27, 13])[0]

    if risk == "low":
        return {
            "employee_id"             : employee_id,
            "timestamp"               : datetime.utcnow().isoformat(),
            "typing_speed_variation"  : round(random.gauss(38, 12), 2),
            "idle_time_pct"           : round(random.uniform(5, 28), 2),
            "session_duration_hrs"    : round(random.uniform(4.5, 7.5), 2),
            "break_irregularity_index": round(random.uniform(0.5, 3.0), 2),
            "work_hour_deviation"     : round(random.uniform(0, 1.5), 2),
            "task_completion_rate"    : round(random.uniform(68, 100), 2),
        }
    elif risk == "medium":
        return {
            "employee_id"             : employee_id,
            "timestamp"               : datetime.utcnow().isoformat(),
            "typing_speed_variation"  : round(random.gauss(24, 10), 2),
            "idle_time_pct"           : round(random.uniform(28, 58), 2),
            "session_duration_hrs"    : round(random.uniform(7.5, 10.5), 2),
            "break_irregularity_index": round(random.uniform(3.0, 6.5), 2),
            "work_hour_deviation"     : round(random.uniform(1.5, 3.5), 2),
            "task_completion_rate"    : round(random.uniform(38, 68), 2),
        }
    else:
        return {
            "employee_id"             : employee_id,
            "timestamp"               : datetime.utcnow().isoformat(),
            "typing_speed_variation"  : round(max(0, random.gauss(14, 8)), 2),
            "idle_time_pct"           : round(random.uniform(58, 92), 2),
            "session_duration_hrs"    : round(random.uniform(10.5, 15.5), 2),
            "break_irregularity_index": round(random.uniform(6.5, 10.0), 2),
            "work_hour_deviation"     : round(random.uniform(3.5, 6.0), 2),
            "task_completion_rate"    : round(random.uniform(8, 38), 2),
        }


# ── Locust User ───────────────────────────────────────────────────────────────
class BurnoutAgentUser(HttpUser):
    """
    Simulates one monitoring agent sending data every 25-35 seconds.
    Represents one remote employee's agent.
    """
    wait_time = between(25, 35)

    def on_start(self):
        """Called when a simulated user starts. Get auth token."""
        self.employee_id = f"load_emp_{random.randint(1000, 9999)}"
        resp = self.client.post(
            "/api/v1/auth/token",
            json={"employee_id": self.employee_id, "role": "employee"},
            name="/api/v1/auth/token",
        )
        if resp.status_code == 200:
            self.token = resp.json().get("token", "test-token")
        else:
            self.token = "test-token"

    def _headers(self) -> dict:
        return {
            "Authorization" : f"Bearer {self.token}",
            "Content-Type"  : "application/json",
        }

    @task(10)
    def log_activity(self):
        """Primary task — send behavioral data (most frequent)."""
        payload = make_payload(self.employee_id)
        self.client.post(
            "/api/v1/activity/log",
            json=payload,
            headers=self._headers(),
            name="/api/v1/activity/log",
        )

    @task(3)
    def get_own_score(self):
        """Check own burnout score (less frequent)."""
        self.client.get(
            f"/api/v1/burnout/score/{self.employee_id}",
            headers=self._headers(),
            name="/api/v1/burnout/score/:id",
        )

    @task(2)
    def predict_only(self):
        """Run prediction without logging (dashboard test)."""
        payload = make_payload(self.employee_id)
        del payload["employee_id"]
        del payload["timestamp"]
        self.client.post(
            "/api/v1/burnout/predict",
            json=payload,
            name="/api/v1/burnout/predict",
        )

    @task(1)
    def health_check(self):
        """Health check (least frequent)."""
        self.client.get("/health", name="/health")


class ManagerUser(HttpUser):
    """
    Simulates a manager checking the dashboard every 30-60 seconds.
    Lower frequency than agent users.
    """
    wait_time = between(30, 60)

    def on_start(self):
        self.manager_id = f"manager_{random.randint(100, 999)}"
        resp = self.client.post(
            "/api/v1/auth/token",
            json={"employee_id": self.manager_id, "role": "manager"},
        )
        self.token = resp.json().get("token", "test-token") \
            if resp.status_code == 200 else "test-token"

    def _headers(self):
        return {"Authorization": f"Bearer {self.token}"}

    @task(5)
    def view_dashboard(self):
        self.client.get(
            "/api/v1/dashboard/summary",
            headers=self._headers(),
            name="/api/v1/dashboard/summary",
        )

    @task(3)
    def view_alerts(self):
        self.client.get(
            "/api/v1/alerts/active",
            headers=self._headers(),
            name="/api/v1/alerts/active",
        )

    @task(2)
    def list_employees(self):
        self.client.get(
            "/api/v1/employees/list?page=1&per_page=50",
            headers=self._headers(),
            name="/api/v1/employees/list",
        )


# ── Custom report ─────────────────────────────────────────────────────────────
@events.quitting.add_listener
def on_quit(environment, **kwargs):
    """Print a summary when the test ends."""
    stats = environment.runner.stats
    print("\n" + "="*60)
    print("  SCALABILITY TEST RESULTS")
    print("="*60)
    for name, entry in stats.entries.items():
        if entry.num_requests == 0:
            continue
        print(f"\n  Endpoint: {name[1]}")
        print(f"    Requests     : {entry.num_requests}")
        print(f"    Failures     : {entry.num_failures}")
        print(f"    Failure Rate : {entry.fail_ratio*100:.1f}%")
        print(f"    Avg Response : {entry.avg_response_time:.0f} ms")
        print(f"    p50 Response : {entry.get_response_time_percentile(0.50):.0f} ms")
        print(f"    p95 Response : {entry.get_response_time_percentile(0.95):.0f} ms")
        print(f"    p99 Response : {entry.get_response_time_percentile(0.99):.0f} ms")
        print(f"    RPS          : {entry.total_rps:.2f}")
    print("\n" + "="*60)
