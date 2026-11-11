#!/usr/bin/env python3
"""
BurnoutSense AI — Part B
File: client/agent.py

Privacy-aware monitoring agent.
Collects 6 behavioral metrics ONLY — no keystrokes, no screen content.
Sends data to Flask API every 30 seconds (configurable).

IMPORTANT: This agent uses SIMULATED data for pipeline demonstration.
           It does NOT make real-world burnout accuracy claims.
           Real deployment would require consenting employees.
"""

import time
import os
import random
import requests
from datetime import datetime
from threading import Thread
import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


try:
    from pynput import keyboard, mouse
    PYNPUT_AVAILABLE = True
except Exception:
    PYNPUT_AVAILABLE = False
    logger.warning("pynput not available — running in simulation mode")


class BurnoutAgent:
    """
    Privacy-aware behavioral monitoring agent.
    Captures timing patterns only. Never captures content.
    """

    def __init__(self, employee_id: str, api_url: str,
                 token: str, interval: int = 30):
        self.employee_id  = employee_id
        self.api_url      = api_url
        self.token        = token
        self.interval     = interval
        self.session_start = time.time()
        self.last_activity = time.time()
        self.last_key_time = None
        self.typing_speeds = []
        self.breaks        = []
        self.mouse_moves   = 0
        logger.info(f"Agent initialised for: {employee_id}")
        logger.info(f"Tracking: {'Real (pynput)' if PYNPUT_AVAILABLE else 'Simulated'}")

    # ── Real listeners ──────────────────────────────────────────────────────
    def on_key_press(self, key):
      """Records timing only — never the key content."""
      now = time.time()

      # Check for prolonged inactivity using overall activity
      # from both keyboard and mouse.
      inactivity_gap = now - self.last_activity
      if inactivity_gap > 300:
        self.breaks.append(inactivity_gap)

      self.last_activity = now

      # Calculate typing speed from keyboard timing only.
      if self.last_key_time:
        gap = now - self.last_key_time

        if gap < 2.0:
            wpm = min(200, 60.0 / (gap * 5.0))
            self.typing_speeds.append(wpm)

      self.last_key_time = now

    def on_mouse_move(self, x, y):
        self.last_activity = time.time()
        self.mouse_moves += 1

    # ── Feature computation ─────────────────────────────────────────────────
    def _typing_variation(self) -> float:
        recent = self.typing_speeds[-200:] if self.typing_speeds else []
        if len(recent) >= 5:
            return round(max(recent) - min(recent), 2)
        # Simulation fallback
        hrs = self._session_hrs()
        base = max(5, 40 - hrs * 3)
        return round(random.gauss(base, 8), 2)

    def _idle_pct(self) -> float:
        now  = time.time()
        idle = now - self.last_activity
        sess = now - self.session_start
        if sess < 10:
            return 0.0
        return round(min(idle / sess * 100, 98), 1)

    def _session_hrs(self) -> float:
        return round((time.time() - self.session_start) / 3600, 2)

    def _break_irregularity(self) -> float:
        if len(self.breaks) >= 2:
            diffs = [abs(self.breaks[i] - self.breaks[i-1])
                     for i in range(1, len(self.breaks))]
            return round(min(10.0, sum(diffs) / len(diffs) / 60), 2)
        hrs = self._session_hrs()
        return round(min(10.0, hrs * 0.8 + random.uniform(0, 1)), 2)

    def _work_hour_deviation(self) -> float:
        return round(abs(datetime.now().hour - 9) + random.uniform(0, 0.5), 2)

    def _task_completion(self) -> float:
        """
        Simulated task completion rate.
        NOTE: In Kaggle experiments this feature was REMOVED due to
        target leakage. Here it is used ONLY for pipeline demonstration.
        """
        hrs  = self._session_hrs()
        rate = 90 - (hrs * 6) + random.uniform(-8, 8)
        return round(max(10.0, min(100.0, rate)), 1)

    def compute_features(self) -> dict:
        return {
            "employee_id"            : self.employee_id,
            "timestamp"              : datetime.utcnow().isoformat(),
            "typing_speed_variation" : self._typing_variation(),
            "idle_time_pct"          : self._idle_pct(),
            "session_duration_hrs"   : self._session_hrs(),
            "break_irregularity_index": self._break_irregularity(),
            "work_hour_deviation"    : self._work_hour_deviation(),
            "task_completion_rate"   : self._task_completion(),
        }

    # ── Send ────────────────────────────────────────────────────────────────
    def send_data(self):
        try:
            payload = self.compute_features()
            print(f"\n--- [{datetime.now().strftime('%H:%M:%S')}] Sending ---")
            for k, v in payload.items():
                if k not in ("employee_id", "timestamp"):
                    print(f"  {k:<32}: {v}")
            resp = requests.post(
                f"{self.api_url}/api/v1/activity/log",
                json=payload,
                headers={"Authorization": f"Bearer {self.token}"},
                timeout=10,
            )
            if resp.status_code == 201:
                r = resp.json()
                level = r.get("risk", {}).get("level", "N/A")
                score = r.get("risk", {}).get("score", "N/A")
                conf  = r.get("risk", {}).get("confidence", 0)
                icons = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}
                print(f"  RESULT: {icons.get(level,'')} {level} | "
                      f"Score={score}/100 | Confidence={conf*100:.1f}%")
            else:
                logger.warning(f"API returned {resp.status_code}")
        except Exception as e:
            logger.error(f"Send failed: {e}")

    def _send_loop(self):
        while True:
            time.sleep(self.interval)
            self.send_data()

    def run(self):
        logger.info("BurnoutSense Agent starting...")
        sender = Thread(target=self._send_loop, daemon=True)
        sender.start()

        if PYNPUT_AVAILABLE:
            logger.info("Real keyboard + mouse tracking active")
            logger.info("Press Ctrl+C to stop\n")
            with keyboard.Listener(on_press=self.on_key_press), \
                 mouse.Listener(on_move=self.on_mouse_move):
                try:
                    while True:
                        time.sleep(1)
                except KeyboardInterrupt:
                    logger.info("Agent stopped.")
        else:
            logger.info("Simulation mode — Press Ctrl+C to stop\n")
            try:
                while True:
                    time.sleep(1)
                    if random.random() < 0.3:
                        self.last_activity = time.time()
            except KeyboardInterrupt:
                logger.info("Agent stopped.")


if __name__ == "__main__":
    agent = BurnoutAgent(
        employee_id=os.getenv("EMPLOYEE_ID", "emp_001"),
        api_url    =os.getenv("API_URL",     "http://localhost:5000"),
        token      =os.getenv("AGENT_TOKEN", "test-token"),
        interval   =int(os.getenv("SEND_INTERVAL", "30")),
    )
    agent.run()
