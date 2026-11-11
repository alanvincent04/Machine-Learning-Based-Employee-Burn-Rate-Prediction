import { useState, useEffect, useCallback } from "react";
import axios from "axios";

const API = process.env.REACT_APP_API_URL || "http://localhost:5000";

// ── Helpers ───────────────────────────────────────────────────────────────────
const RISK_COLORS = {
  High:   { bg: "#FEE2E2", text: "#DC2626", border: "#FCA5A5" },
  Medium: { bg: "#FEF3C7", text: "#D97706", border: "#FCD34D" },
  Low:    { bg: "#DCFCE7", text: "#16A34A", border: "#86EFAC" },
};
const RISK_ICONS = { High: "🔴", Medium: "🟡", Low: "🟢" };

const fmtTime = (iso) =>
  iso ? new Date(iso).toLocaleTimeString() : "—";

// ── Sub-components ────────────────────────────────────────────────────────────
function StatusBadge({ ok, label }) {
  return (
    <span style={{
      background: ok ? "#DCFCE7" : "#FEE2E2",
      color:      ok ? "#16A34A" : "#DC2626",
      padding: "3px 10px", borderRadius: 20,
      fontSize: 12, fontWeight: 600,
    }}>
      {ok ? "✓" : "✗"} {label}
    </span>
  );
}

function KpiBox({ value, label, color }) {
  return (
    <div style={{
      background: "#F9FAFB", border: "1px solid #E5E7EB",
      borderRadius: 10, padding: "16px 20px", textAlign: "center", flex: 1,
    }}>
      <div style={{ fontSize: 34, fontWeight: 800, color, lineHeight: 1 }}>
        {value}
      </div>
      <div style={{ fontSize: 12, color: "#6B7280", marginTop: 4 }}>
        {label}
      </div>
    </div>
  );
}

function RiskCard({ data }) {
  if (!data) return null;
  const c = RISK_COLORS[data.level] || RISK_COLORS.Low;
  const pct = data.score || 0;
  const barColor =
    pct >= 70 ? "linear-gradient(90deg,#f97316,#dc2626)"
    : pct >= 45 ? "linear-gradient(90deg,#fbbf24,#f97316)"
    : "linear-gradient(90deg,#34d399,#16a34a)";

  return (
    <div style={{
      background: c.bg, border: `1.5px solid ${c.border}`,
      borderRadius: 12, padding: 20, marginBottom: 16,
    }}>
      <div style={{ fontSize: 12, color: c.text, fontWeight: 600,
                    textTransform: "uppercase", marginBottom: 12 }}>
        Live Burnout Assessment
      </div>
      <div style={{ display: "flex", gap: 12, marginBottom: 16 }}>
        {[
          [pct, "Score", c.text, 40],
          [RISK_ICONS[data.level] + " " + data.level, "Risk Level", c.text, 20],
          [((data.confidence || 0) * 100).toFixed(1) + "%", "Confidence", "#4F46E5", 24],
        ].map(([val, lbl, col, fs], i) => (
          <div key={i} style={{
            background: "white", borderRadius: 8, padding: "12px 16px",
            textAlign: "center", flex: 1, border: `1px solid ${c.border}`,
          }}>
            <div style={{ fontSize: fs, fontWeight: 700, color: col }}>{val}</div>
            <div style={{ fontSize: 11, color: "#6B7280", marginTop: 4 }}>{lbl}</div>
          </div>
        ))}
      </div>
      <div style={{ height: 12, background: "white",
                    borderRadius: 6, overflow: "hidden",
                    border: `1px solid ${c.border}`, marginBottom: 10 }}>
        <div style={{ width: `${pct}%`, height: "100%",
                      background: barColor, borderRadius: 6,
                      transition: "width 1s ease" }} />
      </div>
      <div style={{ fontSize: 12, color: c.text, background: "white",
                    padding: "8px 12px", borderRadius: 6,
                    border: `1px solid ${c.border}` }}>
        {data.level === "High" &&
          "! Immediate action needed. Schedule wellness check-in."}
        {data.level === "Medium" &&
          " Monitor closely. Suggest regular breaks and flexible hours."}
        {data.level === "Low" &&
          " Healthy patterns detected. Continue regular check-ins."}
      </div>
    </div>
  );
}

function AlertItem({ alert }) {
  const c = RISK_COLORS[
    alert.level === "high" ? "High"
    : alert.level === "medium" ? "Medium" : "Low"
  ] || RISK_COLORS.Low;
  return (
    <div style={{
      display: "flex", gap: 10, padding: "10px 12px",
      borderRadius: 8, marginBottom: 6,
      background: c.bg, borderLeft: `3px solid ${c.text}`,
    }}>
      <div style={{ flex: 1 }}>
        <div style={{ fontSize: 12, fontWeight: 600, color: "#111827" }}>
          {alert.employee_id} — {alert.level?.toUpperCase()} Risk
          (Score: {alert.score})
        </div>
        <div style={{ fontSize: 11, color: "#6B7280", marginTop: 2 }}>
          {alert.message}
        </div>
      </div>
      <div style={{ fontSize: 10, color: "#9CA3AF" }}>
        {fmtTime(alert.timestamp)}
      </div>
    </div>
  );
}

// ── Main App ──────────────────────────────────────────────────────────────────
export default function App() {
  const [health,     setHealth]     = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [summary,    setSummary]    = useState(null);
  const [alerts,     setAlerts]     = useState([]);
  const [lastUpdate, setLastUpdate] = useState(null);
  const [loading,    setLoading]    = useState(true);
  const [activeTab,  setActiveTab]  = useState("dashboard");
  const [employees,  setEmployees]  = useState([]);

  // ── Fetch functions ─────────────────────────────────────────────────────
  const fetchHealth = useCallback(async () => {
    try {
      const { data } = await axios.get(`${API}/health`);
      setHealth(data);
    } catch { setHealth(null); }
  }, []);

  const fetchPrediction = useCallback(async () => {
    try {
      const now = Date.now();
      const { data } = await axios.post(`${API}/api/v1/burnout/predict`, {
        typing_speed_variation   : Math.random() * 40 + 5,
        idle_time_pct            : Math.random() * 70 + 10,
        session_duration_hrs     : (now - startRef.current) / 3600000,
        break_irregularity_index : Math.random() * 8 + 1,
        work_hour_deviation      : Math.abs(new Date().getHours() - 9),
        task_completion_rate     : Math.random() * 60 + 20,
      });
      setPrediction(data);
      setLastUpdate(new Date().toLocaleTimeString());
      setLoading(false);
    } catch (e) { console.error("Prediction failed:", e); }
  }, []);

  const startRef = { current: Date.now() };

  // ── Initial load + polling ──────────────────────────────────────────────
  useEffect(() => {
    fetchHealth();
    fetchPrediction();
    const interval = setInterval(() => {
      fetchHealth();
      fetchPrediction();
    }, 30000);
    return () => clearInterval(interval);
  }, [fetchHealth, fetchPrediction]);

  // ── Tabs ────────────────────────────────────────────────────────────────
  const tabs = [
    { key: "dashboard", label: "Dashboard" },
    {/* { key: "privacy",   label: "Privacy & Ethics" },
    { key: "about",     label: "About" }, */}
  ];

  const tabStyle = (key) => ({
    padding: "7px 16px", border: "none", cursor: "pointer",
    fontFamily: "Arial", fontSize: 13,
    borderBottom: activeTab === key ? "2px solid #4F46E5" : "2px solid transparent",
    color: activeTab === key ? "#4F46E5" : "#6B7280",
    fontWeight: activeTab === key ? 600 : 400,
    background: "none",
  });

  return (
    <div style={{ fontFamily: "Arial", maxWidth: 900,
                  margin: "0 auto", padding: "20px 16px" }}>

      {/* Header */}
      <div style={{
        background: "#4F46E5", color: "white",
        padding: "20px 24px", borderRadius: 12, marginBottom: 16,
        display: "flex", justifyContent: "space-between", alignItems: "center",
      }}>
        <div>
          <div style={{ fontSize: 24, fontWeight: 800, letterSpacing: -0.5 }}>
            BurnoutSense AI
          </div>
          <div style={{ fontSize: 13, opacity: 0.8, marginTop: 4 }}>
            Privacy-Aware Remote Worker Monitoring — System Demo
          </div>
        </div>
        <div style={{ textAlign: "right" }}>
          <div style={{ fontSize: 12, opacity: 0.8 }}>
            <span style={{ display: "inline-block", width: 8, height: 8,
                           borderRadius: "50%", background: "#22c55e",
                           marginRight: 5 }} />
            Live
          </div>
          {lastUpdate && (
            <div style={{ fontSize: 11, opacity: 0.6, marginTop: 3 }}>
              Updated: {lastUpdate}
            </div>
          )}
        </div>
      </div>


      {/* Tabs */}
      <div style={{ display: "flex", borderBottom: "1px solid #E5E7EB",
                    marginBottom: 16 }}>
        {tabs.map(t => (
          <button key={t.key} style={tabStyle(t.key)}
                  onClick={() => setActiveTab(t.key)}>
            {t.label}
          </button>
        ))}
      </div>

      {/* ── DASHBOARD TAB ── */}
      {activeTab === "dashboard" && (
        <>
          {/* KPIs */}
          <div style={{ display: "flex", gap: 12, marginBottom: 16 }}>
            <KpiBox value={prediction?.score ?? "—"}
                    label="Current Score"
                    color={
                      (prediction?.score ?? 0) >= 70 ? "#DC2626"
                      : (prediction?.score ?? 0) >= 45 ? "#D97706"
                      : "#16A34A"
                    } />
            <KpiBox value={prediction?.level ?? "—"}
                    label="Risk Level"
                    color={
                      RISK_COLORS[prediction?.level]?.text || "#4F46E5"
                    } />
            <KpiBox value={prediction ? ((prediction.confidence || 0) * 100).toFixed(0) + "%" : "—"}
                    label="Confidence"
                    color="#4F46E5" />
            <KpiBox value={lastUpdate || "—"}
                    label="Last Updated"
                    color="#6B7280" />
          </div>

          {/* Risk Card */}
          {loading ? (
            <div style={{
              background: "#F9FAFB", border: "1px solid #E5E7EB",
              borderRadius: 12, padding: 40, textAlign: "center",
              color: "#9CA3AF"
            }}>
              ⏳ Loading burnout assessment...
            </div>
          ) : (
            <RiskCard data={prediction} />
          )}

          {/* Notice */}
          <div style={{
            background: "#FEF3C7", border: "1px solid #FCD34D",
            borderRadius: 8, padding: "10px 14px", marginBottom: 16,
            fontSize: 12, color: "#78350F",
          }}>
            <b>System Demo:</b> This dashboard demonstrates the technical
            pipeline using simulated behavioral data. Academic ML accuracy results
            are in the Part A report (BurnoutSense_PartA_Report.pdf).
          </div>
        </>
      )}

      <div style={{ textAlign: "center", fontSize: 11,
                    color: "#D1D5DB", marginTop: 20 }}>
        BurnoutSense AI v1.0 — Privacy-Aware ML Monitoring
      </div>
    </div>
  );
}
