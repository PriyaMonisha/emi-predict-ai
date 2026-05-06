# filename: pages/3_System_Health.py
# purpose:  Live API health check, latency test, and stack reference
# version:  1.0

import time
from datetime import datetime

import streamlit as st

from src.ui.api_client import check_health, predict_single

# Standalone payload — same field values as SAMPLE_CUSTOMER in page 1.
# Defined here independently so this page has no cross-page import dependency.
SAMPLE_PAYLOAD: dict = {
    "customer_id":            "HEALTHCHECK-UI",
    "age":                    34,
    "gender":                 "Male",
    "marital_status":         "Married",
    "education":              "Graduate",
    "monthly_salary":         72000.0,
    "employment_type":        "Private",
    "years_of_employment":    6.0,
    "company_type":           "Mnc",
    "house_type":             "Own",
    "monthly_rent":           0.0,
    "family_size":            3,
    "dependents":             1,
    "school_fees":            2000.0,
    "college_fees":           0.0,
    "travel_expenses":        3500.0,
    "groceries_utilities":    8000.0,
    "other_monthly_expenses": 1500.0,
    "existing_loans":         "No",
    "current_emi_amount":     0.0,
    "credit_score":           720.0,
    "bank_balance":           280000.0,
    "emergency_fund":         120000.0,
    "emi_scenario":           "Personal Loan Emi",
    "requested_amount":       500000.0,
    "requested_tenure":       36.0,
}

STACK_TABLE = [
    {"Service": "PostgreSQL",         "Port": "5432 (internal)", "Role": "Airflow metadata DB"},
    {"Service": "Redis",              "Port": "6379 (internal)", "Role": "Feature store / cache"},
    {"Service": "MLflow",             "Port": "5000",            "Role": "Experiment tracking UI"},
    {"Service": "FastAPI",            "Port": "8000",            "Role": "Prediction API"},
    {"Service": "Airflow Init",       "Port": "—",               "Role": "One-shot DB setup"},
    {"Service": "Airflow Webserver",  "Port": "8080",            "Role": "DAG management UI"},
    {"Service": "Airflow Scheduler",  "Port": "—",               "Role": "DAG execution"},
    {"Service": "Prometheus",         "Port": "9090",            "Role": "Metrics scraping"},
    {"Service": "Grafana",            "Port": "3000",            "Role": "Monitoring dashboards"},
    {"Service": "Streamlit (this UI)","Port": "8501",            "Role": "Prediction UI"},
]

st.title("🔍 System Health")
st.caption("Live API health check, inference latency test, and stack reference.")

api_url = st.session_state.get("api_url", "http://localhost:8000")
api_key = st.session_state.get("api_key", "")

# ── Health check section ───────────────────────────────────────────────────────
col_refresh, col_ts = st.columns([1, 3])
with col_refresh:
    do_check = st.button("Refresh Health", type="primary")
with col_ts:
    if "last_health_check" in st.session_state:
        st.caption(f"Last checked: {st.session_state['last_health_check']}")

if do_check or "health_result" not in st.session_state:
    with st.spinner("Checking API health…"):
        health = check_health(api_url, api_key)
    st.session_state["health_result"] = health
    st.session_state["last_health_check"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

health = st.session_state.get("health_result", {})

if health.get("status") == "ok":
    st.success("API is healthy")
else:
    err = health.get("error", "Unknown error")
    st.error(f"API unreachable — {err}")

c1, c2, c3 = st.columns(3)
with c1:
    redis_ok = health.get("redis_ok", False)
    st.metric("Redis", "UP" if redis_ok else "DOWN",
              delta="connected" if redis_ok else "unreachable")
with c2:
    models_ok = health.get("models_loaded", False)
    st.metric("ML Models", "Loaded" if models_ok else "NOT LOADED",
              delta="ready" if models_ok else "check logs")
with c3:
    st.metric("API Status", health.get("status", "unknown").upper())

st.divider()

# ── Latency test ───────────────────────────────────────────────────────────────
st.markdown("### Inference Latency Test")
st.caption("Fires a real `/predict` request with a sample payload and measures round-trip time.")

if st.button("Run Latency Test"):
    t0 = time.perf_counter()
    try:
        with st.spinner("Running inference…"):
            result = predict_single(SAMPLE_PAYLOAD, api_url, api_key)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        prev_latency = st.session_state.get("last_latency_ms")
        st.session_state["last_latency_ms"] = elapsed_ms

        delta = None
        if prev_latency is not None:
            delta = f"{elapsed_ms - prev_latency:+.0f} ms vs last run"

        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Round-trip Latency", f"{elapsed_ms:.0f} ms", delta=delta)
        with c2:
            st.metric("API-reported Latency", f"{result.get('latency_ms', 0):.0f} ms")
        with c3:
            st.metric("Cache Hit", "Yes" if result.get("cache_hit") else "No")

        st.info(
            f"Result: conf_zone=**{result['conf_zone']}** · "
            f"proba=**{result['clf_proba']:.4f}** · "
            f"predicted_emi=**₹{result['predicted_emi']:,.0f}**"
        )
    except RuntimeError as exc:
        st.error(str(exc))

st.divider()

# ── Stack reference ────────────────────────────────────────────────────────────
with st.expander("Stack Architecture (10 services)"):
    import pandas as pd
    st.table(pd.DataFrame(STACK_TABLE))
    st.caption(
        "Note: Services are checked individually only for FastAPI (/health). "
        "The other service statuses shown here are static reference information — "
        "use the Grafana dashboard (port 3000) for live multi-service monitoring."
    )
