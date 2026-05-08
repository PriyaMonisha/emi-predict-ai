# filename: streamlit_app.py
# purpose:  Multi-page Streamlit UI entry point for EMI Predict AI
# version:  1.2

import streamlit as st
from src.ui.api_client import _get_defaults, check_health

st.set_page_config(
    layout="wide",
    page_title="EMI Predict AI",
    page_icon="🏦",
    initial_sidebar_state="auto",   # "collapsed" hides pages/ nav — "auto" keeps it visible
)

# Initialise session state with API defaults on first load
if "api_url" not in st.session_state:
    url, key = _get_defaults()
    st.session_state["api_url"] = url
    st.session_state["api_key"] = key

# ── Sidebar — API Configuration ────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ API Configuration")

    st.session_state["api_url"] = st.text_input(
        "API URL",
        value=st.session_state["api_url"],
        help="Base URL of the FastAPI backend",
    )
    st.session_state["api_key"] = st.text_input(
        "API Key",
        value=st.session_state["api_key"],
        type="password",
        help="X-API-Key header value",
    )

    if st.button("Test Connection", use_container_width=True):
        with st.spinner("Connecting…"):
            result = check_health(
                st.session_state["api_url"],
                st.session_state["api_key"],
            )
        if result.get("status") == "ok":
            redis_status = "UP" if result["redis_ok"] else "DOWN"
            model_status = "Loaded" if result["models_loaded"] else "NOT LOADED"
            st.success(f"API OK — Redis: {redis_status} · Models: {model_status}")
        else:
            st.error(f"Unreachable — {result.get('error', 'unknown error')}")

    st.caption("⚠️ First request on Render free tier may take 30–60s (cold start).")
    st.divider()
    st.caption("EMI Predict AI · Section 14")

# ── Home page ──────────────────────────────────────────────────────────────────
# pages/ folder routing is automatic in all Streamlit versions — no st.navigation() needed.
# Streamlit discovers pages/*.py at startup and injects them into the sidebar nav.

st.title("🏦 EMI Predict AI")
st.markdown(
    "**Production-grade EMI risk prediction for Indian financial institutions.** "
    "Dual-model pipeline: LightGBM eligibility classifier + XGBoost EMI regressor."
)

col1, col2, col3 = st.columns(3)
col1.metric("Classifier AUC", "0.9999", "LightGBM")
col2.metric("Regressor RMSE", "₹671.85", "XGBoost")
col3.metric("Training Rows", "387,287", "stratified split")

st.divider()
st.markdown(
    "**Use the sidebar to navigate:**\n"
    "- 🎯 **Single Prediction** — score one applicant instantly\n"
    "- 📂 **Batch Upload** — upload a CSV and download scored results\n"
    "- 🔍 **System Health** — check API, Redis, and model status"
)
st.info(
    "Configure the API URL and Key in the sidebar, then click **Test Connection** "
    "to verify the backend is reachable before scoring.",
    icon="ℹ️",
)
