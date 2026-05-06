# filename: src/ui/api_client.py
# purpose:  HTTP client wrapper for EMI Predict AI FastAPI service
# version:  1.0

import os
import requests
import pandas as pd


def _get_defaults() -> tuple[str, str]:
    """Resolve API URL and key from Streamlit secrets (Cloud) or env vars (Docker/local)."""
    try:
        import streamlit as st  # lazy — only valid inside a running Streamlit context
        url = st.secrets.get("API_URL", os.environ.get("API_URL", "http://localhost:8000"))
        key = st.secrets.get("API_KEY", os.environ.get("API_KEY", ""))
    except Exception:
        url = os.environ.get("API_URL", "http://localhost:8000")
        key = os.environ.get("API_KEY", "")
    return url, key


def check_health(api_url: str, api_key: str) -> dict:
    """GET /health — no auth required. Returns status dict; never raises."""
    try:
        r = requests.get(f"{api_url}/health", timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as exc:
        return {
            "status": "error",
            "redis_ok": False,
            "models_loaded": False,
            "error": str(exc),
        }


def predict_single(payload: dict, api_url: str, api_key: str) -> dict:
    """POST /predict — score one customer. Raises RuntimeError on any failure."""
    headers = {"X-API-Key": api_key, "Content-Type": "application/json"}
    try:
        r = requests.post(f"{api_url}/predict", json=payload, headers=headers, timeout=60)
    except requests.exceptions.Timeout:
        raise RuntimeError(
            "Request timed out. Render free tier may be cold-starting (30–60s). "
            "Please wait and retry."
        )
    except requests.exceptions.ConnectionError:
        raise RuntimeError(
            f"Cannot connect to API at {api_url}. "
            "Verify the URL in the sidebar is correct."
        )
    if r.status_code != 200:
        detail = r.json().get("detail", r.text[:200]) if r.content else r.status_code
        raise RuntimeError(f"API error {r.status_code}: {detail}")
    return r.json()


def predict_batch(df: pd.DataFrame, api_url: str, api_key: str) -> dict:
    """POST /predict/batch — score up to 500 customers. Raises RuntimeError on failure."""
    # NaN → None so JSON serialisation sends null for optional fields
    records = df.where(pd.notna(df), other=None).to_dict(orient="records")
    headers = {"X-API-Key": api_key, "Content-Type": "application/json"}
    try:
        r = requests.post(
            f"{api_url}/predict/batch",
            json={"customers": records},
            headers=headers,
            timeout=120,
        )
    except requests.exceptions.Timeout:
        raise RuntimeError(
            "Request timed out. Render free tier may be cold-starting (30–60s). "
            "Please wait and retry."
        )
    except requests.exceptions.ConnectionError:
        raise RuntimeError(
            f"Cannot connect to API at {api_url}. "
            "Verify the URL in the sidebar is correct."
        )
    if r.status_code != 200:
        detail = r.json().get("detail", r.text[:200]) if r.content else r.status_code
        raise RuntimeError(f"API error {r.status_code}: {detail}")
    return r.json()


def build_results_dataframe(response: dict) -> pd.DataFrame:
    """Flatten BatchPredictResponse.predictions[] into a display DataFrame."""
    rows = response.get("predictions", [])
    df = pd.DataFrame(rows)
    if "clf_label" in df.columns:
        df["clf_label"] = df["clf_label"].map({1: "Eligible", 0: "Not Eligible"})
    col_order = [
        "customer_id", "conf_zone", "clf_proba", "clf_label",
        "predicted_emi", "cache_hit", "latency_ms",
    ]
    present = [c for c in col_order if c in df.columns]
    return df[present]
