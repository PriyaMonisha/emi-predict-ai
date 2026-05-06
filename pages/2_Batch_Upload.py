# filename: pages/2_Batch_Upload.py
# purpose:  Batch CSV upload, validation, scoring, and results download
# version:  1.0

import io
from pathlib import Path

import pandas as pd
import streamlit as st

from src.ui.api_client import build_results_dataframe, predict_batch

REQUIRED_COLS = [
    "age", "gender", "marital_status", "education", "monthly_salary",
    "employment_type", "years_of_employment", "company_type", "house_type",
    "monthly_rent", "family_size", "dependents", "school_fees", "college_fees",
    "travel_expenses", "groceries_utilities", "other_monthly_expenses",
    "existing_loans", "current_emi_amount", "credit_score", "bank_balance",
    "emergency_fund", "emi_scenario", "requested_amount", "requested_tenure",
]   # customer_id is Optional — NOT required

ZONE_COLORS = {
    "auto_approve": "#0d6e3f",
    "human_review": "#b45309",
    "auto_reject":  "#991b1b",
}

st.title("📂 Batch Upload")
st.caption("Upload a CSV of up to 500 customers and score them in one request.")

# ── Section A — Template download ─────────────────────────────────────────────
st.markdown("### Step 1 — Download the CSV template")
template_path = Path("data/processed/sample_batch_template.csv")
if template_path.exists():
    template_bytes = template_path.read_bytes()
else:
    # Fallback: generate minimal template on the fly
    template_bytes = (",".join(["customer_id"] + REQUIRED_COLS) + "\n").encode()

st.download_button(
    label="Download CSV Template (3 example rows)",
    data=template_bytes,
    file_name="emi_batch_template.csv",
    mime="text/csv",
    use_container_width=False,
)
st.caption(
    "The template contains 3 pre-filled example rows with verified categorical values. "
    "Row 3 has blank optional fields (credit_score, bank_balance, emergency_fund) to "
    "demonstrate null handling."
)

st.divider()

# ── Section B — Upload & validate ─────────────────────────────────────────────
st.markdown("### Step 2 — Upload your CSV")
uploaded = st.file_uploader("Upload CSV (max 500 rows)", type=["csv"])

if uploaded is not None:
    df = pd.read_csv(uploaded)

    if len(df) == 0:
        st.error("Uploaded file is empty.")
        st.stop()

    if len(df) > 500:
        st.error(f"File has {len(df)} rows. Maximum allowed is 500.")
        st.stop()

    missing_cols = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing_cols:
        st.error(f"Missing required columns: {missing_cols}")
        st.stop()

    c1, c2 = st.columns(2)
    with c1:
        st.metric("Rows to score", len(df))
    with c2:
        st.metric("Columns detected", len(df.columns))

    st.markdown("**Preview (first 5 rows):**")
    st.dataframe(df.head(5), use_container_width=True)

    st.divider()

    # ── Section C — Score ──────────────────────────────────────────────────────
    st.markdown("### Step 3 — Score")
    if st.button("Score Batch", type="primary", use_container_width=False):
        api_url = st.session_state.get("api_url", "http://localhost:8000")
        api_key = st.session_state.get("api_key", "")

        with st.spinner(f"Scoring {len(df)} customers…"):
            try:
                response = predict_batch(df, api_url, api_key)
            except RuntimeError as exc:
                st.error(str(exc))
                st.stop()

        results_df = build_results_dataframe(response)
        st.session_state["batch_results"] = results_df
        st.session_state["batch_summary"] = {
            "total":        response["total_scored"],
            "auto_approve": response["auto_approve"],
            "human_review": response["human_review"],
            "auto_reject":  response["auto_reject"],
            "latency_ms":   response["total_latency_ms"],
        }

# ── Section D — Results ────────────────────────────────────────────────────────
if "batch_results" in st.session_state:
    st.divider()
    st.markdown("### Results")

    summary = st.session_state["batch_summary"]
    results_df = st.session_state["batch_results"]

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Total Scored", summary["total"])
    with c2:
        st.metric("Auto-Approve", summary["auto_approve"],
                  delta=f"{summary['auto_approve']/max(summary['total'],1):.0%}")
    with c3:
        st.metric("Human Review", summary["human_review"],
                  delta=f"{summary['human_review']/max(summary['total'],1):.0%}")
    with c4:
        st.metric("Auto-Reject", summary["auto_reject"],
                  delta=f"{summary['auto_reject']/max(summary['total'],1):.0%}")

    st.caption(f"Total batch latency: {summary['latency_ms']:.0f} ms")

    # Zone distribution bar chart
    zone_counts = pd.DataFrame({
        "Zone":  ["Auto-Approve", "Human Review", "Auto-Reject"],
        "Count": [summary["auto_approve"], summary["human_review"], summary["auto_reject"]],
    }).set_index("Zone")
    st.bar_chart(zone_counts)

    # Results table with styled conf_zone column
    def _zone_style(val: str) -> str:
        colors = {
            "auto_approve": "background-color:#0d6e3f;color:white",
            "human_review": "background-color:#b45309;color:white",
            "auto_reject":  "background-color:#991b1b;color:white",
        }
        return colors.get(val, "")

    styled = results_df.style.map(_zone_style, subset=["conf_zone"])
    st.dataframe(styled, use_container_width=True)

    # Download
    csv_bytes = results_df.to_csv(index=False).encode()
    st.download_button(
        label="Download Results CSV",
        data=csv_bytes,
        file_name="emi_predictions.csv",
        mime="text/csv",
        use_container_width=False,
    )
