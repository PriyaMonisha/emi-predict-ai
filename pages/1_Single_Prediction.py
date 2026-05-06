# filename: pages/1_Single_Prediction.py
# purpose:  Single-customer EMI eligibility and amount prediction form
# version:  1.4

import hashlib
import json
import uuid

import streamlit as st
from src.ui.api_client import predict_single

# ── Verified categorical options from data/processed/train_features.csv ────────
CATEGORICAL_OPTIONS: dict[str, list[str]] = {
    "gender":          ["Female", "Male"],
    "marital_status":  ["Married", "Single"],
    "education":       ["Graduate", "Post Graduate", "High School", "Professional", "Unknown"],
    "employment_type": ["Private", "Self-Employed", "Government"],
    "company_type":    ["Large Indian", "Mnc", "Mid-Size", "Startup", "Small"],
    "house_type":      ["Own", "Rented", "Family"],
    "existing_loans":  ["No", "Yes"],
    "emi_scenario":    [
        "Vehicle Emi",
        "Personal Loan Emi",
        "Education Emi",
        "E-Commerce Shopping Emi",
        "Home Appliances Emi",
    ],
}

SAMPLE_CUSTOMER: dict = {
    "customer_id":            "CUST-DEMO-001",
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

# zone → (display label, banner colour, icon)
ZONE_MAP: dict[str, tuple[str, str, str]] = {
    "auto_approve": ("AUTO APPROVE", "#0d6e3f", "✅"),
    "human_review": ("HUMAN REVIEW", "#b45309", "⚠️"),
    "auto_reject":  ("AUTO REJECT",  "#991b1b", "❌"),
}

# ── Session state defaults — initialised once per browser session ─────────────
_DEFAULTS: dict = {
    "age":                    30,
    "gender":                 "Female",
    "marital_status":         "Married",
    "education":              "Graduate",
    "employment_type":        "Private",
    "company_type":           "Large Indian",
    "years_of_employment":    1.0,
    "monthly_salary":         30000.0,
    "house_type":             "Own",
    "monthly_rent":           0.0,
    "family_size":            2,
    "dependents":             0,
    "school_fees":            0.0,
    "college_fees":           0.0,
    "travel_expenses":        0.0,
    "groceries_utilities":    5000.0,
    "other_monthly_expenses": 0.0,
    "existing_loans":         "No",
    "current_emi_amount":     0.0,
    "credit_score":           None,
    "bank_balance":           None,
    "emergency_fund":         None,
    "emi_scenario":           "Vehicle Emi",
    "requested_amount":       100000.0,
    "requested_tenure":       12.0,
}

for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

if "customer_id" not in st.session_state:
    st.session_state["customer_id"] = f"CUST-{uuid.uuid4().hex[:8].upper()}"

# ── Page Header ───────────────────────────────────────────────────────────────
st.title("🏦 EMI Eligibility Scoring")
st.caption(
    "Evaluate a customer's EMI eligibility and predicted repayment capacity "
    "in real-time using ML models."
)
st.divider()

# ── Load Sample button (outside form — triggers rerun to fill widgets) ─────────
if st.button("Load Sample Data", help="Pre-fill the form with a demo customer"):
    for k, v in SAMPLE_CUSTOMER.items():
        st.session_state[k] = v
    st.rerun()

# ── Two-column layout: form | result ──────────────────────────────────────────
col_form, col_result = st.columns([1, 1], gap="large")

with col_form:
    with st.form("predict_form"):

        # ── Personal Profile ──────────────────────────────────────────────────
        st.markdown("**Personal Profile**")
        r1c1, r1c2 = st.columns(2)
        with r1c1:
            age = st.number_input("Age", min_value=18, max_value=100, step=1, key="age")
        with r1c2:
            gender = st.selectbox("Gender", CATEGORICAL_OPTIONS["gender"], key="gender")

        r2c1, r2c2 = st.columns(2)
        with r2c1:
            marital_status = st.selectbox(
                "Marital Status", CATEGORICAL_OPTIONS["marital_status"], key="marital_status"
            )
        with r2c2:
            education = st.selectbox(
                "Education", CATEGORICAL_OPTIONS["education"], key="education"
            )

        st.divider()

        # ── Employment ────────────────────────────────────────────────────────
        st.markdown("**Employment**")
        r1c1, r1c2 = st.columns(2)
        with r1c1:
            employment_type = st.selectbox(
                "Employment Type", CATEGORICAL_OPTIONS["employment_type"], key="employment_type"
            )
        with r1c2:
            company_type = st.selectbox(
                "Company Type", CATEGORICAL_OPTIONS["company_type"], key="company_type"
            )

        r2c1, r2c2 = st.columns(2)
        with r2c1:
            years_of_employment = st.number_input(
                "Years Employed", min_value=0.0, step=0.5, key="years_of_employment"
            )
        with r2c2:
            monthly_salary = st.number_input(
                "Monthly Salary (₹)", min_value=0.0, step=1000.0, key="monthly_salary"
            )

        st.divider()

        # ── Housing & Family ──────────────────────────────────────────────────
        st.markdown("**Housing & Family**")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            house_type = st.selectbox(
                "House Type", CATEGORICAL_OPTIONS["house_type"], key="house_type"
            )
        with c2:
            monthly_rent = st.number_input(
                "Monthly Rent (₹)", min_value=0.0, step=500.0, key="monthly_rent"
            )
        with c3:
            family_size = st.number_input(
                "Family Size", min_value=1, step=1, key="family_size"
            )
        with c4:
            dependents = st.number_input(
                "Dependents", min_value=0, step=1, key="dependents"
            )

        st.divider()

        # ── Monthly Expenses ──────────────────────────────────────────────────
        st.markdown("**Monthly Expenses (₹)**")
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            school_fees = st.number_input(
                "School Fees", min_value=0.0, step=500.0, key="school_fees"
            )
        with c2:
            college_fees = st.number_input(
                "College Fees", min_value=0.0, step=500.0, key="college_fees"
            )
        with c3:
            travel_expenses = st.number_input(
                "Travel", min_value=0.0, step=500.0, key="travel_expenses"
            )
        with c4:
            groceries_utilities = st.number_input(
                "Groceries & Utilities", min_value=0.0, step=500.0, key="groceries_utilities"
            )
        with c5:
            other_monthly_expenses = st.number_input(
                "Other", min_value=0.0, step=500.0, key="other_monthly_expenses"
            )

        st.divider()

        # ── Existing Debt ─────────────────────────────────────────────────────
        st.markdown("**Existing Debt**")
        c1, c2 = st.columns(2)
        with c1:
            existing_loans = st.radio(
                "Existing Loans",
                options=["No", "Yes"],
                horizontal=True,
                key="existing_loans",
            )
        with c2:
            current_emi_amount = st.number_input(
                "Current EMI Amount (₹)", min_value=0.0, step=500.0, key="current_emi_amount"
            )

        st.divider()

        # ── Optional: Financial Profile ───────────────────────────────────────
        with st.expander("Optional — Credit & Savings (leave blank if unknown)"):
            st.caption("The model will impute missing values — providing them improves accuracy.")
            c1, c2, c3 = st.columns(3)
            with c1:
                credit_score = st.number_input(
                    "Credit Score", min_value=300.0, max_value=900.0, step=1.0, key="credit_score"
                )
            with c2:
                bank_balance = st.number_input(
                    "Bank Balance (₹)", min_value=0.0, step=10000.0, key="bank_balance"
                )
            with c3:
                emergency_fund = st.number_input(
                    "Emergency Fund (₹)", min_value=0.0, step=10000.0, key="emergency_fund"
                )

        st.divider()

        # ── Loan Request ──────────────────────────────────────────────────────
        st.markdown("**Loan Request**")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            emi_scenario = st.selectbox(
                "EMI Scenario", CATEGORICAL_OPTIONS["emi_scenario"], key="emi_scenario"
            )
        with c2:
            requested_amount = st.number_input(
                "Requested Amount (₹)", min_value=1.0, step=10000.0, key="requested_amount"
            )
        with c3:
            requested_tenure = st.number_input(
                "Tenure (months)", min_value=1.0, step=6.0, key="requested_tenure"
            )
        with c4:
            customer_id = st.text_input(
                "Customer ID (optional)", placeholder="CUST-001", key="customer_id"
            )

        submitted = st.form_submit_button(
            "Predict EMI Eligibility", use_container_width=True, type="primary"
        )

# ── Result card (right column) ────────────────────────────────────────────────
with col_result:
    if submitted:
        payload = {
            "customer_id":            customer_id or None,
            "age":                    age,
            "gender":                 gender,
            "marital_status":         marital_status,
            "education":              education,
            "monthly_salary":         monthly_salary,
            "employment_type":        employment_type,
            "years_of_employment":    years_of_employment,
            "company_type":           company_type,
            "house_type":             house_type,
            "monthly_rent":           monthly_rent,
            "family_size":            family_size,
            "dependents":             dependents,
            "school_fees":            school_fees,
            "college_fees":           college_fees,
            "travel_expenses":        travel_expenses,
            "groceries_utilities":    groceries_utilities,
            "other_monthly_expenses": other_monthly_expenses,
            "existing_loans":         existing_loans,
            "current_emi_amount":     current_emi_amount,
            "credit_score":           credit_score,
            "bank_balance":           bank_balance,
            "emergency_fund":         emergency_fund,
            "emi_scenario":           emi_scenario,
            "requested_amount":       requested_amount,
            "requested_tenure":       requested_tenure,
        }

        # Content-addressed cache key: different feature values → different Redis key
        _feat_str = json.dumps(
            {k: v for k, v in payload.items() if k != "customer_id"},
            sort_keys=True, default=str,
        )
        payload["customer_id"] = f"UI-{hashlib.sha256(_feat_str.encode()).hexdigest()[:12]}"

        api_url = st.session_state.get("api_url", "http://localhost:8000")
        api_key = st.session_state.get("api_key", "")

        with st.spinner("Scoring…"):
            try:
                result = predict_single(payload, api_url, api_key)
            except RuntimeError as exc:
                st.error(str(exc))
                st.stop()

        zone_text, color, icon = ZONE_MAP.get(
            result["conf_zone"], ("UNKNOWN", "#6e7681", "❓")
        )
        clf_proba     = result["clf_proba"]
        predicted_emi = result["predicted_emi"]
        clf_label     = "Eligible" if result.get("clf_label") == 1 else "Not Eligible"
        latency_ms    = int(result.get("latency_ms", 0))
        cache_str     = "Yes" if result.get("cache_hit", False) else "No"
        conf_label    = (
            "Very High" if clf_proba > 0.85
            else "Moderate" if clf_proba >= 0.40
            else "Low"
        )

        html = f"""
        <div style="background:#161b22;border:1px solid #30363d;
                    padding:24px;border-radius:12px;font-family:sans-serif">

            <div style="background:{color};padding:16px 24px;border-radius:8px;
                        text-align:center;margin-bottom:24px">
                <div style="font-size:2rem;margin-bottom:4px">{icon}</div>
                <div style="font-size:1.25rem;font-weight:700;letter-spacing:2px;
                            color:white;text-transform:uppercase">{zone_text}</div>
            </div>

            <div style="text-align:center;margin-bottom:24px">
                <div style="font-size:3rem;font-weight:800;color:#f0f6ff;line-height:1">
                    &#8377;{predicted_emi:,.0f}
                </div>
                <div style="font-size:0.78rem;color:#8b949e;margin-top:6px;
                            letter-spacing:2px;text-transform:uppercase">
                    Estimated Monthly EMI
                </div>
            </div>

            <div style="display:flex;gap:12px;margin-bottom:20px">
                <div style="flex:1;background:#0d1117;padding:14px 16px;border-radius:8px;
                            text-align:center;border:1px solid #21262d">
                    <div style="font-size:0.68rem;color:#8b949e;margin-bottom:6px;
                                letter-spacing:1.5px;text-transform:uppercase">
                        Confidence
                    </div>
                    <div style="font-size:1.4rem;font-weight:700;color:#f0f6ff">
                        {clf_proba:.1%}
                    </div>
                    <div style="font-size:0.78rem;color:#8b949e;margin-top:2px">
                        {conf_label}
                    </div>
                </div>
                <div style="flex:1;background:#0d1117;padding:14px 16px;border-radius:8px;
                            text-align:center;border:1px solid #21262d">
                    <div style="font-size:0.68rem;color:#8b949e;margin-bottom:6px;
                                letter-spacing:1.5px;text-transform:uppercase">
                        Decision
                    </div>
                    <div style="font-size:1.4rem;font-weight:700;color:#f0f6ff">
                        {clf_label}
                    </div>
                </div>
            </div>

            <div style="background:#0d1117;padding:12px 16px;border-radius:8px;
                        text-align:center;border:1px solid #21262d">
                <div style="font-size:0.68rem;color:#6e7681;margin-bottom:6px;
                            letter-spacing:1.5px;text-transform:uppercase">
                    Technical Details
                </div>
                <div style="font-size:0.8rem;color:#6e7681">
                    Latency: {latency_ms} ms &nbsp;|&nbsp;
                    Cache: {cache_str} &nbsp;|&nbsp;
                    Raw Score: {clf_proba:.4f}
                </div>
            </div>

        </div>
        """

        st.html(html)
        st.info(
            "**Thresholds:** Auto-Approve > 85% · Human Review 40–85% · Auto-Reject < 40%"
        )
    else:
        st.html(
            '<div style="background:#161b22;border:1px solid #30363d;padding:24px;'
            'border-radius:12px;display:flex;align-items:center;justify-content:center;'
            'min-height:320px;color:#6e7681;font-size:0.95rem;text-align:center;'
            'font-family:sans-serif">'
            "Fill the form on the left and click<br>"
            "<strong style='color:#8b949e'>Predict EMI Eligibility</strong>"
            " to see the result here."
            "</div>"
        )
