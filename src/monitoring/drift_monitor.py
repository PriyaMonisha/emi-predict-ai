# filename: src/monitoring/drift_monitor.py
# purpose:  Section 10 — two-layer Evidently drift detection for EMI batch pipeline
# version:  1.0

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# ── Feature sets ───────────────────────────────────────────────────────────────
# Layer 1: raw 25 input features — detects population/distribution shift
# (business dashboard: "are our applicants changing?")
RAW_FEATURES = [
    "age", "monthly_salary", "years_of_employment", "monthly_rent",
    "family_size", "dependents", "school_fees", "college_fees",
    "travel_expenses", "groceries_utilities", "other_monthly_expenses",
    "current_emi_amount", "credit_score", "bank_balance", "emergency_fund",
    "requested_amount", "requested_tenure",
    "gender", "marital_status", "education", "employment_type",
    "company_type", "house_type", "existing_loans", "emi_scenario",
]

# Layer 2: 4 key engineered features — detects ML signal degradation
# (model health: "is the financial stress / risk profile of applicants shifting?")
LAYER2_FEATURES = [
    "expense_ratio",        # total_expenses / monthly_salary (financial stress)
    "emi_burden_ratio",     # current_emi / monthly_salary (debt load)
    "loan_to_income_ratio", # requested_amount / monthly_salary (affordability)
    "credit_score_band",    # ordinal categorical used directly in OHE model input
]


def _extract_drift_summary(report_dict: dict) -> dict:
    """Parse Evidently report dict → simplified drift summary."""
    metrics = report_dict.get("metrics", [])
    drift_detected = False
    share_drifted   = 0.0
    drifted_cols    = []

    for m in metrics:
        if m.get("metric") == "DatasetDriftMetric":
            r = m.get("result", {})
            drift_detected = r.get("dataset_drift", False)
            share_drifted  = r.get("share_of_drifted_columns", 0.0)

        if m.get("metric") == "DataDriftTable":
            by_col = m.get("result", {}).get("drift_by_columns", {})
            drifted_cols = [
                col for col, data in by_col.items()
                if data.get("drift_detected", False)
            ]

    return {
        "drift_detected":    drift_detected,
        "share_drifted":     round(share_drifted, 4),
        "drifted_columns":   drifted_cols,
    }


def run_drift_report(
    reference_path: str,
    current_path: str,
    output_dir: str,
    batch_date: str,
) -> dict:
    """
    Two-layer drift detection using Evidently.

    Layer 1: 25 raw input features  — population drift (business dashboard)
    Layer 2: 4 key engineered features — ML signal drift (model health)

    Non-blocking: returns gracefully if either file is missing or Evidently fails.
    The Airflow DAG retries are the error handler — this task must never crash the DAG.

    Args:
        reference_path : path to train_features.csv (reference distribution)
        current_path   : path to today's features_{date}.csv (current distribution)
        output_dir     : directory to save HTML + JSON reports
        batch_date     : used for output file naming (YYYY-MM-DD)

    Returns:
        {
            "drift_detected"        : bool  — True if EITHER layer detects drift
            "layer1_drift"          : bool
            "layer1_drifted_features": list[str]
            "layer2_drift"          : bool
            "layer2_drifted_features": list[str]
            "report_path"           : str   — path to Layer 1 HTML report
            "skipped"               : bool  — True if files missing or error
            "skip_reason"           : str   — populated if skipped=True
        }
    """
    _empty = {
        "drift_detected": False,
        "layer1_drift": False, "layer1_drifted_features": [],
        "layer2_drift": False, "layer2_drifted_features": [],
        "report_path": "", "skipped": True, "skip_reason": "",
    }

    # ── Guard: missing input files ─────────────────────────────────────────────
    ref_path = Path(reference_path)
    cur_path = Path(current_path)

    if not ref_path.exists():
        reason = f"Reference file not found: {reference_path}"
        logger.warning(f"[drift_monitor] Skipping — {reason}")
        return {**_empty, "skip_reason": reason}

    if not cur_path.exists():
        reason = f"Current batch file not found: {current_path}"
        logger.warning(f"[drift_monitor] Skipping — {reason}")
        return {**_empty, "skip_reason": reason}

    try:
        from evidently.metric_preset import DataDriftPreset
        from evidently.report import Report
    except ImportError as exc:
        reason = f"Evidently not available: {exc}"
        logger.warning(f"[drift_monitor] Skipping — {reason}")
        return {**_empty, "skip_reason": reason}

    try:
        ref_df = pd.read_csv(ref_path)
        cur_df = pd.read_csv(cur_path)
    except Exception as exc:
        reason = f"Failed to read CSV: {exc}"
        logger.warning(f"[drift_monitor] Skipping — {reason}")
        return {**_empty, "skip_reason": reason}

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # ── Layer 1: raw 25 input features ────────────────────────────────────────
    l1_result = {"drift_detected": False, "drifted_columns": []}
    l1_report_path = ""
    try:
        l1_cols = [c for c in RAW_FEATURES if c in ref_df.columns and c in cur_df.columns]
        if len(l1_cols) < 5:
            logger.warning(f"[drift_monitor] Layer 1: only {len(l1_cols)} matching columns — skipping layer")
        else:
            l1_report = Report(metrics=[DataDriftPreset()])
            l1_report.run(
                reference_data=ref_df[l1_cols],
                current_data=cur_df[l1_cols],
            )
            l1_html = output_path / f"drift_layer1_raw_{batch_date}.html"
            l1_report.save_html(str(l1_html))
            l1_result = _extract_drift_summary(l1_report.as_dict())
            l1_report_path = str(l1_html)
            logger.info(
                f"[drift_monitor] Layer 1 — drift={l1_result['drift_detected']}  "
                f"share={l1_result['share_drifted']:.1%}  "
                f"drifted={l1_result['drifted_columns']}"
            )
    except Exception as exc:
        logger.warning(f"[drift_monitor] Layer 1 failed (non-fatal): {exc}")

    # ── Layer 2: 4 key engineered features ────────────────────────────────────
    l2_result = {"drift_detected": False, "drifted_columns": []}
    try:
        l2_cols = [c for c in LAYER2_FEATURES if c in ref_df.columns and c in cur_df.columns]
        if len(l2_cols) < 2:
            logger.warning(f"[drift_monitor] Layer 2: only {len(l2_cols)} matching columns — skipping layer")
        else:
            l2_report = Report(metrics=[DataDriftPreset()])
            l2_report.run(
                reference_data=ref_df[l2_cols],
                current_data=cur_df[l2_cols],
            )
            l2_html = output_path / f"drift_layer2_engineered_{batch_date}.html"
            l2_report.save_html(str(l2_html))
            l2_result = _extract_drift_summary(l2_report.as_dict())
            logger.info(
                f"[drift_monitor] Layer 2 — drift={l2_result['drift_detected']}  "
                f"drifted={l2_result['drifted_columns']}"
            )
    except Exception as exc:
        logger.warning(f"[drift_monitor] Layer 2 failed (non-fatal): {exc}")

    combined_drift = l1_result["drift_detected"] or l2_result["drift_detected"]
    if combined_drift:
        logger.warning(
            f"[drift_monitor] DRIFT DETECTED on {batch_date} — "
            f"L1={l1_result['drifted_columns']}  L2={l2_result['drifted_columns']}"
        )
    else:
        logger.info(f"[drift_monitor] No drift detected for batch {batch_date}")

    return {
        "drift_detected":          combined_drift,
        "layer1_drift":            l1_result["drift_detected"],
        "layer1_drifted_features": l1_result["drifted_columns"],
        "layer2_drift":            l2_result["drift_detected"],
        "layer2_drifted_features": l2_result["drifted_columns"],
        "report_path":             l1_report_path,
        "skipped":                 False,
        "skip_reason":             "",
    }
