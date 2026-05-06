# filename: tests/test_preprocess.py
# purpose:  Unit tests for src/data/preprocess.py
# version:  1.0

import numpy as np
import pandas as pd
import pytest

from src.data.preprocess import preprocess_data


def _run(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """Run preprocessing with safe defaults for tests (no disk writes)."""
    return preprocess_data(df.copy(), save_unlabeled=False, **kwargs)


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_preprocess_returns_dataframe(raw_df):
    result = _run(raw_df)
    assert isinstance(result, pd.DataFrame)
    assert len(result) > 0


def test_removes_duplicates(raw_df):
    doubled = pd.concat([raw_df, raw_df], ignore_index=True)
    result = _run(doubled)
    # Duplicates from the concat should be removed
    assert len(result) <= len(raw_df)


def test_creates_credit_score_missing_flag():
    """credit_score null in raw data → credit_score_missing flag column created."""
    df = pd.DataFrame({
        "age": [30], "gender": ["Male"], "marital_status": ["Married"],
        "education": ["Graduate"], "monthly_salary": [50000.0],
        "employment_type": ["Private"], "years_of_employment": [5.0],
        "company_type": ["Private"], "house_type": ["Own"],
        "monthly_rent": [0.0], "family_size": [3], "dependents": [1],
        "school_fees": [0.0], "college_fees": [0.0], "travel_expenses": [2000.0],
        "groceries_utilities": [5000.0], "other_monthly_expenses": [1000.0],
        "existing_loans": ["No"], "current_emi_amount": [0.0],
        "credit_score": [np.nan],   # <-- null to trigger flag
        "bank_balance": [100000.0], "emergency_fund": [20000.0],
        "emi_scenario": ["Personal Loan Emi"], "requested_amount": [300000.0],
        "requested_tenure": [36.0], "emi_eligibility": ["Eligible"],
        "max_monthly_emi": [8000.0],
    })
    result = _run(df)
    assert "credit_score_missing" in result.columns
    assert result["credit_score_missing"].iloc[0] == 1


def test_target_encoding_maps_labels(raw_df):
    """Eligible → 1, Not_Eligible → 0 in training mode."""
    result = _run(raw_df, is_training=True)
    assert set(result["emi_eligibility"].unique()).issubset({0, 1})
    assert result["emi_eligibility"].dtype in (
        np.dtype("int64"), np.dtype("float64"), np.dtype("int32")
    )


def test_inference_mode_skips_target_encoding(raw_df):
    """is_training=False — emi_eligibility stays as string, not encoded."""
    result = _run(raw_df, is_training=False)
    assert result["emi_eligibility"].dtype == object
    assert set(result["emi_eligibility"].unique()).issubset(
        {"Eligible", "Not_Eligible"}
    )


def test_value_standardization_dirty_gender():
    """'M' and 'F' abbreviations → 'Male' / 'Female'."""
    df = pd.DataFrame({
        "age": [28, 35], "gender": ["M", "F"],
        "marital_status": ["Single", "Married"],
        "education": ["Graduate", "Graduate"],
        "monthly_salary": [50000.0, 60000.0],
        "employment_type": ["Private", "Private"],
        "years_of_employment": [3.0, 5.0],
        "company_type": ["Private", "Private"],
        "house_type": ["Own", "Own"],
        "monthly_rent": [0.0, 0.0], "family_size": [2, 3],
        "dependents": [0, 1], "school_fees": [0.0, 0.0],
        "college_fees": [0.0, 0.0], "travel_expenses": [1000.0, 1500.0],
        "groceries_utilities": [4000.0, 5000.0],
        "other_monthly_expenses": [500.0, 800.0],
        "existing_loans": ["No", "No"], "current_emi_amount": [0.0, 0.0],
        "credit_score": [700.0, 720.0],
        "bank_balance": [80000.0, 100000.0],
        "emergency_fund": [15000.0, 20000.0],
        "emi_scenario": ["Personal Loan Emi", "Vehicle Emi"],
        "requested_amount": [200000.0, 300000.0],
        "requested_tenure": [24.0, 36.0],
        "emi_eligibility": ["Eligible", "Not_Eligible"],
        "max_monthly_emi": [8000.0, 6000.0],
    })
    result = _run(df, is_training=True)
    genders = result["gender"].astype(str).unique()
    assert "Male" in genders or "Female" in genders
    assert "M" not in genders
    assert "F" not in genders


def test_outlier_capping_applied(raw_df):
    """Values beyond 1st–99th percentile are clipped (monthly_salary capped)."""
    df = raw_df.copy()
    # Inject an extreme outlier
    df.loc[0, "monthly_salary"] = 1e9
    result = _run(df, is_training=True)
    assert result["monthly_salary"].max() < 1e9
