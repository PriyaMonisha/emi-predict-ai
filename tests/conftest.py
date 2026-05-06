# filename: tests/conftest.py
# purpose:  Shared pytest fixtures for EMI Predict AI test suite
# version:  1.0

import os

import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock

# Set before any src.api import so verify_api_key passes in endpoint tests
os.environ.setdefault("API_KEY", "test-key")

from src.features.feature_engineering import FeatureEngineer


# ── Pre-import main.py safely ────────────────────────────────────────────────
# main.py replaces sys.stdout/sys.stderr at module level (Windows UTF-8 fix).
# When pytest captures output, sys.stdout is a TextIOWrapper whose .buffer is
# a BytesIO owned by pytest's capture.  Wrapping that BytesIO a second time
# causes "I/O operation on closed file" on subsequent test teardowns.
# Fix: import src.api.main NOW (conftest module level), with a fake stdout that
# has no .buffer attribute → hasattr check in main.py is False → no-op.
# Python module cache guarantees the module-level code never runs again.
import sys as _sys
import io as _io

if "src.api.main" not in _sys.modules:
    class _NoBuf:
        """Fake stdout/stderr without .buffer — blocks main.py's UTF-8 patch."""
        encoding = "utf-8"
        errors   = "replace"
        def write(self, s): pass
        def flush(self): pass
        def fileno(self): raise _io.UnsupportedOperation("fileno")

    _saved_out, _saved_err = _sys.stdout, _sys.stderr
    _sys.stdout = _sys.stderr = _NoBuf()
    try:
        import src.api.main  # noqa: F401 — exhausts module-level side effects
    except Exception:
        pass
    finally:
        _sys.stdout, _sys.stderr = _saved_out, _saved_err


# ── Core data fixtures ─────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def raw_df() -> pd.DataFrame:
    """10-row DataFrame with all 27 expected columns in canonical form."""
    return pd.DataFrame({
        "age":                    [25, 30, 35, 28, 45, 22, 38, 55, 32, 40],
        "gender":                 ["Male", "Female", "Male", "Female", "Male",
                                   "Female", "Male", "Female", "Male", "Female"],
        "marital_status":         ["Single", "Married", "Married", "Single",
                                   "Married", "Single", "Married", "Married",
                                   "Single", "Married"],
        "education":              ["Graduate", "Post-Graduate", "Graduate",
                                   "Undergraduate", "Graduate", "Post-Graduate",
                                   "Graduate", "Undergraduate", "Graduate",
                                   "Post-Graduate"],
        "monthly_salary":         [50000.0, 60000.0, 45000.0, 70000.0, 80000.0,
                                   30000.0, 55000.0, 90000.0, 40000.0, 65000.0],
        "employment_type":        ["Private", "Government", "Private",
                                   "Self-Employed", "Government", "Private",
                                   "Government", "Private", "Self-Employed",
                                   "Private"],
        "years_of_employment":    [3.0, 8.0, 5.0, 2.0, 15.0,
                                   1.0, 10.0, 20.0, 4.0, 12.0],
        "company_type":           ["Private", "Government", "Mnc",
                                   "Self-Employed", "Large Indian", "Private",
                                   "Government", "Mnc", "Mid-Size", "Private"],
        "house_type":             ["Rented", "Own", "Rented", "Family", "Own",
                                   "Rented", "Own", "Own", "Rented", "Family"],
        "monthly_rent":           [10000.0, 0.0, 8000.0, 0.0, 0.0,
                                   7000.0, 0.0, 0.0, 9000.0, 0.0],
        "family_size":            [3, 4, 2, 3, 5, 2, 4, 3, 2, 4],
        "dependents":             [1, 2, 0, 1, 3, 0, 2, 1, 0, 2],
        "school_fees":            [0.0, 5000.0, 3000.0, 0.0, 8000.0,
                                   0.0, 4000.0, 6000.0, 0.0, 5000.0],
        "college_fees":           [0.0, 0.0, 0.0, 0.0, 15000.0,
                                   0.0, 0.0, 10000.0, 0.0, 0.0],
        "travel_expenses":        [2000.0, 3000.0, 1500.0, 2500.0, 4000.0,
                                   1000.0, 3500.0, 5000.0, 2000.0, 3000.0],
        "groceries_utilities":    [5000.0, 6000.0, 4000.0, 7000.0, 8000.0,
                                   3000.0, 6500.0, 9000.0, 4500.0, 7000.0],
        "other_monthly_expenses": [1000.0, 2000.0, 500.0, 1500.0, 2000.0,
                                   500.0, 1500.0, 3000.0, 1000.0, 2000.0],
        "existing_loans":         ["No", "Yes", "No", "Yes", "No",
                                   "No", "Yes", "No", "Yes", "No"],
        "current_emi_amount":     [0.0, 5000.0, 0.0, 8000.0, 0.0,
                                   0.0, 6000.0, 0.0, 4000.0, 0.0],
        "credit_score":           [720.0, 680.0, 750.0, 620.0, 800.0,
                                   580.0, 700.0, 760.0, 640.0, 730.0],
        "bank_balance":           [100000.0, 150000.0, 80000.0, 200000.0,
                                   300000.0, 50000.0, 120000.0, 400000.0,
                                   90000.0, 180000.0],
        "emergency_fund":         [20000.0, 30000.0, 15000.0, 40000.0, 60000.0,
                                   10000.0, 25000.0, 80000.0, 18000.0, 35000.0],
        "emi_scenario":           ["Personal Loan Emi", "Vehicle Emi",
                                   "Education Emi", "Personal Loan Emi",
                                   "Home Appliances Emi", "Personal Loan Emi",
                                   "Vehicle Emi", "Education Emi",
                                   "Personal Loan Emi", "Vehicle Emi"],
        "requested_amount":       [200000.0, 500000.0, 300000.0, 800000.0,
                                   1000000.0, 150000.0, 400000.0, 600000.0,
                                   250000.0, 700000.0],
        "requested_tenure":       [24.0, 36.0, 12.0, 48.0, 60.0,
                                   18.0, 36.0, 24.0, 12.0, 48.0],
        "emi_eligibility":        ["Not_Eligible", "Eligible", "Not_Eligible",
                                   "Eligible", "Eligible", "Not_Eligible",
                                   "Eligible", "Eligible", "Not_Eligible",
                                   "Eligible"],
        "max_monthly_emi":        [5000.0, 10000.0, 7000.0, 15000.0, 20000.0,
                                   4000.0, 9000.0, 18000.0, 6000.0, 12000.0],
    })


@pytest.fixture(scope="session")
def fitted_fe(raw_df) -> FeatureEngineer:
    """FeatureEngineer fitted on the 10-row test DataFrame."""
    fe = FeatureEngineer()
    fe.fit(raw_df)
    return fe


# ── Model mocks ────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def mock_clf():
    """Minimal sklearn-compatible classifier mock (no feature_names_in_)."""
    clf = MagicMock(spec=["predict_proba", "predict"])
    clf.predict_proba.side_effect = lambda X: np.column_stack([
        np.ones(len(X)) * 0.1,
        np.ones(len(X)) * 0.9,
    ])
    clf.predict.side_effect = lambda X: np.ones(len(X), dtype=int)
    return clf


@pytest.fixture(scope="session")
def mock_reg():
    """Minimal sklearn-compatible regressor mock (no feature_names_in_)."""
    reg = MagicMock(spec=["predict"])
    reg.predict.side_effect = lambda X: np.ones(len(X)) * 8000.0
    return reg


# ── API helpers ────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def valid_payload() -> dict:
    """Full valid PredictRequest payload (all 25 feature fields)."""
    return {
        "age": 30,
        "gender": "Male",
        "marital_status": "Married",
        "education": "Graduate",
        "monthly_salary": 60000.0,
        "employment_type": "Private",
        "years_of_employment": 5.0,
        "company_type": "Mnc",
        "house_type": "Rented",
        "monthly_rent": 10000.0,
        "family_size": 3,
        "dependents": 1,
        "school_fees": 0.0,
        "college_fees": 0.0,
        "travel_expenses": 2000.0,
        "groceries_utilities": 5000.0,
        "other_monthly_expenses": 1000.0,
        "existing_loans": "No",
        "current_emi_amount": 0.0,
        "credit_score": 720.0,
        "bank_balance": 100000.0,
        "emergency_fund": 20000.0,
        "emi_scenario": "Personal Loan Emi",
        "requested_amount": 300000.0,
        "requested_tenure": 36.0,
    }
