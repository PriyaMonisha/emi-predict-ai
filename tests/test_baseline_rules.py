# filename: tests/test_baseline_rules.py
# purpose:  Unit tests for src/models/baseline_rules.py
# version:  1.0

import numpy as np
import pandas as pd
import pytest

from src.models.baseline_rules import (
    PROBA_ELIGIBLE,
    PROBA_REJECT,
    RuleBasedClassifier,
)


def _make_row(credit_score=720, expense_ratio=0.40, monthly_salary=50000,
              total_expenses=None) -> pd.DataFrame:
    """Build a 1-row DataFrame for rule evaluation."""
    if total_expenses is None:
        total_expenses = monthly_salary * expense_ratio
    return pd.DataFrame({
        "credit_score":       [credit_score],
        "monthly_salary":     [monthly_salary],
        "monthly_rent":       [total_expenses],
        "school_fees":        [0.0],
        "college_fees":       [0.0],
        "travel_expenses":    [0.0],
        "groceries_utilities":[0.0],
        "other_monthly_expenses": [0.0],
        "current_emi_amount": [0.0],
    })


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_fit_returns_self(raw_df):
    clf = RuleBasedClassifier()
    returned = clf.fit(raw_df)
    assert returned is clf
    assert clf.is_fitted_


def test_predict_before_fit_raises():
    clf = RuleBasedClassifier()
    with pytest.raises(ValueError, match="fit"):
        clf.predict_proba(_make_row())


def test_predict_proba_shape(raw_df):
    clf = RuleBasedClassifier().fit(raw_df)
    proba = clf.predict_proba(raw_df)
    assert proba.shape == (len(raw_df), 2)
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_eligible_case_gives_high_proba():
    """High credit, low expenses, good salary → auto-approve probability."""
    clf = RuleBasedClassifier()
    df = _make_row(credit_score=750, expense_ratio=0.30, monthly_salary=60000,
                   total_expenses=18000)
    clf.fit(df)
    proba = clf.predict_proba(df)[:, 1]
    assert float(proba[0]) == PROBA_ELIGIBLE


def test_reject_case_gives_low_proba():
    """Low credit, high expenses → auto-reject probability."""
    clf = RuleBasedClassifier()
    df = _make_row(credit_score=400, expense_ratio=0.90, monthly_salary=15000,
                   total_expenses=13500)
    clf.fit(df)
    proba = clf.predict_proba(df)[:, 1]
    assert float(proba[0]) == PROBA_REJECT


def test_confidence_zone_summary_keys(raw_df):
    clf = RuleBasedClassifier().fit(raw_df)
    summary = clf.confidence_zone_summary(raw_df)
    required_keys = {
        "auto_approve", "human_review", "auto_reject", "total",
        "auto_approve_pct", "human_review_pct", "auto_reject_pct",
    }
    assert required_keys.issubset(summary.keys())
    assert summary["total"] == len(raw_df)
    assert (
        summary["auto_approve"] + summary["human_review"] + summary["auto_reject"]
        == summary["total"]
    )
