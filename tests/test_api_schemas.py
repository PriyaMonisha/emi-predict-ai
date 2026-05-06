# filename: tests/test_api_schemas.py
# purpose:  Unit tests for src/api/schemas.py Pydantic validation
# version:  1.0

import pytest
from pydantic import ValidationError

from src.api.schemas import BatchPredictRequest, PredictRequest


def _valid_request(**overrides) -> dict:
    base = {
        "age": 30, "gender": "Male", "marital_status": "Married",
        "education": "Graduate", "monthly_salary": 60000.0,
        "employment_type": "Private", "years_of_employment": 5.0,
        "company_type": "Mnc", "house_type": "Rented",
        "monthly_rent": 10000.0, "family_size": 3, "dependents": 1,
        "school_fees": 0.0, "college_fees": 0.0,
        "travel_expenses": 2000.0, "groceries_utilities": 5000.0,
        "other_monthly_expenses": 1000.0, "existing_loans": "No",
        "current_emi_amount": 0.0, "credit_score": 720.0,
        "bank_balance": 100000.0, "emergency_fund": 20000.0,
        "emi_scenario": "Personal Loan Emi", "requested_amount": 300000.0,
        "requested_tenure": 36.0,
    }
    base.update(overrides)
    return base


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_valid_predict_request_parses():
    req = PredictRequest(**_valid_request())
    assert req.age == 30
    assert req.monthly_salary == 60000.0
    assert req.credit_score == 720.0


def test_age_below_minimum_raises():
    with pytest.raises(ValidationError, match="age"):
        PredictRequest(**_valid_request(age=17))


def test_age_above_maximum_raises():
    with pytest.raises(ValidationError, match="age"):
        PredictRequest(**_valid_request(age=101))


def test_negative_salary_raises():
    with pytest.raises(ValidationError, match="monthly_salary"):
        PredictRequest(**_valid_request(monthly_salary=-1.0))


def test_requested_amount_zero_raises():
    with pytest.raises(ValidationError, match="requested_amount"):
        PredictRequest(**_valid_request(requested_amount=0.0))


def test_optional_fields_can_be_none():
    req = PredictRequest(**_valid_request(
        customer_id=None, credit_score=None,
        bank_balance=None, emergency_fund=None,
    ))
    assert req.customer_id is None
    assert req.credit_score is None


def test_string_fields_stripped():
    req = PredictRequest(**_valid_request(gender="  Male  "))
    assert req.gender == "Male"


def test_batch_request_empty_list_raises():
    with pytest.raises(ValidationError):
        BatchPredictRequest(customers=[])


def test_batch_request_over_500_raises():
    items = [_valid_request() for _ in range(501)]
    with pytest.raises(ValidationError):
        BatchPredictRequest(customers=items)


def test_batch_request_valid_single_item():
    req = BatchPredictRequest(customers=[_valid_request()])
    assert len(req.customers) == 1
