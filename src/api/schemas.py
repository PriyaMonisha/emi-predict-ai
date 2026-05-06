# filename: src/api/schemas.py
# purpose:  Pydantic request/response models for EMI prediction API
# version:  1.0

from typing import Optional
from pydantic import BaseModel, Field, field_validator


class PredictRequest(BaseModel):
    """Raw customer features for a single EMI prediction request.

    customer_id is optional — supply it to enable Redis cache lookups.
    credit_score, bank_balance, and emergency_fund may be null; the
    preprocessing pipeline computes missing-flag columns automatically.
    """
    customer_id:            Optional[str]   = None
    age:                    int             = Field(..., ge=18, le=100)
    gender:                 str
    marital_status:         str
    education:              str
    monthly_salary:         float           = Field(..., ge=0)
    employment_type:        str
    years_of_employment:    float           = Field(..., ge=0)
    company_type:           str
    house_type:             str
    monthly_rent:           float           = Field(..., ge=0)
    family_size:            int             = Field(..., ge=1)
    dependents:             int             = Field(..., ge=0)
    school_fees:            float           = Field(..., ge=0)
    college_fees:           float           = Field(..., ge=0)
    travel_expenses:        float           = Field(..., ge=0)
    groceries_utilities:    float           = Field(..., ge=0)
    other_monthly_expenses: float           = Field(..., ge=0)
    existing_loans:         str
    current_emi_amount:     float           = Field(..., ge=0)
    credit_score:           Optional[float] = None
    bank_balance:           Optional[float] = None
    emergency_fund:         Optional[float] = None
    emi_scenario:           str
    requested_amount:       float           = Field(..., gt=0)
    requested_tenure:       float           = Field(..., gt=0)

    @field_validator("gender", "marital_status", "education", "employment_type",
                     "company_type", "house_type", "existing_loans", "emi_scenario",
                     mode="before")
    @classmethod
    def strip_strings(cls, v: str) -> str:
        return v.strip() if isinstance(v, str) else v


class PredictResponse(BaseModel):
    customer_id:   Optional[str]
    clf_proba:     float
    clf_label:     int
    conf_zone:     str
    predicted_emi: float
    cache_hit:     bool
    latency_ms:    float


class BatchPredictRequest(BaseModel):
    customers: list[PredictRequest] = Field(..., min_length=1, max_length=500)


class BatchPredictResponse(BaseModel):
    predictions:   list[PredictResponse]
    total_scored:  int
    auto_approve:  int
    human_review:  int
    auto_reject:   int
    total_latency_ms: float


class HealthResponse(BaseModel):
    status:        str
    redis_ok:      bool
    models_loaded: bool
