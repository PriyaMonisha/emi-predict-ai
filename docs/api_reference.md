# EMI Predict AI — API Reference

**Base URL:** `http://localhost:8000`
**API version:** 1.1.0
**Interactive docs:** http://localhost:8000/docs (Swagger UI)

---

## Authentication

All prediction endpoints require an API key in the request header:

```
X-API-Key: <your-api-key>
```

The key is set via the `API_KEY` environment variable (`.env` file or Docker Compose env). Health and metrics endpoints do not require authentication.

**Error response when key is missing or invalid:**

```json
HTTP 403 Forbidden
{"detail": "Invalid or missing API key"}
```

---

## Endpoints

### GET /health

Liveness and readiness check. No authentication required.

**Request:**

```bash
curl http://localhost:8000/health
```

**Response `200 OK`:**

```json
{
  "status": "ok",
  "redis_ok": true,
  "models_loaded": true
}
```

| Field | Type | Description |
|---|---|---|
| `status` | string | Always `"ok"` when the endpoint responds |
| `redis_ok` | boolean | `true` if Redis is reachable |
| `models_loaded` | boolean | `true` if classifier, regressor, and feature engineer are loaded in memory |

---

### POST /predict

Score a single customer for EMI eligibility and predicted monthly EMI amount.

Cache-aside: if `customer_id` is supplied and Redis holds the feature vector from a prior call, preprocessing and feature engineering are skipped (~10× faster).

**Request headers:**

```
Content-Type: application/json
X-API-Key: <your-api-key>
```

**Request body — `PredictRequest`:**

| Field | Type | Required | Constraints | Description |
|---|---|---|---|---|
| `customer_id` | string | No | — | Optional identifier; enables Redis caching |
| `age` | integer | Yes | 18 – 100 | Applicant age |
| `gender` | string | Yes | — | e.g. `"Male"`, `"Female"` |
| `marital_status` | string | Yes | — | e.g. `"Single"`, `"Married"` |
| `education` | string | Yes | — | e.g. `"Graduate"`, `"Post Graduate"` |
| `monthly_salary` | float | Yes | ≥ 0 | Gross monthly salary (₹) |
| `employment_type` | string | Yes | — | e.g. `"Salaried"`, `"Self-Employed"` |
| `years_of_employment` | float | Yes | ≥ 0 | Total years in current employment |
| `company_type` | string | Yes | — | e.g. `"Private"`, `"Public"`, `"Government"` |
| `house_type` | string | Yes | — | e.g. `"Owned"`, `"Rented"`, `"Family-Owned"` |
| `monthly_rent` | float | Yes | ≥ 0 | Monthly rent paid (₹); 0 if owned |
| `family_size` | integer | Yes | ≥ 1 | Total household members |
| `dependents` | integer | Yes | ≥ 0 | Number of financial dependents |
| `school_fees` | float | Yes | ≥ 0 | Monthly school fee expenses (₹) |
| `college_fees` | float | Yes | ≥ 0 | Monthly college fee expenses (₹) |
| `travel_expenses` | float | Yes | ≥ 0 | Monthly travel expenses (₹) |
| `groceries_utilities` | float | Yes | ≥ 0 | Monthly groceries + utilities (₹) |
| `other_monthly_expenses` | float | Yes | ≥ 0 | All other monthly expenses (₹) |
| `existing_loans` | string | Yes | — | e.g. `"None"`, `"Home Loan"`, `"Personal Loan"` |
| `current_emi_amount` | float | Yes | ≥ 0 | Total existing EMI payments per month (₹) |
| `credit_score` | float | No | — | Credit bureau score; `null` triggers missing-flag |
| `bank_balance` | float | No | — | Current bank balance (₹); `null` triggers missing-flag |
| `emergency_fund` | float | No | — | Emergency savings (₹); `null` triggers missing-flag |
| `emi_scenario` | string | Yes | — | e.g. `"Conservative"`, `"Moderate"`, `"Aggressive"` |
| `requested_amount` | float | Yes | > 0 | Loan amount requested (₹) |
| `requested_tenure` | float | Yes | > 0 | Loan tenure in months |

**Example request:**

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key-here" \
  -d '{
    "customer_id": "CUST-001",
    "age": 32,
    "gender": "Male",
    "marital_status": "Married",
    "education": "Graduate",
    "monthly_salary": 65000,
    "employment_type": "Salaried",
    "years_of_employment": 5.0,
    "company_type": "Private",
    "house_type": "Owned",
    "monthly_rent": 0,
    "family_size": 3,
    "dependents": 1,
    "school_fees": 2000,
    "college_fees": 0,
    "travel_expenses": 3000,
    "groceries_utilities": 8000,
    "other_monthly_expenses": 1500,
    "existing_loans": "None",
    "current_emi_amount": 0,
    "credit_score": 720,
    "bank_balance": 250000,
    "emergency_fund": 100000,
    "emi_scenario": "Conservative",
    "requested_amount": 500000,
    "requested_tenure": 36
  }'
```

**Response `200 OK` — `PredictResponse`:**

```json
{
  "customer_id": "CUST-001",
  "clf_proba": 0.924371,
  "clf_label": 1,
  "conf_zone": "auto_approve",
  "predicted_emi": 15832.50,
  "cache_hit": false,
  "latency_ms": 42.7
}
```

| Field | Type | Description |
|---|---|---|
| `customer_id` | string / null | Echoed from request |
| `clf_proba` | float | Eligibility probability ∈ [0, 1] (6 decimal places) |
| `clf_label` | integer | 0 = Not Eligible, 1 = Eligible (threshold 0.5) |
| `conf_zone` | string | `"auto_approve"` / `"human_review"` / `"auto_reject"` |
| `predicted_emi` | float | Predicted max monthly EMI (₹), clipped to ₹500–₹34,750 |
| `cache_hit` | boolean | `true` if Redis feature vector was used |
| `latency_ms` | float | End-to-end inference time in milliseconds |

---

### POST /predict/batch

Score up to 500 customers in a single request. More efficient than calling `/predict` in a loop — preprocessing and feature engineering run once for the whole batch.

**Request body — `BatchPredictRequest`:**

```json
{
  "customers": [<PredictRequest>, <PredictRequest>, ...]
}
```

- `customers`: array of `PredictRequest` objects
- Minimum: 1 customer
- Maximum: 500 customers

**Example request:**

```bash
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key-here" \
  -d '{
    "customers": [
      {
        "customer_id": "CUST-001",
        "age": 32,
        "gender": "Male",
        "marital_status": "Married",
        "education": "Graduate",
        "monthly_salary": 65000,
        "employment_type": "Salaried",
        "years_of_employment": 5.0,
        "company_type": "Private",
        "house_type": "Owned",
        "monthly_rent": 0,
        "family_size": 3,
        "dependents": 1,
        "school_fees": 2000,
        "college_fees": 0,
        "travel_expenses": 3000,
        "groceries_utilities": 8000,
        "other_monthly_expenses": 1500,
        "existing_loans": "None",
        "current_emi_amount": 0,
        "credit_score": 720,
        "bank_balance": 250000,
        "emergency_fund": 100000,
        "emi_scenario": "Conservative",
        "requested_amount": 500000,
        "requested_tenure": 36
      }
    ]
  }'
```

**Response `200 OK` — `BatchPredictResponse`:**

```json
{
  "predictions": [
    {
      "customer_id": "CUST-001",
      "clf_proba": 0.924371,
      "clf_label": 1,
      "conf_zone": "auto_approve",
      "predicted_emi": 15832.50,
      "cache_hit": false,
      "latency_ms": 8.5
    }
  ],
  "total_scored": 1,
  "auto_approve": 1,
  "human_review": 0,
  "auto_reject": 0,
  "total_latency_ms": 8.5
}
```

| Field | Type | Description |
|---|---|---|
| `predictions` | array | One `PredictResponse` per input customer, in order |
| `total_scored` | integer | Count of customers scored |
| `auto_approve` | integer | Count in `auto_approve` zone |
| `human_review` | integer | Count in `human_review` zone |
| `auto_reject` | integer | Count in `auto_reject` zone |
| `total_latency_ms` | float | Wall-clock time for the entire batch |

---

### GET /metrics

Prometheus scrape endpoint. Returns text-format metrics. No authentication required. Not included in Swagger UI (`include_in_schema=False`).

```bash
curl http://localhost:8000/metrics
```

**Key metrics exposed:**

| Metric | Type | Labels | Description |
|---|---|---|---|
| `emi_predictions_total` | Counter | `conf_zone` | Cumulative predictions by confidence zone |
| `emi_inference_duration_seconds` | Histogram | `endpoint` | Inference latency distribution |
| `emi_cache_hits_total` | Counter | — | Redis feature cache hits |
| `emi_cache_misses_total` | Counter | — | Redis feature cache misses |
| `emi_redis_up` | Gauge | — | 1 if Redis reachable, 0 if down |
| `http_requests_total` | Counter | `method`, `endpoint`, `status_code` | All HTTP requests |
| `http_request_duration_seconds` | Histogram | `method`, `endpoint` | HTTP request latency |

---

## Error Codes

| HTTP Status | When | Response body |
|---|---|---|
| `200 OK` | Successful prediction or health check | PredictResponse / HealthResponse |
| `403 Forbidden` | Missing or invalid `X-API-Key` | `{"detail": "Invalid or missing API key"}` |
| `422 Unprocessable Entity` | Request body fails Pydantic validation | Pydantic error detail with field paths |
| `500 Internal Server Error` | Inference pipeline error (preprocessing, FE, or model) | `{"detail": "Inference error: <message>"}` |

---

## Notes

- String fields (`gender`, `marital_status`, etc.) are stripped of leading/trailing whitespace by the Pydantic validator before processing.
- `credit_score`, `bank_balance`, and `emergency_fund` are nullable. Passing `null` is valid — the preprocessing pipeline adds corresponding `_missing` flag columns automatically.
- `predicted_emi` is always clipped to ₹500–₹34,750 (training distribution range), regardless of the requested amount.
- The `/predict/batch` endpoint does NOT perform per-row cache reads (no Redis GET per customer). It does perform a bulk cache write after scoring, so repeat calls to `/predict` for the same customers will be faster.
