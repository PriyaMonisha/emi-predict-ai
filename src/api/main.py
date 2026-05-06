# filename: src/api/main.py
# purpose:  FastAPI serving + Prometheus instrumentation for EMI prediction API
# version:  1.2

import contextlib
import io
import os
import sys
import time
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()   # loads .env from cwd — works when started from project root

# Force UTF-8 stdout so preprocess.py emoji print statements don't crash on Windows
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "buffer"):
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Response
from prometheus_client import (
    Counter, Histogram, Gauge,
    generate_latest, CONTENT_TYPE_LATEST,
)
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request as StarletteRequest

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.api.dependencies import ModelState, get_model_state, verify_api_key
from src.api.schemas import (
    BatchPredictRequest,
    BatchPredictResponse,
    HealthResponse,
    PredictRequest,
    PredictResponse,
)
from src.features.feature_store import health_check, read_features, write_features
from src.pipelines.predict_pipeline import (
    predict_from_preloaded,
    preprocess_for_inference,
)

logger = logging.getLogger(__name__)

# ── Prometheus metrics ─────────────────────────────────────────────────────────
_HTTP_REQUESTS = Counter(
    "http_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status_code"],
)
_HTTP_DURATION = Histogram(
    "http_request_duration_seconds",
    "HTTP request duration in seconds",
    ["method", "endpoint"],
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
)
_PREDICTIONS_TOTAL = Counter(
    "emi_predictions_total",
    "Total EMI predictions by confidence zone",
    ["conf_zone"],
)
_INFERENCE_DURATION = Histogram(
    "emi_inference_duration_seconds",
    "End-to-end inference duration (preprocess+FE+score)",
    ["endpoint"],
    buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
)
_CACHE_HITS   = Counter("emi_cache_hits_total",   "Redis feature cache hits")
_CACHE_MISSES = Counter("emi_cache_misses_total",  "Redis feature cache misses")
_REDIS_UP     = Gauge("emi_redis_up", "Redis reachability (1=up, 0=down)")


class _PrometheusMiddleware(BaseHTTPMiddleware):
    """Records HTTP request count and latency for every endpoint."""
    async def dispatch(self, request: StarletteRequest, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        duration = time.perf_counter() - start
        path = request.url.path
        _HTTP_REQUESTS.labels(request.method, path, str(response.status_code)).inc()
        _HTTP_DURATION.labels(request.method, path).observe(duration)
        return response


# ── Columns that go into the model (everything except customer_id) ─────────────
_FEATURE_COLS = [
    "age", "gender", "marital_status", "education", "monthly_salary",
    "employment_type", "years_of_employment", "company_type", "house_type",
    "monthly_rent", "family_size", "dependents", "school_fees", "college_fees",
    "travel_expenses", "groceries_utilities", "other_monthly_expenses",
    "existing_loans", "current_emi_amount", "credit_score", "bank_balance",
    "emergency_fund", "emi_scenario", "requested_amount", "requested_tenure",
]


# ── Lifespan: load models once at startup ─────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load directly from pkl files — avoids MLflow artifact URI resolution
    # which stores Windows paths (file:///C:/Users/...) incompatible with
    # Linux Docker containers. MODELS_DIR is /app/models in Docker, ./models locally.
    models_dir = Path(os.environ.get("MODELS_DIR", str(ROOT / "models")))
    logger.info(f"Loading champion models from {models_dir} …")

    import joblib
    fe  = joblib.load(models_dir / "feature_engineer.pkl")
    clf = joblib.load(models_dir / "best_classifier.pkl")
    reg = joblib.load(models_dir / "best_regressor.pkl")

    app.state.models = ModelState(clf=clf, reg=reg, fe=fe)
    logger.info("Models loaded — API ready")
    yield


# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="EMI Predict AI",
    description="Real-time EMI eligibility and amount prediction API",
    version="1.1.0",
    lifespan=lifespan,
)
app.add_middleware(_PrometheusMiddleware)


@app.get("/metrics", tags=["ops"], include_in_schema=False)
def metrics():
    """Prometheus scrape endpoint."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


# ── Helpers ────────────────────────────────────────────────────────────────────
def _request_to_df(req: PredictRequest) -> pd.DataFrame:
    """Build a 1-row DataFrame from a PredictRequest (excludes customer_id)."""
    return pd.DataFrame([{col: getattr(req, col) for col in _FEATURE_COLS}])


def _run_inference(
    df_raw: pd.DataFrame,
    state: ModelState,
    customer_id: str | None,
) -> tuple[pd.DataFrame, bool]:
    """
    Full inference pipeline: preprocess → FE transform → score.
    Returns (predictions_df, cache_hit).

    If customer_id is provided and Redis has a cache hit, skips preprocessing
    and FE transform — uses the stored 51-col feature vector directly.
    """
    cache_hit = False

    if customer_id:
        cached = read_features(customer_id)
        if cached is not None:
            df_features = pd.DataFrame([cached])
            cache_hit = True
            _CACHE_HITS.inc()
            logger.info(f"Cache hit for {customer_id}")
        else:
            _CACHE_MISSES.inc()
            with contextlib.redirect_stdout(io.StringIO()):
                df_preprocessed = preprocess_for_inference(df_raw)
            df_features = state.fe.transform(df_preprocessed)
            feature_dict = {k: df_features.iloc[0][k] for k in df_features.columns}
            write_features(customer_id, feature_dict)
    else:
        _CACHE_MISSES.inc()
        with contextlib.redirect_stdout(io.StringIO()):
            df_preprocessed = preprocess_for_inference(df_raw)
        df_features = state.fe.transform(df_preprocessed)

    if customer_id:
        df_features = df_features.copy()
        df_features["customer_id"] = customer_id

    preds = predict_from_preloaded(
        df_features,
        clf=state.clf,
        reg=state.reg,
        use_cache=False,   # cache already handled above for single predictions
    )
    return preds, cache_hit


# ── Endpoints ──────────────────────────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse, tags=["ops"])
def health():
    """Liveness + readiness check. No auth required."""
    redis_ok = health_check()
    _REDIS_UP.set(1 if redis_ok else 0)
    return HealthResponse(
        status="ok",
        redis_ok=redis_ok,
        models_loaded=hasattr(app.state, "models"),
    )


@app.post("/predict", response_model=PredictResponse, tags=["prediction"])
def predict(
    req: PredictRequest,
    state: ModelState = Depends(get_model_state),
    _: str = Depends(verify_api_key),
) -> PredictResponse:
    """
    Score a single customer for EMI eligibility and predicted EMI amount.

    Cache-aside: if customer_id is provided and Redis has the feature vector,
    preprocessing and feature engineering are skipped (~10x faster).
    """
    t0 = time.perf_counter()

    df_raw = _request_to_df(req)
    try:
        preds, cache_hit = _run_inference(df_raw, state, req.customer_id)
    except Exception as exc:
        logger.error(f"Inference failed: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Inference error: {exc}")

    latency = time.perf_counter() - t0
    row = preds.iloc[0]
    zone = str(row.conf_zone)

    _PREDICTIONS_TOTAL.labels(conf_zone=zone).inc()
    _INFERENCE_DURATION.labels(endpoint="/predict").observe(latency)

    return PredictResponse(
        customer_id=req.customer_id,
        clf_proba=float(row.clf_proba),
        clf_label=int(row.clf_label),
        conf_zone=zone,
        predicted_emi=float(row.predicted_emi),
        cache_hit=cache_hit,
        latency_ms=round(latency * 1000, 2),
    )


@app.post("/predict/batch", response_model=BatchPredictResponse, tags=["prediction"])
def predict_batch_endpoint(
    req: BatchPredictRequest,
    state: ModelState = Depends(get_model_state),
    _: str = Depends(verify_api_key),
) -> BatchPredictResponse:
    """
    Score up to 500 customers in a single request.

    Builds a single DataFrame and runs preprocessing + FE once for the whole
    batch — more efficient than calling /predict in a loop.
    No per-row cache check (the batch write after scoring populates the cache
    for subsequent individual /predict calls).
    """
    t0 = time.perf_counter()

    rows = [{col: getattr(r, col) for col in _FEATURE_COLS} for r in req.customers]
    df_raw = pd.DataFrame(rows)

    try:
        with contextlib.redirect_stdout(io.StringIO()):
            df_preprocessed = preprocess_for_inference(df_raw)
        df_features = state.fe.transform(df_preprocessed)

        # Attach customer_ids so the cache write inside predict_from_preloaded works
        ids = [r.customer_id for r in req.customers]
        if any(ids):
            df_features = df_features.copy()
            df_features["customer_id"] = ids

        preds = predict_from_preloaded(
            df_features,
            clf=state.clf,
            reg=state.reg,
            use_cache=True,
            customer_id_col="customer_id",
        )
    except Exception as exc:
        logger.error(f"Batch inference failed: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Batch inference error: {exc}")

    total_latency = time.perf_counter() - t0
    total_ms = round(total_latency * 1000, 2)
    zone_counts = preds["conf_zone"].value_counts().to_dict()

    for zone, count in zone_counts.items():
        _PREDICTIONS_TOTAL.labels(conf_zone=zone).inc(count)
    _INFERENCE_DURATION.labels(endpoint="/predict/batch").observe(total_latency)

    predictions = [
        PredictResponse(
            customer_id=req.customers[i].customer_id,
            clf_proba=float(row.clf_proba),
            clf_label=int(row.clf_label),
            conf_zone=str(row.conf_zone),
            predicted_emi=float(row.predicted_emi),
            cache_hit=False,
            latency_ms=round(total_ms / len(preds), 2),
        )
        for i, (_, row) in enumerate(preds.iterrows())
    ]

    return BatchPredictResponse(
        predictions=predictions,
        total_scored=len(preds),
        auto_approve=zone_counts.get("auto_approve", 0),
        human_review=zone_counts.get("human_review", 0),
        auto_reject=zone_counts.get("auto_reject", 0),
        total_latency_ms=total_ms,
    )
