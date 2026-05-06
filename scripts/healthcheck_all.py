# filename: scripts/healthcheck_all.py
# purpose:  Hit all EMI Predict AI service endpoints and print pass/fail summary
# version:  1.0

import os
import sys
import json
import time
import subprocess
from typing import NamedTuple

try:
    import requests
except ImportError:
    print("requests not installed — run: pip install requests")
    sys.exit(1)

API_KEY = os.environ.get("API_KEY", "")
BASE_URL = os.environ.get("EMI_API_URL", "http://localhost:8000")
TIMEOUT  = 10

SAMPLE_PAYLOAD = {
    "customer_id": "HEALTHCHECK-001",
    "age": 35,
    "gender": "Male",
    "marital_status": "Married",
    "education": "Graduate",
    "monthly_salary": 60000.0,
    "employment_type": "Salaried",
    "years_of_employment": 7.0,
    "company_type": "Private",
    "house_type": "Owned",
    "monthly_rent": 0.0,
    "family_size": 3,
    "dependents": 1,
    "school_fees": 1500.0,
    "college_fees": 0.0,
    "travel_expenses": 2500.0,
    "groceries_utilities": 7000.0,
    "other_monthly_expenses": 1000.0,
    "existing_loans": "None",
    "current_emi_amount": 0.0,
    "credit_score": 700.0,
    "bank_balance": 200000.0,
    "emergency_fund": 80000.0,
    "emi_scenario": "Conservative",
    "requested_amount": 400000.0,
    "requested_tenure": 36.0,
}


class CheckResult(NamedTuple):
    name: str
    passed: bool
    detail: str
    latency_ms: float


def _get(url: str, **kwargs) -> tuple:
    t0 = time.perf_counter()
    try:
        r = requests.get(url, timeout=TIMEOUT, **kwargs)
        ms = (time.perf_counter() - t0) * 1000
        return r, ms, None
    except Exception as e:
        ms = (time.perf_counter() - t0) * 1000
        return None, ms, str(e)


def _post(url: str, **kwargs) -> tuple:
    t0 = time.perf_counter()
    try:
        r = requests.post(url, timeout=TIMEOUT, **kwargs)
        ms = (time.perf_counter() - t0) * 1000
        return r, ms, None
    except Exception as e:
        ms = (time.perf_counter() - t0) * 1000
        return None, ms, str(e)


def check_fastapi_health() -> CheckResult:
    r, ms, err = _get(f"{BASE_URL}/health")
    if err:
        return CheckResult("FastAPI /health", False, err, ms)
    if r.status_code != 200:
        return CheckResult("FastAPI /health", False, f"HTTP {r.status_code}", ms)
    body = r.json()
    if body.get("status") != "ok":
        return CheckResult("FastAPI /health", False, f"status={body.get('status')}", ms)
    detail = f"redis_ok={body['redis_ok']}  models_loaded={body['models_loaded']}"
    passed = body["redis_ok"] and body["models_loaded"]
    return CheckResult("FastAPI /health", passed, detail, ms)


def check_fastapi_auth_rejected() -> CheckResult:
    r, ms, err = _post(
        f"{BASE_URL}/predict",
        headers={"Content-Type": "application/json", "X-API-Key": "wrong-key"},
        json=SAMPLE_PAYLOAD,
    )
    if err:
        return CheckResult("FastAPI auth rejection", False, err, ms)
    passed = r.status_code == 403
    return CheckResult("FastAPI auth rejection (wrong key → 403)", passed, f"HTTP {r.status_code}", ms)


def check_fastapi_predict() -> CheckResult:
    if not API_KEY:
        return CheckResult("FastAPI POST /predict", False, "API_KEY env var not set — skipped", 0.0)
    headers = {"Content-Type": "application/json", "X-API-Key": API_KEY}
    r, ms, err = _post(f"{BASE_URL}/predict", headers=headers, json=SAMPLE_PAYLOAD)
    if err:
        return CheckResult("FastAPI POST /predict", False, err, ms)
    if r.status_code != 200:
        return CheckResult("FastAPI POST /predict", False, f"HTTP {r.status_code}: {r.text[:200]}", ms)
    body = r.json()
    zone = body.get("conf_zone", "?")
    proba = body.get("clf_proba", "?")
    emi = body.get("predicted_emi", "?")
    detail = f"conf_zone={zone}  proba={proba}  predicted_emi=₹{emi}  cache_hit={body.get('cache_hit')}"
    return CheckResult("FastAPI POST /predict", True, detail, ms)


def check_fastapi_metrics() -> CheckResult:
    r, ms, err = _get(f"{BASE_URL}/metrics")
    if err:
        return CheckResult("FastAPI GET /metrics", False, err, ms)
    if r.status_code != 200:
        return CheckResult("FastAPI GET /metrics", False, f"HTTP {r.status_code}", ms)
    has_emi = "emi_predictions_total" in r.text
    has_redis = "emi_redis_up" in r.text
    passed = has_emi and has_redis
    detail = f"emi_predictions_total={'found' if has_emi else 'MISSING'}  emi_redis_up={'found' if has_redis else 'MISSING'}"
    return CheckResult("FastAPI GET /metrics", passed, detail, ms)


def check_mlflow() -> CheckResult:
    r, ms, err = _get("http://localhost:5000/health")
    if err:
        return CheckResult("MLflow :5000", False, err, ms)
    passed = r.status_code == 200
    return CheckResult("MLflow :5000 /health", passed, f"HTTP {r.status_code}", ms)


def check_airflow() -> CheckResult:
    r, ms, err = _get("http://localhost:8080/health")
    if err:
        return CheckResult("Airflow :8080", False, err, ms)
    if r.status_code != 200:
        return CheckResult("Airflow :8080 /health", False, f"HTTP {r.status_code}", ms)
    body = r.json()
    scheduler_status = body.get("scheduler", {}).get("status", "?")
    passed = scheduler_status == "healthy"
    return CheckResult("Airflow :8080 /health", passed, f"scheduler={scheduler_status}", ms)


def check_prometheus() -> CheckResult:
    r, ms, err = _get("http://localhost:9090/-/healthy")
    if err:
        return CheckResult("Prometheus :9090", False, err, ms)
    passed = r.status_code == 200
    return CheckResult("Prometheus :9090 /-/healthy", passed, f"HTTP {r.status_code}", ms)


def check_grafana() -> CheckResult:
    r, ms, err = _get("http://localhost:3000/api/health")
    if err:
        return CheckResult("Grafana :3000", False, err, ms)
    passed = r.status_code == 200
    return CheckResult("Grafana :3000 /api/health", passed, f"HTTP {r.status_code}", ms)


def check_streamlit() -> CheckResult:
    r, ms, err = _get("http://localhost:8501/_stcore/health")
    if err:
        return CheckResult("Streamlit :8501", False, err, ms)
    passed = r.status_code == 200
    return CheckResult("Streamlit :8501 /_stcore/health", passed, f"HTTP {r.status_code}", ms)


def check_redis_docker() -> CheckResult:
    t0 = time.perf_counter()
    try:
        result = subprocess.run(
            ["docker", "exec", "emi-redis", "redis-cli", "ping"],
            capture_output=True, text=True, timeout=5,
        )
        ms = (time.perf_counter() - t0) * 1000
        passed = result.stdout.strip() == "PONG"
        return CheckResult("Redis (docker exec ping)", passed, result.stdout.strip() or result.stderr.strip(), ms)
    except Exception as e:
        ms = (time.perf_counter() - t0) * 1000
        return CheckResult("Redis (docker exec ping)", False, str(e), ms)


def main() -> None:
    print()
    print("=" * 65)
    print("  EMI Predict AI — Stack Health Check")
    print("=" * 65)
    if not API_KEY:
        print("  TIP: set API_KEY env var to test POST /predict")
        print("       PowerShell: $env:API_KEY='your-key'; python scripts/healthcheck_all.py")
    print()

    checks = [
        check_redis_docker,
        check_mlflow,
        check_fastapi_health,
        check_fastapi_metrics,
        check_fastapi_auth_rejected,
        check_fastapi_predict,
        check_airflow,
        check_prometheus,
        check_grafana,
        check_streamlit,
    ]

    results: list[CheckResult] = []
    for fn in checks:
        result = fn()
        results.append(result)
        icon = "✅" if result.passed else "❌"
        latency = f"{result.latency_ms:>6.1f} ms"
        print(f"  {icon}  {result.name:<42}  {latency}  {result.detail}")

    passed = sum(1 for r in results if r.passed)
    failed = len(results) - passed

    print()
    print("-" * 65)
    print(f"  Results: {passed}/{len(results)} passed", end="")
    if failed:
        print(f"  ({failed} FAILED)")
        print()
        print("  Failed checks:")
        for r in results:
            if not r.passed:
                print(f"    ❌  {r.name}: {r.detail}")
    else:
        print("  — all systems operational")
    print("=" * 65)
    print()

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
