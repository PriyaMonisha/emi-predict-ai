# EMI Predict AI — Operations Runbook

## Prerequisites

- Docker Desktop running
- Working directory: project root (`Proj 3 EMI prediction/`)
- All commands assume PowerShell unless noted

---

## 1. Start / Stop the Full Stack

### Start all 9 services

```powershell
docker compose up -d
```

Wait ~2 minutes for `airflow-init` to complete, then verify:

```powershell
docker compose ps
python scripts/healthcheck_all.py
```

### Stop all services (preserve data volumes)

```powershell
docker compose down
```

### Stop and remove all volumes (full reset)

```powershell
docker compose down -v
```

> **Warning:** `-v` deletes PostgreSQL, Prometheus, and Grafana data volumes. MLflow data (`mlflow.db`, `mlruns/`) and predictions (`data/processed/predictions/`) are bind-mounted and are NOT deleted.

### Rebuild images after code changes

```powershell
docker compose build --no-cache
docker compose up -d
```

---

## 2. Restart a Single Service

```powershell
# FastAPI (e.g. after model update)
docker compose restart fastapi

# Airflow webserver
docker compose restart airflow-webserver

# Airflow scheduler
docker compose restart airflow-scheduler

# Redis
docker compose restart redis

# MLflow
docker compose restart mlflow
```

Check the service came back healthy:

```powershell
docker compose ps <service-name>
docker compose logs --tail=50 <service-name>
```

---

## 3. Trigger the Batch DAG Manually

The `emi_batch_prediction` DAG runs automatically at 2 AM daily. To trigger immediately:

### Via Airflow UI

1. Open http://localhost:8080 (admin / admin)
2. Find `emi_batch_prediction` in the DAG list
3. Toggle it ON (blue) if paused
4. Click the **▶ Trigger DAG** button (play icon)
5. Monitor the run in **Graph** view

### Via Airflow CLI (inside container)

```powershell
docker exec emi-airflow-scheduler airflow dags trigger emi_batch_prediction
```

### Check DAG run status

```powershell
docker exec emi-airflow-scheduler airflow dags list-runs -d emi_batch_prediction
```

### View task logs

```powershell
# Logs are bind-mounted to airflow/logs/
Get-ChildItem "airflow\logs\dag_id=emi_batch_prediction\" -Recurse | Sort-Object LastWriteTime -Descending | Select-Object -First 5
```

---

## 4. Roll Back a Model

Use this procedure if a newly promoted champion model degrades API performance.

### Step 1 — Identify the previous good version

```powershell
# Open MLflow UI and inspect the registry
Start-Process "http://localhost:5000/#/models/emi_eligibility_classifier"
```

Or via CLI:

```powershell
docker exec emi-mlflow mlflow models list --name emi_eligibility_classifier
```

### Step 2 — Promote the previous version to `@champion`

```python
import mlflow
from mlflow.tracking import MlflowClient

client = MlflowClient("sqlite:///mlflow.db")
# Set version N (the previous good version) as champion
client.set_registered_model_alias(
    name="emi_eligibility_classifier",
    alias="champion",
    version="N",
)
```

Repeat for `emi_amount_regressor` if needed.

### Step 3 — Restart FastAPI to load the rolled-back model

```powershell
docker compose restart fastapi
```

### Step 4 — Verify

```powershell
python scripts/healthcheck_all.py
```

---

## 5. Handle a Grafana Alert

### Locate the firing alert

1. Open http://localhost:3000 (admin / admin)
2. Navigate to **Alerting → Alert rules**
3. Click the firing rule to see the query and threshold

### Silence an alert

1. Go to **Alerting → Silences**
2. Click **+ New silence**
3. Set the label matcher (e.g. `alertname=HighInferenceLatency`)
4. Set duration (e.g. 2 hours for a scheduled maintenance window)

### Common alerts and remediation

| Alert | Likely cause | Action |
|---|---|---|
| `emi_redis_up = 0` | Redis container down | `docker compose restart redis` |
| High `emi_inference_duration_seconds` P99 | Cache miss storm or cold start | Check Redis health; warm cache by running a small batch |
| Low `emi_cache_hits_total / total` | New customers not seen before (expected) | No action needed; ratio improves over time |
| `http_requests_total{status_code=~"5.."}` spike | Model load failure or preprocess error | Check `docker compose logs fastapi` |

---

## 6. Debug Redis Cache Misses

### Check Redis health

```powershell
docker exec emi-redis redis-cli ping
# Expected: PONG
```

### Inspect cache keys

```powershell
# Count all EMI feature keys
docker exec emi-redis redis-cli --scan --pattern "emi:features:*" | Measure-Object -Line

# Inspect a specific key (replace <hash> with the first 16 chars of SHA256 of customer_id)
docker exec emi-redis redis-cli GET "emi:features:<hash>"
```

### Check cache stats

```powershell
docker exec emi-redis redis-cli INFO stats | Select-String "keyspace_hits|keyspace_misses"
```

### Flush the cache (caution — all feature vectors lost)

```powershell
docker exec emi-redis redis-cli FLUSHDB
```

### Warm the cache from a recent batch

Trigger or wait for the nightly batch DAG — it calls `batch_write()` after scoring, which populates Redis for all 17,488 high-risk customers.

---

## 7. Check Drift Reports

Drift reports are saved to `data/processed/drift_reports/` after each batch run.

```powershell
Get-ChildItem "data\processed\drift_reports\" | Sort-Object LastWriteTime -Descending | Select-Object -First 5
```

Each report is a JSON file. Key fields:

```json
{
  "layer1_drift_detected": false,
  "layer2_results": {
    "credit_score": {"drift_detected": false, "p_value": 0.23},
    "monthly_salary": {"drift_detected": true, "p_value": 0.03},
    ...
  }
}
```

If drift is detected, initiate a retraining cycle (see `docs/model_card.md` — Retraining Procedure).

---

## 8. Inspect Batch Predictions

Predictions are saved in versioned directories:

```
data/processed/predictions/{execution_date}/{run_id}/predictions.csv
```

```powershell
# List all batch runs
Get-ChildItem "data\processed\predictions\" -Recurse -Filter "predictions.csv" |
    Sort-Object LastWriteTime -Descending | Select-Object -First 5

# Quick summary of latest run
$latest = Get-ChildItem "data\processed\predictions\" -Recurse -Filter "predictions.csv" |
    Sort-Object LastWriteTime -Descending | Select-Object -First 1
python -c "import pandas as pd; df=pd.read_csv('$($latest.FullName)'); print(df['conf_zone'].value_counts())"
```

---

## 9. View Service Logs

```powershell
# Tail logs for a specific service
docker compose logs -f fastapi
docker compose logs -f airflow-scheduler
docker compose logs -f mlflow

# All services, last 100 lines
docker compose logs --tail=100

# Filter for errors
docker compose logs fastapi 2>&1 | Select-String "ERROR|CRITICAL"
```

---

## 10. Prometheus / Grafana Queries

### Useful PromQL queries

```promql
# Prediction rate per minute
rate(emi_predictions_total[1m])

# Confidence zone breakdown
sum by (conf_zone) (emi_predictions_total)

# P99 inference latency
histogram_quantile(0.99, rate(emi_inference_duration_seconds_bucket[5m]))

# Cache hit rate
rate(emi_cache_hits_total[5m]) / (rate(emi_cache_hits_total[5m]) + rate(emi_cache_misses_total[5m]))

# API error rate
rate(http_requests_total{status_code=~"5.."}[1m])
```

Access Prometheus directly: http://localhost:9090/graph

---

## 11. Environment Variables Reference

| Variable | Default | Description |
|---|---|---|
| `API_KEY` | (required) | X-API-Key for `/predict` and `/predict/batch` |
| `MLFLOW_TRACKING_URI` | `sqlite:///mlflow.db` | MLflow backend store URI |
| `MODELS_DIR` | `./models` | Directory containing `best_classifier.pkl` etc. |
| `REDIS_HOST` | `localhost` | Redis hostname |
| `REDIS_PORT` | `6379` | Redis port |

In Docker Compose, `REDIS_HOST` is overridden to `redis` (service name) and `MLFLOW_TRACKING_URI` to `sqlite:////app/mlflow.db`.

---

## 12. Common Errors and Fixes

| Error | Root Cause | Fix |
|---|---|---|
| `FileNotFoundError: best_classifier.pkl` | Models not trained yet | Run `notebooks/05_model_training.py`, then `notebooks/06_mlflow_experiments.py` |
| `ImportError: libgomp.so.1` in Airflow container | Missing `libgomp1` system package | Already fixed in `docker/airflow/Dockerfile` — rebuild with `docker compose build --no-cache` |
| `redis.exceptions.ConnectionError` | Redis down or wrong host | `docker compose restart redis`; check `REDIS_HOST` env var |
| `mlflow.exceptions.MlflowException: Model not found` | Champion alias not set | Run Section 6 notebook or manually set alias in MLflow UI |
| DAG stuck in `queued` state | Airflow scheduler not running | `docker compose restart airflow-scheduler` |
| `422 Unprocessable Entity` from `/predict` | Invalid field in request body | Check `docs/api_reference.md` for required field types and ranges |
