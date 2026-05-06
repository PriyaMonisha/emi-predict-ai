# EMI Predict AI — System Architecture

## 1. Service Dependency Graph

All 9 Docker Compose services with startup order and healthcheck dependencies.

```mermaid
graph TD
    postgres[(PostgreSQL :5432\nAirflow metadata)]
    redis[(Redis :6379\nFeature store)]
    mlflow[MLflow :5000\nExperiment tracking]
    fastapi[FastAPI :8000\nPrediction API]
    airflow_init[airflow-init\none-shot DB setup]
    airflow_web[Airflow Webserver :8080]
    airflow_sched[Airflow Scheduler]
    prometheus[Prometheus :9090\nMetrics scraping]
    grafana[Grafana :3000\nDashboards]

    postgres -->|healthy| airflow_init
    airflow_init -->|completed successfully| airflow_web
    airflow_init -->|completed successfully| airflow_sched
    redis -->|healthy| fastapi
    mlflow -->|healthy| fastapi
    fastapi -->|healthy| prometheus
    prometheus -->|healthy| grafana

    airflow_web -.->|reads DAGs| airflow_sched
    airflow_sched -.->|uses| redis
    airflow_sched -.->|uses| mlflow
```

**Startup sequence (approx):**
1. `postgres` + `redis` — no dependencies, start immediately
2. `mlflow` — no dependencies, starts in parallel
3. `airflow-init` — waits for `postgres` healthy
4. `fastapi` — waits for `redis` + `mlflow` healthy (~90 s total)
5. `airflow-webserver` + `airflow-scheduler` — wait for `airflow-init` completed
6. `prometheus` — waits for `fastapi` healthy
7. `grafana` — waits for `prometheus` healthy

---

## 2. Real-Time Prediction Path

Single-customer scoring via `POST /predict`. Cache-aside pattern reduces P99 latency ~10× on repeat customers.

```mermaid
sequenceDiagram
    autonumber
    participant C as API Consumer
    participant F as FastAPI :8000
    participant R as Redis :6379
    participant P as Preprocessor
    participant FE as FeatureEngineer
    participant CLF as LightGBM
    participant REG as XGBoost

    C->>F: POST /predict\n{customer_id, 25 raw features}\nX-API-Key: ...

    F->>F: Validate PredictRequest\n(Pydantic + field validators)

    F->>R: GET emi:features:{sha256[:16]}

    alt Cache HIT (repeat customer)
        R-->>F: 42-col feature vector (JSON)
        note over F: Skips steps 5-7
    else Cache MISS (new customer)
        F->>P: preprocess_for_inference(df_raw)
        P-->>F: cleaned DataFrame (32 cols)
        F->>FE: FeatureEngineer.transform()
        FE-->>F: engineered DataFrame (42 cols)
        F->>R: SET emi:features:{sha256} TTL=86400s
    end

    F->>CLF: predict_proba() → eligibility score
    CLF-->>F: proba ∈ [0, 1]

    F->>REG: predict() → EMI amount
    REG-->>F: ₹500 – ₹34,750

    F->>F: Assign confidence zone\n> 0.85 → auto_approve\n0.40–0.85 → human_review\n< 0.40 → auto_reject

    F->>F: Increment Prometheus counters\n(emi_predictions_total, emi_inference_duration_seconds)

    F-->>C: PredictResponse\n{clf_proba, clf_label, conf_zone,\npredicted_emi, cache_hit, latency_ms}
```

**Typical latencies (local):**

| Path | P50 | P99 |
|---|---|---|
| Cache hit | ~5 ms | ~15 ms |
| Cache miss (full pipeline) | ~45 ms | ~120 ms |

---

## 3. Batch Prediction Path

Nightly DAG runs at 2 AM, processes `unlabeled_for_prediction.csv` (17,488 high-risk rows), writes versioned output, and checks for feature drift.

```mermaid
flowchart TD
    A([Airflow Scheduler\n2 AM daily trigger]) --> B

    subgraph DAG["emi_batch_prediction DAG (6 tasks)"]
        B[load_batch_data\nread unlabeled_for_prediction.csv]
        C[preprocess_batch\npreprocess_for_inference()]
        D[engineer_features\nFeatureEngineer.transform()]
        E[score_batch\nLightGBM + XGBoost predict]
        F[save_predictions\ndata/processed/predictions/{ds}/{run_id}/]
        G[retrain_stub\ndrift_monitor.run_drift_check()]

        B --> C --> D --> E --> F --> G
    end

    subgraph Storage
        H[(data/processed/\npredictions/\n{ds}/{run_id}/\npredictions.csv)]
        I[(data/processed/\ndrift_reports/)]
        J[(Redis\ncache warm-up)]
    end

    E -->|batch_write| J
    F --> H
    G --> I

    G --> K{Drift\ndetected?}
    K -->|Layer 1 or 2| L[Log warning\nAlert retrain stub]
    K -->|Clean| M([DAG complete])
    L --> M
```

**DAG configuration:**
- Schedule: `0 2 * * *` (2 AM daily)
- `catchup=False` — never back-fills missed runs
- `max_active_runs=1` — prevents concurrent batch overlap
- `retries=3`, `retry_delay=5 min` per task
- Output path: `data/processed/predictions/{execution_date}/{run_id}/predictions.csv` (immutable per run)

---

## 4. Feature Engineering Pipeline

Data flow from 25 raw API fields to 42 model-ready features.

```mermaid
flowchart LR
    subgraph Input["Raw Input (25 cols)"]
        A[Demographics\nage, gender, education...]
        B[Financial\nsalary, credit_score, bank_balance...]
        C[Loan request\nrequested_amount, tenure, scenario]
    end

    subgraph Preprocess["Preprocessing (17 steps)"]
        D[Column normalisation]
        E[Missing value imputation\ncredit_score, bank_balance, emergency_fund]
        F[Missing-flag columns × 5]
        G[Outlier capping 1st–99th pct]
        H[Categorical encoding prep]
    end

    subgraph FE["Feature Engineering (+21 new)"]
        I[Ratio features\ndebt_to_income, emi_to_income\nrent_to_income...]
        J[Affordability features\nnet_income, disposable_income\naffordability_ratio...]
        K[Risk composite\ncredit_score_band, risk_score]
        L[Interaction features\nfinancial_stability_score\nloan_burden_ratio...]
    end

    subgraph Output["Model Input (42 cols)"]
        M[42 features\nfloat64 + categorical\nfor OHE in predict_pipeline]
    end

    A & B & C --> D --> E --> F --> G --> H
    H --> I & J & K & L
    I & J & K & L --> M
```

---

## 5. Monitoring & Observability

```mermaid
graph LR
    subgraph API["FastAPI :8000"]
        A[PrometheusMiddleware\nHTTP request count + latency]
        B[emi_predictions_total\nlabelled by conf_zone]
        C[emi_inference_duration_seconds]
        D[emi_cache_hits_total\nemi_cache_misses_total]
        E[emi_redis_up gauge]
    end

    subgraph Drift["Drift Monitor (Evidently)"]
        F[Layer 1: dataset-level\nDataDriftPreset]
        G[Layer 2: feature-level\nColumnDriftMetric × 4 key features]
    end

    subgraph Scrape
        H[Prometheus :9090\nscrape /metrics every 10s]
    end

    subgraph Viz
        I[Grafana :3000\n10-panel dashboard]
    end

    A & B & C & D & E --> H
    F & G -->|retrain_stub| J[Airflow DAG log]
    H --> I
```

**Grafana dashboard panels (10 total):**
1. Predictions per minute (rate)
2. Confidence zone distribution (pie)
3. Inference latency P50/P99 (histogram)
4. Cache hit rate (gauge)
5. Redis up/down (stat)
6. HTTP request rate by endpoint
7. HTTP error rate (4xx/5xx)
8. HTTP latency heatmap
9. Auto-approve / human-review / auto-reject counts (bar)
10. Models loaded status (stat)
