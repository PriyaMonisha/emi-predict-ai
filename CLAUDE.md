# EMI Predict AI — Project Intelligence

## Who I Am
You are a senior ML engineer and MLOps architect working on "EMI Predict AI"
— a production-grade end-to-end machine learning system for EMI risk
prediction and eligibility scoring for an Indian financial institution.

## My Working Style (STRICT)
Before writing ANY code:
1. State your plan in clear steps
2. Call out assumptions explicitly
3. Flag any risks or design decisions
4. Ask me to confirm

Only after I say "yes" or "go" — write the code.

If a task has multiple valid approaches, present them with tradeoffs.
Let me choose. Never silently pick one.

## Section Completion Checklist (MANDATORY — no exceptions)
At the end of EVERY section, before declaring it complete or moving on:
- [ ] All files saved
- [ ] Tests / verification passed
- [ ] `git add` all changed files
- [ ] `git commit -m "section-X: description + what was built + bugs fixed"`
- [ ] `git log --oneline` — confirm commit appears
- [ ] `git status` — confirm working tree clean
- [ ] CLAUDE.md progress table updated (move section from In Progress → Completed)
- [ ] Next section dependencies confirmed
Only AFTER all boxes checked → ask if ready for next section.

## How to Start Each Chat
I will open with:
"Continuing EMI Predict AI. Completed: [X]. Now building: [Y]."
You respond with your plan for Y — steps, risks, decisions — and WAIT
for my confirmation before writing any code.

---

## Current Status
**Active Section:** None — all 14 sections complete
**Last Working File:** docker-compose.yml, streamlit_app.py
**Last Decision Made:** S11 complete. 9-service Docker Compose stack written and config-validated. MLflow SQLite path explicitly handled (3 different paths for host/fastapi/airflow via env var overrides). PYTHONPATH=/opt/airflow in all Airflow containers. Airflow Variables set via AIRFLOW_VAR_* env vars + airflow_settings.yaml import. Grafana dashboard provisioning wired. Full DAG execution requires docker compose up -d then manual trigger in Airflow UI.

---

## Progress Tracker

### Completed ✅
- [x] src/data/load_data.py
- [x] src/data/preprocess.py (v4, 17-step pipeline)
- [x] notebooks/00_data_audit.py
- [x] notebooks/01_data_cleaning.py
- [x] notebooks/02_eda.py (v3, 8 charts, Dark2_r palette)
- [x] docs/section1_problem_definition.md
- [x] requirements.txt
- [x] Section 3: src/models/baseline_rules.py, baseline_logistic.py, notebooks/03_baseline.py, docs/section_03_baseline.md
  - Rule-Based AUC 0.7956 | Logistic Regression AUC 0.9763 (floor to beat)
- [x] Section 4: src/features/feature_engineering.py, notebooks/04_feature_engineering.py, docs/section_04_feature_engineering.md
  - 21 new features, 42 total; train_features.csv, test_features.csv, feature_engineer.pkl

- [x] Section 5: src/models/train_classifier.py (v2), train_regressor.py (v2), notebooks/05_model_training.py (v2), docs/section_05_model_training.md, src/utils/leakage_checks.py
  - FAST_MODE: 3 trials × 3-fold CV × 50k stratified sample. Final model on full X_train.
  - Best classifier: LightGBM (AUC 0.9999, F1 0.9922)
  - Best regressor: XGBoost (RMSE ₹671.85, R²=0.9916, MAPE 7.59%)
  - Leakage audit: PASS. FULL_MODE Optuna leakage fixed. 6dp metric storage.

- [x] Section 6: MLflow experiment tracking
  - LightGBM = champion classifier (AUC 0.9999, F1 0.9922)
  - XGBoost = champion regressor (RMSE ₹672, R²=0.9916)
  - All 9 models logged and registered
  - Backend: SQLite (mlflow.db) — not committed to git (runtime artifact)
  - UI: mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000

- [x] Section 7: Airflow ETL pipeline (v1.1)
  - airflow/dags/emi_batch_prediction_dag.py — 6-task DAG (added retrain_stub)
  - src/pipelines/predict_pipeline.py (v1.1) — reusable prediction logic (used by DAG + S9 FastAPI)
  - Schedule: 0 2 * * * (2 AM daily) | catchup=False | max_active_runs=1 | retries=3
  - Versioned output: {predictions_dir}/{ds}/{run_id}/predictions.csv (immutable per run)
  - FeatureEngineer loaded from MLflow champion artifacts (not from disk Variable)
  - Runs in Docker (Section 11): apache/airflow:2.8.4-python3.11

- [x] Section 8: Redis Feature Store
  - src/features/feature_store.py (v1.0) — 7 functions, cache-aside, JSON-only, graceful degradation
  - Key pattern: emi:features:{sha256[:16]} | TTL: 86400s | decode_responses=True
  - All 7 functions smoke-tested: health_check, write, read, invalidate, batch_write — PASS
  - Redis container running: docker run -d -p 6379:6379 --name emi-redis redis:7-alpine

- [x] Section 9: FastAPI serving
  - src/api/main.py (v1.0) — 3 endpoints: GET /health, POST /predict, POST /predict/batch
  - src/api/schemas.py (v1.0) — PredictRequest (25 raw features), PredictResponse, Batch variants
  - src/api/dependencies.py (v1.0) — ModelState dataclass, X-API-Key auth, get_model_state
  - src/pipelines/predict_pipeline.py (v1.2) — predict_from_preloaded() for FastAPI pre-loaded models
  - src/features/feature_store.py — NumpyEncoder added (numpy scalars → JSON-native)
  - Cache-aside: Redis hit skips preprocess+FE (~3x faster); miss runs full pipeline + writes cache
  - Run: $env:PYTHONPATH="."; uvicorn src.api.main:app --reload --port 8000

- [x] Section 10: Prometheus + Grafana monitoring
  - src/api/main.py (v1.1) — Prometheus middleware + 6 metrics (HTTP + 4 EMI business metrics)
  - src/monitoring/drift_monitor.py (v1.0) — two-layer Evidently drift, non-blocking
  - airflow/dags/emi_batch_prediction_dag.py (v1.2) — retrain_stub wired to drift monitor
  - configs/prometheus.yml — scrape config (Docker-ready, host.docker.internal)
  - configs/grafana/ — datasource provisioning + 10-panel dashboard JSON
  - GET /metrics verified: emi_predictions_total, emi_inference_duration_seconds, cache hit/miss, redis_up
  - Drift: missing file → graceful skip; sample test → Layer 1 clean, Layer 2 detected 1/4 features

- [x] Section 11: Docker Compose deployment
  - docker-compose.yml — 9 services, YAML anchors, healthchecks, depends_on chains
  - Dockerfile — multi-stage (builder+runtime), non-root user, curl healthcheck
  - .dockerignore — excludes data/raw, mlruns, *.pkl, __pycache__, .git
  - docker/airflow/Dockerfile — extends airflow:2.8.4-python3.11, installs DAG deps
  - docker/airflow/requirements.txt — lightgbm, xgboost, mlflow, evidently, redis
  - docker/airflow/airflow_settings.yaml — 6 Variables with Docker-internal paths
  - configs/prometheus.docker.yml — target: fastapi:8000 (service name, not host.docker.internal)
  - configs/grafana/provisioning/dashboards/dashboard.yml — dashboard file provider
  - MLflow path: host sqlite:///mlflow.db → fastapi sqlite:////app/mlflow.db → airflow sqlite:////opt/airflow/mlflow.db
  - docker compose config VALID — no YAML errors

- [x] Section 12: pytest test suite
  - 54 tests, 76% coverage (60% floor enforced via pytest.ini + .coveragerc)
  - tests/conftest.py — session fixtures: raw_df, fitted_fe, mock_clf, mock_reg, valid_payload
  - 9 test modules: load_data, preprocess, feature_engineering, feature_store, baseline_rules, leakage_checks, predict_pipeline, api_schemas, api_endpoints
  - Key fix: conftest pre-imports src.api.main with fake stdout (no .buffer) to prevent pytest capture corruption on Windows

- [x] Section 13: Documentation + production checklist
  - README.md (comprehensive — badges, Mermaid architecture, quick-start, API examples, build table)
  - docs/architecture.md (5 Mermaid diagrams: service deps, real-time path, batch path, FE pipeline, monitoring)
  - docs/model_card.md (classifier + regressor performance, limitations, fairness, DPDP notes)
  - docs/runbook.md (12 operational procedures)
  - docs/api_reference.md (all 4 endpoints, schemas, error codes, curl examples)
  - docs/production_checklist.md (50 items × 6 categories with verification commands)
  - scripts/healthcheck_all.py (10 checks after S14, ✅/❌ summary, exit code 1 on any failure)

- [x] Section 14: Streamlit UI + public deployment
  - streamlit_app.py — multi-page entry point, st.navigation(), shared sidebar with API config
  - pages/1_Single_Prediction.py — full 25-field form, Load Sample button, colour-coded zone banner
  - pages/2_Batch_Upload.py — CSV upload/validate/score/download, REQUIRED_COLS enforced
  - pages/3_System_Health.py — /health check, latency test, stack reference table
  - src/ui/api_client.py — predict_single (60s), predict_batch (120s), lazy st import
  - docker/streamlit/Dockerfile — multi-stage, non-root appuser, ~300 MB
  - docker-compose.yml — service #10 (streamlit:8501), depends on fastapi:healthy
  - render.yaml — Render blueprint (deploy branch, render-requirements.txt)
  - render-requirements.txt — FastAPI runtime deps only (no evidently/mlflow/airflow)
  - streamlit-requirements.txt — 3 packages for Streamlit Community Cloud
  - Deployment: deploy branch (PKLs force-added) → Render + main → Streamlit Cloud

### In Progress 🔄
None — all 14 sections complete.

### Remaining 📋
None.


---

## Project Specifications (Locked — Do Not Change Without Explicit Decision)

### Stack
Python 3.11 | XGBoost + LightGBM | FastAPI + Uvicorn | MLflow
Redis | Airflow | Prometheus + Grafana | PostgreSQL | Docker Compose
Streamlit UI | pytest | Evidently (drift) | Optuna (hypertuning)

### Data Facts
- Raw: 404,800 rows → Training: 387,287 → High-risk saved: 17,488
- Target 1: emi_eligibility (0/1) — 80.8% Not Eligible / 19.2% Eligible
- Target 2: max_monthly_emi — range ₹500–₹34,750, mean ₹6,461
- Class imbalance: 4.2:1 ratio
- 32 columns, 5 missing-flag columns added in preprocessing
- Outliers capped at 1st–99th percentile

### Locked Decisions
- class_weight='balanced' on ALL classifiers — no exceptions
- Primary metrics: ROC-AUC + F1 (accuracy is misleading here)
- PYTHONPATH set in Dockerfile from Day 1
- Confidence thresholds:
  - > 0.85 → auto-approve
  - 0.40–0.85 → human review
  - < 0.40 → auto-reject
- Phase 1: Docker Compose local → Phase 2: AWS ECS (zero rewrites)
- Palettes: Dark2_r (binary) | Paired_r (multi-category) | Accent_r (histograms)
- Best classifier selection: primary=ROC-AUC, tie-break=F1, final tie-break=speed (LightGBM > XGBoost > RF > ET > LR)
- Best regressor selection: primary=RMSE (minimize)
- Metric storage: 6 decimal places in JSON/pkl artifacts; console display rounds to 4dp
- Optuna tuning: NEVER use X_test inside objective — FAST_MODE uses CV on 50k sample, FULL_MODE uses 5-fold CV on X_train
- FAST_MODE=True for dev/interview (3 trials, 50k sample, 3-fold CV); FAST_MODE=False for production (25 trials, 5-fold CV on full X_train)

### Known Infrastructure Issues
- Airflow image requires `libgomp1` system package (LightGBM dependency) — must be in `docker/airflow/Dockerfile` apt-get install line; `lightgbm`, `xgboost`, and `joblib` must be in `docker/airflow/requirements.txt`. Without `libgomp1`, any DAG task that imports LightGBM will fail at runtime even if pip install succeeded.

### Known Data Quirks (Never "Fix" These)
- monthly_rent and years_of_employment are zero-heavy (annotate, not hide)
- Bank balance filled with salary-bracket median (not global median)
- High-risk rows (17,488) → data/processed/unlabeled_for_prediction.csv
- Age histogram 4-spike pattern = real data pattern, NOT a bug

---

## Code Standards (Enforced by hooks — no exceptions)

### Every File
```python
# filename: src/models/baseline_rules.py
# purpose:  Rule-based EMI eligibility classifier (Section 3 baseline)
# version:  1.0
```

- Header block (filename / purpose / version) is mandatory on every .py file
- version increments on breaking changes (1.0 → 2.0) or significant additions (1.0 → 1.1)

### Constants
```python
RANDOM_STATE = 42          # all models, splits, sampling
AUTO_APPROVE  = 0.85       # confidence threshold — locked
AUTO_REJECT   = 0.40       # confidence threshold — locked
FAST_MODE     = True       # switch: dev vs production run
```

### Imports
- stdlib → third-party → internal (`src.*`) — one blank line between groups
- No wildcard imports (`from x import *`)

### Functions
- Type-annotated signatures: `def fn(X: pd.DataFrame, y: pd.Series) -> dict:`
- Single-line docstring only when the name alone is insufficient
- No multi-paragraph docstrings

### Metrics & Artifacts
- All JSON metrics stored to 6 decimal places; console display rounds to 4dp
- Models saved to `models/` alongside their preprocessor
- Figures saved to `docs/figures/` with section prefix (e.g. `05_classifier_comparison.png`)

### Forbidden
- `accuracy_score` as primary or sole metric
- `GridSearchCV` — use Optuna
- `fit()` or `fit_transform()` called on test/validation data
- Hardcoded hyperparameters (use Optuna trial suggestions)
- Training code without MLflow logging block (Section 6 onward)