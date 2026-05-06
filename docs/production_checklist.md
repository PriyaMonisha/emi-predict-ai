# EMI Predict AI — Production Readiness Checklist

50-item gate organised by category. Run each verification command before signing off.

**Status legend:** ✅ Pass · ❌ Fail · ⚠️ Warning (proceed with plan) · N/A Not applicable

---

## Category 1 — Infrastructure (Items 1–10)

| # | Item | Verification Command | Expected Result |
|---|---|---|---|
| 1 | All 9 containers healthy | `docker compose ps` | All `STATUS` = `healthy` or `running` |
| 2 | PostgreSQL accepting connections | `docker exec emi-postgres pg_isready -U airflow` | `accepting connections` |
| 3 | Redis accepting connections | `docker exec emi-redis redis-cli ping` | `PONG` |
| 4 | MLflow server responding | `curl -s http://localhost:5000/health` | `{"status":"OK"}` or 200 |
| 5 | FastAPI health endpoint | `curl -s http://localhost:8000/health` | `{"status":"ok","redis_ok":true,"models_loaded":true}` |
| 6 | Airflow webserver responding | `curl -s -o /dev/null -w "%{http_code}" http://localhost:8080/health` | `200` |
| 7 | Prometheus scraping FastAPI | `curl -s http://localhost:9090/api/v1/targets \| python -m json.tool \| findstr "emi-api"` | Target present with `state: up` |
| 8 | Grafana dashboard accessible | `curl -s -o /dev/null -w "%{http_code}" http://localhost:3000/api/health` | `200` |
| 9 | Ports not conflicting with host services | `netstat -an \| findstr "3000 5000 6379 8000 8080 9090"` | Only expected Docker-bound entries |
| 10 | Docker volumes created and not full | `docker system df` | No volume near capacity |

---

## Category 2 — Model Health (Items 11–20)

| # | Item | Verification Command | Expected Result |
|---|---|---|---|
| 11 | Champion classifier alias set in MLflow | Open http://localhost:5000/#/models/emi_eligibility_classifier | Version tagged `@champion` |
| 12 | Champion regressor alias set in MLflow | Open http://localhost:5000/#/models/emi_amount_regressor | Version tagged `@champion` |
| 13 | Model PKL files present | `Test-Path models\best_classifier.pkl; Test-Path models\best_regressor.pkl; Test-Path models\feature_engineer.pkl` | `True` for all three |
| 14 | Classifier AUC ≥ 0.9763 (logistic floor) | Check MLflow run metrics in UI | AUC displayed ≥ 0.9763 |
| 15 | Regressor RMSE ≤ 2000 (sanity cap) | Check MLflow run metrics in UI | RMSE ≤ ₹2,000 |
| 16 | Leakage audit passes | `python -c "from src.utils.leakage_checks import run_all_checks; run_all_checks()"` | `LEAKAGE AUDIT: PASS` |
| 17 | Feature engineer column count correct | `python -c "import joblib; fe=joblib.load('models/feature_engineer.pkl'); print(len(fe.feature_names_out_))"` | 42 (or expected count) |
| 18 | Inference returns all three conf_zones | Send requests with low/mid/high credit scores, check `conf_zone` field | All three zones observed |
| 19 | `predicted_emi` always in ₹500–₹34,750 | Make 20 diverse `/predict` calls; check `predicted_emi` | All values within range |
| 20 | Batch endpoint processes 500 rows | `python -c "import requests, json; r=requests.post('http://localhost:8000/predict/batch', headers={'X-API-Key':'<key>','Content-Type':'application/json'}, json={'customers':[...500 rows...]}); print(r.json()['total_scored'])"` | `500` |

---

## Category 3 — Security (Items 21–28)

| # | Item | Verification Command | Expected Result |
|---|---|---|---|
| 21 | `/predict` rejects missing API key | `curl -s -o /dev/null -w "%{http_code}" -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{}'` | `403` |
| 22 | `/predict` rejects wrong API key | `curl -s -o /dev/null -w "%{http_code}" -X POST http://localhost:8000/predict -H "X-API-Key: wrong" -H "Content-Type: application/json" -d '{}'` | `403` |
| 23 | API_KEY not hardcoded in source | `Select-String -Path src\api\dependencies.py -Pattern "API_KEY\s*=\s*[\"']"` | No match (key loaded from env) |
| 24 | `.env` file not tracked in git | `git ls-files .env` | No output (file not tracked) |
| 25 | Sensitive files in `.gitignore` | `Select-String -Path .gitignore -Pattern "\.env|mlflow\.db|models/"` | All three present |
| 26 | Fernet key not committed to git history | `git log --all -p -- docker-compose.yml \| Select-String "FERNET_KEY"` | No real key in git history |
| 27 | FastAPI runs as non-root in Docker | `docker exec emi-fastapi whoami` | `appuser` (not `root`) |
| 28 | Airflow admin password is non-default | Confirm `AIRFLOW__WEBSERVER__SECRET_KEY` and admin password changed from `admin/admin` in production | Changed before production cutover |

---

## Category 4 — Monitoring (Items 29–36)

| # | Item | Verification Command | Expected Result |
|---|---|---|---|
| 29 | Prometheus scrape target up | `curl -s http://localhost:9090/api/v1/query?query=up{job="emi-api"}` | `value: [<ts>, "1"]` |
| 30 | `emi_predictions_total` incrementing | Make a `/predict` call; `curl -s http://localhost:8000/metrics \| Select-String "emi_predictions_total"` | Counter incremented |
| 31 | `emi_redis_up` gauge correct | `curl -s http://localhost:8000/metrics \| Select-String "emi_redis_up"` | `emi_redis_up 1.0` when Redis is up |
| 32 | Grafana datasource connected | http://localhost:3000/datasources → Prometheus → Test | `Data source is working` |
| 33 | Grafana dashboard loaded | http://localhost:3000/dashboards → `EMI Monitoring` | Dashboard visible with 10 panels |
| 34 | Drift monitor runs without error | Trigger DAG; check `retrain_stub` task log | No exception; drift report JSON written |
| 35 | Drift report directory writable | `Test-Path data\processed\drift_reports` | `True` |
| 36 | Log rotation configured | `(Get-ChildItem airflow\logs -Recurse).Count` | No unbounded growth; configure `AIRFLOW__LOG_RETENTION_DAYS` before production |

---

## Category 5 — Disaster Recovery / Rollback (Items 37–43)

| # | Item | Verification Command | Expected Result |
|---|---|---|---|
| 37 | Previous champion version still exists in MLflow | Open MLflow registry; count versions for each model | ≥ 2 versions; previous version accessible |
| 38 | Rollback procedure documented and tested | Follow `docs/runbook.md` Section 4 | FastAPI loads rolled-back model and `/health` returns `models_loaded: true` |
| 39 | Batch predictions are versioned and immutable | `Get-ChildItem data\processed\predictions -Recurse -Filter predictions.csv` | One CSV per run in `{ds}/{run_id}/` path |
| 40 | `docker compose down && docker compose up -d` recovers cleanly | Run the commands; then `python scripts/healthcheck_all.py` | All checks pass after restart |
| 41 | PostgreSQL data persisted across restarts | `docker compose down; docker compose up -d; docker exec emi-airflow-webserver airflow dags list` | DAG list returns without re-running init |
| 42 | MLflow database backed up | `Copy-Item mlflow.db mlflow.db.backup` | Backup file created; verify size matches |
| 43 | Model PKL files backed up | `Copy-Item models\ models_backup\ -Recurse` | All three PKL files in backup directory |

---

## Category 6 — DPDP Act Compliance (India) — Items 44–50

The Digital Personal Data Protection Act 2023 applies to personal financial data processed by this system.

| # | Item | Verification Step | Status |
|---|---|---|---|
| 44 | **Data minimisation:** only fields necessary for EMI scoring are collected | Review `src/api/schemas.py` PredictRequest — confirm no fields beyond the 25 listed | Verified by schema review |
| 45 | **Purpose limitation:** predictions used only for EMI eligibility, not for profiling or marketing | Confirm with business stakeholder that `/predict` output is not fed to other systems without explicit consent | Business sign-off required |
| 46 | **Retention policy documented:** prediction records and Redis cache have defined retention periods | Redis TTL = 86,400 s (1 day). Prediction CSVs — define and document maximum retention period (e.g. 90 days) | Retention policy document required |
| 47 | **Data Principal rights:** mechanism exists for a customer to request deletion of their cached feature vector | Test: `docker exec emi-redis redis-cli DEL "emi:features:<customer_hash>"` — verify key removed | Manual delete process documented |
| 48 | **Sensitive data not logged:** credit score, bank balance, salary not written to application logs | `docker compose logs fastapi \| Select-String "credit_score\|bank_balance\|monthly_salary"` | No sensitive values in log output |
| 49 | **Data Fiduciary designation:** organisation processing the data has registered or documented its DPDP obligations | Confirm with legal/compliance team — registration requirements depend on organisation size | Legal sign-off required |
| 50 | **Cross-border data transfer:** MLflow, Prometheus, Grafana data must remain within India (or under SCCs) | Confirm all Docker volumes and external storage are on India-region infrastructure in production | Infrastructure sign-off required |

---

## Sign-Off

| Category | Items | Passed | Failed | Warnings | Sign-off |
|---|---|---|---|---|---|
| Infrastructure | 1–10 | | | | |
| Model Health | 11–20 | | | | |
| Security | 21–28 | | | | |
| Monitoring | 29–36 | | | | |
| DR / Rollback | 37–43 | | | | |
| DPDP Compliance | 44–50 | | | | |
| **TOTAL** | **50** | | | | |

**Production go/no-go decision:**

> All infrastructure, model health, security, monitoring, and DR items must pass (0 failures) before deployment.
> DPDP items 44–48 must pass; items 49–50 require documented business/legal sign-off.
