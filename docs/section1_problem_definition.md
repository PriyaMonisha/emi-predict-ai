# Section 1: Problem Definition & Decision Framework
**Project:** EMI Predict AI — Intelligent Financial Risk Assessment  
**Date:** 2026-04-30  
**Status:** Complete ✅

---

## 1. Business Problem

Manual EMI decisions are:
- Slow (3-7 days per application)
- Expensive (officer time per application)
- Inconsistent (human judgment varies)
- Unscalable (404,800 applications)
- Cannot predict MAX EMI amount

---

## 2. Dataset Facts

| Metric | Value |
|--------|-------|
| Total Applications | 404,800 |
| Training Ready | 387,287 |
| High Risk (pending) | 17,488 |
| Features | 32 |
| EMI Scenarios | 5 |

---

## 3. Target Variables

### Task 1 — Classification
- **Column:** emi_eligibility
- **Values:** Eligible (1) / Not Eligible (0)
- **Distribution:** 19.2% Eligible / 80.8% Not Eligible
- **Challenge:** Class imbalance (4.2:1 ratio)

### Task 2 — Regression
- **Column:** max_monthly_emi
- **Range:** ₹500 to ₹34,750
- **Mean:** ₹6,461
- **Median:** ₹3,920

---

## 4. Why ML Over Rules?

### Rule-Based Fails Because:
- Cannot capture feature INTERACTIONS
- One rule set cannot handle 5 EMI scenarios
- Cannot predict MAX EMI amount (only YES/NO)
- Rules become stale as economy changes
- 80.8% naive accuracy looks good but catches ZERO eligible customers

### ML Solves This:
- Learns complex patterns automatically
- Handles all 5 scenarios in one model
- Predicts exact EMI capacity
- Retrains when data drifts

---

## 5. ML Solution Design

Customer Application
│
▼
[Task 1] Classifier → Eligible / Not Eligible
│
YES │ NO
│ └──→ Reject with reason
▼
[Task 2] Regressor → Max EMI = ₹15,000/month
│
▼
Final Decision by Bank Officer


---

## 6. Models Selected

### Classification (4 models):
| Model | Why |
|-------|-----|
| Logistic Regression | Baseline + explainable |
| Random Forest | Feature importance |
| XGBoost | Industry standard |
| LightGBM | Fastest on large data |

### Regression (4 models):
| Model | Why |
|-------|-----|
| Linear Regression | Baseline |
| Random Forest Regressor | Non-linear patterns |
| XGBoost Regressor | Best accuracy |
| LightGBM Regressor | Speed + accuracy |

---

## 7. Success Metrics

### Classification:
| Metric | Target |
|--------|--------|
| ROC-AUC | > 0.85 |
| F1-Score | > 0.75 |
| Precision | > 0.80 |
| Recall | > 0.70 |
| Accuracy | > 0.82 |

### Regression:
| Metric | Target |
|--------|--------|
| R-Squared | > 0.85 |
| RMSE | < ₹2,000 |
| MAE | < ₹1,500 |
| MAPE | < 15% |

### Business:
| Metric | Target |
|--------|--------|
| Decision Speed | < 2 seconds |
| Cost Per Decision | ₹0.001 (vs ₹500-1000 manual) |
| Throughput | 1000+ apps/minute |

---

## 8. Risk & Fallback

| Risk | Mitigation |
|------|------------|
| Wrong prediction | Confidence threshold + human review |
| Data drift | Prometheus + Evidently daily monitoring |
| Model bias | Fairness metrics per gender/age |
| API failure | Rule-based fallback system |
| Train-serve mismatch | Same preprocess.py for both |

### Confidence Threshold System:
- Probability > 0.85 → Auto approve
- Probability 0.40-0.85 → Human review
- Probability < 0.40 → Auto reject

---

## 9. Tech Stack Decisions

| Tool | Purpose |
|------|---------|
| PostgreSQL | Store all records + predictions |
| Redis | Cache features (< 1ms lookup) |
| MLflow | Track 8 model experiments |
| Airflow | Daily ETL + retraining pipeline |
| FastAPI | Serve predictions (< 100ms) |
| Prometheus | Collect metrics every 15 seconds |
| Grafana | Real-time dashboard + alerts |
| Evidently | Statistical drift detection |
| Docker | Package everything consistently |

---

## 10. Data Cleaning Summary

| Step | Result |
|------|--------|
| Raw rows | 404,800 |
| After cleaning | 387,287 |
| Nulls before | 12,027 |
| Nulls after | 0 |
| New features added | 5 flag columns |
| High risk saved | 17,488 rows |
| Credit violations fixed | 4,576 rows |
| Outliers capped | 1st-99th percentile |

---

## 11. Next Steps

- [ ] Section 2: EDA & Data Audit
- [ ] Section 3: Baseline Model
- [ ] Section 4: Feature Engineering
- [ ] Section 5: Model Training (8 models)
- [ ] Section 6: MLflow Evaluation
- [ ] Section 7: Airflow ETL Pipeline
- [ ] Section 8: Redis Feature Store
- [ ] Section 9: FastAPI Serving
- [ ] Section 10: Prometheus + Grafana
- [ ] Section 11: Docker Deployment
- [ ] Section 12: Testing
- [ ] Section 13: Documentation