# EMI Predict AI — Model Card

## Model Overview

| Attribute | Classifier | Regressor |
|---|---|---|
| **Task** | Binary classification (eligibility) | Regression (EMI amount) |
| **Algorithm** | LightGBM (LGBMClassifier) | XGBoost (XGBRegressor) |
| **Target column** | `emi_eligibility` (0 / 1) | `max_monthly_emi` (₹) |
| **MLflow model name** | `emi_eligibility_classifier` | `emi_amount_regressor` |
| **MLflow alias** | `@champion` | `@champion` |
| **Artifact path** | `models/best_classifier.pkl` | `models/best_regressor.pkl` |
| **Version** | Section 5, FAST_MODE=True | Section 5, FAST_MODE=True |

---

## Training Data

| Property | Value |
|---|---|
| Raw dataset | 404,800 rows × 32 columns |
| Training split | 387,287 rows (after unlabeled extraction) |
| Test split | held-out 20% stratified |
| Features | 42 (25 raw + 17 engineered) |
| Class balance (target 1) | 80.8% Not Eligible / 19.2% Eligible — 4.2:1 ratio |
| EMI range (target 2) | ₹500 – ₹34,750 (mean ₹6,461) |
| Domain | Indian personal lending |
| Collection period | Not disclosed by data provider |
| Known quirks | `monthly_rent` and `years_of_employment` are zero-heavy (valid business reality, not data errors); bank balance filled with salary-bracket median; age distribution has a 4-spike pattern (real data, not artefact) |

---

## Performance

### Classifier — LightGBM

| Metric | Value | Notes |
|---|---|---|
| ROC-AUC | **0.9999** | Primary selection metric |
| F1-score | **0.9922** | Macro-averaged |
| Precision | 0.9924 | Class 1 (Eligible) |
| Recall | 0.9921 | Class 1 (Eligible) |
| Baseline comparison | Rule-based AUC 0.7956 → LR AUC 0.9763 | +2.4 ppt over LR |

> AUC of 0.9999 on a real dataset warrants scrutiny. Leakage audit (Section 5) confirmed:
> - No test-set rows used in Optuna CV objective
> - No target columns present in feature matrix at inference time
> - `emi_eligibility` and `max_monthly_emi` dropped from inference path in `preprocess_for_inference()`
> - Feature importance analysis shows `credit_score` and `disposable_income` as top drivers — both are legitimate pre-application signals

### Regressor — XGBoost

| Metric | Value |
|---|---|
| RMSE | **₹671.85** |
| R² | **0.9916** |
| MAPE | **7.59%** |
| Baseline comparison | Predicts EMI within ±₹672 on average |

---

## Confidence Zones (Classifier)

| Zone | Probability Threshold | Business Action |
|---|---|---|
| `auto_approve` | > 0.85 | Instantly approve — no human review required |
| `human_review` | 0.40 – 0.85 | Route to underwriter for manual assessment |
| `auto_reject` | < 0.40 | Instantly decline — no manual review |

These thresholds are **locked decisions** — changing them requires an explicit business review. They are constants in `src/pipelines/predict_pipeline.py` (`AUTO_APPROVE = 0.85`, `AUTO_REJECT = 0.40`).

---

## Feature Importance (Top 10 — Classifier)

| Rank | Feature | Type |
|---|---|---|
| 1 | `credit_score` | Raw numeric |
| 2 | `disposable_income` | Engineered ratio |
| 3 | `debt_to_income` | Engineered ratio |
| 4 | `bank_balance` | Raw numeric |
| 5 | `monthly_salary` | Raw numeric |
| 6 | `affordability_ratio` | Engineered ratio |
| 7 | `net_income` | Engineered sum |
| 8 | `risk_score` | Engineered composite |
| 9 | `requested_amount` | Raw numeric |
| 10 | `existing_loans` | Categorical (OHE'd) |

See `docs/figures/05_feature_importance.png` for the full importance chart.

---

## Hyperparameter Tuning

Both models were tuned with **Optuna** (FAST_MODE: 3 trials × 3-fold CV on 50k stratified sample; FULL_MODE: 25 trials × 5-fold CV on full training set). `GridSearchCV` is forbidden by project standards — only Optuna is used.

Key LightGBM hyperparameters (champion run):
```
num_leaves, learning_rate, n_estimators, min_child_samples,
colsample_bytree, subsample — all Optuna-tuned
class_weight='balanced'  ← locked, never changed
```

---

## Class Imbalance Handling

- `class_weight='balanced'` on ALL classifiers — locked decision, no exceptions
- Rationale: 4.2:1 imbalance means accuracy is misleading; balanced weighting prevents the model from trivially predicting the majority class
- Primary metric is ROC-AUC; F1 is the tie-break. `accuracy_score` is never used as a primary or sole metric (project standard)

---

## Limitations

1. **Domain specificity:** Trained on a single Indian lending institution's portfolio. Performance on data from different lenders, regions, or economic periods is unknown.
2. **Temporal validity:** No collection date is available. The model should be retrained when drift is detected (see `src/monitoring/drift_monitor.py`).
3. **Missing-value sensitivity:** `credit_score`, `bank_balance`, and `emergency_fund` can be null at inference. Missing-flag columns (`credit_score_missing`, etc.) partially compensate, but high rates of nulls will degrade performance.
4. **Synthetic high-risk data:** 17,488 rows were separated as unlabeled before training (high-risk profile). The model has not been validated on this population.
5. **EMI range:** Regressor is clipped to ₹500–₹34,750 (training range). Requests outside this range will be capped silently.
6. **FAST_MODE artefacts:** The production models in this repo were trained with `FAST_MODE=True` (3 Optuna trials). Set `FAST_MODE=False` and retrain for a full production deployment (25 trials × 5-fold CV).

---

## Fairness and Bias Notes

- `gender`, `marital_status`, and `education` are present as raw features and are OHE'd into the model. No fairness constraints have been applied in training.
- Recommended before production deployment: run disparate impact analysis across `gender` and `marital_status` subgroups using the held-out test set.
- DPDP Act (India) 2023 consideration: the model processes sensitive personal financial data. Any deployment must comply with data minimisation and purpose limitation requirements.

---

## Drift Monitoring

Two-layer Evidently drift check runs after each nightly batch (Airflow `retrain_stub` task):

| Layer | Check | Trigger |
|---|---|---|
| Layer 1 | `DataDriftPreset` — dataset-level | Any dataset drift detected |
| Layer 2 | `ColumnDriftMetric` on 4 key features: `credit_score`, `monthly_salary`, `debt_to_income`, `disposable_income` | Any feature drift detected |

Drift does not block predictions — it logs a warning and alerts the retrain stub. Retrain is manual in the current version.

---

## Retraining Procedure

1. Collect new labelled data into `data/raw/`
2. Re-run `src/data/preprocess.py` → `src/features/feature_engineering.py`
3. Run `notebooks/05_model_training.py` with `FAST_MODE=False`
4. Review MLflow experiment: confirm new AUC ≥ 0.9763 (logistic regression floor)
5. Promote the new model version to `@champion` in the MLflow Registry
6. Restart the FastAPI container: `docker compose restart fastapi`
7. Verify via `GET /health` and `python scripts/healthcheck_all.py`
