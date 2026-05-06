# Section 5 — Model Training

## Objective
Train 5 classifiers and 4 regressors on engineered features (Section 4 output),
optimise with Optuna, and select the best model for each task.

- **Classification target:** `emi_eligibility` (0/1)
- **Regression target:** `max_monthly_emi` (₹500–₹34,750)
- **Baseline to beat:** Logistic Regression AUC 0.9763 (Section 3, raw features)

---

## Data Inputs
| File | Rows | Columns | Source |
|---|---|---|---|
| `data/processed/train_features.csv` | ~309,830 | 58+ | Section 4 |
| `data/processed/test_features.csv` | ~77,457 | 58+ | Section 4 |

Feature columns after one-hot encoding: 42 engineered + OHE expansions of 9 categoricals.

---

## Design Decisions

| Decision | Choice | Reason |
|---|---|---|
| Hyperparameter tuning | Optuna TPESampler | Bayesian, efficient, reproducible |
| Classifier imbalance | `class_weight='balanced'` (sklearn) / `scale_pos_weight` (XGBoost) | Locked 4.2:1 ratio |
| Optuna trials | LR=10, trees=25 | Crash prevention; still thorough |
| MLflow | Deferred to Section 6 | Keeps Section 5 focused |
| Model persistence | `joblib` bundles (model + feature_names) | Safe for serving in Section 9 |
| Best classifier | Highest ROC-AUC | Primary metric per locked decisions |
| Best regressor | Lowest RMSE | Primary regression metric |

---

## Classifiers (5 models)

| Model | Class Balance | Key Hyperparams Tuned |
|---|---|---|
| Logistic Regression | `class_weight='balanced'` | C, solver, penalty |
| Random Forest | `class_weight='balanced'` | n_estimators, max_depth, min_samples, max_features |
| XGBoost | `scale_pos_weight=4.2` | n_estimators, max_depth, lr, subsample, colsample, reg_alpha/lambda |
| LightGBM | `class_weight='balanced'` | n_estimators, num_leaves, max_depth, lr, subsample, min_child_samples |
| Extra Trees | `class_weight='balanced'` | n_estimators, max_depth, min_samples, max_features |

**Performance targets:** XGBoost / LightGBM ROC-AUC > 0.85 (locked in ml-code.md)

---

## Regressors (4 models)

| Model | Key Hyperparams Tuned |
|---|---|
| Random Forest | n_estimators, max_depth, min_samples, max_features |
| XGBoost | n_estimators, max_depth, lr, subsample, colsample, reg_alpha/lambda |
| LightGBM | n_estimators, num_leaves, max_depth, lr, subsample, min_child_samples |
| Extra Trees | n_estimators, max_depth, min_samples, max_features |

**Performance targets:** R² > 0.75, MAPE < 20% (locked in ml-code.md)

---

## Saved Artifacts

| File | Contents |
|---|---|
| `models/best_classifier.pkl` | `{name, model, feature_names}` — best ROC-AUC classifier |
| `models/best_regressor.pkl` | `{name, model, feature_names}` — lowest RMSE regressor |
| `reports/model_metrics.json` | All metrics for classifiers + regressors |
| `docs/figures/05_classifier_comparison.png` | ROC-AUC & F1 bar chart (Paired_r) |
| `docs/figures/05_regressor_comparison.png` | RMSE & MAE bar chart (Paired_r) |
| `docs/figures/05_feature_importance.png` | Top 20 features from best classifier (Dark2_r) |

---

## Source Files

| File | Purpose |
|---|---|
| `src/models/train_classifier.py` | 5 classifier trainer functions + `evaluate_classifier()` |
| `src/models/train_regressor.py` | 4 regressor trainer functions + `evaluate_regressor()` |
| `notebooks/05_model_training.py` | End-to-end training run |

---

## Next Section
**Section 6 — MLflow Experiment Tracking**
Log all training runs to MLflow, register best models in the model registry,
and build experiment comparison views for the full Section 5 run history.
