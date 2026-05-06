---
paths:
  - "src/models/**/*.py"
  - "src/training/**/*.py"
  - "scripts/train*.py"
---

# ML Code Rules — EMI Predict AI

## Absolute Rules (Never Override Without Explicit User Decision)
- class_weight='balanced' on EVERY classifier — check this first always
- Primary metrics: ROC-AUC + F1 — never report accuracy as primary
- random_state=42 on all models, splits, and sampling operations
- Stratified splits always: StratifiedKFold, stratify=y in train_test_split
- 5-fold CV minimum — never use single train/val split for final evaluation
- Optuna for tuning — never GridSearchCV or manual search

## MLflow Logging (Mandatory in All Training Files)
Every training run must log:
- All hyperparameters: mlflow.log_params(model.get_params())
- All metrics: roc_auc, f1, precision, recall (+ rmse, mae, r2 for regressors)
- Tags: section, model_type, data_version, train_rows, class_ratio
- Model artifact: mlflow.sklearn.log_model(model, "model")
- Preprocessor artifact: mlflow.log_artifact("path/to/preprocessor.joblib")
- Confusion matrix or residuals plot as figure artifact

## Model Persistence
- Save to: models/{model_name}/{timestamp}/
- Save preprocessor alongside model — always together, never separately
- Save metrics.json alongside model — for quick loading without MLflow
- Never save model without preprocessor in same directory

## Performance Targets
Section 3 — Baseline floor to establish:
  Rule-based: compute and record AUC (this is the floor)
  Logistic regression: target ROC-AUC > 0.72

Section 5 — Production targets:
  XGBoost classifier:  ROC-AUC > 0.85
  LightGBM classifier: ROC-AUC > 0.85
  Regressors (max_monthly_emi): R² > 0.75, MAPE < 20%

## Forbidden Patterns
- accuracy_score as primary or only metric
- GridSearchCV (use Optuna)
- fit() or fit_transform() called on test/val data
- Model saved without preprocessor in same location
- Hardcoded hyperparameters (use configs/ YAML files)
- Training code without MLflow logging block