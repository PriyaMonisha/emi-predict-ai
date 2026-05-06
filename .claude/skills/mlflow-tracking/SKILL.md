---
name: mlflow-tracking
description: MLflow experiment tracking standard patterns for EMI Predict AI
user-invocable: true
---

# MLflow Tracking — EMI Predict AI Patterns

## Experiment Structure
emi_eligibility_classification/
  runs:
    logistic_baseline_v1
    random_forest_v1
    xgboost_v1
    lightgbm_v1

emi_amount_regression/
  runs:
    linear_baseline_v1
    ridge_v1
    xgboost_regressor_v1
    lightgbm_regressor_v1

## Standard Logging Block (use this template in every training file)
```python
with mlflow.start_run(run_name=f"{model_name}_v{version}") as run:

    # Hyperparameters
    mlflow.log_params(model.get_params())

    # Tags for filtering runs later
    mlflow.set_tags({
        "section": "5",
        "model_type": "classifier",        # or "regressor"
        "data_version": "v4",
        "train_rows": str(len(X_train)),
        "class_ratio": "4.2:1",
        "class_weight": "balanced"
    })

    # Classification metrics
    mlflow.log_metrics({
        "roc_auc_cv_mean": float(cv_roc_auc.mean()),
        "roc_auc_cv_std":  float(cv_roc_auc.std()),
        "f1_cv_mean":      float(cv_f1.mean()),
        "f1_cv_std":       float(cv_f1.std()),
        "precision":       float(precision),
        "recall":          float(recall),
        "train_time_s":    float(train_time)
    })

    # Model + preprocessor artifacts
    mlflow.sklearn.log_model(
        model,
        "model",
        input_example=X_train.head(1)
    )
    mlflow.log_artifact("models/preprocessor.joblib")

    # Visual artifacts
    mlflow.log_figure(confusion_matrix_fig, "confusion_matrix.png")

    run_id = run.info.run_id
    logger.info(f"MLflow run complete. Run ID: {run_id}")