# filename: airflow/dags/emi_batch_prediction_dag.py
# purpose:  Daily batch EMI prediction pipeline with drift monitoring (S7 + S10)
# version:  1.3

"""
EMI Batch Prediction DAG
========================
Scores 17,488 high-risk applicants from unlabeled_for_prediction.csv
using champion models loaded from the MLflow registry.

Pipeline (runs at 02:00 daily):
  validate_input
      → preprocess_raw        (preprocess_data, is_training=False)
      → engineer_features     (FeatureEngineer paired with champion via MLflow)
      → run_predictions       (MLflow @champion classifier + regressor)
      → save_results          ({predictions_dir}/{ds}/{run_id}/predictions.csv)
      → retrain_stub          (no-op placeholder — wired in Section 10)

Airflow Variables required (set via Airflow UI or CLI before first run):
  batch_input_path     : absolute path to unlabeled_for_prediction.csv
  mlflow_tracking_uri  : sqlite:////<abs_path>/mlflow.db  (or http://mlflow:5000 in Docker)
  predictions_dir      : absolute path to data/processed/predictions/
  batch_staging_dir    : absolute path for intermediate files (data/processed/batch/)

Docker note (Section 11):
  mlflow_tracking_uri  → http://mlflow:5000
  All path variables   → /opt/airflow/... (mounted volume paths)
"""

import sys
import logging
from datetime import datetime, timedelta
from pathlib import Path

from airflow.decorators import dag, task
from airflow.models import Variable
from airflow.exceptions import AirflowSkipException
from airflow.operators.python import get_current_context

# Add project root to path so src.pipelines is importable inside tasks
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

logger = logging.getLogger(__name__)

# ── DAG defaults (per infrastructure.md) ──────────────────────────────────────
_DEFAULT_ARGS = {
    "owner":            "emi_predict_ai",
    "retries":          3,
    "retry_delay":      timedelta(minutes=5),
    "email_on_failure": False,
    "email_on_retry":   False,
}


@dag(
    dag_id="emi_batch_prediction",
    description="Daily batch scoring of high-risk EMI applicants via champion models",
    schedule="0 2 * * *",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["emi", "batch", "prediction", "section7"],
    default_args=_DEFAULT_ARGS,
)
def emi_batch_prediction():
    """EMI batch prediction pipeline — scores unlabeled high-risk applicants."""

    # ── Task 1: Validate Input ─────────────────────────────────────────────────
    @task(task_id="validate_input")
    def validate_input(ds: str = None) -> dict:
        """
        Check that the input file exists and has the expected shape.
        Returns a metadata dict passed downstream via XCom.
        """
        import pandas as pd

        batch_path = Variable.get("batch_input_path")
        path       = Path(batch_path)

        if not path.exists():
            raise FileNotFoundError(
                f"[validate_input] Batch input not found: {batch_path}\n"
                f"Set Airflow Variable 'batch_input_path' to the correct path."
            )

        # Lightweight check — read only header + row count
        df_head = pd.read_csv(path, nrows=5)
        row_count = sum(1 for _ in open(path)) - 1  # minus header

        if row_count == 0:
            raise AirflowSkipException(
                f"[validate_input] Input file is empty: {batch_path}"
            )

        batch_date = ds or datetime.utcnow().strftime("%Y-%m-%d")

        ctx          = get_current_context()
        safe_run_id  = (
            ctx["run_id"]
            .replace(":", "_")
            .replace("+", "_")
            .replace("/", "_")
        )

        logger.info(
            f"[validate_input] Input OK — {row_count:,} rows × {df_head.shape[1]} cols  "
            f"run={safe_run_id}"
        )
        return {
            "input_path":   str(path),
            "row_count":    row_count,
            "batch_date":   batch_date,
            "n_cols":       df_head.shape[1],
            "safe_run_id":  safe_run_id,
        }

    # ── Task 2: Preprocess ─────────────────────────────────────────────────────
    @task(task_id="preprocess_raw")
    def preprocess_raw(input_info: dict) -> str:
        """
        Load batch CSV and run preprocess_data(is_training=False).
        Writes cleaned DataFrame to staging dir; returns file path via XCom.
        """
        import pandas as pd
        from src.pipelines.predict_pipeline import (
            load_batch_data,
            preprocess_for_inference,
        )

        staging_dir = Variable.get("batch_staging_dir")
        batch_date  = input_info["batch_date"]
        out_path    = Path(staging_dir) / f"preprocessed_{batch_date}.csv"
        out_path.parent.mkdir(parents=True, exist_ok=True)

        df      = load_batch_data(input_info["input_path"])
        cleaned = preprocess_for_inference(df)
        cleaned.to_csv(out_path, index=False)

        logger.info(
            f"[preprocess_raw] {len(cleaned):,} rows → {out_path.name}"
        )
        return str(out_path)

    # ── Task 3: Feature Engineering ───────────────────────────────────────────
    @task(task_id="engineer_features")
    def engineer_features(preprocessed_path: str) -> str:
        """
        Load the fitted FeatureEngineer directly from pkl file and apply transform().

        Uses direct joblib loading via the models_dir Variable instead of MLflow
        artifact download — MLflow stores Windows paths (file:///C:/Users/...) that
        are incompatible with Linux Docker containers.
        """
        import joblib
        import pandas as pd

        models_dir = Variable.get("models_dir", default_var="/opt/airflow/models")
        fe_path    = Path(models_dir) / "feature_engineer.pkl"
        batch_date = Path(preprocessed_path).stem.replace("preprocessed_", "")
        out_path   = Path(preprocessed_path).parent / f"features_{batch_date}.csv"

        df          = pd.read_csv(preprocessed_path)
        fe          = joblib.load(str(fe_path))
        transformed = fe.transform(df)
        transformed.to_csv(out_path, index=False)

        logger.info(
            f"[engineer_features] {df.shape[1]} → {transformed.shape[1]} cols "
            f"(loaded from {fe_path}) → {out_path.name}"
        )
        return str(out_path)

    # ── Task 4: Run Predictions ────────────────────────────────────────────────
    @task(task_id="run_predictions")
    def run_predictions(features_path: str) -> str:
        """
        Load champion models directly from pkl files and score all rows.

        Uses predict_from_preloaded() with direct joblib loading instead of
        predict_batch() which loads from MLflow registry — MLflow artifact URIs
        contain Windows paths incompatible with Linux Docker containers.
        """
        import joblib
        import pandas as pd
        from src.pipelines.predict_pipeline import predict_from_preloaded

        models_dir  = Variable.get("models_dir", default_var="/opt/airflow/models")
        batch_date  = Path(features_path).stem.replace("features_", "")
        staging_dir = Path(features_path).parent
        out_path    = staging_dir / f"raw_predictions_{batch_date}.csv"

        clf = joblib.load(str(Path(models_dir) / "best_classifier.pkl"))
        reg = joblib.load(str(Path(models_dir) / "best_regressor.pkl"))

        df    = pd.read_csv(features_path)
        preds = predict_from_preloaded(df, clf=clf, reg=reg, use_cache=False)
        preds.to_csv(out_path, index=False)

        zone_counts = preds["conf_zone"].value_counts().to_dict()
        logger.info(
            f"[run_predictions] Scored {len(preds):,} rows | "
            f"approve={zone_counts.get('auto_approve', 0):,}  "
            f"review={zone_counts.get('human_review', 0):,}  "
            f"reject={zone_counts.get('auto_reject', 0):,}"
        )
        return str(out_path)

    # ── Task 5: Save Results ───────────────────────────────────────────────────
    @task(task_id="save_results")
    def save_results(predictions_path: str, input_info: dict) -> dict:
        """
        Copy final predictions to a versioned, immutable output path.

        Path: {predictions_dir}/{batch_date}/{safe_run_id}/predictions.csv
        Using run_id in the path means a 2:05 AM retry never clobbers the
        2:00 AM run — every Airflow run produces a distinct output directory.
        """
        import pandas as pd
        import shutil

        preds_dir   = Variable.get("predictions_dir")
        batch_date  = input_info["batch_date"]
        safe_run_id = input_info["safe_run_id"]

        # Immutable per-run output path
        final_path  = Path(preds_dir) / batch_date / safe_run_id / "predictions.csv"
        final_path.parent.mkdir(parents=True, exist_ok=True)

        shutil.copy2(predictions_path, final_path)
        preds = pd.read_csv(final_path)

        zone_counts  = preds["conf_zone"].value_counts().to_dict()
        eligible_pct = round(preds["clf_label"].mean() * 100, 2)
        median_emi   = round(float(preds.loc[preds["clf_label"] == 1, "predicted_emi"].median()), 2)

        summary = {
            "batch_date":     batch_date,
            "safe_run_id":    safe_run_id,
            "total_scored":   len(preds),
            "auto_approve":   zone_counts.get("auto_approve", 0),
            "human_review":   zone_counts.get("human_review", 0),
            "auto_reject":    zone_counts.get("auto_reject", 0),
            "eligible_pct":   eligible_pct,
            "median_emi_inr": median_emi,
            "output_path":    str(final_path),
        }

        logger.info(
            f"[save_results] Batch {batch_date} complete\n"
            f"  Run     : {safe_run_id}\n"
            f"  Total   : {summary['total_scored']:,}\n"
            f"  Approve : {summary['auto_approve']:,}\n"
            f"  Review  : {summary['human_review']:,}\n"
            f"  Reject  : {summary['auto_reject']:,}\n"
            f"  Eligible: {eligible_pct:.1f}%\n"
            f"  Saved   : {final_path}"
        )
        return summary

    # ── Task 6: Drift Monitor (formerly retrain_stub) ─────────────────────────
    @task(task_id="retrain_stub")
    def retrain_stub(summary: dict) -> None:
        """
        Two-layer Evidently drift check after each batch run.

        Non-blocking: if the features file is missing or Evidently errors,
        logs a warning and returns — never fails the DAG.
        Retraining remains deferred; this task surfaces the signal for S11+.

        Layer 1: 25 raw input features (population drift)
        Layer 2: 4 key engineered features (ML signal drift)
        """
        from src.monitoring.drift_monitor import run_drift_report

        batch_date    = summary.get("batch_date", "unknown")
        staging_dir   = Variable.get("batch_staging_dir", default_var="")
        reference_path = Variable.get(
            "reference_features_path",
            default_var=str(_PROJECT_ROOT / "data" / "processed" / "train_features.csv"),
        )
        drift_output_dir = Variable.get(
            "drift_reports_dir",
            default_var=str(_PROJECT_ROOT / "data" / "processed" / "drift_reports"),
        )

        # Infer current batch features path from staging dir + batch_date
        current_path = (
            str(Path(staging_dir) / f"features_{batch_date}.csv")
            if staging_dir
            else ""
        )

        result = run_drift_report(
            reference_path=reference_path,
            current_path=current_path,
            output_dir=drift_output_dir,
            batch_date=batch_date,
        )

        if result["skipped"]:
            logger.warning(
                f"[retrain_stub] Drift check skipped for {batch_date}: "
                f"{result['skip_reason']}"
            )
            return

        if result["drift_detected"]:
            logger.warning(
                f"[retrain_stub] DRIFT DETECTED — batch {batch_date} | "
                f"L1={result['layer1_drifted_features']} | "
                f"L2={result['layer2_drifted_features']} | "
                f"report={result['report_path']}"
            )
        else:
            logger.info(
                f"[retrain_stub] No drift detected for batch {batch_date}. "
                f"Scored {summary.get('total_scored', 0):,} rows."
            )

    # ── Wire up the pipeline ───────────────────────────────────────────────────
    _input_info        = validate_input()
    _preprocessed_path = preprocess_raw(_input_info)
    _features_path     = engineer_features(_preprocessed_path)
    _predictions_path  = run_predictions(_features_path)
    _summary           = save_results(_predictions_path, _input_info)
    retrain_stub(_summary)


# Instantiate the DAG
dag_instance = emi_batch_prediction()
