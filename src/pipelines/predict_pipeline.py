# filename: src/pipelines/predict_pipeline.py
# purpose:  Reusable batch prediction logic — called by Airflow DAG (S7) and FastAPI (S9)
# version:  1.2

import sys
import logging
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.data.preprocess import preprocess_data
from src.features.feature_engineering import FeatureEngineer

TARGET_CLF   = "emi_eligibility"
TARGET_REG   = "max_monthly_emi"
AUTO_APPROVE = 0.85
AUTO_REJECT  = 0.40
EMI_MIN      = 500.0
EMI_MAX      = 34_750.0

CATEGORICAL_COLS = [
    "gender", "marital_status", "education",
    "employment_type", "company_type", "house_type",
    "existing_loans", "emi_scenario", "credit_score_band",
]


def load_batch_data(file_path: str) -> pd.DataFrame:
    """Load a batch CSV. Normalises column names and validates shape."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Batch input not found: {file_path}")
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip().str.lower()
    if df.empty:
        raise ValueError(f"Batch file is empty: {file_path}")
    logger.info(f"Loaded {len(df):,} rows × {df.shape[1]} cols from {path.name}")
    return df


def preprocess_for_inference(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply full preprocessing pipeline in inference mode.
    - is_training=False  → skips unlabeled extraction (step 7) and target encoding (step 16)
    - Drops any target columns before returning so downstream steps see features only.
    """
    cleaned = preprocess_data(df, is_training=False, save_unlabeled=False)
    drop = [c for c in [TARGET_CLF, TARGET_REG] if c in cleaned.columns]
    if drop:
        cleaned = cleaned.drop(columns=drop)
        logger.info(f"Dropped target columns for inference: {drop}")
    logger.info(f"After preprocessing: {len(cleaned):,} rows × {cleaned.shape[1]} cols")
    return cleaned


def engineer_features_for_inference(
    df: pd.DataFrame,
    fe_path: str,
) -> pd.DataFrame:
    """Load the fitted FeatureEngineer (from Section 4) and transform inference data."""
    fe: FeatureEngineer = joblib.load(fe_path)
    transformed = fe.transform(df)
    logger.info(
        f"Feature engineering: {df.shape[1]} → {transformed.shape[1]} cols  "
        f"(+{transformed.shape[1] - df.shape[1]} new)"
    )
    return transformed


def load_feature_engineer_from_mlflow(
    mlflow_tracking_uri: str,
    clf_registry_uri: str = "models:/emi_eligibility_classifier@champion",
) -> FeatureEngineer:
    """
    Load the FeatureEngineer pkl that was logged alongside the champion classifier.

    Resolves the champion model's run_id via the MLflow registry alias, then
    downloads the paired artifact so model and feature engineer are always in sync.
    Promotes a new champion → correct feature engineer is loaded automatically.
    """
    import mlflow
    import mlflow.artifacts
    from mlflow.tracking import MlflowClient

    mlflow.set_tracking_uri(mlflow_tracking_uri)

    # Parse "models:/NAME@ALIAS" → (NAME, ALIAS)
    uri_path   = clf_registry_uri.removeprefix("models:/")
    model_name, alias = uri_path.rsplit("@", 1)

    mv       = MlflowClient().get_model_version_by_alias(model_name, alias)
    art_uri  = f"runs:/{mv.run_id}/preprocessor/feature_engineer.pkl"
    local_path = mlflow.artifacts.download_artifacts(art_uri)

    fe: FeatureEngineer = joblib.load(local_path)
    logger.info(
        f"Loaded FeatureEngineer from MLflow run {mv.run_id[:8]}… "
        f"(champion alias: {alias})"
    )
    return fe


def _ohe_and_align(df: pd.DataFrame, model) -> pd.DataFrame:
    """
    One-hot encode categorical columns, then reindex to match the model's
    expected feature set (model.feature_names_in_).

    Missing OHE columns are filled with 0; extra columns are dropped.
    This guarantees inference always matches training schema exactly.
    """
    # Match both object and category dtype — category dtype is preserved when
    # DataFrames are passed in-memory (FastAPI). Batch DAG avoids this silently
    # via CSV round-trip, but in-memory inference breaks without this check.
    cat_present = [
        c for c in CATEGORICAL_COLS
        if c in df.columns and not pd.api.types.is_numeric_dtype(df[c])
    ]
    X = pd.get_dummies(df, columns=cat_present, drop_first=True)

    # Cast bool columns that get_dummies may produce
    bool_cols = X.select_dtypes("bool").columns.tolist()
    if bool_cols:
        X[bool_cols] = X[bool_cols].astype(int)

    X = X.replace([float("inf"), float("-inf")], float("nan")).fillna(0)

    # Align to training schema using model's stored feature names
    expected: list | None = None
    if hasattr(model, "feature_names_in_"):
        expected = list(model.feature_names_in_)
    elif hasattr(model, "feature_name_"):          # LightGBM sklearn attribute — list, not callable
        expected = list(model.feature_name_)

    if expected:
        missing = [c for c in expected if c not in X.columns]
        extra   = [c for c in X.columns if c not in expected]
        if missing:
            logger.warning(f"Adding {len(missing)} missing OHE columns as 0: {missing[:5]}...")
        if extra:
            logger.debug(f"Dropping {len(extra)} extra columns not in training schema")
        X = X.reindex(columns=expected, fill_value=0)

    return X


def predict_batch(
    df: pd.DataFrame,
    mlflow_tracking_uri: str,
    clf_registry_uri: str = "models:/emi_eligibility_classifier@champion",
    reg_registry_uri: str = "models:/emi_amount_regressor@champion",
    use_cache: bool = True,
    customer_id_col: str = "customer_id",
) -> pd.DataFrame:
    """
    Load champion models from MLflow registry and score all rows.

    Returns DataFrame with columns:
      clf_proba, clf_label, conf_zone, predicted_emi

    When use_cache=True and customer_id_col is present, writes the scored
    feature vectors to Redis after inference so FastAPI (S9) can serve
    repeat lookups without re-running feature engineering.
    Cache write is non-blocking — a Redis outage does not affect inference.
    """
    import mlflow
    import mlflow.sklearn

    mlflow.set_tracking_uri(mlflow_tracking_uri)
    logger.info(f"Loading classifier  : {clf_registry_uri}")
    clf = mlflow.sklearn.load_model(clf_registry_uri)
    logger.info(f"Loading regressor   : {reg_registry_uri}")
    reg = mlflow.sklearn.load_model(reg_registry_uri)

    X_clf = _ohe_and_align(df, clf)
    X_reg = _ohe_and_align(df, reg)

    clf_proba = clf.predict_proba(X_clf)[:, 1]
    reg_preds = reg.predict(X_reg).clip(EMI_MIN, EMI_MAX)

    def _zone(p: float) -> str:
        if p > AUTO_APPROVE:
            return "auto_approve"
        if p >= AUTO_REJECT:
            return "human_review"
        return "auto_reject"

    preds = pd.DataFrame({
        "clf_proba":     np.round(clf_proba, 6),
        "clf_label":     (clf_proba >= 0.5).astype(int),
        "conf_zone":     [_zone(p) for p in clf_proba],
        "predicted_emi": np.round(reg_preds, 2),
    })

    # ── Cache-aside: populate Redis after inference ────────────────────────────
    # Batch runs write feature vectors so FastAPI (S9) can skip feature
    # engineering on repeat lookups. Skipped silently if Redis is down.
    if use_cache and customer_id_col in df.columns:
        from src.features.feature_store import batch_write, health_check
        feature_cols = [c for c in df.columns if c != customer_id_col]
        if health_check():
            result = batch_write(
                records=df.reset_index(drop=True).to_dict("records"),
                id_col=customer_id_col,
                feature_cols=feature_cols,
            )
            logger.info(
                f"[predict_batch] Redis cache: "
                f"{result['written']:,} written, {result['errors']} errors"
            )
        else:
            logger.warning(
                "[predict_batch] Redis unavailable — cache write skipped, "
                "inference results unchanged"
            )
    # ── End cache block ────────────────────────────────────────────────────────

    zones = preds["conf_zone"].value_counts().to_dict()
    logger.info(
        f"Scored {len(preds):,} rows — "
        f"approve={zones.get('auto_approve', 0):,}  "
        f"review={zones.get('human_review', 0):,}  "
        f"reject={zones.get('auto_reject', 0):,}"
    )
    return preds


def predict_from_preloaded(
    df: pd.DataFrame,
    clf,
    reg,
    use_cache: bool = True,
    customer_id_col: str = "customer_id",
) -> pd.DataFrame:
    """
    Score a FE-transformed DataFrame using pre-loaded models (no MLflow I/O).

    df must be the 51-col FE-transformed representation (raw categoricals intact).
    _ohe_and_align() handles the OHE step before scoring — do NOT pass OHE'd data.

    Called by FastAPI endpoints where clf/reg/fe are loaded once at startup.
    Reuses the same _ohe_and_align + confidence zone + cache-aside logic as
    predict_batch(), without the per-call MLflow model loading overhead.

    Returns DataFrame with columns: clf_proba, clf_label, conf_zone, predicted_emi
    """
    X_clf = _ohe_and_align(df, clf)
    X_reg = _ohe_and_align(df, reg)

    clf_proba = clf.predict_proba(X_clf)[:, 1].astype(float)
    # Cast to float64 before rounding — XGBoost returns float32 which causes
    # precision artefacts (e.g. 10176.080078125 instead of 10176.08) after round()
    reg_preds = reg.predict(X_reg).astype(float).clip(EMI_MIN, EMI_MAX)

    def _zone(p: float) -> str:
        if p > AUTO_APPROVE:
            return "auto_approve"
        if p >= AUTO_REJECT:
            return "human_review"
        return "auto_reject"

    preds = pd.DataFrame({
        "clf_proba":     np.round(clf_proba, 6),
        "clf_label":     (clf_proba >= 0.5).astype(int),
        "conf_zone":     [_zone(p) for p in clf_proba],
        "predicted_emi": np.round(reg_preds, 2),
    })

    # ── Cache-aside: write FE-transformed features to Redis after scoring ──────
    if use_cache and customer_id_col in df.columns:
        from src.features.feature_store import batch_write, health_check
        feature_cols = [c for c in df.columns if c != customer_id_col]
        if health_check():
            result = batch_write(
                records=df.reset_index(drop=True).to_dict("records"),
                id_col=customer_id_col,
                feature_cols=feature_cols,
            )
            logger.info(
                f"[predict_from_preloaded] Redis cache: "
                f"{result['written']:,} written, {result['errors']} errors"
            )
        else:
            logger.warning(
                "[predict_from_preloaded] Redis unavailable — cache write skipped"
            )

    zones = preds["conf_zone"].value_counts().to_dict()
    logger.info(
        f"Scored {len(preds):,} rows — "
        f"approve={zones.get('auto_approve', 0):,}  "
        f"review={zones.get('human_review', 0):,}  "
        f"reject={zones.get('auto_reject', 0):,}"
    )
    return preds


def save_predictions(preds: pd.DataFrame, output_path: str) -> str:
    """Write predictions CSV; creates parent directories if needed."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    preds.to_csv(path, index=False)
    logger.info(f"Saved {len(preds):,} predictions → {path}")
    return str(path)
