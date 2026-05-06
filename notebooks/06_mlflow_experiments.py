# filename: notebooks/06_mlflow_experiments.py
# purpose:  Section 6 — MLflow experiment tracking for all 9 trained models
# version:  1.0

# ── Cell 1: Setup ─────────────────────────────────────────────────────────────
import sys
import time
import logging
import warnings
warnings.filterwarnings("ignore")

# Force UTF-8 stdout so ₹, ─, and other non-ASCII characters print on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from mlflow.models import infer_signature

np.random.seed(42)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

ROOT              = Path(__file__).resolve().parent.parent
DATA_DIR          = ROOT / "data" / "processed"
MODELS_DIR        = ROOT / "models"
PREPROCESSOR_PATH = MODELS_DIR / "feature_engineer.pkl"

sys.path.insert(0, str(ROOT))
from src.models.train_classifier import (
    train_logistic_regression, train_random_forest,
    train_xgboost, train_lightgbm, train_extra_trees,
)
from src.models.train_regressor import (
    train_rf_regressor, train_xgb_regressor,
    train_lgbm_regressor, train_et_regressor,
)

# Palettes (locked — same as Section 5)
PR   = sns.color_palette("Paired_r",  12)
D2   = sns.color_palette("Dark2_r",    8)
AC   = sns.color_palette("Accent_r",   8)
DARK = "#2D3436"

plt.rcParams.update({
    "figure.dpi":        130,
    "font.size":         10,
    "axes.titlesize":    12,
    "axes.titleweight":  "bold",
    "axes.labelsize":    10,
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

# ── FAST_MODE config ───────────────────────────────────────────────────────────
FAST_MODE     = True
OPTUNA_TRIALS = 3      if FAST_MODE else 25
OPTUNA_CV     = 3
TUNE_SAMPLE   = 50_000 if FAST_MODE else None

# ── MLflow config ──────────────────────────────────────────────────────────────
MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
EXPERIMENT_CLF      = "emi_eligibility_classification"
EXPERIMENT_REG      = "emi_amount_regression"
REGISTRY_NAME_CLF   = "emi_eligibility_classifier"
REGISTRY_NAME_REG   = "emi_amount_regressor"
DATA_VERSION        = "section4_v1"
BASELINE_AUC        = 0.9763

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
logger.info(f"MLflow tracking URI: {MLFLOW_TRACKING_URI}")

# ── Cell 2: Load Data ──────────────────────────────────────────────────────────
logger.info("Loading feature data...")
train_df = pd.read_csv(DATA_DIR / "train_features.csv")
test_df  = pd.read_csv(DATA_DIR / "test_features.csv")

TARGET_CLF = "emi_eligibility"
TARGET_REG = "max_monthly_emi"

print("=" * 65)
print(f"SECTION 6: MLflow EXPERIMENT TRACKING  {'[FAST_MODE]' if FAST_MODE else '[FULL MODE]'}")
print("=" * 65)
print(f"\nTrain : {train_df.shape[0]:,} rows × {train_df.shape[1]} columns")
print(f"Test  : {test_df.shape[0]:,} rows × {test_df.shape[1]} columns")
print(f"Class balance (train): {train_df[TARGET_CLF].value_counts().to_dict()}")

# ── Cell 3: Encode Categoricals & Prepare Feature Matrices ────────────────────
CATEGORICAL_COLS = [
    "gender", "marital_status", "education",
    "employment_type", "company_type", "house_type",
    "existing_loans", "emi_scenario",
    "credit_score_band",
]
cat_present = [c for c in CATEGORICAL_COLS if c in train_df.columns
               and train_df[c].dtype == object]

train_enc = pd.get_dummies(train_df, columns=cat_present, drop_first=True)
test_enc  = pd.get_dummies(test_df,  columns=cat_present, drop_first=True)
test_enc  = test_enc.reindex(columns=train_enc.columns, fill_value=0)

drop_cols   = [c for c in [TARGET_CLF, TARGET_REG] if c in train_enc.columns]
X_train     = train_enc.drop(columns=drop_cols)
y_train_clf = train_enc[TARGET_CLF]
y_train_reg = train_enc[TARGET_REG]

X_test     = test_enc.drop(columns=drop_cols)
y_test_clf = test_enc[TARGET_CLF]
y_test_reg = test_enc[TARGET_REG]

bool_cols = X_train.select_dtypes("bool").columns.tolist()
if bool_cols:
    X_train[bool_cols] = X_train[bool_cols].astype(int)
    X_test[bool_cols]  = X_test[bool_cols].astype(int)

X_train = X_train.replace([np.inf, -np.inf], np.nan).fillna(0)
X_test  = X_test.replace([np.inf, -np.inf], np.nan).fillna(0)

logger.info(f"Feature matrix: {X_train.shape[1]} columns after OHE")

# Compute class stats once — reused in all run tags
N_TRAIN     = len(X_train)
N0          = int((y_train_clf == 0).sum())
N1          = int((y_train_clf == 1).sum())
CLASS_RATIO = round(N0 / N1, 4)

# FAST_MODE: stratified 50k tuning sample
if TUNE_SAMPLE and TUNE_SAMPLE < len(X_train):
    _, tune_idx = train_test_split(
        np.arange(len(X_train)),
        test_size=TUNE_SAMPLE,
        stratify=y_train_clf,
        random_state=42,
    )
    X_tune     = X_train.iloc[tune_idx].reset_index(drop=True)
    y_tune_clf = y_train_clf.iloc[tune_idx].reset_index(drop=True)
    y_tune_reg = y_train_reg.iloc[tune_idx].reset_index(drop=True)
    print(f"\nFAST_MODE ON  : tune sample = {len(X_tune):,} rows "
          f"({y_tune_clf.mean()*100:.1f}% eligible)")
    print(f"               final train  = {len(X_train):,} rows (full data)")
    print(f"               {OPTUNA_TRIALS} trials × {OPTUNA_CV}-fold CV per model")
    print(f"               Est. total time: ~5–8 min")
else:
    X_tune = y_tune_clf = y_tune_reg = None
    print("\nFAST_MODE OFF : tuning on full training data")

# ── Cell 4: Helper Functions ───────────────────────────────────────────────────

def _make_confusion_matrix_fig(
    y_true: pd.Series,
    y_pred: np.ndarray,
    model_name: str,
) -> plt.Figure:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Dark2_r",
        xticklabels=["Not Eligible", "Eligible"],
        yticklabels=["Not Eligible", "Eligible"],
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(
        f"Confusion Matrix — {model_name.replace('_', ' ').title()}\n"
        f"Section 6 MLflow Run",
        pad=10,
    )
    plt.tight_layout()
    return fig


def _make_residuals_fig(
    y_true: pd.Series,
    y_pred: np.ndarray,
    model_name: str,
) -> plt.Figure:
    residuals = y_true.values - y_pred
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    ax1.scatter(y_true, y_pred, alpha=0.3, s=2, color=PR[1])
    max_val = max(float(y_true.max()), float(y_pred.max()))
    ax1.plot([0, max_val], [0, max_val], color=D2[0], linewidth=1.5, linestyle="--")
    ax1.set_xlabel("Actual EMI (₹)")
    ax1.set_ylabel("Predicted EMI (₹)")
    ax1.set_title("Actual vs Predicted")

    ax2.hist(residuals, bins=60, color=AC[2], alpha=0.85)
    ax2.axvline(0, color=D2[0], linewidth=1.5, linestyle="--")
    ax2.set_xlabel("Residual (₹)")
    ax2.set_ylabel("Count")
    ax2.set_title("Residual Distribution")

    fig.suptitle(
        f"Residuals — {model_name.replace('_', ' ').title()} | Section 6 MLflow Run",
    )
    plt.tight_layout()
    return fig


def _sanitize_params(params: dict) -> dict:
    return {k: str(v) for k, v in params.items() if v is not None}


def _build_clf_tags(model_name: str) -> dict:
    return {
        "section":       "06",
        "model_type":    model_name,
        "task":          "classification",
        "data_version":  DATA_VERSION,
        "train_rows":    str(N_TRAIN),
        "class_ratio":   str(CLASS_RATIO),
        "class_weight":  "balanced",
        "fast_mode":     str(FAST_MODE),
        "optuna_trials": str(OPTUNA_TRIALS),
        "optuna_cv":     str(OPTUNA_CV),
    }


def _build_reg_tags(model_name: str) -> dict:
    return {
        "section":       "06",
        "model_type":    model_name,
        "task":          "regression",
        "data_version":  DATA_VERSION,
        "train_rows":    str(N_TRAIN),
        "fast_mode":     str(FAST_MODE),
        "optuna_trials": str(OPTUNA_TRIALS),
        "optuna_cv":     str(OPTUNA_CV),
    }


def _log_clf_run(
    model_name: str,
    run_name: str,
    train_fn,
) -> dict:
    """Run one classifier inside an MLflow context; return result dict + run_id."""
    print(f"\n{'-'*65}\nMLflow Run: {run_name}  ({OPTUNA_TRIALS} trials)\n{'-'*65}")
    t0 = time.time()

    with mlflow.start_run(run_name=run_name, tags=_build_clf_tags(model_name)) as run:
        result = train_fn(
            X_train, y_train_clf, X_test, y_test_clf,
            n_trials=OPTUNA_TRIALS,
            X_tune=X_tune, y_tune=y_tune_clf,
            cv=OPTUNA_CV if FAST_MODE else None,
        )
        m = result["metrics"]

        mlflow.log_params(_sanitize_params({
            **m["best_params"],
            "n_trials":  OPTUNA_TRIALS,
            "fast_mode": FAST_MODE,
            "cv_folds":  OPTUNA_CV if FAST_MODE else 5,
        }))
        mlflow.log_metrics({
            "roc_auc":            m["roc_auc"],
            "f1":                 m["f1"],
            "precision":          m["precision"],
            "recall":             m["recall"],
            "accuracy":           m["accuracy"],
            "auto_approve_count": float(m["confidence_zones"]["auto_approve"]),
            "human_review_count": float(m["confidence_zones"]["human_review"]),
            "auto_reject_count":  float(m["confidence_zones"]["auto_reject"]),
        })

        y_pred_run = result["model"].predict(X_test)
        fig_cm = _make_confusion_matrix_fig(y_test_clf, y_pred_run, model_name)
        mlflow.log_figure(fig_cm, "confusion_matrix.png")
        plt.close(fig_cm)

        sig = infer_signature(X_test, result["model"].predict_proba(X_test))
        mlflow.sklearn.log_model(
            sk_model=result["model"],
            artifact_path="model",
            signature=sig,
            input_example=X_test.iloc[:5],
        )
        mlflow.log_artifact(str(PREPROCESSOR_PATH), artifact_path="preprocessor")
        run_id = run.info.run_id

    print(f"  ROC-AUC : {m['roc_auc']:.4f}  F1: {m['f1']:.4f}")
    print(f"  Run ID  : {run_id}")
    print(f"  Elapsed : {time.time()-t0:.1f}s")
    return {**result, "run_id": run_id}


def _log_reg_run(
    model_name: str,
    run_name: str,
    train_fn,
) -> dict:
    """Run one regressor inside an MLflow context; return result dict + run_id."""
    print(f"\n{'-'*65}\nMLflow Run: {run_name}  ({OPTUNA_TRIALS} trials)\n{'-'*65}")
    t0 = time.time()

    with mlflow.start_run(run_name=run_name, tags=_build_reg_tags(model_name)) as run:
        result = train_fn(
            X_train, y_train_reg, X_test, y_test_reg,
            n_trials=OPTUNA_TRIALS,
            X_tune=X_tune, y_tune=y_tune_reg,
            cv=OPTUNA_CV if FAST_MODE else None,
        )
        m = result["metrics"]

        mlflow.log_params(_sanitize_params({
            **m["best_params"],
            "n_trials":  OPTUNA_TRIALS,
            "fast_mode": FAST_MODE,
            "cv_folds":  OPTUNA_CV if FAST_MODE else 5,
        }))
        mlflow.log_metrics({
            "rmse": m["rmse"],
            "mae":  m["mae"],
            "r2":   m["r2"],
            "mape": m["mape"],
        })

        y_pred_run = result["model"].predict(X_test)
        fig_res = _make_residuals_fig(y_test_reg, y_pred_run, model_name)
        mlflow.log_figure(fig_res, "residuals.png")
        plt.close(fig_res)

        sig = infer_signature(X_test, result["model"].predict(X_test))
        mlflow.sklearn.log_model(
            sk_model=result["model"],
            artifact_path="model",
            signature=sig,
            input_example=X_test.iloc[:5],
        )
        mlflow.log_artifact(str(PREPROCESSOR_PATH), artifact_path="preprocessor")
        run_id = run.info.run_id

    print(f"  RMSE: ₹{m['rmse']:,.0f}  R²: {m['r2']:.4f}  MAPE: {m['mape']:.1f}%")
    print(f"  Run ID: {run_id}")
    print(f"  Elapsed: {time.time()-t0:.1f}s")
    return {**result, "run_id": run_id}


# ── Cell 5: Experiment 1 — Classification ─────────────────────────────────────
mlflow.set_experiment(EXPERIMENT_CLF)
print(f"\nExperiment: {EXPERIMENT_CLF}")
print(f"Tracking URI: {MLFLOW_TRACKING_URI}")
all_clf_results: dict = {}

# ── Cell 6: Classifier 1 / 5 — Logistic Regression ───────────────────────────
all_clf_results["logistic_regression"] = _log_clf_run(
    "logistic_regression", "logistic_regression_v1", train_logistic_regression,
)

# ── Cell 7: Classifier 2 / 5 — Random Forest ──────────────────────────────────
all_clf_results["random_forest"] = _log_clf_run(
    "random_forest", "random_forest_v1", train_random_forest,
)

# ── Cell 8: Classifier 3 / 5 — XGBoost ───────────────────────────────────────
all_clf_results["xgboost"] = _log_clf_run(
    "xgboost", "xgboost_v1", train_xgboost,
)

# ── Cell 9: Classifier 4 / 5 — LightGBM ──────────────────────────────────────
all_clf_results["lightgbm"] = _log_clf_run(
    "lightgbm", "lightgbm_v1", train_lightgbm,
)

# ── Cell 10: Classifier 5 / 5 — Extra Trees ───────────────────────────────────
all_clf_results["extra_trees"] = _log_clf_run(
    "extra_trees", "extra_trees_v1", train_extra_trees,
)

# Select best classifier: primary=AUC, tie-break=F1, final=speed preference
_CLF_SPEED_PREF = ["lightgbm", "xgboost", "random_forest", "extra_trees", "logistic_regression"]
best_clf_name = max(
    all_clf_results,
    key=lambda n: (
        all_clf_results[n]["metrics"]["roc_auc"],
        all_clf_results[n]["metrics"]["f1"],
        -_CLF_SPEED_PREF.index(n) if n in _CLF_SPEED_PREF else -99,
    ),
)
best_clf_run_id = all_clf_results[best_clf_name]["run_id"]

print("\n" + "=" * 65)
print("ALL 5 CLASSIFIERS LOGGED TO MLflow")
print("=" * 65)
for _n in sorted(all_clf_results, key=lambda n: -all_clf_results[n]["metrics"]["roc_auc"]):
    _m     = all_clf_results[_n]["metrics"]
    marker = " <- CHAMPION" if _n == best_clf_name else ""
    print(f"  {_n:22}: AUC {_m['roc_auc']:.4f}  F1 {_m['f1']:.4f}{marker}")
print(f"\nBest classifier: {best_clf_name}  "
      f"(AUC {all_clf_results[best_clf_name]['metrics']['roc_auc']:.4f})")

# ── Cell 11: Experiment 2 — Regression ────────────────────────────────────────
mlflow.set_experiment(EXPERIMENT_REG)
print(f"\nExperiment: {EXPERIMENT_REG}")
all_reg_results: dict = {}

# ── Cell 12: Regressor 1 / 4 — Random Forest ──────────────────────────────────
all_reg_results["random_forest"] = _log_reg_run(
    "random_forest", "random_forest_v1", train_rf_regressor,
)

# ── Cell 13: Regressor 2 / 4 — XGBoost ───────────────────────────────────────
all_reg_results["xgboost"] = _log_reg_run(
    "xgboost", "xgboost_v1", train_xgb_regressor,
)

# ── Cell 14: Regressor 3 / 4 — LightGBM ──────────────────────────────────────
all_reg_results["lightgbm"] = _log_reg_run(
    "lightgbm", "lightgbm_v1", train_lgbm_regressor,
)

# ── Cell 15: Regressor 4 / 4 — Extra Trees ────────────────────────────────────
all_reg_results["extra_trees"] = _log_reg_run(
    "extra_trees", "extra_trees_v1", train_et_regressor,
)

# Select best regressor: primary=RMSE (minimize)
best_reg_name   = min(all_reg_results, key=lambda n: all_reg_results[n]["metrics"]["rmse"])
best_reg_run_id = all_reg_results[best_reg_name]["run_id"]

print("\n" + "=" * 65)
print("ALL 4 REGRESSORS LOGGED TO MLflow")
print("=" * 65)
for _n in sorted(all_reg_results, key=lambda n: all_reg_results[n]["metrics"]["rmse"]):
    _m     = all_reg_results[_n]["metrics"]
    marker = " <- CHAMPION" if _n == best_reg_name else ""
    print(f"  {_n:22}: RMSE ₹{_m['rmse']:>6,.0f}  R²={_m['r2']:.4f}  MAPE={_m['mape']:.1f}%{marker}")
print(f"\nBest regressor: {best_reg_name}  "
      f"(RMSE ₹{all_reg_results[best_reg_name]['metrics']['rmse']:,.0f})")

# ── Cell 16: Model Registry ────────────────────────────────────────────────────
client = MlflowClient()


def _ensure_registered_model(name: str, description: str) -> None:
    """Create registered model if absent; silently skip if it already exists."""
    try:
        client.create_registered_model(name=name, description=description)
        logger.info(f"Created registered model: {name}")
    except mlflow.exceptions.MlflowException as e:
        if "already exists" in str(e).lower() or "RESOURCE_ALREADY_EXISTS" in str(e):
            logger.info(f"Registered model already exists: {name}")
        else:
            raise


_ensure_registered_model(
    name=REGISTRY_NAME_CLF,
    description=(
        "EMI eligibility binary classifier. "
        f"Predicts P(emi_eligibility=1). Champion: {best_clf_name}. "
        "Thresholds: auto_approve>0.85 | human_review=0.40-0.85 | auto_reject<0.40."
    ),
)
clf_mv = client.create_model_version(
    name=REGISTRY_NAME_CLF,
    source=f"runs:/{best_clf_run_id}/model",
    run_id=best_clf_run_id,
    description=(
        f"Section 6 FAST_MODE={FAST_MODE}. "
        f"Model: {best_clf_name}. "
        f"AUC={all_clf_results[best_clf_name]['metrics']['roc_auc']:.6f}  "
        f"F1={all_clf_results[best_clf_name]['metrics']['f1']:.6f}."
    ),
    tags={
        "model_name":   best_clf_name,
        "section":      "06",
        "data_version": DATA_VERSION,
        "fast_mode":    str(FAST_MODE),
    },
)
client.set_registered_model_alias(
    name=REGISTRY_NAME_CLF,
    alias="champion",
    version=str(clf_mv.version),
)
print(f"\nRegistered: {REGISTRY_NAME_CLF} v{clf_mv.version}  alias=champion")

_ensure_registered_model(
    name=REGISTRY_NAME_REG,
    description=(
        "EMI amount regressor. "
        f"Predicts max_monthly_emi in INR (₹500–₹34,750). Champion: {best_reg_name}."
    ),
)
reg_mv = client.create_model_version(
    name=REGISTRY_NAME_REG,
    source=f"runs:/{best_reg_run_id}/model",
    run_id=best_reg_run_id,
    description=(
        f"Section 6 FAST_MODE={FAST_MODE}. "
        f"Model: {best_reg_name}. "
        f"RMSE=₹{all_reg_results[best_reg_name]['metrics']['rmse']:,.2f}  "
        f"R2={all_reg_results[best_reg_name]['metrics']['r2']:.6f}."
    ),
    tags={
        "model_name":   best_reg_name,
        "section":      "06",
        "data_version": DATA_VERSION,
        "fast_mode":    str(FAST_MODE),
    },
)
client.set_registered_model_alias(
    name=REGISTRY_NAME_REG,
    alias="champion",
    version=str(reg_mv.version),
)
print(f"Registered: {REGISTRY_NAME_REG} v{reg_mv.version}  alias=champion")

# ── Cell 17: Verification ──────────────────────────────────────────────────────
print("\n" + "=" * 65)
print(f"VERIFICATION — {EXPERIMENT_CLF}")
print("=" * 65)
clf_exp     = mlflow.get_experiment_by_name(EXPERIMENT_CLF)
clf_runs_df = mlflow.search_runs(
    experiment_ids=[clf_exp.experiment_id],
    order_by=["metrics.roc_auc DESC"],
)
for _, row in clf_runs_df.iterrows():
    run_model = row.get("tags.model_type", "?")
    marker    = " <- champion" if run_model == best_clf_name else ""
    print(
        f"  {str(row.get('tags.mlflow.runName', '?')):30} "
        f"AUC={row['metrics.roc_auc']:.4f}  "
        f"F1={row['metrics.f1']:.4f}{marker}"
    )

print("\n" + "=" * 65)
print(f"VERIFICATION — {EXPERIMENT_REG}")
print("=" * 65)
reg_exp     = mlflow.get_experiment_by_name(EXPERIMENT_REG)
reg_runs_df = mlflow.search_runs(
    experiment_ids=[reg_exp.experiment_id],
    order_by=["metrics.rmse ASC"],
)
for _, row in reg_runs_df.iterrows():
    run_model = row.get("tags.model_type", "?")
    marker    = " <- champion" if run_model == best_reg_name else ""
    print(
        f"  {str(row.get('tags.mlflow.runName', '?')):30} "
        f"RMSE=₹{row['metrics.rmse']:>7,.0f}  "
        f"R2={row['metrics.r2']:.4f}{marker}"
    )

print("\n" + "=" * 65)
print("MODEL REGISTRY")
print("=" * 65)
for reg_name in [REGISTRY_NAME_CLF, REGISTRY_NAME_REG]:
    champ_mv = client.get_model_version_by_alias(reg_name, "champion")
    print(f"\n  {reg_name}")
    print(f"    champion → version {champ_mv.version}")
    print(f"    run_id   : {champ_mv.run_id}")
    print(f"    model    : {champ_mv.tags.get('model_name', '?')}")

# ── Last Cell: Section 6 Complete ─────────────────────────────────────────────
_best_clf_m = all_clf_results[best_clf_name]["metrics"]
_best_reg_m = all_reg_results[best_reg_name]["metrics"]

print("\n" + "=" * 70)
print(f"   SECTION 6 COMPLETE  {'[FAST_MODE]' if FAST_MODE else '[FULL MODE]'}")
print("=" * 70)

print(f"""
EXPERIMENTS CREATED:
  {EXPERIMENT_CLF}
    5 runs: logistic_regression_v1, random_forest_v1, xgboost_v1,
            lightgbm_v1, extra_trees_v1
    Champion: {best_clf_name}
      AUC {_best_clf_m['roc_auc']:.4f}  F1 {_best_clf_m['f1']:.4f}
      Confidence zones: approve={_best_clf_m['confidence_zones']['auto_approve']:,}  \
review={_best_clf_m['confidence_zones']['human_review']:,}  \
reject={_best_clf_m['confidence_zones']['auto_reject']:,}

  {EXPERIMENT_REG}
    4 runs: random_forest_v1, xgboost_v1, lightgbm_v1, extra_trees_v1
    Champion: {best_reg_name}
      RMSE ₹{_best_reg_m['rmse']:,.0f}  MAE ₹{_best_reg_m['mae']:,.0f}  \
R²={_best_reg_m['r2']:.4f}  MAPE={_best_reg_m['mape']:.1f}%

MODEL REGISTRY:
  {REGISTRY_NAME_CLF}  v{clf_mv.version}  alias=champion
  {REGISTRY_NAME_REG}  v{reg_mv.version}  alias=champion

LOAD IN SECTION 9 (FastAPI):
  clf = mlflow.sklearn.load_model("models:/{REGISTRY_NAME_CLF}@champion")
  reg = mlflow.sklearn.load_model("models:/{REGISTRY_NAME_REG}@champion")

ARTIFACTS PER RUN:
  model/              — sklearn-flavored model + signature + input_example
  preprocessor/       — feature_engineer.pkl (Section 4 FeatureEngineer)
  confusion_matrix.png or residuals.png

BACKEND: {MLFLOW_TRACKING_URI}  (→ PostgreSQL in Section 11)

TO LAUNCH UI:
  mlflow ui --backend-store-uri {MLFLOW_TRACKING_URI} --port 5000
  then open: http://localhost:5000

DECISIONS MADE:
  - Re-trained all 9 models inside MLflow contexts (FAST_MODE={FAST_MODE})
  - Params: Optuna best_params subset only (no None-valued keys from get_params)
  - Aliases (not stages) — set_registered_model_alias("champion")
  - Figures logged via mlflow.log_figure() — no temp files on disk

WATCH FOR IN SECTION 7 (Airflow):
  - Set MLFLOW_TRACKING_URI env var so DAG workers connect to the same backend
  - PostgreSQL backend for MLflow configured in Section 11 docker-compose.yml
""")
