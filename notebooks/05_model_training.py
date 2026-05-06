# filename: notebooks/05_model_training.py
# purpose:  Section 5 — Train 5 classifiers and 4 regressors with Optuna tuning
# version:  2.0

# ── Cell 1: Setup ─────────────────────────────────────────────────────────────
import sys
import json
import time
import logging
import warnings
warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split

np.random.seed(42)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

ROOT        = Path(__file__).resolve().parent.parent
DATA_DIR    = ROOT / "data" / "processed"
MODELS_DIR  = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = ROOT / "docs" / "figures"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT))
from src.models.train_classifier import (
    train_logistic_regression, train_random_forest,
    train_xgboost, train_lightgbm, train_extra_trees,
)
from src.models.train_regressor import (
    train_rf_regressor, train_xgb_regressor,
    train_lgbm_regressor, train_et_regressor,
)

# Palettes (locked)
PR   = sns.color_palette("Paired_r",  12)
D2   = sns.color_palette("Dark2_r",    8)
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

BASELINE_AUC = 0.9763  # Section 3 LR on raw features


def save_fig(name: str) -> None:
    path = FIGURES_DIR / name
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved figure: {name}")


# ── FAST_MODE config ───────────────────────────────────────────────────────────
# Set FAST_MODE = False to restore full production training (25 trials, full data)
FAST_MODE     = True
OPTUNA_TRIALS = 3       if FAST_MODE else 25
OPTUNA_CV     = 3       # folds used inside Optuna objective (FAST_MODE only)
TUNE_SAMPLE   = 50_000  if FAST_MODE else None

# ── Cell 2: Load Data ──────────────────────────────────────────────────────────
logger.info("Loading feature data...")
train_df = pd.read_csv(DATA_DIR / "train_features.csv")
test_df  = pd.read_csv(DATA_DIR / "test_features.csv")

TARGET_CLF = "emi_eligibility"
TARGET_REG = "max_monthly_emi"

print("=" * 60)
print(f"SECTION 5: MODEL TRAINING  {'[FAST_MODE]' if FAST_MODE else '[FULL MODE]'}")
print("=" * 60)
print(f"\nTrain : {train_df.shape[0]:,} rows × {train_df.shape[1]} columns")
print(f"Test  : {test_df.shape[0]:,} rows × {test_df.shape[1]} columns")
print(f"Class balance (train): {train_df[TARGET_CLF].value_counts().to_dict()}")

# ── Cell 3: Encode Categoricals & Prepare Feature Matrices ────────────────────
CATEGORICAL_COLS = [
    "gender", "marital_status", "education",
    "employment_type", "company_type", "house_type",
    "existing_loans", "emi_scenario",
    "credit_score_band",   # engineered categorical (Poor/Fair/Good/Excellent)
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

X_test      = test_enc.drop(columns=drop_cols)
y_test_clf  = test_enc[TARGET_CLF]
y_test_reg  = test_enc[TARGET_REG]

bool_cols = X_train.select_dtypes("bool").columns.tolist()
if bool_cols:
    X_train[bool_cols] = X_train[bool_cols].astype(int)
    X_test[bool_cols]  = X_test[bool_cols].astype(int)

X_train = X_train.replace([np.inf, -np.inf], np.nan).fillna(0)
X_test  = X_test.replace([np.inf, -np.inf], np.nan).fillna(0)

logger.info(f"Feature matrix: {X_train.shape[1]} columns after OHE")

# ── FAST_MODE: create stratified 50k tuning sample ────────────────────────────
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

metrics_path   = REPORTS_DIR / "model_metrics.json"
all_clf_metrics: dict = {}

# ── Cell 4: Model 1 / 5 — Logistic Regression ─────────────────────────────────
print(f"\n{'─'*60}")
print(f"Model 1 / 5: Logistic Regression  ({OPTUNA_TRIALS} trials)")
print(f"{'─'*60}")
t0 = time.time()

lr_result = train_logistic_regression(
    X_train, y_train_clf, X_test, y_test_clf,
    n_trials=OPTUNA_TRIALS,
    X_tune=X_tune, y_tune=y_tune_clf,
    cv=OPTUNA_CV if FAST_MODE else None,
)
all_clf_metrics["logistic_regression"] = lr_result["metrics"]

m     = lr_result["metrics"]
delta = m["roc_auc"] - BASELINE_AUC
sign  = "+" if delta >= 0 else ""
print(f"  ROC-AUC  : {m['roc_auc']:.4f}  ({sign}{delta:.4f} vs Section 3 baseline)")
print(f"  F1       : {m['f1']:.4f}")
print(f"  Precision: {m['precision']:.4f}")
print(f"  Recall   : {m['recall']:.4f}")
print(f"  Confidence zones: {m['confidence_zones']}")
print(f"  Elapsed  : {time.time()-t0:.1f}s")

with open(metrics_path, "w") as f:
    json.dump({"classifiers": all_clf_metrics}, f, indent=2)

# ── Cell 5: Model 2 / 5 — Random Forest ───────────────────────────────────────
print(f"\n{'─'*60}")
print(f"Model 2 / 5: Random Forest  ({OPTUNA_TRIALS} trials)")
print(f"{'─'*60}")
t0 = time.time()

rf_result = train_random_forest(
    X_train, y_train_clf, X_test, y_test_clf,
    n_trials=OPTUNA_TRIALS,
    X_tune=X_tune, y_tune=y_tune_clf,
    cv=OPTUNA_CV if FAST_MODE else None,
)
all_clf_metrics["random_forest"] = rf_result["metrics"]

m_rf     = rf_result["metrics"]
delta_rf = m_rf["roc_auc"] - BASELINE_AUC
sign_rf  = "+" if delta_rf >= 0 else ""
print(f"  ROC-AUC  : {m_rf['roc_auc']:.4f}  ({sign_rf}{delta_rf:.4f} vs baseline)")
print(f"  F1       : {m_rf['f1']:.4f}")
print(f"  Precision: {m_rf['precision']:.4f}")
print(f"  Recall   : {m_rf['recall']:.4f}")
print(f"  Confidence zones: {m_rf['confidence_zones']}")
print(f"  Elapsed  : {time.time()-t0:.1f}s")

with open(metrics_path, "w") as f:
    json.dump({"classifiers": all_clf_metrics}, f, indent=2)

# ── Cell 6: Model 3 / 5 — XGBoost ─────────────────────────────────────────────
print(f"\n{'─'*60}")
print(f"Model 3 / 5: XGBoost  ({OPTUNA_TRIALS} trials)")
print(f"{'─'*60}")
t0 = time.time()

xgb_result = train_xgboost(
    X_train, y_train_clf, X_test, y_test_clf,
    n_trials=OPTUNA_TRIALS,
    X_tune=X_tune, y_tune=y_tune_clf,
    cv=OPTUNA_CV if FAST_MODE else None,
)
all_clf_metrics["xgboost"] = xgb_result["metrics"]

m_xgb     = xgb_result["metrics"]
delta_xgb = m_xgb["roc_auc"] - BASELINE_AUC
sign_xgb  = "+" if delta_xgb >= 0 else ""
print(f"  ROC-AUC  : {m_xgb['roc_auc']:.4f}  ({sign_xgb}{delta_xgb:.4f} vs baseline)")
print(f"  F1       : {m_xgb['f1']:.4f}")
print(f"  Precision: {m_xgb['precision']:.4f}")
print(f"  Recall   : {m_xgb['recall']:.4f}")
print(f"  Confidence zones: {m_xgb['confidence_zones']}")
print(f"  Elapsed  : {time.time()-t0:.1f}s")

with open(metrics_path, "w") as f:
    json.dump({"classifiers": all_clf_metrics}, f, indent=2)

# ── Cell 7: Model 4 / 5 — LightGBM ───────────────────────────────────────────
print(f"\n{'─'*60}")
print(f"Model 4 / 5: LightGBM  ({OPTUNA_TRIALS} trials)")
print(f"{'─'*60}")
t0 = time.time()

lgbm_result = train_lightgbm(
    X_train, y_train_clf, X_test, y_test_clf,
    n_trials=OPTUNA_TRIALS,
    X_tune=X_tune, y_tune=y_tune_clf,
    cv=OPTUNA_CV if FAST_MODE else None,
)
all_clf_metrics["lightgbm"] = lgbm_result["metrics"]

m_lgbm     = lgbm_result["metrics"]
delta_lgbm = m_lgbm["roc_auc"] - BASELINE_AUC
sign_lgbm  = "+" if delta_lgbm >= 0 else ""
print(f"  ROC-AUC  : {m_lgbm['roc_auc']:.4f}  ({sign_lgbm}{delta_lgbm:.4f} vs baseline)")
print(f"  F1       : {m_lgbm['f1']:.4f}")
print(f"  Precision: {m_lgbm['precision']:.4f}")
print(f"  Recall   : {m_lgbm['recall']:.4f}")
print(f"  Confidence zones: {m_lgbm['confidence_zones']}")
print(f"  Elapsed  : {time.time()-t0:.1f}s")

with open(metrics_path, "w") as f:
    json.dump({"classifiers": all_clf_metrics}, f, indent=2)

# ── Cell 8: Model 5 / 5 — Extra Trees ─────────────────────────────────────────
print(f"\n{'─'*60}")
print(f"Model 5 / 5: Extra Trees  ({OPTUNA_TRIALS} trials)")
print(f"{'─'*60}")
t0 = time.time()

et_result = train_extra_trees(
    X_train, y_train_clf, X_test, y_test_clf,
    n_trials=OPTUNA_TRIALS,
    X_tune=X_tune, y_tune=y_tune_clf,
    cv=OPTUNA_CV if FAST_MODE else None,
)
all_clf_metrics["extra_trees"] = et_result["metrics"]

m_et     = et_result["metrics"]
delta_et = m_et["roc_auc"] - BASELINE_AUC
sign_et  = "+" if delta_et >= 0 else ""
print(f"  ROC-AUC  : {m_et['roc_auc']:.4f}  ({sign_et}{delta_et:.4f} vs baseline)")
print(f"  F1       : {m_et['f1']:.4f}")
print(f"  Precision: {m_et['precision']:.4f}")
print(f"  Recall   : {m_et['recall']:.4f}")
print(f"  Confidence zones: {m_et['confidence_zones']}")
print(f"  Elapsed  : {time.time()-t0:.1f}s")

# Save best classifier
all_clf_results = {
    "logistic_regression": lr_result,
    "random_forest":       rf_result,
    "xgboost":             xgb_result,
    "lightgbm":            lgbm_result,
    "extra_trees":         et_result,
}
# Selection: primary = ROC-AUC, tie-break = F1, final tie-break = production speed preference
_CLF_SPEED_PREF = ["lightgbm", "xgboost", "random_forest", "extra_trees", "logistic_regression"]
best_clf_name = max(
    all_clf_results,
    key=lambda n: (
        all_clf_results[n]["metrics"]["roc_auc"],
        all_clf_results[n]["metrics"]["f1"],
        -_CLF_SPEED_PREF.index(n) if n in _CLF_SPEED_PREF else -99,
    ),
)
joblib.dump(
    {
        "name":          best_clf_name,
        "model":         all_clf_results[best_clf_name]["model"],
        "feature_names": all_clf_results[best_clf_name]["feature_names"],
    },
    MODELS_DIR / "best_classifier.pkl",
)
with open(metrics_path, "w") as f:
    json.dump({"classifiers": all_clf_metrics}, f, indent=2)
print(f"\nBest classifier: {best_clf_name}  "
      f"(AUC {all_clf_metrics[best_clf_name]['roc_auc']:.4f})")
print(f"Saved → models/best_classifier.pkl")

# ── Classifier Midpoint Summary ───────────────────────────────────────────────
print("\n" + "=" * 60)
print("ALL 5 CLASSIFIERS COMPLETE")
print("=" * 60)
for _name, _m in sorted(all_clf_metrics.items(), key=lambda x: -x[1]["roc_auc"]):
    marker = " <- BEST" if _name == best_clf_name else ""
    print(f"  {_name:22}: AUC {_m['roc_auc']:.4f}  F1 {_m['f1']:.4f}{marker}")
print(f"\n  Baseline (raw features): AUC {BASELINE_AUC}")
_beat = all_clf_metrics[best_clf_name]["roc_auc"] > BASELINE_AUC
print(f"  {'BEAT BASELINE' if _beat else 'Did not beat baseline — investigate'}")

all_reg_metrics: dict = {}

# ── Cell 9: Regressor 1 / 4 — Random Forest ───────────────────────────────────
print(f"\n{'─'*60}")
print(f"Regressor 1 / 4: Random Forest  ({OPTUNA_TRIALS} trials)")
print(f"Target: R² > 0.75  |  MAPE < 20%")
print(f"{'─'*60}")
t0 = time.time()

rf_reg_result = train_rf_regressor(
    X_train, y_train_reg, X_test, y_test_reg,
    n_trials=OPTUNA_TRIALS,
    X_tune=X_tune, y_tune=y_tune_reg,
    cv=OPTUNA_CV if FAST_MODE else None,
)
all_reg_metrics["random_forest"] = rf_reg_result["metrics"]

m_rfr = rf_reg_result["metrics"]
print(f"  RMSE : ₹{m_rfr['rmse']:,.0f}")
print(f"  MAE  : ₹{m_rfr['mae']:,.0f}")
print(f"  R²   : {m_rfr['r2']:.4f}  ({'PASS' if m_rfr['r2'] > 0.75 else 'FAIL'} target > 0.75)")
print(f"  MAPE : {m_rfr['mape']:.1f}%  ({'PASS' if m_rfr['mape'] < 20.0 else 'FAIL'} target < 20%)")
print(f"  Elapsed: {time.time()-t0:.1f}s")

with open(metrics_path, "w") as f:
    json.dump({"classifiers": all_clf_metrics, "regressors": all_reg_metrics}, f, indent=2)

# ── Cell 10: Regressor 2 / 4 — XGBoost ────────────────────────────────────────
print(f"\n{'─'*60}")
print(f"Regressor 2 / 4: XGBoost  ({OPTUNA_TRIALS} trials)")
print(f"{'─'*60}")
t0 = time.time()

xgb_reg_result = train_xgb_regressor(
    X_train, y_train_reg, X_test, y_test_reg,
    n_trials=OPTUNA_TRIALS,
    X_tune=X_tune, y_tune=y_tune_reg,
    cv=OPTUNA_CV if FAST_MODE else None,
)
all_reg_metrics["xgboost"] = xgb_reg_result["metrics"]

m_xgbr = xgb_reg_result["metrics"]
print(f"  RMSE : ₹{m_xgbr['rmse']:,.0f}")
print(f"  MAE  : ₹{m_xgbr['mae']:,.0f}")
print(f"  R²   : {m_xgbr['r2']:.4f}  ({'PASS' if m_xgbr['r2'] > 0.75 else 'FAIL'} target > 0.75)")
print(f"  MAPE : {m_xgbr['mape']:.1f}%  ({'PASS' if m_xgbr['mape'] < 20.0 else 'FAIL'} target < 20%)")
print(f"  Elapsed: {time.time()-t0:.1f}s")

with open(metrics_path, "w") as f:
    json.dump({"classifiers": all_clf_metrics, "regressors": all_reg_metrics}, f, indent=2)

# ── Cell 11: Regressor 3 / 4 — LightGBM ──────────────────────────────────────
print(f"\n{'─'*60}")
print(f"Regressor 3 / 4: LightGBM  ({OPTUNA_TRIALS} trials)")
print(f"{'─'*60}")
t0 = time.time()

lgbm_reg_result = train_lgbm_regressor(
    X_train, y_train_reg, X_test, y_test_reg,
    n_trials=OPTUNA_TRIALS,
    X_tune=X_tune, y_tune=y_tune_reg,
    cv=OPTUNA_CV if FAST_MODE else None,
)
all_reg_metrics["lightgbm"] = lgbm_reg_result["metrics"]

m_lgbmr = lgbm_reg_result["metrics"]
print(f"  RMSE : ₹{m_lgbmr['rmse']:,.0f}")
print(f"  MAE  : ₹{m_lgbmr['mae']:,.0f}")
print(f"  R²   : {m_lgbmr['r2']:.4f}  ({'PASS' if m_lgbmr['r2'] > 0.75 else 'FAIL'} target > 0.75)")
print(f"  MAPE : {m_lgbmr['mape']:.1f}%  ({'PASS' if m_lgbmr['mape'] < 20.0 else 'FAIL'} target < 20%)")
print(f"  Elapsed: {time.time()-t0:.1f}s")

with open(metrics_path, "w") as f:
    json.dump({"classifiers": all_clf_metrics, "regressors": all_reg_metrics}, f, indent=2)

# ── Cell 12: Regressor 4 / 4 — Extra Trees ────────────────────────────────────
print(f"\n{'─'*60}")
print(f"Regressor 4 / 4: Extra Trees  ({OPTUNA_TRIALS} trials)")
print(f"{'─'*60}")
t0 = time.time()

et_reg_result = train_et_regressor(
    X_train, y_train_reg, X_test, y_test_reg,
    n_trials=OPTUNA_TRIALS,
    X_tune=X_tune, y_tune=y_tune_reg,
    cv=OPTUNA_CV if FAST_MODE else None,
)
all_reg_metrics["extra_trees"] = et_reg_result["metrics"]

m_etr = et_reg_result["metrics"]
print(f"  RMSE : ₹{m_etr['rmse']:,.0f}")
print(f"  MAE  : ₹{m_etr['mae']:,.0f}")
print(f"  R²   : {m_etr['r2']:.4f}  ({'PASS' if m_etr['r2'] > 0.75 else 'FAIL'} target > 0.75)")
print(f"  MAPE : {m_etr['mape']:.1f}%  ({'PASS' if m_etr['mape'] < 20.0 else 'FAIL'} target < 20%)")
print(f"  Elapsed: {time.time()-t0:.1f}s")

# Save best regressor
all_reg_results = {
    "random_forest": rf_reg_result,
    "xgboost":       xgb_reg_result,
    "lightgbm":      lgbm_reg_result,
    "extra_trees":   et_reg_result,
}
best_reg_name = min(all_reg_results, key=lambda n: all_reg_results[n]["metrics"]["rmse"])
joblib.dump(
    {
        "name":          best_reg_name,
        "model":         all_reg_results[best_reg_name]["model"],
        "feature_names": all_reg_results[best_reg_name]["feature_names"],
    },
    MODELS_DIR / "best_regressor.pkl",
)
with open(metrics_path, "w") as f:
    json.dump({"classifiers": all_clf_metrics, "regressors": all_reg_metrics}, f, indent=2)
print(f"\nBest regressor: {best_reg_name}  "
      f"(RMSE ₹{all_reg_metrics[best_reg_name]['rmse']:,.0f})")
print(f"Saved → models/best_regressor.pkl  |  reports/model_metrics.json")

# ── Cell 13: Chart 1 — Classifier Comparison ──────────────────────────────────
clf_names  = list(all_clf_metrics.keys())
clf_aucs   = [all_clf_metrics[n]["roc_auc"] for n in clf_names]
clf_f1s    = [all_clf_metrics[n]["f1"]      for n in clf_names]
clf_colors = [PR[i * 2 + 1] for i in range(len(clf_names))]
x_clf = np.arange(len(clf_names))
w     = 0.35

fig, ax = plt.subplots(figsize=(14, 6))
bars_auc = ax.bar(x_clf - w/2, clf_aucs, w, color=clf_colors,
                  edgecolor="white", alpha=0.92, label="ROC-AUC")
bars_f1  = ax.bar(x_clf + w/2, clf_f1s,  w, color=clf_colors,
                  edgecolor="white", alpha=0.55, label="F1-Score", hatch="//")
ax.axhline(BASELINE_AUC, color=D2[0], linestyle="--", linewidth=2,
           label=f"Baseline AUC {BASELINE_AUC} (Section 3 LR, raw features)")
for bar, val in zip(bars_auc, clf_aucs):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
            f"{val:.4f}", ha="center", fontsize=8, fontweight="bold", color=DARK)
for bar, val in zip(bars_f1, clf_f1s):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
            f"{val:.4f}", ha="center", fontsize=8, color=DARK)
ax.set_xticks(x_clf)
ax.set_xticklabels([n.replace("_", "\n") for n in clf_names], fontsize=9)
ax.set_ylabel("Score")
ax.set_ylim(0.80, 1.02)
ax.set_title("Section 5 — Classifier Comparison: ROC-AUC & F1\n"
             "All models vs Section 3 baseline (dashed)", pad=12)
ax.legend(fontsize=9)
plt.tight_layout()
save_fig("05_classifier_comparison.png")

# ── Cell 14: Chart 2 — Regressor Comparison ───────────────────────────────────
reg_names  = list(all_reg_metrics.keys())
reg_rmses  = [all_reg_metrics[n]["rmse"] for n in reg_names]
reg_maes   = [all_reg_metrics[n]["mae"]  for n in reg_names]
reg_colors = [PR[i * 2] for i in range(len(reg_names))]
x_reg = np.arange(len(reg_names))

fig, ax = plt.subplots(figsize=(12, 6))
bars_rmse = ax.bar(x_reg - w/2, reg_rmses, w, color=reg_colors,
                   edgecolor="white", alpha=0.92, label="RMSE (₹)")
bars_mae  = ax.bar(x_reg + w/2, reg_maes,  w, color=reg_colors,
                   edgecolor="white", alpha=0.55, label="MAE (₹)", hatch="//")
for bar, val in zip(bars_rmse, reg_rmses):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 10,
            f"₹{val:,.0f}", ha="center", fontsize=8, fontweight="bold", color=DARK)
for bar, val in zip(bars_mae, reg_maes):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 10,
            f"₹{val:,.0f}", ha="center", fontsize=8, color=DARK)
ax.set_xticks(x_reg)
ax.set_xticklabels([n.replace("_", "\n") for n in reg_names], fontsize=9)
ax.set_ylabel("Error (₹)")
ax.set_title("Section 5 — Regressor Comparison: RMSE & MAE\n"
             "Predicting max_monthly_emi (₹500–₹34,750)", pad=12)
ax.legend(fontsize=9)
plt.tight_layout()
save_fig("05_regressor_comparison.png")

# ── Cell 15: Chart 3 — Feature Importance (best classifier) ───────────────────
best_clf_model = all_clf_results[best_clf_name]["model"]
feat_names     = all_clf_results[best_clf_name]["feature_names"]

if hasattr(best_clf_model, "feature_importances_"):
    importances = best_clf_model.feature_importances_
else:  # LR Pipeline — coef_ inside named_steps
    importances = np.abs(best_clf_model.named_steps["clf"].coef_[0])

imp_df = (
    pd.DataFrame({"feature": feat_names, "importance": importances})
    .sort_values("importance", ascending=False)
    .head(20)
    .sort_values("importance")
)
bar_colors = [D2[1] if i >= 10 else D2[6] for i in range(len(imp_df))]

fig, ax = plt.subplots(figsize=(12, 8))
ax.barh(imp_df["feature"], imp_df["importance"],
        color=bar_colors[::-1], edgecolor="white", alpha=0.92)
ax.set_xlabel("Feature Importance")
ax.set_title(
    f"Top 20 Feature Importances — {best_clf_name.replace('_', ' ').title()}\n"
    f"ROC-AUC {all_clf_metrics[best_clf_name]['roc_auc']:.4f}",
    pad=12,
)
plt.tight_layout()
save_fig("05_feature_importance.png")

# ── Last Cell: Section 5 Complete ─────────────────────────────────────────────
print("\n" + "=" * 65)
print(f"   SECTION 5 COMPLETE  {'[FAST_MODE — 3 trials × 3-fold CV × 50k sample]' if FAST_MODE else '[FULL MODE]'}")
print("=" * 65)

print("\nCLASSIFIERS (target: beat AUC 0.9763):")
for _n, _m in sorted(all_clf_metrics.items(), key=lambda x: -x[1]["roc_auc"]):
    marker = " <- BEST" if _n == best_clf_name else ""
    print(f"  {_n:22}: AUC {_m['roc_auc']:.4f}  F1 {_m['f1']:.4f}{marker}")
_beat = all_clf_metrics[best_clf_name]["roc_auc"] > BASELINE_AUC
print(f"\n  Baseline: {BASELINE_AUC}  |  {'BEAT BASELINE' if _beat else 'DID NOT BEAT BASELINE'}")

print("\nREGRESSORS (targets: R² > 0.75, MAPE < 20%):")
for _n, _m in sorted(all_reg_metrics.items(), key=lambda x: x[1]["rmse"]):
    marker  = " <- BEST" if _n == best_reg_name else ""
    r2_lbl  = "PASS" if _m["r2"]   > 0.75 else "FAIL"
    mpe_lbl = "PASS" if _m["mape"] < 20.0 else "FAIL"
    print(f"  {_n:22}: RMSE ₹{_m['rmse']:>6,.0f}  "
          f"R²={_m['r2']:.4f}({r2_lbl})  MAPE={_m['mape']:.1f}%({mpe_lbl}){marker}")

print(f"""
SAVED ARTIFACTS:
  models/best_classifier.pkl  ({best_clf_name})
  models/best_regressor.pkl   ({best_reg_name})
  reports/model_metrics.json
  docs/figures/05_classifier_comparison.png
  docs/figures/05_regressor_comparison.png
  docs/figures/05_feature_importance.png

NEXT → Section 6: MLflow Experiment Tracking
{"=" * 65}""")
