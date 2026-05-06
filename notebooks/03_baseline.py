# ================================================================
# EMI PREDICT AI — Baseline Model (Section 3)
# filename : notebooks/03_baseline.py
# purpose  : Compare rule-based and logistic regression baselines
# section  : 3
# version  : 1.0
# date     : 2026-05-02
# ================================================================

import sys
import os
import json
import logging
import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
    confusion_matrix, roc_curve, classification_report,
)

from src.models.baseline_rules    import RuleBasedClassifier
from src.models.baseline_logistic import LogisticRegressionBaseline

# ── Reproducibility ────────────────────────────────────────────
np.random.seed(42)

# ── Logging ────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s'
)
logger = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────
CLEAN_PATH  = os.path.join(PROJECT_ROOT, 'data', 'processed', 'emi_cleaned.csv')
FIGURES_DIR = os.path.join(PROJECT_ROOT, 'docs', 'figures')
REPORTS_DIR = os.path.join(PROJECT_ROOT, 'reports')
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

# ── Palettes (locked project standards) ───────────────────────
D2       = sns.color_palette('Dark2_r',  8)
PR       = sns.color_palette('Paired_r', 12)
AC       = sns.color_palette('Accent_r', 8)
NOT_CLR  = D2[6]   # orange-red  — Not Eligible
ELIG_CLR = D2[7]   # teal-green  — Eligible
RULES_CLR = PR[1]  # dark blue   — Rule-Based model
LR_CLR    = PR[3]  # dark green  — Logistic Regression model
DARK      = '#2D3436'

plt.rcParams.update({
    'figure.dpi'       : 130,
    'font.size'        : 10,
    'axes.titlesize'   : 12,
    'axes.titleweight' : 'bold',
    'axes.labelsize'   : 10,
    'axes.spines.top'  : False,
    'axes.spines.right': False,
})


def save_fig(name):
    path = os.path.join(FIGURES_DIR, name)
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved: {name}")


# ================================================================
# CELL 3 — LOAD DATA  (always from data/processed/)
# ================================================================
logger.info("Loading cleaned data...")
df = pd.read_csv(CLEAN_PATH)
logger.info(f"Loaded: {df.shape[0]:,} rows x {df.shape[1]} cols")

vc = df['emi_eligibility'].value_counts()
logger.info(
    f"Class distribution — "
    f"Not Eligible: {vc.get(0,0):,} ({vc.get(0,0)/len(df)*100:.1f}%)  "
    f"Eligible: {vc.get(1,0):,} ({vc.get(1,0)/len(df)*100:.1f}%)"
)


# ================================================================
# CELL 4 — COMPUTE expense_ratio  (before split — uses raw cols)
# ================================================================
EXPENSE_COLS = [
    'monthly_rent', 'school_fees', 'college_fees',
    'travel_expenses', 'groceries_utilities',
    'other_monthly_expenses', 'current_emi_amount',
]
df['total_expenses'] = df[EXPENSE_COLS].fillna(0).sum(axis=1)
df['expense_ratio']  = (
    df['total_expenses'] / df['monthly_salary'].replace(0, np.nan)
).clip(0, 2).fillna(1.0)
df.drop(columns=['total_expenses'], inplace=True)
logger.info("expense_ratio computed and added before split.")


# ================================================================
# CELL 5 — TRAIN / TEST SPLIT  (stratified 80/20, random_state=42)
# ================================================================
X = df.drop(columns=['emi_eligibility', 'max_monthly_emi'])
y = df['emi_eligibility'].values

X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    stratify=y,
    random_state=42,
)

logger.info(f"Train: {X_train_raw.shape[0]:,} rows  |  Test: {X_test_raw.shape[0]:,} rows")
train_pos = y_train.sum() / len(y_train) * 100
test_pos  = y_test.sum()  / len(y_test)  * 100
logger.info(f"Train eligible: {train_pos:.1f}%  |  Test eligible: {test_pos:.1f}%")


# ================================================================
# CELL 6 — ENCODE CATEGORICALS  (one-hot, drop_first avoids dummy trap)
# ================================================================
CAT_COLS = [
    'gender', 'marital_status', 'education', 'employment_type',
    'company_type', 'house_type', 'existing_loans', 'emi_scenario',
]

X_train_enc = pd.get_dummies(X_train_raw, columns=CAT_COLS, drop_first=True, dtype=int)
X_test_enc  = pd.get_dummies(X_test_raw,  columns=CAT_COLS, drop_first=True, dtype=int)
# Align test columns to train schema (handles any missing dummies in test set)
X_test_enc  = X_test_enc.reindex(columns=X_train_enc.columns, fill_value=0)

logger.info(f"Encoded feature count: {X_train_enc.shape[1]} columns")


# ================================================================
# CELL 7 — FIT BOTH MODELS
# ================================================================
logger.info("Fitting RuleBasedClassifier...")
rules_model = RuleBasedClassifier()
rules_model.fit(X_train_enc, y_train)

logger.info("Fitting LogisticRegressionBaseline...")
lr_model = LogisticRegressionBaseline(random_state=42, max_iter=1000)
lr_model.fit(X_train_enc, y_train)


# ================================================================
# CELL 8 — EVALUATE  (threshold=0.5 for both — fair comparison)
# ================================================================
def evaluate(model, X, y, name):
    y_proba = model.predict_proba(X)[:, 1]
    y_pred  = (y_proba >= 0.50).astype(int)
    return {
        'model'    : name,
        'roc_auc'  : round(roc_auc_score(y, y_proba), 4),
        'f1'       : round(f1_score(y, y_pred, zero_division=0), 4),
        'precision': round(precision_score(y, y_pred, zero_division=0), 4),
        'recall'   : round(recall_score(y, y_pred, zero_division=0), 4),
    }, y_pred, y_proba


rules_metrics, rules_pred, rules_proba = evaluate(
    rules_model, X_test_enc, y_test, 'Rule-Based'
)
lr_metrics, lr_pred, lr_proba = evaluate(
    lr_model, X_test_enc, y_test, 'Logistic Regression'
)

print("\n" + "=" * 60)
print("   BASELINE MODEL COMPARISON  (threshold = 0.50)")
print("=" * 60)
results = pd.DataFrame([rules_metrics, lr_metrics])
print(results.to_string(index=False))
print("=" * 60)

print("\nRule-Based — Classification Report:")
print(classification_report(y_test, rules_pred,
                             target_names=['Not Eligible', 'Eligible']))

print("Logistic Regression — Classification Report:")
print(classification_report(y_test, lr_pred,
                             target_names=['Not Eligible', 'Eligible']))


# ================================================================
# CELL 9 — CHART 1: CONFUSION MATRICES (Dark2_r palette, 2-panel)
# ================================================================
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Confusion Matrices — Baseline Models', y=0.98)

for ax, y_pred, name, color, m in [
    (axes[0], rules_pred, 'Rule-Based Classifier',  RULES_CLR, rules_metrics),
    (axes[1], lr_pred,    'Logistic Regression',     LR_CLR,    lr_metrics),
]:
    cm   = confusion_matrix(y_test, y_pred)
    norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    sns.heatmap(
        cm, annot=True, fmt='d', ax=ax,
        cmap=sns.light_palette(color, as_cmap=True),
        linewidths=1.5, linecolor='white',
        xticklabels=['Not Eligible', 'Eligible'],
        yticklabels=['Not Eligible', 'Eligible'],
        cbar=False,
    )
    # % annotations below raw counts
    for i in range(2):
        for j in range(2):
            ax.text(j + 0.5, i + 0.73,
                    f'({norm[i,j]*100:.1f}%)',
                    ha='center', va='center',
                    fontsize=9, color='dimgray')

    ax.set_title(
        f'{name}\nROC-AUC: {m["roc_auc"]:.4f}   F1: {m["f1"]:.4f}'
    )
    ax.set_xlabel('Predicted', labelpad=8)
    ax.set_ylabel('Actual',    labelpad=8)

plt.tight_layout(pad=2.0)
save_fig('03_confusion_matrices.png')


# ================================================================
# CELL 10 — CHART 2: ROC CURVES (overlaid, both models)
# ================================================================
fig, ax = plt.subplots(figsize=(10, 7))

for proba, name, color in [
    (rules_proba, f'Rule-Based  (AUC = {rules_metrics["roc_auc"]:.4f})', RULES_CLR),
    (lr_proba,    f'Logistic LR (AUC = {lr_metrics["roc_auc"]:.4f})',    LR_CLR),
]:
    fpr, tpr, _ = roc_curve(y_test, proba)
    ax.plot(fpr, tpr, linewidth=2.5, color=color, label=name)
    ax.fill_between(fpr, tpr, alpha=0.07, color=color)

ax.plot([0, 1], [0, 1], 'k--', linewidth=1.2, alpha=0.4,
        label='Random Classifier (AUC = 0.50)')
ax.set_xlabel('False Positive Rate  (1 - Specificity)')
ax.set_ylabel('True Positive Rate  (Recall / Sensitivity)')
ax.set_title('ROC Curves — Baseline Models\n'
             'Higher and further left = better', pad=12)
ax.legend(loc='lower right', fontsize=10)
ax.set_xlim([0, 1])
ax.set_ylim([0, 1.02])

plt.tight_layout()
save_fig('03_roc_curves.png')


# ================================================================
# CELL 11 — CHART 3: LR TOP-15 FEATURE COEFFICIENTS
# ================================================================
top_feats = lr_model.get_top_features(n=15).sort_values('coefficient')
bar_colors = [ELIG_CLR if c >= 0 else NOT_CLR for c in top_feats['coefficient']]

fig, ax = plt.subplots(figsize=(12, 7))
bars = ax.barh(top_feats['feature'], top_feats['coefficient'],
               color=bar_colors, edgecolor='white', alpha=0.92)
ax.axvline(0, color=DARK, linewidth=0.9, alpha=0.45)
ax.set_xlabel('Coefficient Value  (features are StandardScaled)')
ax.set_title(
    'Logistic Regression — Top 15 Feature Coefficients\n'
    'Teal = pushes toward Eligible  |  Orange = pushes toward Not Eligible',
    pad=12
)

for bar, val in zip(bars, top_feats['coefficient']):
    xp  = bar.get_width()
    ha  = 'left' if xp >= 0 else 'right'
    off = 0.012 if xp >= 0 else -0.012
    ax.text(xp + off, bar.get_y() + bar.get_height() / 2,
            f'{val:+.3f}', va='center', ha=ha,
            fontsize=8.5, color=DARK)

plt.tight_layout()
save_fig('03_lr_coefficients.png')


# ================================================================
# CELL 12 — CHART 4: CONFIDENCE ZONE DISTRIBUTION
# ================================================================
rules_zones = rules_model.confidence_zone_summary(X_test_enc)
lr_zones    = lr_model.confidence_zone_summary(X_test_enc)

zone_labels = ['Auto-Approve\n(prob > 0.85)', 'Human Review\n(0.40 – 0.85)', 'Auto-Reject\n(prob < 0.40)']
zone_keys   = ['auto_approve_pct', 'human_review_pct', 'auto_reject_pct']
zone_colors = [ELIG_CLR, AC[3], NOT_CLR]

x = np.arange(len(zone_labels))
w = 0.35

fig, ax = plt.subplots(figsize=(12, 6))

for i, (key, zcolor) in enumerate(zip(zone_keys, zone_colors)):
    ax.bar(x[i] - w/2, rules_zones[key], w,
           color=RULES_CLR, edgecolor='white', alpha=0.90,
           label='Rule-Based' if i == 0 else '_')
    ax.bar(x[i] + w/2, lr_zones[key], w,
           color=LR_CLR, edgecolor='white', alpha=0.90,
           label='Logistic Regression' if i == 0 else '_')

    ax.text(x[i] - w/2, rules_zones[key] + 0.6,
            f"{rules_zones[key]}%",
            ha='center', fontsize=9, fontweight='bold', color=DARK)
    ax.text(x[i] + w/2, lr_zones[key] + 0.6,
            f"{lr_zones[key]}%",
            ha='center', fontsize=9, fontweight='bold', color=DARK)

ax.set_xticks(x)
ax.set_xticklabels(zone_labels)
ax.set_ylabel('% of Test Applicants')
ax.set_title(
    'Confidence Zone Distribution — Baseline Models\n'
    'Based on business thresholds: >0.85 auto-approve | 0.40–0.85 review | <0.40 auto-reject',
    pad=12
)
ax.legend(fontsize=10)
ax.set_ylim(0, max(
    max(rules_zones[k] for k in zone_keys),
    max(lr_zones[k]    for k in zone_keys),
) * 1.18)

plt.tight_layout()
save_fig('03_confidence_zones.png')


# ================================================================
# CELL 13 — SAVE METRICS JSON  (used by MLflow in Section 6)
# ================================================================
metrics_out = {
    'section'     : 3,
    'description' : 'Baseline models — rule-based and logistic regression',
    'models'      : [rules_metrics, lr_metrics],
    'confidence_zones': {
        'rule_based'          : rules_zones,
        'logistic_regression' : lr_zones,
    },
    'data_split'  : {
        'train_size'  : int(len(y_train)),
        'test_size'   : int(len(y_test)),
        'strategy'    : 'stratified 80/20',
        'random_state': 42,
    },
    'decisions': {
        'class_weight'       : 'balanced',
        'eval_threshold'     : 0.50,
        'encoding'           : 'one-hot drop_first',
        'rule_credit_floor'  : 650,
        'rule_expense_ceil'  : 0.60,
        'rule_salary_floor'  : 20000,
    }
}

out_path = os.path.join(REPORTS_DIR, 'baseline_metrics.json')
with open(out_path, 'w') as f:
    json.dump(metrics_out, f, indent=2)
logger.info(f"Metrics saved: {out_path}")


# ================================================================
# LAST CELL — SUMMARY
# ================================================================
print(f"""
{'=' * 65}
   SECTION 3 COMPLETE — BASELINE MODEL SUMMARY
{'=' * 65}

RESULTS (test set, threshold=0.50):
  Rule-Based          ROC-AUC: {rules_metrics['roc_auc']}   F1: {rules_metrics['f1']}
  Logistic Regression ROC-AUC: {lr_metrics['roc_auc']}   F1: {lr_metrics['f1']}

CONFIDENCE ZONES (Rule-Based on test set):
  Auto-Approve  : {rules_zones['auto_approve_pct']}%
  Human Review  : {rules_zones['human_review_pct']}%
  Auto-Reject   : {rules_zones['auto_reject_pct']}%

CONFIDENCE ZONES (Logistic Regression on test set):
  Auto-Approve  : {lr_zones['auto_approve_pct']}%
  Human Review  : {lr_zones['human_review_pct']}%
  Auto-Reject   : {lr_zones['auto_reject_pct']}%

DECISIONS MADE:
  - Rule thresholds: credit >= 650 | expense_ratio <= 0.60 | salary >= 20k
  - class_weight='balanced' on LR (locked — 4.2:1 imbalance)
  - Encoding: one-hot drop_first (avoids dummy variable trap)
  - Both models use same threshold (0.50) for fair F1 comparison
  - Metrics saved to reports/baseline_metrics.json for Section 6 MLflow

WHAT TO WATCH IN SECTION 4 (Feature Engineering):
  - expense_ratio already strong — explore variants (salary utilisation ratio)
  - credit_score is top predictor — consider binning + interaction terms
  - LR coefficients highlight which features need log-transform (skewed)
  - Salary brackets showed non-linear approval — tree-friendly binned features

SAVED:
  docs/figures/03_confusion_matrices.png
  docs/figures/03_roc_curves.png
  docs/figures/03_lr_coefficients.png
  docs/figures/03_confidence_zones.png
  reports/baseline_metrics.json
{'=' * 65}
""")
