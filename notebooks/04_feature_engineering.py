# ================================================================
# EMI PREDICT AI — Feature Engineering (Section 4)
# filename : notebooks/04_feature_engineering.py
# purpose  : Build, validate, and save engineered features
# section  : 4
# version  : 1.0
# date     : 2026-05-02
# ================================================================

import sys
import os
import json
import logging
import joblib
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

from src.features.feature_engineering import FeatureEngineer, NEW_FEATURES

# ── Reproducibility ────────────────────────────────────────────
np.random.seed(42)

# ── Logging ────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s'
)
logger = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────
CLEAN_PATH        = os.path.join(PROJECT_ROOT, 'data', 'processed', 'emi_cleaned.csv')
TRAIN_PATH        = os.path.join(PROJECT_ROOT, 'data', 'processed', 'train_features.csv')
TEST_PATH         = os.path.join(PROJECT_ROOT, 'data', 'processed', 'test_features.csv')
FEATURE_COLS_PATH = os.path.join(PROJECT_ROOT, 'data', 'processed', 'feature_columns.json')
MODELS_DIR        = os.path.join(PROJECT_ROOT, 'models')
ENGINEER_PATH     = os.path.join(MODELS_DIR, 'feature_engineer.pkl')
FIGURES_DIR       = os.path.join(PROJECT_ROOT, 'docs', 'figures')
os.makedirs(MODELS_DIR,  exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# ── Palettes (locked project standards) ───────────────────────
D2       = sns.color_palette('Dark2_r',  8)
PR       = sns.color_palette('Paired_r', 12)
AC       = sns.color_palette('Accent_r', 8)
NOT_CLR  = D2[6]
ELIG_CLR = D2[7]
DARK     = '#2D3436'

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
# CELL 3 — LOAD DATA
# ================================================================
logger.info("Loading cleaned data...")
df = pd.read_csv(CLEAN_PATH)
logger.info(f"Loaded: {df.shape[0]:,} rows x {df.shape[1]} cols")


# ================================================================
# CELL 4 — EXPLORE: fit_transform on full dataset (EDA only)
# This step is for correlation analysis only.
# Actual training will use train-only fit to prevent leakage.
# ================================================================
fe_explore = FeatureEngineer()
df_feat    = fe_explore.fit_transform(df)

n_original  = df.shape[1] - 2          # minus 2 targets
n_engineered = df_feat.shape[1] - 2
n_new        = len(NEW_FEATURES)

print(f"\n{'=' * 55}")
print(f"  FEATURE ENGINEERING SUMMARY")
print(f"{'=' * 55}")
print(f"  Original features  : {n_original}")
print(f"  New features added : {n_new}")
print(f"  Total features     : {n_engineered}")
print(f"{'=' * 55}\n")
print("  New features by group:")
groups = {
    'Financial Ratios (8)': NEW_FEATURES[:8],
    'Credit Banding  (2)': NEW_FEATURES[8:10],
    'Loan Capacity   (2)': NEW_FEATURES[10:12],
    'Interactions    (2)': NEW_FEATURES[12:14],
    'Binary Flags    (4)': NEW_FEATURES[14:18],
    'Log-transforms  (3)': NEW_FEATURES[18:],
}
for group, feats in groups.items():
    print(f"  {group}: {', '.join(feats)}")
print()


# ================================================================
# CELL 5 — CORRELATION ANALYSIS
# ================================================================
# Numerical columns only (exclude targets, string categoricals)
NUM_COLS = [
    c for c in df_feat.select_dtypes(include=[np.number]).columns
    if c not in ['emi_eligibility', 'max_monthly_emi']
]
new_set = set(NEW_FEATURES) - {'credit_score_band'}   # band is string, not numeric

corr = (
    df_feat[NUM_COLS + ['emi_eligibility']]
    .corr()['emi_eligibility']
    .drop('emi_eligibility')
)
corr_sorted = corr.reindex(corr.abs().sort_values().index)

print("Top 10 features by |correlation| with emi_eligibility:")
top10 = corr.abs().sort_values(ascending=False).head(10)
for feat, val in top10.items():
    sign  = '+' if corr[feat] > 0 else '-'
    tag   = ' ★ NEW' if feat in new_set else ''
    print(f"  {feat:30}: {sign}{val:.4f}{tag}")


# ================================================================
# CELL 6 — CHART 1: FEATURE CORRELATION RANKING (top 30)
# ================================================================
top30     = corr.abs().sort_values(ascending=False).head(30)
top30_corr = corr[top30.index].sort_values()

bar_colors = [ELIG_CLR if v >= 0 else NOT_CLR for v in top30_corr]
labels     = [
    f"{f}  ★" if f in new_set else f
    for f in top30_corr.index
]

fig, ax = plt.subplots(figsize=(14, 12))
bars = ax.barh(labels, top30_corr.values,
               color=bar_colors, edgecolor='white', alpha=0.90)
ax.axvline(0, color=DARK, linewidth=0.8, alpha=0.45)
ax.set_xlabel('Pearson r with EMI Eligibility')
ax.set_title(
    'Feature Correlation with EMI Eligibility — Top 30\n'
    'Teal = positive  |  Orange = negative  |  ★ = engineered feature',
    pad=12
)

for bar, val in zip(bars, top30_corr.values):
    xp  = bar.get_width()
    ha  = 'left' if xp >= 0 else 'right'
    off = 0.003 if xp >= 0 else -0.003
    ax.text(xp + off, bar.get_y() + bar.get_height() / 2,
            f'{val:+.3f}', va='center', ha=ha,
            fontsize=8, color=DARK)

# Legend patches
import matplotlib.patches as mpatches
ax.legend(handles=[
    mpatches.Patch(facecolor=ELIG_CLR, label='Positive correlation (→ Eligible)'),
    mpatches.Patch(facecolor=NOT_CLR,  label='Negative correlation (→ Not Eligible)'),
    mpatches.Patch(facecolor='white',  edgecolor='gray',
                   label='★ = engineered feature'),
], fontsize=9, loc='lower right')

plt.tight_layout()
save_fig('04_feature_correlations.png')


# ================================================================
# CELL 7 — CHART 2: KDE DISTRIBUTIONS FOR TOP 6 NEW FEATURES
# ================================================================
# Pick top 6 new numerical features by |correlation|
new_num_feats = [f for f in NEW_FEATURES if f != 'credit_score_band']
new_corr      = corr[new_num_feats].abs().sort_values(ascending=False)
top6_new      = new_corr.head(6).index.tolist()

elig_df = df_feat[df_feat['emi_eligibility'] == 1]
not_df  = df_feat[df_feat['emi_eligibility'] == 0]
SAMPLE  = 15000
e_s = elig_df.sample(min(SAMPLE, len(elig_df)), random_state=42)
n_s = not_df.sample(min(SAMPLE, len(not_df)),   random_state=42)

from scipy.stats import gaussian_kde

def kde_fill(ax, data, color, label, clip_lo=None, clip_hi=None,
             linestyle='-', alpha_fill=0.20, lw=2.5, n_points=400):
    data = data.dropna()
    if len(data) < 10:
        return
    lo   = clip_lo if clip_lo is not None else float(data.quantile(0.005))
    hi   = clip_hi if clip_hi is not None else float(data.quantile(0.995))
    xgrd = np.linspace(lo, hi, n_points)
    try:
        kde  = gaussian_kde(data, bw_method='scott')
        ygrd = kde(xgrd)
    except Exception:
        return
    ax.plot(xgrd, ygrd, color=color, linewidth=lw,
            linestyle=linestyle, label=label, zorder=3)
    ax.fill_between(xgrd, ygrd, alpha=alpha_fill, color=color, zorder=2)


fig, axes = plt.subplots(2, 3, figsize=(19, 11))
fig.suptitle(
    'Top 6 Engineered Features — Distribution by EMI Eligibility\n'
    '(Ranked by correlation with target)',
    y=0.98
)

for idx, feat in enumerate(top6_new):
    ax     = axes[idx // 3][idx % 3]
    r_val  = corr[feat]
    e_data = e_s[feat].dropna()
    n_data = n_s[feat].dropna()

    kde_fill(ax, n_data, NOT_CLR,
             f'Not Eligible (μ={n_data.mean():.3f})',
             linestyle='-', alpha_fill=0.15, lw=2.5)
    kde_fill(ax, e_data, ELIG_CLR,
             f'Eligible (μ={e_data.mean():.3f})',
             linestyle='--', alpha_fill=0.28, lw=2.5)

    ax.axvline(n_data.mean(), color=NOT_CLR,  linestyle=':', lw=1.4, alpha=0.7)
    ax.axvline(e_data.mean(), color=ELIG_CLR, linestyle=':', lw=1.4, alpha=0.7)

    ax.set_title(f'{feat}  (r={r_val:+.3f})')
    ax.set_xlabel(feat)
    ax.set_ylabel('Density')
    ax.legend(fontsize=8, framealpha=0.9)

plt.tight_layout(pad=2.0)
save_fig('04_engineered_feature_kdes.png')


# ================================================================
# CELL 8 — PROPER SPLIT + FIT ON TRAIN ONLY
# No leakage: FeatureEngineer learns only from training data
# ================================================================
X = df.drop(columns=['emi_eligibility', 'max_monthly_emi'])
y = df['emi_eligibility'].values

X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    stratify=y,
    random_state=42,
)

logger.info(f"Train: {len(y_train):,}  |  Test: {len(y_test):,}")

fe = FeatureEngineer()
X_train_feat = fe.fit_transform(X_train_raw)   # learns from train only
X_test_feat  = fe.transform(X_test_raw)         # same transform, no fit

logger.info(f"Train features shape: {X_train_feat.shape}")
logger.info(f"Test  features shape: {X_test_feat.shape}")


# ================================================================
# CELL 9 — SAVE ENRICHED TRAIN / TEST CSVs
# ================================================================
train_out = X_train_feat.copy()
train_out['emi_eligibility'] = y_train
train_out['max_monthly_emi'] = df.loc[X_train_raw.index, 'max_monthly_emi'].values

test_out = X_test_feat.copy()
test_out['emi_eligibility'] = y_test
test_out['max_monthly_emi'] = df.loc[X_test_raw.index, 'max_monthly_emi'].values

train_out.to_csv(TRAIN_PATH, index=False)
test_out.to_csv(TEST_PATH,   index=False)
logger.info(f"Saved: train_features.csv  ({len(train_out):,} rows)")
logger.info(f"Saved: test_features.csv   ({len(test_out):,} rows)")

# ── Feature column list (numerical only, no targets, no string cats) ──
feature_cols = [
    c for c in train_out.select_dtypes(include=[np.number]).columns
    if c not in ['emi_eligibility', 'max_monthly_emi']
]
feature_meta = {
    'feature_columns': feature_cols,
    'n_features'     : len(feature_cols),
    'new_features'   : [f for f in NEW_FEATURES if f != 'credit_score_band'],
    'original_count' : n_original,
    'engineered_count': len(feature_cols),
}
with open(FEATURE_COLS_PATH, 'w') as f:
    json.dump(feature_meta, f, indent=2)
logger.info(f"Saved: feature_columns.json  ({len(feature_cols)} features)")

# ── Save fitted FeatureEngineer for serving (Section 9) ──────────────
joblib.dump(fe, ENGINEER_PATH)
logger.info(f"Saved: feature_engineer.pkl")


# ================================================================
# LAST CELL — SUMMARY
# ================================================================
top5_new = new_corr.head(5)
print(f"""
{'=' * 60}
   SECTION 4 COMPLETE — FEATURE ENGINEERING SUMMARY
{'=' * 60}

FEATURE COUNT:
  Original  : {n_original} features
  Engineered: {len(feature_cols)} features (+{n_new} new)

TOP 5 NEW FEATURES (by correlation with emi_eligibility):
""")
for feat, val in top5_new.items():
    sign = '+' if corr[feat] > 0 else '-'
    print(f"  {feat:30}: {sign}{val:.4f}")

print(f"""
KEY FINDINGS:
  - total_emi_ratio and requested_emi_ratio confirm: loan affordability
    is the strongest signal after credit_score
  - credit_x_income interaction captures joint high-score + high-income effect
  - employment_stability separates stable vs unstable earners cleanly
  - log-transforms reduce skew on bank_balance and salary for LR models

SAVED:
  data/processed/train_features.csv   ({len(train_out):,} rows)
  data/processed/test_features.csv    ({len(test_out):,} rows)
  data/processed/feature_columns.json ({len(feature_cols)} feature names)
  models/feature_engineer.pkl         (fitted transformer for serving)
  docs/figures/04_feature_correlations.png
  docs/figures/04_engineered_feature_kdes.png

NEXT: Section 5 — Model Training (8 models: 4 classifiers + 4 regressors)
  Load from: data/processed/train_features.csv
  Beat:       ROC-AUC 0.9763  (Logistic Regression baseline)
{'=' * 60}
""")
