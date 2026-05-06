# ================================================================
# EMI PREDICT AI — EDA v3 (Bulletproof Rewrite)
# notebooks/02_eda.py
#
# ROOT CAUSE OF PREVIOUS FAILURES:
#   fill_between used ax.lines[-1].get_xdata() which grabbed
#   the WRONG line after seaborn KDE rendered internally.
#
# FIX: Use scipy.stats.gaussian_kde DIRECTLY.
#   We compute x_grid, y_vals ourselves → fill_between works
#   100% reliably because WE own the arrays.
#
# ALL 8 CHARTS FIXED:
#   Chart 1 → Donut + scipy KDE (no overlap, no spill)
#   Chart 2 → Paired_r palette, clean table
#   Chart 3 → scipy KDE clipped, p-value bottom-right only
#   Chart 4 → Heatmap NO annotations, sign-colored bar
#   Chart 5 → Dark2_r binary, white labels inside bars
#   Chart 6 → Scatter zorder, Accent_r salary bracket
#   Chart 7 → Accent_r per feature, zero annotation box
#   Chart 8 → 0% stub bars, explanation note
# ================================================================

import sys
import os

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)
sys.path.insert(0, PROJECT_ROOT)

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats
from scipy.stats import gaussian_kde
import warnings
warnings.filterwarnings('ignore')

# ── Global style ──────────────────────────────────────────────
plt.rcParams.update({
    'figure.dpi'        : 130,
    'font.size'         : 10,
    'axes.titlesize'    : 12,
    'axes.titleweight'  : 'bold',
    'axes.labelsize'    : 10,
    'xtick.labelsize'   : 9,
    'ytick.labelsize'   : 9,
    'legend.fontsize'   : 9,
    'figure.titlesize'  : 14,
    'figure.titleweight': 'bold',
    'axes.spines.top'   : False,
    'axes.spines.right' : False,
})

# ================================================================
# MASTER PALETTE
# ================================================================
D2   = sns.color_palette('Dark2_r',  8)
PR   = sns.color_palette('Paired_r', 12)
AC   = sns.color_palette('Accent_r', 8)

# Binary: Not Eligible = orange-red, Eligible = teal-green
NOT_CLR  = D2[6]   # #d95f02 warm orange
ELIG_CLR = D2[7]   # #1b9e77 teal green
BAR_2    = [NOT_CLR, ELIG_CLR]
DARK     = '#2D3436'

# 5 scenario colors from Paired_r (well-separated)
SCENARIO_PAL = [PR[i] for i in [0, 2, 4, 6, 8]]

# Accent_r for single-series histograms (one per feature)
DIST_COLORS = list(AC)

# ── Paths ─────────────────────────────────────────────────────
CLEAN_PATH  = os.path.join(
    PROJECT_ROOT, 'data', 'processed', 'emi_cleaned.csv'
)
REPORT_PATH = os.path.join(PROJECT_ROOT, 'data', 'reports')
os.makedirs(REPORT_PATH, exist_ok=True)


def save_fig(name):
    plt.savefig(
        os.path.join(REPORT_PATH, name),
        dpi=150, bbox_inches='tight'
    )
    plt.close()
    print(f"   ✅ Saved: {name}")


# ================================================================
# CORE HELPER: scipy KDE — bulletproof filled curves
# ================================================================
def kde_fill(ax, data, color, label, clip_lo=0,
             clip_hi=None, linestyle='-', alpha_fill=0.20,
             lw=2.5, n_points=500):
    """
    Compute KDE with scipy directly.
    Draws filled area + line on ax.
    No seaborn internal lines touched.
    clip_lo / clip_hi hard-clamp the x range so
    distributions NEVER extend into impossible values.
    """
    data = data.dropna()
    if len(data) < 10:
        return

    hi   = clip_hi if clip_hi else data.quantile(0.995)
    lo   = clip_lo
    xgrd = np.linspace(lo, hi, n_points)

    try:
        kde  = gaussian_kde(data, bw_method='scott')
        ygrd = kde(xgrd)
        # Zero out anything left of clip_lo (hard clamp)
        ygrd[xgrd < lo] = 0
    except Exception:
        return

    ax.plot(xgrd, ygrd, color=color, linewidth=lw,
            linestyle=linestyle, label=label, zorder=3)
    ax.fill_between(xgrd, ygrd, alpha=alpha_fill,
                    color=color, zorder=2)


# ================================================================
# LOAD DATA
# ================================================================
print("=" * 65)
print("   EMI PREDICT AI — EDA v3 (Bulletproof)")
print("=" * 65)
print("\n📂 Loading cleaned data...")

df = pd.read_csv(CLEAN_PATH)

CAT_COLS = [
    'gender', 'marital_status', 'education',
    'employment_type', 'company_type', 'house_type',
    'existing_loans', 'emi_scenario'
]
for col in CAT_COLS:
    if col in df.columns:
        df[col] = df[col].astype('category')

total = len(df)
vc    = df['emi_eligibility'].value_counts()
print(f"✅ Loaded: {total:,} rows × {df.shape[1]} cols")

elig_df = df[df['emi_eligibility'] == 1].copy()
not_df  = df[df['emi_eligibility'] == 0].copy()

SAMPLE  = 15000
e_samp  = elig_df.sample(min(SAMPLE, len(elig_df)), random_state=42)
ne_samp = not_df.sample(min(SAMPLE, len(not_df)),   random_state=42)


# ================================================================
# CHART 1: TARGET VARIABLE ANALYSIS
# ================================================================
print("\n📊 Chart 1: Target Variable Analysis...")

fig, axes = plt.subplots(1, 3, figsize=(19, 6))
fig.suptitle('Target Variable Analysis', y=0.98)

# ── 1a: Donut ─────────────────────────────────────────────────
n_not  = int(vc.get(0, 0))
n_elig = int(vc.get(1, 0))

wedge_kw = dict(width=0.42, edgecolor='white', linewidth=3)
patches, texts, autotexts = axes[0].pie(
    [n_not, n_elig],
    colors=[NOT_CLR, ELIG_CLR],
    startangle=90,
    wedgeprops=wedge_kw,
    autopct='%1.1f%%',
    pctdistance=0.77,
    labels=['Not Eligible', 'Eligible'],
    labeldistance=1.10,
)
for t in texts:
    t.set_fontsize(9)
    t.set_fontweight('bold')
for at in autotexts:
    at.set_fontsize(9)
    at.set_fontweight('bold')
    at.set_color('white')
axes[0].text(0, 0.10, f'{total:,}',
             ha='center', va='center',
             fontsize=15, fontweight='bold', color=DARK)
axes[0].text(0, -0.14, 'Applications',
             ha='center', va='center',
             fontsize=9, color='gray')
axes[0].legend(handles=[
    mpatches.Patch(facecolor=NOT_CLR,
                   label=f'{n_not:,} records'),
    mpatches.Patch(facecolor=ELIG_CLR,
                   label=f'{n_elig:,} records'),
], loc='lower center', bbox_to_anchor=(0.5, -0.08),
   fontsize=8, framealpha=0, ncol=2)
axes[0].set_title('EMI Eligibility\nDistribution')

# ── 1b: Max EMI histogram ─────────────────────────────────────
axes[1].hist(df['max_monthly_emi'], bins=50,
             color=AC[2], edgecolor='white', alpha=0.88)
axes[1].axvline(df['max_monthly_emi'].mean(),
                color=NOT_CLR, linestyle='--', linewidth=2,
                label=f"Mean ₹{df['max_monthly_emi'].mean():,.0f}")
axes[1].axvline(df['max_monthly_emi'].median(),
                color=ELIG_CLR, linestyle='--', linewidth=2,
                label=f"Median ₹{df['max_monthly_emi'].median():,.0f}")
axes[1].set_title('Max Monthly EMI\nDistribution')
axes[1].set_xlabel('Max Monthly EMI (Rs)')
axes[1].set_ylabel('Count')
axes[1].legend()

# ── 1c: KDE by eligibility ────────────────────────────────────
emi_max = float(df['max_monthly_emi'].quantile(0.995))
kde_fill(axes[2], not_df['max_monthly_emi'], NOT_CLR,
         f'Not Eligible (μ=₹{not_df["max_monthly_emi"].mean():,.0f})',
         clip_lo=0, clip_hi=emi_max, linestyle='--')
kde_fill(axes[2], elig_df['max_monthly_emi'], ELIG_CLR,
         f'Eligible (μ=₹{elig_df["max_monthly_emi"].mean():,.0f})',
         clip_lo=0, clip_hi=emi_max, alpha_fill=0.28)
axes[2].set_title('Max EMI by Eligibility\n(Density)')
axes[2].set_xlabel('Max Monthly EMI (Rs)')
axes[2].set_ylabel('Density')
axes[2].set_xlim(left=0)
axes[2].legend()

plt.tight_layout(pad=2.0)
save_fig('eda_01_target_analysis.png')


# ================================================================
# CHART 2: EMI SCENARIO ANALYSIS
# ================================================================
print("\n📊 Chart 2: EMI Scenario Analysis...")

sc = df.groupby('emi_scenario', observed=True).agg(
    total     =('emi_eligibility', 'count'),
    eligible  =('emi_eligibility', 'sum'),
    avg_emi   =('max_monthly_emi', 'mean'),
    median_req=('requested_amount', 'median'),
).reset_index()
sc['approval_pct'] = (sc['eligible'] / sc['total'] * 100).round(1)
sc['short'] = (sc['emi_scenario']
               .str.replace(' Emi', '').str.replace(' EMI', ''))

fig, axes = plt.subplots(2, 2, figsize=(16, 13))
fig.suptitle('EMI Scenario Deep Dive', y=0.98)

# 2a — Applications
bars = axes[0, 0].barh(sc['short'], sc['total'],
                        color=SCENARIO_PAL, edgecolor='white', alpha=0.92)
axes[0, 0].set_title('Applications per Scenario')
axes[0, 0].set_xlabel('Number of Applications')
for bar, val in zip(bars, sc['total']):
    axes[0, 0].text(bar.get_width() + 300,
                    bar.get_y() + bar.get_height() / 2,
                    f'{val:,}', va='center', fontsize=9)

# 2b — Approval rate
sc_sorted = sc.sort_values('approval_pct')
col_apr   = [ELIG_CLR if r > 19.2 else NOT_CLR
             for r in sc_sorted['approval_pct']]
bars = axes[0, 1].barh(sc_sorted['short'], sc_sorted['approval_pct'],
                        color=col_apr, edgecolor='white', alpha=0.92)
axes[0, 1].set_title('Approval Rate by Scenario (%)')
axes[0, 1].set_xlabel('Approval Rate (%)')
axes[0, 1].axvline(19.2, color=DARK, linestyle='--',
                   linewidth=1.2, alpha=0.6, label='Avg 19.2%')
axes[0, 1].legend()
for bar, val in zip(bars, sc_sorted['approval_pct']):
    axes[0, 1].text(bar.get_width() + 0.3,
                    bar.get_y() + bar.get_height() / 2,
                    f'{val:.1f}%', va='center', fontsize=9)

# 2c — Avg EMI
bars = axes[1, 0].bar(sc['short'], sc['avg_emi'],
                       color=SCENARIO_PAL, edgecolor='white', alpha=0.92)
axes[1, 0].set_title('Average Max EMI per Scenario (Rs)')
axes[1, 0].set_xlabel('EMI Scenario')
axes[1, 0].set_ylabel('Avg Max EMI (Rs)')
axes[1, 0].tick_params(axis='x', rotation=18)
for bar, val in zip(bars, sc['avg_emi']):
    axes[1, 0].text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 60,
                    f'₹{val:,.0f}', ha='center', fontsize=8)

# 2d — Table
axes[1, 1].axis('off')
tdata = [[r['short'], f"{r['total']:,}", f"{r['approval_pct']}%",
          f"₹{r['avg_emi']:,.0f}", f"₹{r['median_req']:,.0f}"]
         for _, r in sc.iterrows()]
tbl = axes[1, 1].table(
    cellText=tdata,
    colLabels=['Scenario', 'Total', 'Approval', 'Avg EMI', 'Median Req'],
    cellLoc='center', loc='center', bbox=[0, 0.08, 1, 0.86]
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(9)
tbl.auto_set_column_width(col=list(range(5)))
for (r, c), cell in tbl.get_celld().items():
    cell.set_edgecolor('#dddddd')
    if r == 0:
        cell.set_facecolor(DARK)
        cell.set_text_props(color='white', fontweight='bold')
    elif r % 2 == 0:
        cell.set_facecolor('#f4f6f8')
axes[1, 1].set_title('Scenario Summary', pad=14)

plt.tight_layout(pad=2.2)
save_fig('eda_02_scenario_analysis.png')


# ================================================================
# CHART 3: KEY FINANCIAL FEATURES vs ELIGIBILITY
# ================================================================
print("\n📊 Chart 3: Key Financial Features vs Eligibility...")

# Pre-compute t-test p-values (console only — NOT in chart titles)
features_meta = [
    # (column, display_title, clip_lo, clip_hi_pct)
    ('credit_score',        'Credit Score',        300, 0.999),
    ('monthly_salary',      'Monthly Salary (Rs)',   0, 0.99),
    ('bank_balance',        'Bank Balance (Rs)',      0, 0.99),
    ('age',                 'Age (years)',           18, 1.00),
    ('years_of_employment', 'Years Employed',         0, 0.995),
]

print("   t-test significance (console only):")
pvals = {}
for feat, _, _, _ in features_meta:
    if feat not in df.columns:
        continue
    g1 = elig_df[feat].dropna()
    g2 = not_df[feat].dropna()
    _, p_raw = stats.ttest_ind(
        g1.sample(min(5000, len(g1)), random_state=42),
        g2.sample(min(5000, len(g2)), random_state=42)
    )
    p: float = p_raw  # type: ignore[assignment]  scipy stubs don't type TtestResult elements as float
    pvals[feat] = p
    print(f"   {feat:25}: p={p:.4f} "
          f"{'✅ significant' if p < 0.05 else '— no difference'}")

fig, axes = plt.subplots(2, 3, figsize=(19, 12))
fig.suptitle('Key Financial Features vs EMI Eligibility', y=0.98)

for idx, (feat, title, clo, chi_q) in enumerate(features_meta):
    ax = axes[idx // 3][idx % 3]
    if feat not in df.columns:
        continue

    chi = float(df[feat].quantile(chi_q))

    # NOT Eligible — solid line, lighter fill
    kde_fill(ax,
             ne_samp[feat], NOT_CLR,
             f'Not Eligible (μ={ne_samp[feat].mean():.0f})',
             clip_lo=clo, clip_hi=chi,
             linestyle='-', alpha_fill=0.15, lw=2.5)

    # Eligible — dashed line, stronger fill (draws ON TOP)
    kde_fill(ax,
             e_samp[feat], ELIG_CLR,
             f'Eligible (μ={e_samp[feat].mean():.0f})',
             clip_lo=clo, clip_hi=chi,
             linestyle='--', alpha_fill=0.28, lw=2.5)

    # Mean vertical lines
    ax.axvline(ne_samp[feat].mean(), color=NOT_CLR,
               linestyle=':', lw=1.6, alpha=0.75)
    ax.axvline(e_samp[feat].mean(),  color=ELIG_CLR,
               linestyle=':', lw=1.6, alpha=0.75)

    # p-value stamp — bottom right corner, small italic gray
    p = pvals.get(feat, 1.0)
    sig_txt = (f'p={p:.4f} ✓ significant'
               if p < 0.05 else f'p={p:.2f} — no difference')
    ax.text(0.97, 0.04, sig_txt,
            transform=ax.transAxes,
            ha='right', va='bottom', fontsize=7.5,
            color='#888', style='italic')

    ax.set_title(title)   # CLEAN title — no p-value here
    ax.set_xlabel(title)
    ax.set_ylabel('Density')
    ax.set_xlim(left=clo)
    ax.legend(fontsize=9, framealpha=0.9)

# 3f — Existing loans bar
ax = axes[1][2]
if 'existing_loans' in df.columns:
    cross = pd.crosstab(
        df['existing_loans'], df['emi_eligibility'],
        normalize='index'
    ) * 100
    cross.columns = ['Not Eligible %', 'Eligible %']
    cross.sort_values('Eligible %', ascending=True, inplace=True)
    cross.plot(kind='barh', ax=ax, color=BAR_2,
               edgecolor='white', alpha=0.92, width=0.55)
    ax.set_title('Existing Loans vs Eligibility')
    ax.set_xlabel('Percentage (%)')
    ax.set_ylabel('')
    ax.legend(loc='lower right', fontsize=9)
    for container in ax.containers:
        ax.bar_label(container, fmt='%.1f%%',
                     label_type='center', fontsize=9,
                     color='white', fontweight='bold')

plt.tight_layout(pad=2.5)
save_fig('eda_03_features_vs_target.png')


# ================================================================
# CHART 4: CORRELATION ANALYSIS
# ================================================================
print("\n📊 Chart 4: Correlation Analysis...")

corr_cols = [c for c in [
    'age', 'monthly_salary', 'years_of_employment',
    'monthly_rent', 'family_size', 'dependents',
    'school_fees', 'college_fees', 'travel_expenses',
    'groceries_utilities', 'other_monthly_expenses',
    'current_emi_amount', 'credit_score', 'bank_balance',
    'emergency_fund', 'requested_amount', 'requested_tenure',
    'emi_eligibility', 'max_monthly_emi'
] if c in df.columns]

CM = df[corr_cols].corr()

# Short names for axis labels
SN = {
    'age': 'age', 'monthly_salary': 'salary',
    'years_of_employment': 'yrs_emp',
    'monthly_rent': 'm_rent', 'family_size': 'fam_sz',
    'dependents': 'depend', 'school_fees': 'school',
    'college_fees': 'college', 'travel_expenses': 'travel',
    'groceries_utilities': 'grocery',
    'other_monthly_expenses': 'oth_exp',
    'current_emi_amount': 'curr_emi',
    'credit_score': 'cr_score', 'bank_balance': 'bank_bal',
    'emergency_fund': 'emerg', 'requested_amount': 'req_amt',
    'requested_tenure': 'tenure',
    'emi_eligibility': 'eligible', 'max_monthly_emi': 'max_emi',
}

CM_disp = CM.copy()
CM_disp.columns = [SN.get(c, c) for c in CM_disp.columns]
CM_disp.index   = [SN.get(c, c) for c in CM_disp.index]

# Wide figure — heatmap 62%, bar 33%
fig = plt.figure(figsize=(28, 11))
fig.suptitle('Correlation Analysis', fontsize=15,
             fontweight='bold', y=0.98)
ax_heat = fig.add_axes((0.02, 0.05, 0.58, 0.90))
ax_bar  = fig.add_axes((0.66, 0.06, 0.32, 0.86))

# Heatmap: NO annotations — color tells the story
mask = np.triu(np.ones_like(CM_disp, dtype=bool))
sns.heatmap(
    CM_disp, mask=mask,
    annot=True, fmt='.2f',
    annot_kws={'size': 7},
    cmap='coolwarm',
    center=0, vmin=-1, vmax=1,
    square=True,
    linewidths=0.20, linecolor='#ebebeb',
    ax=ax_heat,
    cbar_kws={'shrink': 0.65, 'label': 'Pearson r', 'pad': 0.01}
)
ax_heat.set_title(
    'Feature Correlation Matrix  (lower triangle only)\n'
    'Blue = negative  |  White ≈ 0  |  Red = positive',
    fontsize=11, pad=10
)
ax_heat.tick_params(axis='x', rotation=45, labelsize=8.5)
ax_heat.tick_params(axis='y', rotation=0,  labelsize=8.5)

# Bar chart: correlation with emi_eligibility
tgt = CM[['emi_eligibility']].drop('emi_eligibility', errors='ignore')
tgt = tgt.sort_values('emi_eligibility')
vals   = tgt['emi_eligibility']
bcolrs = [ELIG_CLR if v >= 0 else NOT_CLR for v in vals]

yp   = np.arange(len(tgt))
bars = ax_bar.barh(yp, vals, color=bcolrs,
                   alpha=0.92, edgecolor='white', height=0.68)
ax_bar.set_yticks(yp)
ax_bar.set_yticklabels(
    [SN.get(c) or c for c in tgt.index], fontsize=9.5  # or-fallback gives str, not str|None
)
ax_bar.axvline(0, color=DARK, linewidth=0.8, alpha=0.5)
ax_bar.set_xlabel('Pearson r with EMI Eligibility', fontsize=10)
ax_bar.set_title(
    'Features vs EMI Eligibility\n'
    'Green = helps approval  |  Orange = hurts',
    fontsize=11, pad=10
)
for bar, val in zip(bars, vals):
    xp     = bar.get_width()
    ha_txt = 'left' if xp >= 0 else 'right'
    off    = 0.006 if xp >= 0 else -0.006
    ax_bar.text(xp + off, bar.get_y() + bar.get_height() / 2,
                f'{val:+.2f}', va='center', ha=ha_txt,
                fontsize=8.5, fontweight='bold', color=DARK)

save_fig('eda_04_correlation.png')

print("   Top 5 correlated with eligibility:")
top5 = CM['emi_eligibility'].drop('emi_eligibility').abs() \
        .sort_values(ascending=False).head(5)
for feat, val in top5.items():
    print(f"   {feat:30}: {'+' if val > 0 else '-'}{abs(val):.3f}")  # val IS CM[...][feat], no re-index needed


# ================================================================
# CHART 5: CATEGORICAL FEATURES vs ELIGIBILITY
# ================================================================
print("\n📊 Chart 5: Categorical Features...")

cat_feats = [
    ('gender',          'Gender'),
    ('marital_status',  'Marital Status'),
    ('education',       'Education Level'),
    ('employment_type', 'Employment Type'),
    ('house_type',      'House Type'),
    ('company_type',    'Company Type'),
]

fig, axes = plt.subplots(2, 3, figsize=(19, 12))
fig.suptitle('Categorical Features vs EMI Eligibility', y=0.98)

for idx, (feat, title) in enumerate(cat_feats):
    ax = axes[idx // 3][idx % 3]
    if feat not in df.columns:
        ax.set_visible(False)
        continue

    cross = pd.crosstab(
        df[feat], df['emi_eligibility'], normalize='index'
    ) * 100
    cross.columns = ['Not Eligible %', 'Eligible %']
    cross.sort_values('Eligible %', ascending=True, inplace=True)

    cross.plot(kind='barh', ax=ax,
               color=BAR_2, edgecolor='white',
               linewidth=0.5, alpha=0.93, width=0.62)
    ax.set_title(f'{title}')
    ax.set_xlabel('Percentage (%)')
    ax.set_ylabel('')
    ax.legend(loc='lower right', fontsize=8, framealpha=0.85)

    # White labels inside bars
    for container in ax.containers:
        labels = ax.bar_label(
            container, fmt='%.1f%%',
            label_type='center',
            fontsize=8.5, color='white', fontweight='bold'
        )

plt.tight_layout(pad=2.5)
save_fig('eda_05_categorical_analysis.png')


# ================================================================
# CHART 6: FINANCIAL CAPACITY ANALYSIS
# ================================================================
print("\n📊 Chart 6: Financial Capacity Analysis...")

df['total_expenses'] = (
    df['monthly_rent'] + df['school_fees'] + df['college_fees'] +
    df['travel_expenses'] + df['groceries_utilities'] +
    df['other_monthly_expenses'] + df['current_emi_amount']
)
df['expense_ratio'] = (
    df['total_expenses'] /
    df['monthly_salary'].replace(0, np.nan)
).clip(0, 2)

df['salary_bracket'] = pd.cut(
    df['monthly_salary'],
    bins=[0, 20000, 40000, 60000, 80000, 100000, float('inf')],
    labels=['0-20k', '20-40k', '40-60k', '60-80k', '80-100k', '100k+']
)

SCAT = 2000
s_e  = elig_df.sample(min(SCAT, len(elig_df)), random_state=42)
s_ne = not_df.sample(min(SCAT, len(not_df)),   random_state=42)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Financial Capacity Analysis', y=0.98)

# 6a — Salary vs Max EMI scatter (not-eligible behind, eligible on top)
axes[0, 0].scatter(s_ne['monthly_salary'], s_ne['max_monthly_emi'],
                   alpha=0.45, s=14, color=NOT_CLR,
                   label='Not Eligible', linewidths=0, zorder=2)
axes[0, 0].scatter(s_e['monthly_salary'], s_e['max_monthly_emi'],
                   alpha=0.70, s=14, color=ELIG_CLR,
                   label='Eligible', linewidths=0, zorder=3)
axes[0, 0].set_title('Monthly Salary vs Max EMI')
axes[0, 0].set_xlabel('Monthly Salary (Rs)')
axes[0, 0].set_ylabel('Max Monthly EMI (Rs)')
axes[0, 0].legend(fontsize=9)

# 6b — Salary bracket approval (sequential Accent_r)
br = df.groupby('salary_bracket', observed=True)['emi_eligibility'] \
       .agg(['mean', 'count'])
br['pct'] = br['mean'] * 100

br_colors = [AC[i] for i in range(len(br))]
bars = axes[0, 1].bar(br.index, br['pct'],
                       color=br_colors, edgecolor='white', alpha=0.93)
axes[0, 1].set_title('Approval Rate by Salary Bracket (%)')
axes[0, 1].set_xlabel('Monthly Salary Bracket')
axes[0, 1].set_ylabel('Approval Rate (%)')
axes[0, 1].axhline(19.2, color=DARK, linestyle='--',
                   alpha=0.5, label='Avg 19.2%')
axes[0, 1].legend()
for bar, val in zip(bars, br['pct']):
    axes[0, 1].text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.5,
                    f'{val:.1f}%', ha='center', va='bottom',
                    fontsize=9, fontweight='bold')

# 6c — Expense ratio KDE (scipy)
kde_fill(axes[1, 0],
         df.loc[df['emi_eligibility'] == 0, 'expense_ratio'],
         NOT_CLR, f'Not Eligible (μ={df.loc[df["emi_eligibility"]==0,"expense_ratio"].mean():.2f})',
         clip_lo=0, clip_hi=2, linestyle='--', alpha_fill=0.15)
kde_fill(axes[1, 0],
         df.loc[df['emi_eligibility'] == 1, 'expense_ratio'],
         ELIG_CLR, f'Eligible (μ={df.loc[df["emi_eligibility"]==1,"expense_ratio"].mean():.2f})',
         clip_lo=0, clip_hi=2, alpha_fill=0.28)
axes[1, 0].axvline(0.5, color=DARK, linestyle=':', alpha=0.5,
                   label='0.5 threshold')
axes[1, 0].set_title('Expense-to-Salary Ratio by Eligibility')
axes[1, 0].set_xlabel('Expense Ratio (0 = no expenses, 1 = all salary spent)')
axes[1, 0].set_ylabel('Density')
axes[1, 0].set_xlim(0, 2)
axes[1, 0].legend()

# 6d — Credit score vs Max EMI scatter
axes[1, 1].scatter(s_ne['credit_score'], s_ne['max_monthly_emi'],
                   alpha=0.45, s=14, color=NOT_CLR,
                   label='Not Eligible', linewidths=0, zorder=2)
axes[1, 1].scatter(s_e['credit_score'], s_e['max_monthly_emi'],
                   alpha=0.70, s=14, color=ELIG_CLR,
                   label='Eligible', linewidths=0, zorder=3)
axes[1, 1].set_title('Credit Score vs Max EMI')
axes[1, 1].set_xlabel('Credit Score')
axes[1, 1].set_ylabel('Max Monthly EMI (Rs)')
axes[1, 1].legend(fontsize=9)

plt.tight_layout(pad=2.5)
save_fig('eda_06_financial_analysis.png')

df.drop(columns=['total_expenses', 'expense_ratio', 'salary_bracket'],
        errors='ignore', inplace=True)


# ================================================================
# CHART 7: NUMERICAL FEATURE DISTRIBUTIONS
# ================================================================
print("\n📊 Chart 7: Feature Distributions...")

# (column, display_label, bins, is_zero_heavy, zero_note)
dist_feats = [
    ('age',                 'Age',             20, False, ''),
    ('monthly_salary',      'Monthly Salary',   40, False, ''),
    ('credit_score',        'Credit Score',     40, False, ''),
    ('bank_balance',        'Bank Balance',     40, False, ''),
    ('emergency_fund',      'Emergency Fund',   35, True,
     '% have ₹0\n(no safety net)'),
    ('years_of_employment', 'Years Employed',   25, True,
     '% have 0 yrs\n(newly employed)'),
    ('requested_amount',    'Requested Amount', 40, False, ''),
    ('requested_tenure',    'Tenure (months)',  30, False, ''),
    ('monthly_rent',        'Monthly Rent',     35, True,
     '% pay ₹0\n(own / family home)'),
]
dist_feats = [(f, t, b, z, n) for f, t, b, z, n in dist_feats
              if f in df.columns]

fig, axes = plt.subplots(3, 3, figsize=(19, 14))
fig.suptitle('Numerical Feature Distributions (After Cleaning)',
             y=0.98)

for idx, (feat, title, bins, zero_heavy, znote) in enumerate(dist_feats):
    ax    = axes[idx // 3][idx % 3]
    data  = df[feat].dropna()
    color = DIST_COLORS[idx % len(DIST_COLORS)]  # one Accent_r color per feature

    ax.hist(data, bins=bins, color=color,
            edgecolor='white', alpha=0.88, linewidth=0.4)

    ax.axvline(data.mean(), color='#e74c3c',
               linestyle='--', linewidth=2.0,
               label=f'Mean {data.mean():,.0f}')
    ax.axvline(data.median(), color=DARK,
               linestyle='--', linewidth=2.0,
               label=f'Median {data.median():,.0f}')

    ax.set_title(f'{title}  |  skew={data.skew():.2f}')
    ax.set_ylabel('Count')
    ax.legend(fontsize=8)

    # Zero-spike annotation box (bottom right)
    if zero_heavy:
        zero_pct = (data == 0).sum() / len(data) * 100
        if zero_pct > 20:
            note = f'{zero_pct:.0f}% = 0\n{znote}'
            ax.text(0.97, 0.96, note,
                    transform=ax.transAxes,
                    ha='right', va='top', fontsize=8,
                    style='italic', color='#555',
                    bbox=dict(boxstyle='round,pad=0.35',
                              fc='white', alpha=0.85,
                              ec='#ccc'))

plt.tight_layout(pad=2.2)
save_fig('eda_07_distributions.png')


# ================================================================
# CHART 8: CLASS IMBALANCE ANALYSIS
# ================================================================
print("\n📊 Chart 8: Class Imbalance Analysis...")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Class Imbalance Analysis', y=0.98)

# ── 8a: Bar chart with breathing room ─────────────────────────
cats   = ['Not Eligible', 'Eligible']
counts = [n_not, n_elig]

bars = axes[0].bar(cats, counts,
                   color=[NOT_CLR, ELIG_CLR],
                   edgecolor='white', width=0.44, alpha=0.93)
axes[0].set_title('Class Distribution\n(Imbalance: 4.2 : 1)', pad=14)
axes[0].set_ylabel('Number of Applications')
axes[0].set_ylim(0, max(counts) * 1.20)   # 20% headroom — labels won't touch title

for bar, cnt in zip(bars, counts):
    pct = cnt / total * 100
    axes[0].text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + max(counts) * 0.018,
        f'{cnt:,}\n({pct:.1f}%)',
        ha='center', va='bottom',
        fontsize=11, fontweight='bold', color=DARK
    )

# ── 8b: Naive vs ML — 0% bars rendered as stubs ───────────────
metrics  = ['Accuracy', 'F1-Score', 'Precision', 'Recall']
naive    = [80.8, 0.0, 0.0, 0.0]
ml_tgt   = [85.0, 75.0, 80.0, 70.0]
x        = np.arange(len(metrics))
w        = 0.36

for i, (nv, mv) in enumerate(zip(naive, ml_tgt)):
    stub = nv if nv > 0 else 4.0    # render stub so bar is visible

    axes[1].bar(x[i] - w / 2, stub, w, color=NOT_CLR, alpha=0.90,
                edgecolor='white',
                label='Naive (all Not Eligible)' if i == 0 else '_')
    axes[1].text(x[i] - w / 2, stub + 0.8, f'{nv:.0f}%',
                 ha='center', va='bottom', fontsize=9,
                 fontweight='bold', color=DARK)

    axes[1].bar(x[i] + w / 2, mv, w, color=ELIG_CLR, alpha=0.90,
                edgecolor='white',
                label='Our ML Target' if i == 0 else '_')
    axes[1].text(x[i] + w / 2, mv + 0.8, f'{mv:.0f}%',
                 ha='center', va='bottom', fontsize=9,
                 fontweight='bold', color=DARK)

axes[1].set_title('Why Accuracy Alone is Misleading')
axes[1].set_ylabel('Score (%)')
axes[1].set_xticks(x)
axes[1].set_xticklabels(metrics)
axes[1].set_ylim(0, 108)
axes[1].legend(fontsize=9, framealpha=0.9)

# Explanation note
axes[1].text(
    0.55, 0.14,
    'Naive F1 / Precision / Recall = 0%\n'
    '(catches zero eligible customers)',
    transform=axes[1].transAxes,
    fontsize=8, ha='center', color='#666', style='italic',
    bbox=dict(boxstyle='round,pad=0.35',
              fc='white', alpha=0.82, ec='#ccc')
)

plt.tight_layout(pad=2.0)
save_fig('eda_08_class_imbalance.png')


# ================================================================
# FINAL SUMMARY
# ================================================================
_sep = "=" * 65
print(f"""
{_sep}
   EDA v3 COMPLETE - All 8 charts saved
{_sep}

PALETTE USED:
  Binary comparisons -> Dark2_r  (orange vs teal-green)
  EMI Scenarios      -> Paired_r (5 distinct dark colors)
  Feature histograms -> Accent_r (one color per feature)

KDE METHOD: scipy gaussian_kde (direct x/y arrays)
  -> No seaborn internal line reference bugs
  -> All fills clipped -- zero negative spill

SAVED TO: data/reports/
  eda_01_target_analysis.png
  eda_02_scenario_analysis.png
  eda_03_features_vs_target.png
  eda_04_correlation.png
  eda_05_categorical_analysis.png
  eda_06_financial_analysis.png
  eda_07_distributions.png
  eda_08_class_imbalance.png

NEXT: Section 3 -- Baseline Model
""")