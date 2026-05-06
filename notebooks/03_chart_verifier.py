# ================================================================
# EMI PREDICT AI — Chart Data Verifier
# Prints ALL numbers that go into every chart
# Compare these with what you see in PNGs
# ================================================================

import sys, os
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import pandas as pd
import numpy as np

CLEAN_PATH = os.path.join(PROJECT_ROOT, 'data', 'processed', 'emi_cleaned.csv')
df = pd.read_csv(CLEAN_PATH)
total = len(df)

print("=" * 70)
print("   CHART DATA VERIFIER — Compare with your PNGs")
print("=" * 70)

# ── CHART 1: Target Analysis ────────────────────────────────
print("\n" + "─" * 70)
print("CHART 1: Target Variable Analysis")
print("─" * 70)

vc = df['emi_eligibility'].value_counts()
e_cnt = int(vc.get(1, 0))
ne_cnt = int(vc.get(0, 0))
print(f"  Pie chart labels:")
print(f"    Not Eligible: {ne_cnt:,} ({ne_cnt/total*100:.1f}%)")
print(f"    Eligible:     {e_cnt:,} ({e_cnt/total*100:.1f}%)")

print(f"\n  Max EMI histogram:")
print(f"    Mean:   ₹{df['max_monthly_emi'].mean():,.0f}")
print(f"    Median: ₹{df['max_monthly_emi'].median():,.0f}")

elig_emi = df[df['emi_eligibility'] == 1]['max_monthly_emi']
not_emi  = df[df['emi_eligibility'] == 0]['max_monthly_emi']
print(f"\n  EMI by Eligibility:")
print(f"    Eligible     → Mean: ₹{elig_emi.mean():,.0f} | Median: ₹{elig_emi.median():,.0f}")
print(f"    Not Eligible → Mean: ₹{not_emi.mean():,.0f} | Median: ₹{not_emi.median():,.0f}")

# ── CHART 2: Scenario Analysis ──────────────────────────────
print("\n" + "─" * 70)
print("CHART 2: EMI Scenario Analysis")
print("─" * 70)

scenario_stats = df.groupby('emi_scenario', observed=True).agg(
    total=('emi_eligibility', 'count'),
    eligible=('emi_eligibility', 'sum'),
    avg_emi=('max_monthly_emi', 'mean'),
    median_req=('requested_amount', 'median'),
).reset_index()
scenario_stats['approval_pct'] = (scenario_stats['eligible'] / scenario_stats['total'] * 100).round(1)
scenario_stats['short_name'] = scenario_stats['emi_scenario'].str.replace(' EMI', '').str.replace(' Emi', '')

print(f"  {'Scenario':25} | {'Total':>8} | {'Approval':>9} | {'Avg EMI':>12}")
print(f"  {'-'*25}-+-{'-'*8}-+-{'-'*9}-+-{'-'*12}")
for _, r in scenario_stats.iterrows():
    print(f"  {r['short_name']:25} | {r['total']:>8,} | {r['approval_pct']:>8.1f}% | ₹{r['avg_emi']:>10,.0f}")

# ── CHART 3: Features vs Target ─────────────────────────────
print("\n" + "─" * 70)
print("CHART 3: Key Features vs Eligibility")
print("─" * 70)

from scipy import stats
num_features = [
    'credit_score', 'monthly_salary', 'bank_balance',
    'age', 'years_of_employment'
]
e_df = df[df['emi_eligibility'] == 1]
ne_df = df[df['emi_eligibility'] == 0]

print(f"  {'Feature':25} | {'Elig Mean':>12} | {'NotElig Mean':>13} | {'p-value':>10} | {'Significant':>12}")
print(f"  {'-'*25}-+-{'-'*12}-+-{'-'*13}-+-{'-'*10}-+-{'-'*12}")
for feat in num_features:
    e_val = e_df[feat].mean()
    ne_val = ne_df[feat].mean()
    _, p = stats.ttest_ind(
        e_df[feat].sample(min(5000, len(e_df)), random_state=42),
        ne_df[feat].sample(min(5000, len(ne_df)), random_state=42)
    )
    sig = "YES" if p < 0.05 else "NO"
    print(f"  {feat:25} | {e_val:>12,.0f} | {ne_val:>13,.0f} | {p:>10.4f} | {sig:>12}")

# ── CHART 4: Correlation ────────────────────────────────────
print("\n" + "─" * 70)
print("CHART 4: Correlation with Targets")
print("─" * 70)

corr_cols = [
    'age', 'monthly_salary', 'years_of_employment',
    'monthly_rent', 'family_size', 'dependents',
    'school_fees', 'college_fees', 'travel_expenses',
    'groceries_utilities', 'other_monthly_expenses',
    'current_emi_amount', 'credit_score', 'bank_balance',
    'emergency_fund', 'requested_amount', 'requested_tenure',
    'emi_eligibility', 'max_monthly_emi'
]
corr_cols = [c for c in corr_cols if c in df.columns]
corr = df[corr_cols].corr()

print(f"  Top 5 correlated with emi_eligibility:")
elig_corr = corr['emi_eligibility'].drop('emi_eligibility').abs().sort_values(ascending=False)
for feat, val in elig_corr.head(5).items():
    direction = "+" if corr['emi_eligibility'][feat] > 0 else "-"
    print(f"    {feat:30}: {direction}{val:.3f}")

# ── CHART 5: Categorical ────────────────────────────────────
print("\n" + "─" * 70)
print("CHART 5: Categorical Features — Approval %")
print("─" * 70)

cat_features = ['gender', 'marital_status', 'education',
                'employment_type', 'house_type', 'company_type']
for col in cat_features:
    print(f"\n  {col.upper()}:")
    grp = df.groupby(col, observed=True)['emi_eligibility'].agg(['mean', 'count'])
    for idx, row in grp.iterrows():
        pct = row['mean'] * 100
        print(f"    {str(idx):20}: {pct:.1f}% ({row['count']:,})")

# ── CHART 6: Financial ──────────────────────────────────────
print("\n" + "─" * 70)
print("CHART 6: Financial Capacity")
print("─" * 70)

df['total_expenses'] = (
    df['monthly_rent'] + df['school_fees'] + df['college_fees'] +
    df['travel_expenses'] + df['groceries_utilities'] +
    df['other_monthly_expenses'] + df['current_emi_amount']
)
df['expense_ratio'] = (df['total_expenses'] / df['monthly_salary'].replace(0, np.nan)).clip(0, 2)

e_ratio = df.loc[df['emi_eligibility'] == 1, 'expense_ratio'].mean()
ne_ratio = df.loc[df['emi_eligibility'] == 0, 'expense_ratio'].mean()
print(f"  Expense Ratio:")
print(f"    Eligible     → {e_ratio:.2f} ({e_ratio*100:.0f}% of salary)")
print(f"    Not Eligible → {ne_ratio:.2f} ({ne_ratio*100:.0f}% of salary)")

df['salary_bracket'] = pd.cut(
    df['monthly_salary'],
    bins=[0, 20000, 40000, 60000, 80000, 100000, float('inf')],
    labels=['0-20k', '20-40k', '40-60k', '60-80k', '80-100k', '100k+']
)
bracket_app = df.groupby('salary_bracket', observed=True)['emi_eligibility'].agg(['mean', 'count'])
bracket_app['pct'] = bracket_app['mean'] * 100
print(f"\n  Salary Bracket Approval:")
for idx, row in bracket_app.iterrows():
    print(f"    {str(idx):12}: {row['pct']:.1f}% ({row['count']:,})")

# ── CHART 7: Distributions ──────────────────────────────────
print("\n" + "─" * 70)
print("CHART 7: Feature Distributions (Mean / Median / Skew)")
print("─" * 70)

dist_feats = [
    'age', 'monthly_salary', 'credit_score', 'bank_balance',
    'emergency_fund', 'years_of_employment', 'requested_amount',
    'requested_tenure', 'monthly_rent'
]
print(f"  {'Feature':25} | {'Mean':>12} | {'Median':>12} | {'Skew':>7}")
print(f"  {'-'*25}-+-{'-'*12}-+-{'-'*12}-+-{'-'*7}")
for feat in dist_feats:
    data = df[feat].dropna()
    print(f"  {feat:25} | {data.mean():>12,.0f} | {data.median():>12,.0f} | {data.skew():>7.2f}")

# ── CHART 8: Class Imbalance ────────────────────────────────
print("\n" + "─" * 70)
print("CHART 8: Class Imbalance")
print("─" * 70)
print(f"  Not Eligible (0): {ne_cnt:,} ({ne_cnt/total*100:.1f}%)")
print(f"  Eligible     (1): {e_cnt:,} ({e_cnt/total*100:.1f}%)")
print(f"  Imbalance Ratio:  {ne_cnt/e_cnt:.1f}:1")

df.drop(columns=['total_expenses', 'expense_ratio', 'salary_bracket'], errors='ignore', inplace=True)

print("\n" + "=" * 70)
print("   VERIFICATION COMPLETE")
print("=" * 70)
print("   Cross-check these numbers with your PNG charts")
print("   If numbers match → charts are correct ✅")
print("=" * 70)