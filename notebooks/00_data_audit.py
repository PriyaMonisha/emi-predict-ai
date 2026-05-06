# Audit Current Cleaned CSV

# ================================================================
# EMI PREDICT AI — Data Audit Script
# notebooks/00_data_audit.py
#
# PURPOSE:
# Deeply verify the cleaned dataset
# Find ALL hidden dirty values
# Before we trust this data for modeling
#
# RUN:
# python notebooks/00_data_audit.py
# ================================================================

import sys
import os

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)
sys.path.insert(0, PROJECT_ROOT)

import pandas as pd
import numpy as np

CLEAN_PATH = os.path.join(
    PROJECT_ROOT, 'data', 'processed', 'emi_cleaned.csv'
)

print("=" * 65)
print("   EMI PREDICT AI — DEEP DATA AUDIT")
print("=" * 65)

df = pd.read_csv(CLEAN_PATH)
print(f"\n✅ Loaded: {df.shape[0]:,} rows x {df.shape[1]} cols")

# ── AUDIT 1: Basic shape & dtypes ────────────────────────────
print("\n" + "─" * 65)
print("AUDIT 1: DTYPES")
print("─" * 65)
for col in df.columns:
    null_count = df[col].isnull().sum()
    unique     = df[col].nunique()
    dtype      = df[col].dtype
    null_flag  = "⚠️  HAS NULLS" if null_count > 0 else "✅"
    print(f"  {col:35} | {str(dtype):10} | "
          f"unique={unique:5} | {null_flag}")

# ── AUDIT 2: Categorical unique values ───────────────────────
print("\n" + "─" * 65)
print("AUDIT 2: CATEGORICAL COLUMN UNIQUE VALUES")
print("─" * 65)

cat_cols = [
    'gender', 'marital_status', 'education',
    'employment_type', 'company_type', 'house_type',
    'existing_loans', 'emi_scenario', 'emi_eligibility'
]

for col in cat_cols:
    if col not in df.columns:
        continue
    unique_vals = df[col].unique()
    counts      = df[col].value_counts()
    print(f"\n  {col.upper()} ({len(unique_vals)} unique):")
    for val in unique_vals:
        cnt  = counts.get(val, 0)
        pct  = cnt / len(df) * 100
        flag = ""
        # Flag suspicious values
        if str(val).lower() in [
            'nan', 'none', 'null', 'na',
            'n/a', '--', '', 'unknown'
        ]:
            flag = " ← ⚠️  SUSPICIOUS"
        if col == 'gender' and str(val) in ['M', 'F', 'm', 'f']:
            flag = " ← ❌ ABBREVIATION"
        print(f"    '{val}': {cnt:>7,} ({pct:5.1f}%){flag}")

# ── AUDIT 3: Numerical range check ───────────────────────────
print("\n" + "─" * 65)
print("AUDIT 3: NUMERICAL RANGE VALIDATION")
print("─" * 65)

num_checks = {
    'age'                 : (18, 80),
    'credit_score'        : (300, 900),
    'monthly_salary'      : (0, None),
    'family_size'         : (1, 20),
    'dependents'          : (0, 10),
    'years_of_employment' : (0, 45),
    'requested_tenure'    : (1, 360),
    'monthly_rent'        : (0, None),
    'bank_balance'        : (0, None),
    'emergency_fund'      : (0, None),
    'max_monthly_emi'     : (0, None),
    'requested_amount'    : (0, None),
}

all_clean = True
for col, (mn, mx) in num_checks.items():
    if col not in df.columns:
        continue

    col_data   = df[col].dropna()
    violations = 0

    if mn is not None:
        below = (col_data < mn).sum()
        if below > 0:
            print(f"  ❌ {col:30}: "
                  f"{below:,} values below {mn}")
            violations += below
            all_clean   = False

    if mx is not None:
        above = (col_data > mx).sum()
        if above > 0:
            print(f"  ❌ {col:30}: "
                  f"{above:,} values above {mx}")
            violations += above
            all_clean   = False

    if violations == 0:
        print(f"  ✅ {col:30}: range OK "
              f"[{col_data.min():.0f} - "
              f"{col_data.max():.0f}]")

# ── AUDIT 4: Check for hidden string nulls ───────────────────
print("\n" + "─" * 65)
print("AUDIT 4: HIDDEN STRING NULLS")
print("─" * 65)

suspicious_strings = [
    'nan', 'none', 'null', 'na', 'n/a',
    'NaN', 'None', 'Null', 'NA', 'N/A',
    'Nan', '--', '-', '?', ''
]

found_any = False
for col in df.columns:
    if df[col].dtype == 'object':
        for sus in suspicious_strings:
            count = (df[col].astype(str) == sus).sum()
            if count > 0:
                print(f"  ❌ {col:30}: "
                      f"'{sus}' appears {count:,} times")
                found_any = True

if not found_any:
    print("  ✅ No hidden string nulls found")

# ── AUDIT 5: Flag columns check ──────────────────────────────
print("\n" + "─" * 65)
print("AUDIT 5: FLAG COLUMNS VALIDATION")
print("─" * 65)

flag_cols = [
    'credit_score_missing',
    'bank_balance_missing',
    'emergency_fund_missing',
    'education_missing',
    'monthly_rent_missing'
]

for col in flag_cols:
    if col not in df.columns:
        print(f"  ❌ {col}: MISSING from dataset!")
        continue
    unique = df[col].unique()
    counts = df[col].value_counts()
    valid  = set(unique).issubset({0, 1})
    flag   = "✅" if valid else "❌"
    print(f"  {flag} {col:35}: "
          f"values={sorted(unique)} "
          f"| flagged={counts.get(1, 0):,}")

# ── AUDIT 6: Target variable check ───────────────────────────
print("\n" + "─" * 65)
print("AUDIT 6: TARGET VARIABLE CHECK")
print("─" * 65)

if 'emi_eligibility' in df.columns:
    unique_target = df['emi_eligibility'].unique()
    counts        = df['emi_eligibility'].value_counts()
    print(f"  Unique values: {sorted(unique_target)}")
    print(f"  Expected    : [0, 1]")

    if set(unique_target).issubset({0, 1, 0.0, 1.0}):
        print(f"  ✅ Target is clean binary")
    else:
        print(f"  ❌ Unexpected target values!")

    for val, cnt in counts.items():
        pct = cnt / len(df) * 100
        print(f"  {val}: {cnt:,} ({pct:.1f}%)")

# ── AUDIT 7: Duplicate check ─────────────────────────────────
print("\n" + "─" * 65)
print("AUDIT 7: DUPLICATE CHECK")
print("─" * 65)

dups = df.duplicated().sum()
print(f"  Exact duplicates: {dups:,}")
if dups > 0:
    print(f"  ❌ Duplicates found!")
else:
    print(f"  ✅ No duplicates")

# ── FINAL REPORT ──────────────────────────────────────────────
print("\n" + "=" * 65)
print("   AUDIT COMPLETE")
print("=" * 65)
print(f"""
  Shape      : {df.shape[0]:,} rows x {df.shape[1]} cols
  Total nulls: {df.isnull().sum().sum():,}
  Duplicates : {dups:,}

  Check issues above marked with ❌
  These need to be fixed in preprocess.py
  before modeling begins
""")