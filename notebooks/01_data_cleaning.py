# ============================================================
# EMI PREDICT AI — Data Cleaning Runner
# notebooks/01_data_cleaning.py
# ============================================================

import sys
import os

# ── Path setup ───────────────────────────────────────────────
# This finds project root automatically
# Works no matter where you run the script from
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)
sys.path.insert(0, PROJECT_ROOT)

print(f"Project root : {PROJECT_ROOT}")
print(f"Python path  : {sys.path[0]}")

import pandas as pd
from src.data.load_data  import load_data
from src.data.preprocess import preprocess_data

# ── Config ───────────────────────────────────────────────────
RAW_PATH       = os.path.join(
    PROJECT_ROOT, "data", "raw", "emi_prediction_dataset.csv"
)
PROCESSED_PATH = os.path.join(
    PROJECT_ROOT, "data", "processed", "emi_cleaned.csv"
)
REPORT_PATH    = os.path.join(
    PROJECT_ROOT, "data", "reports", "cleaning_report.txt"
)

print(f"RAW_PATH     : {RAW_PATH}")
print(f"Looking for  : {os.path.exists(RAW_PATH)}")


# ── STEP 1: Load ─────────────────────────────────────────────
print("\nSTEP 1: Loading raw data...")
print("-" * 55)

df_raw = load_data(RAW_PATH, validate=True)

print(f"Raw data shape : {df_raw.shape}")
print(f"Columns        : {list(df_raw.columns)}")


# ── STEP 2: Snapshot before cleaning ─────────────────────────
print("\nSTEP 2: Pre-cleaning snapshot...")
print("-" * 55)

pre_nulls      = df_raw.isnull().sum().sum()
pre_duplicates = df_raw.duplicated().sum()
pre_rows       = len(df_raw)

print(f"Rows            : {pre_rows:,}")
print(f"Total nulls     : {pre_nulls:,}")
print(f"Duplicates      : {pre_duplicates:,}")

print(f"\nNull breakdown:")
null_breakdown = df_raw.isnull().sum()
null_breakdown = null_breakdown[null_breakdown > 0]
if null_breakdown.empty:
    print("  ✅ No nulls found in raw data")
else:
    for col, count in null_breakdown.items():
        pct = (count / pre_rows) * 100
        print(f"  {col:30} : {count:>6,} ({pct:.1f}%)")


# ── STEP 3: Preprocess ───────────────────────────────────────
print("\nSTEP 3: Running preprocessing pipeline...")
print("-" * 55)

df_clean = preprocess_data(df_raw, is_training=True)


# ── STEP 4: Snapshot after cleaning ──────────────────────────
print("\nSTEP 4: Post-cleaning snapshot...")
print("-" * 55)

post_nulls      = df_clean.isnull().sum().sum()
post_duplicates = df_clean.duplicated().sum()
post_rows       = len(df_clean)

print(f"Rows            : {post_rows:,}")
print(f"Total nulls     : {post_nulls:,}")
print(f"Duplicates      : {post_duplicates:,}")
print(f"Columns now     : {df_clean.shape[1]}")

new_cols = [
    col for col in df_clean.columns
    if col not in df_raw.columns
]
if new_cols:
    print(f"\nNew columns added by preprocessing:")
    for col in new_cols:
        print(f"  ✅ {col}")


# ── STEP 5: Target variable check ────────────────────────────
print("\nSTEP 5: Target variable check...")
print("-" * 55)

print("Classification target (emi_eligibility):")
target_counts = df_clean['emi_eligibility'].value_counts()
total         = len(df_clean)

for val, count in target_counts.items():
    label = "Eligible" if val == 1 else "Not Eligible"
    pct   = (count / total) * 100
    bar   = "█" * int(pct / 2)
    print(f"  {label:15} ({val}): {count:>6,} "
          f"({pct:.1f}%) {bar}")

eligible_pct = (
    df_clean['emi_eligibility'].sum() / total * 100
)
if eligible_pct < 20 or eligible_pct > 80:
    print(f"\n  ⚠️  CLASS IMBALANCE DETECTED!")
    print(f"     Will handle with SMOTE or "
          f"class_weight in modeling")
else:
    print(f"\n  ✅ Class balance acceptable")

print(f"\nRegression target (max_monthly_emi):")
print(f"  Min    : {df_clean['max_monthly_emi'].min():>10,.2f}")
print(f"  Max    : {df_clean['max_monthly_emi'].max():>10,.2f}")
print(f"  Mean   : {df_clean['max_monthly_emi'].mean():>10,.2f}")
print(f"  Median : {df_clean['max_monthly_emi'].median():>10,.2f}")


# ── STEP 6: Save cleaned data ─────────────────────────────────
print("\nSTEP 6: Saving cleaned data...")
print("-" * 55)

os.makedirs(os.path.dirname(PROCESSED_PATH), exist_ok=True)
df_clean.to_csv(PROCESSED_PATH, index=False)
print(f"✅ Saved → {PROCESSED_PATH}")

saved_size = os.path.getsize(PROCESSED_PATH) / (1024 * 1024)
print(f"   File size : {saved_size:.1f} MB")
print(f"   Rows      : {post_rows:,}")
print(f"   Columns   : {df_clean.shape[1]}")


# ── STEP 7: Save cleaning report ─────────────────────────────
print("\nSTEP 7: Saving cleaning report...")
print("-" * 55)

os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)

report_lines = [
    "=" * 55,
    "EMI PREDICT AI — CLEANING REPORT",
    "=" * 55,
    f"Raw rows          : {pre_rows:,}",
    f"Cleaned rows      : {post_rows:,}",
    f"Rows removed      : {pre_rows - post_rows:,}",
    f"Columns (raw)     : {df_raw.shape[1]}",
    f"Columns (clean)   : {df_clean.shape[1]}",
    f"New flag columns  : {new_cols}",
    f"Nulls before      : {pre_nulls:,}",
    f"Nulls after       : {post_nulls:,}",
    f"Duplicates removed: {pre_duplicates:,}",
    f"Class balance     : {eligible_pct:.1f}% Eligible",
    "=" * 55,
    "MISSING VALUE STRATEGY:",
    "  monthly_rent    → 0 (own/family) or renter median",
    "  credit_score    → 300 (no credit history)",
    "  bank_balance    → salary bracket median",
    "  emergency_fund  → 0 (no safety net)",
    "  education       → 'Unknown' category",
    "=" * 55,
    "HIGH_RISK rows saved separately",
    "  → data/processed/unlabeled_for_prediction.csv",
    "  → Will get predictions after model training",
    "=" * 55,
]

with open(REPORT_PATH, 'w', encoding='utf-8') as f:
    f.write('\n'.join(report_lines))

print(f"✅ Report saved → {REPORT_PATH}")


# ── FINAL SUMMARY ─────────────────────────────────────────────
print("\n" + "=" * 55)
print("   ✅ CLEANING PIPELINE COMPLETE")
print("=" * 55)
print(f"""
  Raw data    : {pre_rows:,} rows × {df_raw.shape[1]} cols
  Clean data  : {post_rows:,} rows × {df_clean.shape[1]} cols
  Nulls       : {pre_nulls:,} → {post_nulls}
  Duplicates  : {pre_duplicates:,} → 0

  Saved to    : {PROCESSED_PATH}
  Report at   : {REPORT_PATH}

  NEXT STEP   : Section 1 — Problem Definition
""")