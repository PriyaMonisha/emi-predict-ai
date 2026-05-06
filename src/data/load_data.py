# ============================================================
# EMI PREDICT AI — Data Loader (Production Grade)
# src/data/load_data.py
# ============================================================

import pandas as pd
import os
import logging

# ── Logger setup ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s'
)
logger = logging.getLogger(__name__)


# ── Expected columns ─────────────────────────────────────────
# If ANY of these are missing → raise error immediately
# Don't let bad data silently flow into preprocessing

EXPECTED_COLUMNS = [
    'age', 'gender', 'marital_status', 'education',
    'monthly_salary', 'employment_type', 'years_of_employment',
    'company_type', 'house_type', 'monthly_rent',
    'family_size', 'dependents', 'school_fees',
    'college_fees', 'travel_expenses', 'groceries_utilities',
    'other_monthly_expenses', 'existing_loans',
    'current_emi_amount', 'credit_score', 'bank_balance',
    'emergency_fund', 'emi_scenario', 'requested_amount',
    'requested_tenure', 'emi_eligibility', 'max_monthly_emi'
]

# Minimum rows we expect
# If less → something wrong with file
MIN_EXPECTED_ROWS = 1000


# ── Main function ─────────────────────────────────────────────

def load_data(
    file_path: str,
    validate: bool = True
) -> pd.DataFrame:
    """
    Loads EMI dataset from CSV with production-grade checks.

    What this does:
        1. Checks file exists & is CSV
        2. Loads with correct encoding
        3. Strips column name whitespace immediately
        4. Validates expected columns exist
        5. Checks minimum row count
        6. Logs full load summary
        7. Returns raw DataFrame (NO cleaning here)

    Why no cleaning here?
        load_data = just LOAD, nothing else
        preprocess_data = CLEAN & TRANSFORM
        Single responsibility principle ✅

    Args:
        file_path : Path to CSV file
        validate  : Run column & row checks
                    Set False only for unit tests

    Returns:
        pd.DataFrame: Raw loaded dataset

    Raises:
        FileNotFoundError : File doesn't exist
        ValueError        : Wrong format / empty /
                            missing columns
    """

    # ── Step 1: File existence check ─────────────────────────
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        raise FileNotFoundError(
            f"❌ File not found: {file_path}\n"
            f"   Check path and try again."
        )

    # ── Step 2: Extension check ───────────────────────────────
    if not file_path.lower().endswith('.csv'):
        logger.error(f"Wrong file type: {file_path}")
        raise ValueError(
            f"❌ Expected .csv file\n"
            f"   Got: {file_path}"
        )

    # ── Step 3: File size check ───────────────────────────────
    file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
    logger.info(f"File size: {file_size_mb:.1f} MB")

    if file_size_mb == 0:
        raise ValueError(f"❌ File is empty (0 bytes): {file_path}")

    # ── Step 4: Load CSV ──────────────────────────────────────
    logger.info(f"Loading: {file_path}")

    try:
        df = pd.read_csv(
            file_path,
            encoding='utf-8',        # explicit encoding
            low_memory=False,        # needed for mixed types
                                     # in 404k row dataset
        )
    except UnicodeDecodeError:
        # Try fallback encoding if utf-8 fails
        logger.warning("UTF-8 failed, trying latin-1 encoding...")
        df = pd.read_csv(
            file_path,
            encoding='latin-1',
            low_memory=False,
        )

    # ── Step 5: Empty DataFrame check ────────────────────────
    if df.empty:
        raise ValueError(f"❌ File loaded but has no rows: {file_path}")

    # ── Step 6: Strip column whitespace ──────────────────────
    # Do this HERE so validation below works correctly
    df.columns = df.columns.str.strip().str.lower()

    # ── Step 7: Validate expected columns ────────────────────
    if validate:
        missing_cols = _check_expected_columns(df)
        if missing_cols:
            raise ValueError(
                f"❌ Missing expected columns: {missing_cols}\n"
                f"   Found columns: {list(df.columns)}"
            )

    # ── Step 8: Minimum row check ─────────────────────────────
    if validate and len(df) < MIN_EXPECTED_ROWS:
        raise ValueError(
            f"❌ Too few rows: {len(df):,}\n"
            f"   Expected at least: {MIN_EXPECTED_ROWS:,}"
        )

    # ── Step 9: Load summary ──────────────────────────────────
    _log_load_summary(df, file_path, file_size_mb)

    return df


# ── Helper functions ──────────────────────────────────────────

def _check_expected_columns(df: pd.DataFrame) -> list:
    """
    Check all expected columns are present.

    Returns list of MISSING columns.
    Empty list = all columns present ✅
    """
    df_cols_lower    = [c.lower() for c in df.columns]
    expected_lower   = [c.lower() for c in EXPECTED_COLUMNS]
    missing          = [
        col for col in expected_lower
        if col not in df_cols_lower
    ]

    if missing:
        logger.warning(f"Missing columns: {missing}")
    else:
        logger.info("✅ All expected columns present")

    return missing


def _log_load_summary(
    df: pd.DataFrame,
    file_path: str,
    file_size_mb: float
) -> None:
    """
    Print a clean load summary.
    Helps preprocess.py know what it's receiving.
    """

    # Count nulls per column for awareness
    null_counts = df.isnull().sum()
    cols_with_nulls = null_counts[null_counts > 0]

    print("\n" + "=" * 55)
    print("   📂 DATA LOAD SUMMARY")
    print("=" * 55)
    print(f"   File       : {os.path.basename(file_path)}")
    print(f"   Size       : {file_size_mb:.1f} MB")
    print(f"   Rows       : {df.shape[0]:,}")
    print(f"   Columns    : {df.shape[1]}")
    print(f"   Memory     : "
          f"{df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")

    print(f"\n   NULL VALUE SUMMARY:")
    if cols_with_nulls.empty:
        print(f"   ✅ No null values found")
    else:
        for col, count in cols_with_nulls.items():
            pct = (count / len(df)) * 100
            print(f"   ⚠️  {col:30} "
                  f"{count:>6,} nulls ({pct:.1f}%)")

    print(f"\n   DTYPES SUMMARY:")
    dtype_counts = df.dtypes.value_counts()
    for dtype, count in dtype_counts.items():
        print(f"   {str(dtype):15} : {count} columns")

    print("=" * 55 + "\n")

    logger.info(
        f"✅ Loaded: {df.shape[0]:,} rows × "
        f"{df.shape[1]} cols | "
        f"{file_size_mb:.1f} MB"
    )