# ================================================================
# EMI PREDICT AI — Preprocessor (Final Production Version v4)
# src/data/preprocess.py
#
# FIXES IN THIS VERSION:
#   v1: Basic cleaning
#   v2: Smart null filling
#   v3: Business rules + dtype fixes
#   v4: Fixed categorical standardization order
#       - Flag creation BEFORE astype(str)
#       - Value mapping (M→Male, F→Female etc.)
#       - "Nan" string → np.nan → 'Unknown'
#       - Post-processing validation
# ================================================================

import os
import pandas as pd
import numpy as np
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s'
)
logger = logging.getLogger(__name__)


# ================================================================
# COLUMN DEFINITIONS
# ================================================================

NUMERICAL_COLS = [
    'age', 'monthly_salary', 'years_of_employment',
    'monthly_rent', 'family_size', 'dependents',
    'school_fees', 'college_fees', 'travel_expenses',
    'groceries_utilities', 'other_monthly_expenses',
    'current_emi_amount', 'credit_score', 'bank_balance',
    'emergency_fund', 'requested_amount', 'requested_tenure',
    'max_monthly_emi'
]

CATEGORICAL_COLS = [
    'gender', 'marital_status', 'education',
    'employment_type', 'company_type', 'house_type',
    'existing_loans', 'emi_scenario'
]

TARGET_CLASS = 'emi_eligibility'
TARGET_REG   = 'max_monthly_emi'

# Columns where NULL = zero expense
ZERO_FILL_COLS = [
    'school_fees', 'college_fees', 'travel_expenses',
    'other_monthly_expenses', 'current_emi_amount',
    'years_of_employment', 'dependents', 'emergency_fund'
]

# Columns to cap at 1st-99th percentile
CAP_COLS = [
    'monthly_salary', 'bank_balance', 'emergency_fund',
    'requested_amount', 'current_emi_amount',
    'school_fees', 'college_fees', 'travel_expenses',
    'groceries_utilities', 'other_monthly_expenses',
    'monthly_rent', 'max_monthly_emi'
]

# Business rule boundaries (domain knowledge)
BUSINESS_RULES = {
    'age'                 : (18,  80),
    'credit_score'        : (300, 900),
    'family_size'         : (1,   20),
    'dependents'          : (0,   10),
    'years_of_employment' : (0,   45),
    'requested_tenure'    : (1,  360),
}

# No credit history score
NO_CREDIT_SCORE = 300

# ── VALUE STANDARDIZATION MAPS ────────────────────────────────
# These handle ALL case variants and abbreviations
# found in raw data audit

VALUE_MAP = {
    'gender': {
        # Lowercase variants
        'male'        : 'Male',
        'female'      : 'Female',
        # Uppercase variants
        'MALE'        : 'Male',
        'FEMALE'      : 'Female',
        # Abbreviations
        'M'           : 'Male',
        'F'           : 'Female',
        'm'           : 'Male',
        'f'           : 'Female',
    },
    'employment_type': {
        # Self-employed variants
        'self-employed': 'Self-Employed',
        'Self-employed': 'Self-Employed',
        'SELF-EMPLOYED': 'Self-Employed',
        'self employed': 'Self-Employed',
        'Self Employed': 'Self-Employed',
    },
    'company_type': {
        # MNC stays as Mnc (title case standard)
        'MNC'         : 'Mnc',
        'mnc'         : 'Mnc',
        # Mid-size variants
        'mid-size'    : 'Mid-Size',
        'Mid-size'    : 'Mid-Size',
        'MID-SIZE'    : 'Mid-Size',
        'midsize'     : 'Mid-Size',
        'Mid Size'    : 'Mid-Size',
        # Large Indian variants
        'large indian': 'Large Indian',
        'LARGE INDIAN': 'Large Indian',
    },
    'emi_scenario': {
        # Standardize to consistent format
        'Personal Loan EMI'       : 'Personal Loan Emi',
        'Home Appliances EMI'     : 'Home Appliances Emi',
        'Education EMI'           : 'Education Emi',
        'Vehicle EMI'             : 'Vehicle Emi',
        'E-commerce Shopping EMI' : 'E-Commerce Shopping Emi',
        'E-Commerce Shopping EMI' : 'E-Commerce Shopping Emi',
        'e-commerce shopping emi' : 'E-Commerce Shopping Emi',
    },
    'existing_loans': {
        'yes': 'Yes',
        'no' : 'No',
        'YES': 'Yes',
        'NO' : 'No',
        'y'  : 'Yes',
        'n'  : 'No',
    },
    'marital_status': {
        'married': 'Married',
        'single' : 'Single',
        'MARRIED': 'Married',
        'SINGLE' : 'Single',
    },
    'house_type': {
        'rented': 'Rented',
        'own'   : 'Own',
        'family': 'Family',
        'RENTED': 'Rented',
        'OWN'   : 'Own',
        'FAMILY': 'Family',
    },
}

# Strings that look like null but aren't
# After astype(str) these appear from real NaN values
NULL_STRINGS = [
    'nan', 'Nan', 'NaN', 'NAN',
    'none', 'None', 'NONE',
    'null', 'Null', 'NULL',
    'na', 'Na', 'NA',
    'n/a', 'N/A', 'N/a',
    '', ' ', '--', '-', '?'
]

# Unlabeled data save path
UNLABELED_PATH = os.path.join(
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(
                os.path.abspath(__file__)
            )
        )
    ),
    'data', 'processed',
    'unlabeled_for_prediction.csv'
)


# ================================================================
# MAIN FUNCTION
# ================================================================

def preprocess_data(
    df: pd.DataFrame,
    is_training: bool = True,
    save_unlabeled: bool = True
) -> pd.DataFrame:
    """
    Final production-grade preprocessing pipeline v4.

    PIPELINE ORDER:
      1.  Strip column names
      2.  Remove exact duplicates
      3.  Record original nulls (TRUE nulls before any conversion)
      4.  Clean dirty number formatting (Rs, commas)
      5.  Fix numerical dtypes
      6.  Create missing flags (BEFORE string conversion)
      7.  Handle unlabeled target rows
      8.  Drop critical null rows
      9.  Smart missing value fill
      10. Standardize categoricals (AFTER flags & fill)
      11. Apply value maps (M→Male, F→Female etc.)
      12. Replace null strings → proper null → fill
      13. Business rule validation
      14. Convert final dtypes
      15. Cap outliers
      16. Encode target variable
      17. Final dedup & validation

    Args:
        df             : Raw input DataFrame
        is_training    : True = training mode
        save_unlabeled : Save High_Risk rows separately

    Returns:
        pd.DataFrame: Fully cleaned DataFrame
    """
    df           = df.copy()
    initial_rows = len(df)

    print("\n" + "=" * 60)
    print("   EMI PREDICT AI — PREPROCESSING PIPELINE v4")
    print("=" * 60)
    print(f"   Input : {initial_rows:,} rows x {df.shape[1]} cols")
    print(f"   Mode  : {'Training' if is_training else 'Inference'}")
    print("=" * 60)

    # Step 1
    print("\n[1/17] Stripping column names...")
    df = _strip_columns(df)
    print(f"       Done ✅")

    # Step 2
    print("\n[2/17] Removing exact duplicates...")
    df = _remove_duplicates(df)

    # Step 3
    print("\n[3/17] Recording TRUE null values...")
    original_nulls = _get_null_map(df)
    if original_nulls:
        for col, cnt in original_nulls.items():
            pct = cnt / len(df) * 100
            print(f"       {col:30}: {cnt:>6,} ({pct:.1f}%)")
    else:
        print(f"       No nulls in raw data")

    # Step 4
    print("\n[4/17] Cleaning dirty number formats...")
    df = _clean_dirty_values(df)

    # Step 5
    print("\n[5/17] Converting numerical dtypes...")
    df = _fix_dtypes(df)
    _report_new_nans(df, original_nulls)

    # Step 6 — CRITICAL: BEFORE astype(str)!
    print("\n[6/17] Creating missing value flags...")
    df = _create_missing_flags(df, original_nulls)

    # Step 7
    print("\n[7/17] Handling unlabeled target rows...")
    if is_training:
        df = _handle_unlabeled_rows(
            df, save=save_unlabeled
        )

    # Step 8
    print("\n[8/17] Dropping critical null rows...")
    df = _drop_critical_nulls(df)

    # Step 9
    print("\n[9/17] Smart missing value fill...")
    df = _handle_missing_smart(df)

    # Step 10 — AFTER filling, THEN standardize
    print("\n[10/17] Standardizing categorical values...")
    df = _standardize_categoricals(df)

    # Step 11
    print("\n[11/17] Applying value maps...")
    df = _apply_value_maps(df)

    # Step 12
    print("\n[12/17] Replacing null strings...")
    df = _fix_null_strings(df)

    # Step 13
    print("\n[13/17] Business rule validation...")
    df = _apply_business_rules(df)

    # Step 14
    print("\n[14/17] Converting final dtypes...")
    df = _convert_final_dtypes(df)

    # Step 15
    print("\n[15/17] Capping outliers...")
    df = _cap_outliers(df)

    # Step 16
    print("\n[16/17] Encoding target variable...")
    if is_training and TARGET_CLASS in df.columns:
        df = _encode_target(df)

    # Step 17
    print("\n[17/17] Final cleanup & validation...")
    df = df.drop_duplicates()
    _validate_categoricals(df)
    _final_null_check(df)

    # Summary
    print("\n" + "=" * 60)
    print("   PREPROCESSING COMPLETE ✅")
    print("=" * 60)
    print(f"   Input rows     : {initial_rows:,}")
    print(f"   Output rows    : {len(df):,}")
    print(f"   Rows removed   : {initial_rows - len(df):,}")
    print(f"   Output cols    : {df.shape[1]}")
    print(f"   Nulls remaining: {df.isnull().sum().sum()}")
    print("=" * 60)

    return df


# ================================================================
# STEP FUNCTIONS
# ================================================================

def _strip_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Strip whitespace from all column names."""
    df.columns = df.columns.str.strip().str.lower()
    return df


def _remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove exact duplicate rows."""
    before  = len(df)
    df      = df.drop_duplicates()
    removed = before - len(df)
    if removed > 0:
        print(f"       Removed {removed:,} duplicates ✅")
    else:
        print(f"       No duplicates found ✅")
    return df


def _get_null_map(df: pd.DataFrame) -> dict:
    """Returns {column: null_count} for columns WITH nulls."""
    null_counts = df.isnull().sum()
    return {
        col: int(cnt)
        for col, cnt in null_counts.items()
        if cnt > 0
    }


def _clean_dirty_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove currency symbols, commas, spaces
    from numerical columns BEFORE type conversion.

    Handles: Rs, ₹, $, commas, spaces, %
    """
    dirty_chars = ['₹', 'Rs', 'rs', ',', '%', '$']
    cleaned     = []

    for col in NUMERICAL_COLS:
        if col not in df.columns:
            continue
        if df[col].dtype == 'object':
            series = df[col].astype(str)
            for char in dirty_chars:
                series = series.str.replace(
                    char, '', regex=False
                )
            series = series.str.strip()
            series = series.replace(NULL_STRINGS, np.nan)
            df[col] = series
            cleaned.append(col)

    if cleaned:
        print(f"       Cleaned: {cleaned}")
    else:
        print(f"       No dirty chars found ✅")
    return df


def _fix_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Convert numerical columns to float64."""
    for col in NUMERICAL_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col], errors='coerce'
            )
    print(f"       {len(NUMERICAL_COLS)} numerical "
          f"cols → float64 ✅")
    return df


def _report_new_nans(
    df: pd.DataFrame,
    original_nulls: dict
) -> None:
    """Report new NaNs created by type conversion."""
    post_nulls = _get_null_map(df)
    new_nans   = {
        col: post_nulls[col] - original_nulls.get(col, 0)
        for col in post_nulls
        if post_nulls[col] > original_nulls.get(col, 0)
    }
    if new_nans:
        print(f"\n       Dirty values found:")
        for col, cnt in new_nans.items():
            print(f"       {col:30}: "
                  f"{cnt:,} dirty values")
    else:
        print(f"       No dirty values found ✅")


def _create_missing_flags(
    df: pd.DataFrame,
    original_nulls: dict
) -> pd.DataFrame:
    """
    Create binary flag columns for originally null cols.

    MUST run BEFORE _standardize_categoricals()
    because astype(str) converts NaN → 'Nan' string
    which can no longer be detected as null!

    Flag = 1 means value was MISSING in original data.
    Extra signal for the model.
    """
    flag_map = {
        'credit_score'  : 'credit_score_missing',
        'bank_balance'  : 'bank_balance_missing',
        'emergency_fund': 'emergency_fund_missing',
        'education'     : 'education_missing',
        'monthly_rent'  : 'monthly_rent_missing',
    }

    created = []
    for col, flag_col in flag_map.items():
        if col in df.columns and col in original_nulls:
            df[flag_col] = df[col].isnull().astype(int)
            count        = df[flag_col].sum()
            created.append(f"{flag_col}={count:,}")

    if created:
        print(f"       Flags: {', '.join(created)} ✅")
    else:
        print(f"       No flag columns needed ✅")

    return df


def _handle_unlabeled_rows(
    df: pd.DataFrame,
    save: bool = True
) -> pd.DataFrame:
    """
    Save High_Risk rows separately then drop from training.

    RAW DATA HAS 3 TARGET VALUES:
    'Eligible'    → 1 (training)
    'Not_Eligible'→ 0 (training)
    'High_Risk'   → save separately (prediction later)

    High_Risk = bank flagged as extremely risky
    Cannot train on these (no clear label)
    But can predict their max EMI after training
    """
    if TARGET_CLASS not in df.columns:
        return df

    valid_targets = [
        'Eligible', 'Not_Eligible', 'Not Eligible',
        'Not_eligible', 'Not eligible'
    ]

    invalid_mask  = ~df[TARGET_CLASS].isin(valid_targets)
    invalid_count = invalid_mask.sum()

    if invalid_count == 0:
        print(f"       All target rows valid ✅")
        return df

    unlabeled_df  = df[invalid_mask].copy()
    invalid_vals  = (
        df.loc[invalid_mask, TARGET_CLASS]
        .value_counts()
        .to_dict()
    )
    print(f"       Found {invalid_count:,} "
          f"unlabeled rows: {invalid_vals}")

    if save:
        os.makedirs(
            os.path.dirname(UNLABELED_PATH),
            exist_ok=True
        )
        unlabeled_df.to_csv(UNLABELED_PATH, index=False)
        print(f"       Saved → unlabeled_for_prediction.csv")
        print(f"       These get predictions after training")

    df = df[~invalid_mask].copy()
    print(f"       Training rows remaining: {len(df):,}")
    return df


def _drop_critical_nulls(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Drop rows where CRITICAL fields are null.

    requested_amount → core loan field
    requested_tenure → core loan field
    Cannot fill these — would fabricate loan details
    """
    critical_cols = ['requested_amount', 'requested_tenure']
    before        = len(df)

    for col in critical_cols:
        if col not in df.columns:
            continue
        null_count = df[col].isnull().sum()
        if null_count > 0:
            df = df.dropna(subset=[col])
            print(f"       {col}: "
                  f"{null_count:,} nulls → DROPPED")
        else:
            print(f"       {col}: No nulls ✅")

    dropped = before - len(df)
    if dropped > 0:
        print(f"       Total dropped: {dropped:,}")
    return df


def _handle_missing_smart(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Business-logic aware missing value filling.
    Runs BEFORE categorical standardization
    so education NaN is still a real NaN here.
    """

    # monthly_rent
    if 'monthly_rent' in df.columns and \
       df['monthly_rent'].isnull().any():
        own_mask = df['house_type'].isin(
            ['Own', 'own', 'OWN', 'Family',
             'family', 'FAMILY']
        )
        df.loc[
            own_mask & df['monthly_rent'].isnull(),
            'monthly_rent'
        ] = 0

        rented_mask = df['house_type'].str.lower() == 'rented'
        if df.loc[rented_mask, 'monthly_rent'].isnull().any():
            renter_med = (
                df.loc[rented_mask, 'monthly_rent']
                .median()
            )
            df.loc[
                rented_mask & df['monthly_rent'].isnull(),
                'monthly_rent'
            ] = renter_med
            print(f"       monthly_rent → "
                  f"renter median: {renter_med:,.0f}")

        df['monthly_rent'] = df['monthly_rent'].fillna(0)
        print(f"       monthly_rent: done ✅")

    # credit_score
    if 'credit_score' in df.columns and \
       df['credit_score'].isnull().any():
        cnt = df['credit_score'].isnull().sum()
        df['credit_score'] = df['credit_score'].fillna(
            NO_CREDIT_SCORE
        )
        print(f"       credit_score ({cnt:,}) → "
              f"{NO_CREDIT_SCORE} ✅")

    # bank_balance → salary group median
    if 'bank_balance' in df.columns and \
       df['bank_balance'].isnull().any():
        cnt = df['bank_balance'].isnull().sum()
        df['_sal_bracket'] = pd.cut(
            df['monthly_salary'],
            bins=[0, 20000, 40000, 60000,
                  80000, 100000, float('inf')],
            labels=['0-20k', '20-40k', '40-60k',
                    '60-80k', '80-100k', '100k+']
        )
        df['bank_balance'] = (
            df.groupby('_sal_bracket', observed=True)
            ['bank_balance']
            .transform(lambda x: x.fillna(x.median()))
        )
        if df['bank_balance'].isnull().any():
            gm = df['bank_balance'].median()
            df['bank_balance'] = df['bank_balance'].fillna(gm)
        df = df.drop(columns=['_sal_bracket'])
        print(f"       bank_balance ({cnt:,}) → "
              f"salary bracket median ✅")

    # groceries_utilities → salary group median
    if 'groceries_utilities' in df.columns and \
       df['groceries_utilities'].isnull().any():
        cnt = df['groceries_utilities'].isnull().sum()
        df['_sal_bracket'] = pd.cut(
            df['monthly_salary'],
            bins=[0, 20000, 40000, 60000,
                  80000, 100000, float('inf')],
            labels=['0-20k', '20-40k', '40-60k',
                    '60-80k', '80-100k', '100k+']
        )
        df['groceries_utilities'] = (
            df.groupby('_sal_bracket', observed=True)
            ['groceries_utilities']
            .transform(lambda x: x.fillna(x.median()))
        )
        if df['groceries_utilities'].isnull().any():
            gm = df['groceries_utilities'].median()
            df['groceries_utilities'] = (
                df['groceries_utilities'].fillna(gm)
            )
        df = df.drop(columns=['_sal_bracket'])
        print(f"       groceries_utilities ({cnt:,}) → "
              f"salary bracket median ✅")

    # emergency_fund → 0
    if 'emergency_fund' in df.columns and \
       df['emergency_fund'].isnull().any():
        cnt = df['emergency_fund'].isnull().sum()
        df['emergency_fund'] = df['emergency_fund'].fillna(0)
        print(f"       emergency_fund ({cnt:,}) → 0 ✅")

    # education → 'Unknown' (real NaN, not string yet!)
    if 'education' in df.columns and \
       df['education'].isnull().any():
        cnt = df['education'].isnull().sum()
        df['education'] = df['education'].fillna('Unknown')
        print(f"       education ({cnt:,}) → 'Unknown' ✅")

    # Zero-fill expense columns
    for col in ZERO_FILL_COLS:
        if col in df.columns and \
           col != 'emergency_fund' and \
           df[col].isnull().any():
            cnt = df[col].isnull().sum()
            df[col] = df[col].fillna(0)
            print(f"       {col} ({cnt:,}) → 0 ✅")

    # family_size → median
    if 'family_size' in df.columns and \
       df['family_size'].isnull().any():
        cnt = df['family_size'].isnull().sum()
        med = df['family_size'].median()
        df['family_size'] = df['family_size'].fillna(med)
        print(f"       family_size ({cnt:,}) → {med:.0f} ✅")

    # age → median (dirty values)
    if 'age' in df.columns and df['age'].isnull().any():
        cnt = df['age'].isnull().sum()
        med = df['age'].median()
        df['age'] = df['age'].fillna(med)
        print(f"       age ({cnt:,} dirty) → {med:.0f} ✅")

    # monthly_salary → median (dirty values)
    if 'monthly_salary' in df.columns and \
       df['monthly_salary'].isnull().any():
        cnt = df['monthly_salary'].isnull().sum()
        med = df['monthly_salary'].median()
        df['monthly_salary'] = df['monthly_salary'].fillna(med)
        print(f"       monthly_salary ({cnt:,} dirty) → "
              f"{med:,.0f} ✅")

    # Remaining numerics → median
    remaining_num = [
        col for col in NUMERICAL_COLS
        if col in df.columns
        and col != TARGET_REG
        and df[col].isnull().any()
    ]
    if remaining_num:
        for col in remaining_num:
            med      = df[col].median()
            df[col]  = df[col].fillna(med)
            print(f"       {col} → median: {med:.2f} ✅")

    return df


def _standardize_categoricals(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Standardize categorical columns to title case.

    RUNS AFTER _handle_missing_smart() so:
    - education NaN was already filled with 'Unknown'
    - No real NaN left to become 'Nan' string
    - astype(str) is now safe!
    """
    cols_to_standardize = CATEGORICAL_COLS + [TARGET_CLASS]

    for col in cols_to_standardize:
        if col not in df.columns:
            continue
        df[col] = (
            df[col]
            .astype(str)
            .str.strip()
        )

    print(f"       {len(cols_to_standardize)} "
          f"categorical cols standardized ✅")
    return df


def _apply_value_maps(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Apply value standardization maps.

    Fixes:
    - Gender: Male/Female/M/F/MALE/FEMALE → Male/Female
    - Employment: Self-employed → Self-Employed
    - Company: MNC → Mnc, Mid-size → Mid-Size
    - EMI Scenario: consistent title case
    - Existing loans: yes/no → Yes/No
    """
    for col, mapping in VALUE_MAP.items():
        if col not in df.columns:
            continue
        before_unique = df[col].nunique()
        df[col]       = df[col].replace(mapping)
        after_unique  = df[col].nunique()

        if before_unique != after_unique:
            print(f"       {col:20}: "
                  f"{before_unique} → {after_unique} "
                  f"unique values ✅")

    # Apply title case AFTER mapping
    # (mapping handles special cases like MNC)
    title_cols = [
        'marital_status', 'house_type',
        'education', 'existing_loans'
    ]
    for col in title_cols:
        if col in df.columns:
            df[col] = df[col].str.title()

    return df


def _fix_null_strings(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Replace any remaining null-like strings with
    proper values.

    After astype(str) + value maps, check if any
    'Nan', 'None', 'nan' strings still exist.
    These should have been handled already but
    this is a safety net.
    """
    fixed_any = False

    for col in CATEGORICAL_COLS:
        if col not in df.columns:
            continue

        # Check for null strings
        null_mask = df[col].isin(NULL_STRINGS)
        null_cnt  = null_mask.sum()

        if null_cnt > 0:
            if col == 'education':
                df.loc[null_mask, col] = 'Unknown'
            elif col == 'gender':
                # Shouldn't happen after value maps
                # but safety net
                df.loc[null_mask, col] = 'Unknown'
            else:
                mode_val = (
                    df.loc[~null_mask, col].mode()[0]
                    if (~null_mask).any()
                    else 'Unknown'
                )
                df.loc[null_mask, col] = mode_val

            print(f"       Fixed {null_cnt:,} null strings "
                  f"in '{col}' ✅")
            fixed_any = True

    if not fixed_any:
        print(f"       No null strings found ✅")

    return df


def _apply_business_rules(
    df: pd.DataFrame
) -> pd.DataFrame:
    """Apply domain knowledge constraints."""

    for col, (mn, mx) in BUSINESS_RULES.items():
        if col not in df.columns:
            continue
        violations = (
            (df[col] < mn) | (df[col] > mx)
        ).sum()

        if violations > 0:
            df[col] = df[col].clip(
                lower=mn, upper=mx
            )
            print(f"       {col:25}: "
                  f"{violations:,} violations → "
                  f"clipped [{mn}, {mx}]")
        else:
            print(f"       {col:25}: No violations ✅")

    # Expenses must be >= 0
    expense_cols = [
        'monthly_rent', 'school_fees', 'college_fees',
        'travel_expenses', 'groceries_utilities',
        'other_monthly_expenses', 'current_emi_amount',
        'emergency_fund', 'bank_balance'
    ]
    for col in expense_cols:
        if col in df.columns:
            neg = (df[col] < 0).sum()
            if neg > 0:
                df[col] = df[col].clip(lower=0)
                print(f"       {col}: "
                      f"{neg:,} negatives → 0")

    # Salary must be > 0
    if 'monthly_salary' in df.columns:
        zero_sal = (df['monthly_salary'] <= 0).sum()
        if zero_sal > 0:
            pos_med = df[
                df['monthly_salary'] > 0
            ]['monthly_salary'].median()
            df.loc[
                df['monthly_salary'] <= 0,
                'monthly_salary'
            ] = pos_med
            print(f"       monthly_salary: "
                  f"{zero_sal:,} zero/neg → {pos_med:,.0f}")

    return df


def _convert_final_dtypes(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Convert columns to their correct final dtypes.

    INT: age, family_size, dependents
    CATEGORY: all categorical string columns
    """
    # Integer columns (round first)
    int_cols = ['age', 'family_size', 'dependents']
    for col in int_cols:
        if col in df.columns:
            df[col] = df[col].round(0).astype(int)
            print(f"       {col:25}: float → int ✅")

    # Category dtype (memory optimization)
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            unique_cnt = df[col].nunique()
            if unique_cnt < 50:
                df[col] = df[col].astype('category')
                print(f"       {col:25}: "
                      f"object → category "
                      f"({unique_cnt} unique) ✅")

    return df


def _cap_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cap outliers at 1st and 99th percentile.
    Preserves all rows — financial extremes are real.
    """
    capped = []
    for col in CAP_COLS:
        if col not in df.columns:
            continue
        p01   = df[col].quantile(0.01)
        p99   = df[col].quantile(0.99)
        below = (df[col] < p01).sum()
        above = (df[col] > p99).sum()
        df[col] = df[col].clip(lower=p01, upper=p99)
        if below > 0 or above > 0:
            capped.append(
                f"{col}(↓{below:,}↑{above:,})"
            )

    if capped:
        print(f"       Capped: {len(capped)} columns ✅")
    else:
        print(f"       No significant outliers ✅")
    return df


def _encode_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode classification target:
    Eligible     → 1
    Not_Eligible → 0
    """
    if TARGET_CLASS not in df.columns:
        return df

    target_map = {
        'Eligible'    : 1,
        'Not_Eligible': 0,
        'Not Eligible': 0,
        'Not_eligible': 0,
        'Not eligible': 0,
    }

    df[TARGET_CLASS] = df[TARGET_CLASS].map(target_map)

    # Drop any unmapped rows
    unmapped = df[TARGET_CLASS].isnull().sum()
    if unmapped > 0:
        print(f"       WARNING: {unmapped:,} "
              f"unmapped → dropping")
        df = df.dropna(subset=[TARGET_CLASS])

    # Show distribution
    counts = df[TARGET_CLASS].value_counts()
    total  = len(df)
    for val, cnt in counts.items():
        label = 'Eligible' if val == 1 else 'Not Eligible'
        pct   = cnt / total * 100
        print(f"       {label:15} ({int(val)}): "
              f"{cnt:,} ({pct:.1f}%) ✅")

    return df


def _validate_categoricals(
    df: pd.DataFrame
) -> None:
    """
    Post-processing validation.
    Checks all categorical columns have
    only expected clean values.
    Logs warnings for any unexpected values.
    """
    expected = {
        'gender'         : {'Male', 'Female'},
        'marital_status' : {'Married', 'Single'},
        'house_type'     : {'Own', 'Rented', 'Family'},
        'existing_loans' : {'Yes', 'No'},
        'employment_type': {
            'Private', 'Government', 'Self-Employed'
        },
    }

    all_clean = True
    for col, valid_vals in expected.items():
        if col not in df.columns:
            continue

        actual_vals = set(df[col].astype(str).unique())
        unexpected  = actual_vals - valid_vals

        if unexpected:
            print(f"\n       ⚠️  {col}: "
                  f"unexpected values: {unexpected}")
            all_clean = False
        else:
            print(f"       {col:25}: "
                  f"values clean ✅")

    if all_clean:
        print(f"       All categorical values valid ✅")


def _final_null_check(df: pd.DataFrame) -> None:
    """Final validation — zero nulls expected."""
    total = df.isnull().sum().sum()
    if total == 0:
        print(f"       Zero nulls remaining ✅")
    else:
        null_cols = df.isnull().sum()
        null_cols = null_cols[null_cols > 0]
        print(f"\n       ⚠️  {total:,} nulls remain:")
        for col, cnt in null_cols.items():
            print(f"       {col}: {cnt:,}")
        logger.warning(
            f"Preprocessing done but "
            f"{total} nulls remain"
        )