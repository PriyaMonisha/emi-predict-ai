# filename: src/features/feature_engineering.py
# purpose:  Feature engineering pipeline for EMI prediction (Section 4)
# version:  1.0

import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)

EXPENSE_COLS = [
    'monthly_rent', 'school_fees', 'college_fees',
    'travel_expenses', 'groceries_utilities',
    'other_monthly_expenses', 'current_emi_amount',
]

CREDIT_BINS   = [0, 599, 649, 749, 900]
CREDIT_LABELS = ['Poor', 'Fair', 'Good', 'Excellent']
CREDIT_ORD    = {'Poor': 0, 'Fair': 1, 'Good': 2, 'Excellent': 3}

# Canonical list — used by notebook and Section 5 to identify engineered cols
NEW_FEATURES = [
    # Group 1: Financial ratios
    'expense_ratio', 'emi_burden_ratio', 'requested_emi_monthly',
    'requested_emi_ratio', 'total_emi_ratio', 'disposable_income',
    'savings_ratio', 'emergency_months',
    # Group 2: Credit banding
    'credit_score_band', 'credit_score_ord',
    # Group 3: Loan capacity
    'loan_to_income_ratio', 'loan_to_balance_ratio',
    # Group 4: Interaction features
    'credit_x_income', 'employment_stability',
    # Group 5: Binary flags
    'has_emergency_fund', 'is_renter', 'high_credit', 'low_expense_burden',
    # Group 6: Log-transforms (help linear models on skewed features)
    'log_bank_balance', 'log_monthly_salary', 'log_requested_amount',
]


class FeatureEngineer:
    """
    Sklearn-compatible feature engineering pipeline for EMI prediction.

    fit()       — learns one stat from training data: median total_expenses
                  (fallback denominator for emergency_months when expenses=0).
    transform() — appends 21 new features. Originals are never removed.
    fit_transform() — convenience wrapper.

    Input:  cleaned DataFrame from preprocess.py (before categorical encoding).
    Output: same DataFrame with NEW_FEATURES columns appended.

    Usage in training:
        fe = FeatureEngineer()
        X_train_feat = fe.fit_transform(X_train)   # learns from train only
        X_test_feat  = fe.transform(X_test)         # applies same transform

    Usage in serving (Section 9):
        fe = joblib.load('models/feature_engineer.pkl')
        X_feat = fe.transform(X_incoming)
    """

    def __init__(self):
        self.is_fitted_        = False
        self._expense_fallback = None

    # ── Fit ──────────────────────────────────────────────────────────
    def fit(self, X: pd.DataFrame):
        if not isinstance(X, pd.DataFrame):
            raise ValueError("FeatureEngineer requires a pandas DataFrame.")
        cols      = [c for c in EXPENSE_COLS if c in X.columns]
        total_exp = X[cols].fillna(0).sum(axis=1)
        self._expense_fallback = float(total_exp.median())
        self.is_fitted_ = True
        logger.info(
            f"FeatureEngineer fitted — "
            f"expense_fallback = ₹{self._expense_fallback:,.0f}"
        )
        return self

    # ── Transform ────────────────────────────────────────────────────
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted_:
            raise ValueError("Call fit() before transform().")
        if not isinstance(X, pd.DataFrame):
            raise ValueError("FeatureEngineer requires a pandas DataFrame.")

        df = X.copy()

        # Intermediate values (reused across groups, not saved as features)
        expense_cols = [c for c in EXPENSE_COLS if c in df.columns]
        total_exp    = df[expense_cols].fillna(0).sum(axis=1)
        salary       = df['monthly_salary'].replace(0, np.nan)
        curr_emi     = df['current_emi_amount'].fillna(0)
        req_tenure   = df['requested_tenure'].replace(0, np.nan)

        # ── Group 1: Financial Ratios ─────────────────────────────────
        df['expense_ratio']         = (total_exp / salary).clip(0, 2).fillna(1.0)
        df['emi_burden_ratio']      = (curr_emi / salary).clip(0, 1).fillna(0.0)
        df['requested_emi_monthly'] = (df['requested_amount'] / req_tenure).fillna(0.0)
        df['requested_emi_ratio']   = (
            df['requested_emi_monthly'] / salary
        ).clip(0, 2).fillna(1.0)
        df['total_emi_ratio']       = (
            (curr_emi + df['requested_emi_monthly']) / salary
        ).clip(0, 2).fillna(1.0)
        df['disposable_income']     = df['monthly_salary'].fillna(0) - total_exp
        df['savings_ratio']         = (
            df['bank_balance'].fillna(0) / salary
        ).clip(0, 100).fillna(0.0)
        safe_exp = total_exp.replace(0, self._expense_fallback)
        df['emergency_months']      = (
            df['emergency_fund'].fillna(0) / safe_exp
        ).clip(0, 24).fillna(0.0)

        # ── Group 2: Credit Score Banding ────────────────────────────
        df['credit_score_band'] = pd.cut(
            df['credit_score'],
            bins=CREDIT_BINS,
            labels=CREDIT_LABELS,
            include_lowest=True,
        )
        df['credit_score_ord'] = (
            df['credit_score_band'].map(CREDIT_ORD).fillna(0).astype(int)
        )

        # ── Group 3: Loan Capacity ───────────────────────────────────
        df['loan_to_income_ratio']  = (
            df['requested_amount'] / (salary * 12)
        ).clip(0, 20).fillna(0.0)
        bank_bal = df['bank_balance'].replace(0, np.nan)
        df['loan_to_balance_ratio'] = (
            df['requested_amount'] / bank_bal
        ).clip(0, 100).fillna(50.0)

        # ── Group 4: Interaction Features ────────────────────────────
        df['credit_x_income'] = (
            df['credit_score'].fillna(0) * df['monthly_salary'].fillna(0) / 1e7
        )
        working_years = (df['age'].fillna(25) - 18).clip(lower=1)
        df['employment_stability'] = (
            df['years_of_employment'].fillna(0) / working_years
        ).clip(0, 1)

        # ── Group 5: Binary Flags ────────────────────────────────────
        df['has_emergency_fund'] = (df['emergency_fund'].fillna(0) > 0).astype(int)
        df['is_renter']          = (df['house_type'] == 'Rented').astype(int)
        df['high_credit']        = (df['credit_score'].fillna(0) >= 750).astype(int)
        df['low_expense_burden'] = (
            (total_exp / salary).clip(0, 2).fillna(1.0) <= 0.50
        ).astype(int)

        # ── Group 6: Log-transforms ──────────────────────────────────
        df['log_bank_balance']     = np.log1p(df['bank_balance'].fillna(0).clip(lower=0))
        df['log_monthly_salary']   = np.log1p(df['monthly_salary'].fillna(0).clip(lower=0))
        df['log_requested_amount'] = np.log1p(df['requested_amount'].fillna(0).clip(lower=0))

        logger.info(
            f"FeatureEngineer transform complete — "
            f"{df.shape[0]:,} rows | {X.shape[1]} → {df.shape[1]} features "
            f"(+{len(NEW_FEATURES)} new)"
        )
        return df

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.fit(X).transform(X)

    @property
    def new_feature_names(self) -> list:
        return NEW_FEATURES.copy()
