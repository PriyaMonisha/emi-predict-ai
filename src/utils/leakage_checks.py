# filename: src/utils/leakage_checks.py
# purpose:  Leakage sanity checks for Section 5 model validation
# version:  1.0

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score


def shuffled_target_sanity_check(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    n_runs: int = 5,
    tolerance: float = 0.05,
    random_state: int = 42,
) -> dict:
    """Detect target leakage by training on randomly shuffled labels.

    Logic: if a model trained on shuffled (meaningless) targets still achieves
    high AUC on the real test set, the features contain the answer without
    needing the labels — which is target leakage.

    Expected result with clean data: AUC ≈ 0.50 ± tolerance.
    Uses Logistic Regression only (fast, no Optuna — models are NOT retrained).

    Args:
        X_train, y_train: full training feature matrix and labels.
        X_test, y_test:   held-out test set (never used for shuffling).
        n_runs:            number of independent shuffles to average over.
        tolerance:         max acceptable deviation from 0.50 (default 0.05).
        random_state:      seed for reproducibility.

    Returns:
        dict with keys: mean_auc, std, all_runs, passed, verdict.
    """
    rng  = np.random.RandomState(random_state)
    aucs = []

    for _ in range(n_runs):
        y_shuffled = rng.permutation(y_train.values)
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                class_weight="balanced",
                max_iter=500,
                random_state=random_state,
                solver="lbfgs",
            )),
        ])
        pipe.fit(X_train, y_shuffled)
        aucs.append(roc_auc_score(y_test, pipe.predict_proba(X_test)[:, 1]))

    mean_auc = float(np.mean(aucs))
    passed   = abs(mean_auc - 0.5) <= tolerance

    return {
        "mean_shuffled_auc": round(mean_auc, 6),
        "std":               round(float(np.std(aucs)), 6),
        "all_runs":          [round(a, 6) for a in aucs],
        "n_runs":            n_runs,
        "expected":          0.5,
        "tolerance":         tolerance,
        "passed":            passed,
        "verdict": (
            "PASS — shuffled-label AUC near 0.50, no leakage indicator"
            if passed else
            "FAIL — shuffled-label AUC too high, possible target leakage"
        ),
    }


def train_test_overlap_check(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    subset_cols: list = None,
) -> dict:
    """Check for duplicate rows between train and test sets.

    Args:
        train_df, test_df: DataFrames to compare (before OHE/encoding).
        subset_cols:       columns to compare on; if None, uses all shared columns.

    Returns:
        dict with overlap count and verdict.
    """
    cols    = subset_cols or list(set(train_df.columns) & set(test_df.columns))
    overlap = pd.merge(train_df[cols], test_df[cols], on=cols, how="inner")
    n       = len(overlap)

    return {
        "overlap_rows":   n,
        "train_rows":     len(train_df),
        "test_rows":      len(test_df),
        "cols_compared":  len(cols),
        "passed":         n == 0,
        "verdict": (
            "PASS — no duplicate rows found between train and test"
            if n == 0 else
            f"FAIL — {n:,} duplicate rows found between train and test"
        ),
    }
