# filename: tests/test_leakage_checks.py
# purpose:  Unit tests for src/utils/leakage_checks.py
# version:  1.0

import numpy as np
import pandas as pd
import pytest

from src.utils.leakage_checks import (
    shuffled_target_sanity_check,
    train_test_overlap_check,
)


def _make_numeric_split(n_train: int = 200, n_test: int = 50, seed: int = 42):
    """Return X_train, y_train, X_test, y_test with no leakage."""
    rng = np.random.RandomState(seed)
    X_train = pd.DataFrame({"a": rng.randn(n_train), "b": rng.randn(n_train)})
    y_train = pd.Series(rng.randint(0, 2, n_train))
    X_test  = pd.DataFrame({"a": rng.randn(n_test),  "b": rng.randn(n_test)})
    y_test  = pd.Series(rng.randint(0, 2, n_test))
    return X_train, y_train, X_test, y_test


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_shuffled_target_check_returns_pass_on_random_data():
    X_train, y_train, X_test, y_test = _make_numeric_split()
    result = shuffled_target_sanity_check(
        X_train, y_train, X_test, y_test, n_runs=3
    )
    assert result["passed"] is True
    assert abs(result["mean_shuffled_auc"] - 0.5) <= result["tolerance"]


def test_shuffled_target_check_result_has_required_keys():
    X_train, y_train, X_test, y_test = _make_numeric_split()
    result = shuffled_target_sanity_check(
        X_train, y_train, X_test, y_test, n_runs=2
    )
    required = {"mean_shuffled_auc", "std", "all_runs", "n_runs",
                "passed", "verdict"}
    assert required.issubset(result.keys())
    assert len(result["all_runs"]) == 2


def test_overlap_check_no_overlap():
    train = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    test  = pd.DataFrame({"a": [7, 8, 9], "b": [10, 11, 12]})
    result = train_test_overlap_check(train, test)
    assert result["passed"] is True
    assert result["overlap_rows"] == 0


def test_overlap_check_detects_duplicates():
    shared = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    train  = pd.concat([shared, pd.DataFrame({"a": [5], "b": [6]})],
                       ignore_index=True)
    test   = shared.copy()
    result = train_test_overlap_check(train, test)
    assert result["passed"] is False
    assert result["overlap_rows"] == 2
