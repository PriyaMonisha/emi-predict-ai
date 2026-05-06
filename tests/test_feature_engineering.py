# filename: tests/test_feature_engineering.py
# purpose:  Unit tests for src/features/feature_engineering.py
# version:  1.0

import numpy as np
import pandas as pd
import pytest

from src.features.feature_engineering import NEW_FEATURES, FeatureEngineer


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_fit_transform_adds_21_new_features(raw_df):
    fe = FeatureEngineer()
    result = fe.fit_transform(raw_df)
    new_cols = [c for c in result.columns if c not in raw_df.columns]
    assert len(new_cols) == 21


def test_all_new_feature_names_present(raw_df):
    fe = FeatureEngineer()
    result = fe.fit_transform(raw_df)
    for feat in NEW_FEATURES:
        assert feat in result.columns, f"Missing engineered feature: {feat}"


def test_transform_before_fit_raises(raw_df):
    fe = FeatureEngineer()
    with pytest.raises(ValueError, match="fit"):
        fe.transform(raw_df)


def test_new_feature_names_property(fitted_fe):
    names = fitted_fe.new_feature_names
    assert isinstance(names, list)
    assert len(names) == 21
    assert names == NEW_FEATURES


def test_expense_ratio_clipped_between_zero_and_two(raw_df):
    fe = FeatureEngineer()
    result = fe.fit_transform(raw_df)
    assert result["expense_ratio"].between(0, 2).all()


def test_credit_score_banding_values(raw_df):
    fe = FeatureEngineer()
    result = fe.fit_transform(raw_df)
    valid_bands = {"Poor", "Fair", "Good", "Excellent"}
    actual = set(result["credit_score_band"].astype(str).unique())
    assert actual.issubset(valid_bands)
