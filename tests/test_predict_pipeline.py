# filename: tests/test_predict_pipeline.py
# purpose:  Unit tests for src/pipelines/predict_pipeline.py
# version:  1.0

import numpy as np
import pandas as pd
import pytest

from src.pipelines.predict_pipeline import (
    _ohe_and_align,
    load_batch_data,
    predict_from_preloaded,
    save_predictions,
)


# ── load_batch_data ───────────────────────────────────────────────────────────

def test_load_batch_data_returns_dataframe(tmp_path, raw_df):
    csv_path = str(tmp_path / "batch.csv")
    raw_df.to_csv(csv_path, index=False)
    result = load_batch_data(csv_path)
    assert isinstance(result, pd.DataFrame)
    assert len(result) == len(raw_df)


def test_load_batch_data_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_batch_data("/no/such/file.csv")


def test_load_batch_data_empty_csv(tmp_path):
    csv_path = str(tmp_path / "empty.csv")
    pd.DataFrame(columns=["a", "b"]).to_csv(csv_path, index=False)
    with pytest.raises(ValueError, match="empty"):
        load_batch_data(csv_path)


# ── save_predictions ──────────────────────────────────────────────────────────

def test_save_predictions_creates_parent_dirs(tmp_path):
    preds = pd.DataFrame({
        "clf_proba": [0.9], "clf_label": [1],
        "conf_zone": ["auto_approve"], "predicted_emi": [8000.0],
    })
    out_path = str(tmp_path / "run" / "2024-01-01" / "predictions.csv")
    returned = save_predictions(preds, out_path)
    assert returned == out_path
    assert pd.read_csv(out_path).shape == preds.shape


# ── _ohe_and_align ────────────────────────────────────────────────────────────

def test_ohe_align_reindexes_to_model_schema():
    """Columns not in model schema are dropped; missing ones filled with 0."""
    from unittest.mock import MagicMock
    df = pd.DataFrame({
        "age":    [30, 35],
        "gender": ["Male", "Female"],   # in CATEGORICAL_COLS → OHE'd
    })
    model = MagicMock(spec=["predict_proba"])
    model.feature_names_in_ = ["age", "gender_Male"]  # expect specific OHE col
    result = _ohe_and_align(df, model)
    assert list(result.columns) == ["age", "gender_Male"]
    assert result.shape[0] == 2


# ── predict_from_preloaded ────────────────────────────────────────────────────

def test_predict_from_preloaded_returns_required_columns(raw_df, mock_clf, mock_reg):
    preds = predict_from_preloaded(
        raw_df, clf=mock_clf, reg=mock_reg, use_cache=False
    )
    assert set(preds.columns) == {
        "clf_proba", "clf_label", "conf_zone", "predicted_emi"
    }
    assert len(preds) == len(raw_df)


def test_predict_from_preloaded_conf_zones_valid(raw_df, mock_clf, mock_reg):
    preds = predict_from_preloaded(
        raw_df, clf=mock_clf, reg=mock_reg, use_cache=False
    )
    valid_zones = {"auto_approve", "human_review", "auto_reject"}
    assert set(preds["conf_zone"].unique()).issubset(valid_zones)
