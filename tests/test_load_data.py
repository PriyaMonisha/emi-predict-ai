# filename: tests/test_load_data.py
# purpose:  Unit tests for src/data/load_data.py
# version:  1.0

import os
import pandas as pd
import pytest

from src.data.load_data import EXPECTED_COLUMNS, load_data


# ── Helpers ───────────────────────────────────────────────────────────────────

def _write_csv(path: str, df: pd.DataFrame) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)


def _minimal_df(n: int = 10) -> pd.DataFrame:
    """Return a DataFrame with all expected columns and n rows."""
    row = {col: 1 for col in EXPECTED_COLUMNS}
    row.update({
        "gender": "Male", "marital_status": "Single",
        "education": "Graduate", "employment_type": "Private",
        "company_type": "Private", "house_type": "Own",
        "existing_loans": "No", "emi_scenario": "Personal Loan Emi",
        "emi_eligibility": "Eligible",
    })
    return pd.DataFrame([row] * n)


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_load_returns_dataframe(tmp_path):
    csv_path = str(tmp_path / "data.csv")
    _write_csv(csv_path, _minimal_df(1100))
    df = load_data(csv_path)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1100


def test_load_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_data("/nonexistent/path/data.csv")


def test_load_non_csv_extension(tmp_path):
    txt_path = str(tmp_path / "data.txt")
    _minimal_df(5).to_csv(txt_path, index=False)
    with pytest.raises(ValueError, match="Expected .csv"):
        load_data(txt_path)


def test_load_missing_columns_raises(tmp_path):
    csv_path = str(tmp_path / "partial.csv")
    pd.DataFrame({"age": [25, 30], "gender": ["Male", "Female"]}).to_csv(
        csv_path, index=False
    )
    with pytest.raises(ValueError, match="Missing expected columns"):
        load_data(csv_path, validate=True)


def test_load_strips_column_whitespace(tmp_path):
    csv_path = str(tmp_path / "spaces.csv")
    df = _minimal_df(1100)
    df.columns = [f" {c} " for c in df.columns]
    df.to_csv(csv_path, index=False)
    result = load_data(csv_path)
    assert all(c == c.strip() for c in result.columns)


def test_load_validate_false_skips_row_check(tmp_path):
    csv_path = str(tmp_path / "tiny.csv")
    _minimal_df(3).to_csv(csv_path, index=False)
    df = load_data(csv_path, validate=False)
    assert len(df) == 3
