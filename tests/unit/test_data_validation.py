"""Unit tests for raw dataset validation, encoding, separator, and header checks."""

from __future__ import annotations

from pathlib import Path

import pytest

from sentiment_transformer.config import AppConfig
from sentiment_transformer.data.validation import (
    detect_encoding_and_separator,
    find_candidate_column,
    validate_raw_file,
)
from sentiment_transformer.exceptions import DataValidationError


def test_detect_encoding_and_separator(sample_csv_path: Path) -> None:
    enc, sep = detect_encoding_and_separator(sample_csv_path)
    assert enc in ["utf-8", "utf-8-sig", "latin-1"]
    assert sep == ","


def test_find_candidate_column() -> None:
    cols = ["Review_Header", "Review_Body", "Stars", "id"]
    res_text = find_candidate_column(cols, ["review_body", "text", "texto"])
    assert res_text == "Review_Body"

    res_rating = find_candidate_column(cols, ["stars", "rating", "nota"])
    assert res_rating == "Stars"

    res_none = find_candidate_column(cols, ["non_existent_column"])
    assert res_none is None


def test_validate_raw_file_valid(sample_csv_path: Path) -> None:
    cfg = AppConfig()
    res = validate_raw_file(sample_csv_path, cfg)
    assert res["encoding"] is not None
    assert res["delimiter"] == ","
    assert res["resolved_mapping"]["text_col"] == "review_text"
    assert res["resolved_mapping"]["rating_col"] == "overall_rating"


def test_validate_raw_file_empty(temp_dir: Path) -> None:
    empty_f = temp_dir / "empty.csv"
    empty_f.write_text("", encoding="utf-8")
    cfg = AppConfig()
    with pytest.raises(DataValidationError, match="is empty"):
        validate_raw_file(empty_f, cfg)


def test_validate_raw_file_missing_columns(temp_dir: Path) -> None:
    bad_f = temp_dir / "bad_cols.csv"
    bad_f.write_text("col_a,col_b\n1,2\n", encoding="utf-8")
    cfg = AppConfig()
    with pytest.raises(
        DataValidationError, match="Required columns could not be determined"
    ):
        validate_raw_file(bad_f, cfg)
