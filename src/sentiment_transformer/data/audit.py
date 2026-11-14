"""Dataset auditing: statistics, missing values, duplicates, and data integrity checks."""

from __future__ import annotations

import json
import logging
import unicodedata
from pathlib import Path
from typing import Any

import pandas as pd

from sentiment_transformer.config import AppConfig
from sentiment_transformer.constants import (
    COL_OVERALL_RATING,
    COL_REVIEW_ID,
    COL_REVIEW_TEXT,
    COL_REVIEW_TITLE,
)
from sentiment_transformer.data.loader import load_raw_dataset
from sentiment_transformer.data.validation import validate_raw_file
from sentiment_transformer.exceptions import DataAuditError
from sentiment_transformer.reproducibility import calculate_sha256

logger = logging.getLogger(__name__)


def _normalize_text_for_audit(text: str) -> str:
    """Normalize text minimally for audit duplicate detection."""
    if not isinstance(text, str):
        return ""
    normalized = unicodedata.normalize("NFKC", text).strip().lower()
    return " ".join(normalized.split())


def audit_dataset(
    file_path: str | Path,
    config: AppConfig,
    output_json_path: str | Path | None = None,
) -> dict[str, Any]:
    """Perform a comprehensive audit of the raw dataset and return structured audit statistics.

    Args:
        file_path: Path to the raw CSV dataset.
        config: Application configuration.
        output_json_path: Optional path to save audit statistics as JSON.

    Returns:
        Structured audit dictionary.

    Raises:
        DataAuditError: If audit fails.
    """
    path = Path(file_path)
    if not path.is_file():
        raise DataAuditError(f"Raw dataset file not found for audit at: {path}")

    logger.info("Auditing raw dataset file: %s", path)

    validation_info = validate_raw_file(path, config)
    sha256_hash = calculate_sha256(path)

    # Load canonical DataFrame
    df = load_raw_dataset(path, config)
    total_rows = len(df)

    # Missing values
    missing_title_count = int((df[COL_REVIEW_TITLE].str.strip() == "").sum())
    missing_text_count = int((df[COL_REVIEW_TEXT].str.strip() == "").sum())
    missing_rating_count = int(df[COL_OVERALL_RATING].isna().sum())

    # Rating distribution 1 to 5
    rating_series = df[COL_OVERALL_RATING].dropna().astype(int)
    valid_ratings_mask = rating_series.isin([1, 2, 3, 4, 5])
    rating_counts = rating_series[valid_ratings_mask].value_counts().to_dict()
    # Ensure all ratings 1-5 exist in dict
    rating_dist = {str(r): int(rating_counts.get(r, 0)) for r in range(1, 6)}
    invalid_ratings_count = int((~valid_ratings_mask).sum() + missing_rating_count)

    # Length statistics
    raw_title_lengths = df[COL_REVIEW_TITLE].str.len()
    title_lengths = raw_title_lengths[raw_title_lengths > 0]
    raw_text_lengths = df[COL_REVIEW_TEXT].str.len()
    text_lengths = raw_text_lengths[raw_text_lengths > 0]

    title_stats = {
        "count": int(title_lengths.count()),
        "mean_chars": float(title_lengths.mean()) if not title_lengths.empty else 0.0,
        "median_chars": float(title_lengths.median()) if not title_lengths.empty else 0.0,
        "min_chars": int(title_lengths.min()) if not title_lengths.empty else 0,
        "max_chars": int(title_lengths.max()) if not title_lengths.empty else 0,
    }

    text_stats = {
        "count": int(text_lengths.count()),
        "mean_chars": float(text_lengths.mean()) if not text_lengths.empty else 0.0,
        "median_chars": float(text_lengths.median()) if not text_lengths.empty else 0.0,
        "min_chars": int(text_lengths.min()) if not text_lengths.empty else 0,
        "max_chars": int(text_lengths.max()) if not text_lengths.empty else 0,
    }

    # Duplicates detection
    exact_duplicates_count = int(
        df.duplicated(subset=[COL_REVIEW_TITLE, COL_REVIEW_TEXT]).sum()
    )

    # Normalized duplicates detection
    normalized_keys = (
        df[COL_REVIEW_TITLE].apply(_normalize_text_for_audit)
        + " || "
        + df[COL_REVIEW_TEXT].apply(_normalize_text_for_audit)
    )
    normalized_duplicates_count = int(normalized_keys.duplicated().sum())

    # Identify sample invalid records (empty texts or NaN ratings)
    invalid_mask = (df[COL_REVIEW_TEXT].str.strip() == "") | df[COL_OVERALL_RATING].isna()
    invalid_examples: list[dict[str, Any]] = []
    if invalid_mask.any():
        sample_invalids = df[invalid_mask].head(5)
        for _, row in sample_invalids.iterrows():
            invalid_examples.append(
                {
                    "review_id": str(row[COL_REVIEW_ID]),
                    "has_empty_text": bool(str(row[COL_REVIEW_TEXT]).strip() == ""),
                    "rating_value": None
                    if pd.isna(row[COL_OVERALL_RATING])
                    else float(row[COL_OVERALL_RATING]),
                }
            )

    audit_summary: dict[str, Any] = {
        "file_name": path.name,
        "file_size_bytes": validation_info["size_bytes"],
        "sha256": sha256_hash,
        "encoding": validation_info["encoding"],
        "delimiter": validation_info["delimiter"],
        "total_rows": total_rows,
        "resolved_columns": validation_info["resolved_mapping"],
        "rating_distribution": rating_dist,
        "rating_3_count": rating_dist.get("3", 0),
        "invalid_ratings_count": invalid_ratings_count,
        "missing_titles_count": missing_title_count,
        "missing_titles_pct": float(round((missing_title_count / total_rows) * 100, 2))
        if total_rows > 0
        else 0.0,
        "missing_texts_count": missing_text_count,
        "missing_texts_pct": float(round((missing_text_count / total_rows) * 100, 2))
        if total_rows > 0
        else 0.0,
        "title_length_stats": title_stats,
        "text_length_stats": text_stats,
        "exact_duplicates_count": exact_duplicates_count,
        "normalized_duplicates_count": normalized_duplicates_count,
        "invalid_sample_count": len(invalid_examples),
        "invalid_samples": invalid_examples,
    }

    logger.info(
        "Dataset audit completed: %d total rows, %d exact duplicates.",
        total_rows,
        exact_duplicates_count,
    )

    if output_json_path is not None:
        out_p = Path(output_json_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(audit_summary, f, indent=2, ensure_ascii=False)
        logger.info("Saved audit JSON to: %s", out_p)

    return audit_summary
