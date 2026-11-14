"""Raw dataset loading and column standardization."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from sentiment_transformer.config import AppConfig
from sentiment_transformer.constants import (
    COL_OVERALL_RATING,
    COL_REVIEW_ID,
    COL_REVIEW_TITLE,
    COL_REVIEW_TEXT,
)
from sentiment_transformer.data.validation import validate_raw_file
from sentiment_transformer.exceptions import DataValidationError

logger = logging.getLogger(__name__)


def load_raw_dataset(file_path: str | Path, config: AppConfig) -> pd.DataFrame:
    """Load raw dataset CSV and standardize columns to canonical names.

    Args:
        file_path: Path to raw CSV file.
        config: Application configuration.

    Returns:
        DataFrame containing canonical columns: review_id, review_title, review_text, overall_rating.

    Raises:
        DataValidationError: If loading fails or data format is invalid.
    """
    validation_info = validate_raw_file(file_path, config)
    path = Path(file_path)
    encoding = validation_info["encoding"]
    delimiter = validation_info["delimiter"]
    mapping = validation_info["resolved_mapping"]

    logger.info("Reading raw dataset from %s (encoding=%s, sep='%s')...", path, encoding, delimiter)

    try:
        # Load with pandas
        df = pd.read_csv(
            path,
            encoding=encoding,
            sep=delimiter,
            dtype=str,
            low_memory=False,
            on_bad_lines="skip",
        )
    except Exception as e:
        raise DataValidationError(f"Error reading CSV file with pandas from {path}: {e}") from e

    logger.info("Loaded raw DataFrame with shape: %s", df.shape)

    # Standardize column names
    rename_dict = {}
    id_source = mapping["id_col"]
    title_source = mapping["title_col"]
    text_source = mapping["text_col"]
    rating_source = mapping["rating_col"]

    if id_source:
        rename_dict[id_source] = COL_REVIEW_ID
    if title_source:
        rename_dict[title_source] = COL_REVIEW_TITLE
    if text_source:
        rename_dict[text_source] = COL_REVIEW_TEXT
    if rating_source:
        rename_dict[rating_source] = COL_OVERALL_RATING

    # Select only the needed columns or rename
    standardized = df.rename(columns=rename_dict)

    # If ID was not in the original file, generate stable synthetic IDs
    if COL_REVIEW_ID not in standardized.columns:
        logger.info("Generating synthetic IDs as %s was not present in raw dataset.", COL_REVIEW_ID)
        standardized[COL_REVIEW_ID] = [f"review_{i:07d}" for i in range(len(standardized))]
    else:
        # Fill any empty IDs
        standardized[COL_REVIEW_ID] = standardized[COL_REVIEW_ID].fillna("").astype(str)
        empty_ids = standardized[COL_REVIEW_ID] == ""
        if empty_ids.any():
            standardized.loc[empty_ids, COL_REVIEW_ID] = [
                f"gen_id_{i:07d}" for i in range(empty_ids.sum())
            ]

    # If Title was not in the dataset, create empty string column
    if COL_REVIEW_TITLE not in standardized.columns:
        standardized[COL_REVIEW_TITLE] = ""
    else:
        standardized[COL_REVIEW_TITLE] = standardized[COL_REVIEW_TITLE].fillna("").astype(str)

    # Ensure Text and Rating columns exist
    if COL_REVIEW_TEXT not in standardized.columns or COL_OVERALL_RATING not in standardized.columns:
        raise DataValidationError("Mandatory text or rating column missing after column standardization.")

    standardized[COL_REVIEW_TEXT] = standardized[COL_REVIEW_TEXT].fillna("").astype(str)

    # Convert rating to integer/numeric where possible
    standardized[COL_OVERALL_RATING] = pd.to_numeric(
        standardized[COL_OVERALL_RATING], errors="coerce"
    )

    output_df = standardized[
        [COL_REVIEW_ID, COL_REVIEW_TITLE, COL_REVIEW_TEXT, COL_OVERALL_RATING]
    ].copy()

    logger.info("Canonical DataFrame prepared with %d rows.", len(output_df))
    return output_df
