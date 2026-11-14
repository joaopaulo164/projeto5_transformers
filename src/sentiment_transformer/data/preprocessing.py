"""Preprocessing: binary sentiment labeling, rating 3 exclusion, and textual composition."""

from __future__ import annotations

import logging
import unicodedata
from typing import Any

import numpy as np
import pandas as pd

from sentiment_transformer.config import AppConfig
from sentiment_transformer.constants import (
    COL_COMBINED_TEXT,
    COL_LABEL,
    COL_OVERALL_RATING,
    COL_REVIEW_ID,
    COL_REVIEW_TEXT,
    COL_REVIEW_TITLE,
    LABEL_NEGATIVE,
    LABEL_POSITIVE,
    RATING_3_EXCLUSION_JUSTIFICATION,
)

logger = logging.getLogger(__name__)


def normalize_spaces(text: str) -> str:
    """Normalize Unicode (NFKC) and collapse extraneous whitespace."""
    if not isinstance(text, str):
        return ""
    text_nfkc = unicodedata.normalize("NFKC", text)
    # Strip and collapse internal multiple spaces
    return " ".join(text_nfkc.split())


def compose_input_text(title: str, text: str, sep_token: str = "[SEP]") -> str:
    """Combine review title and body text according to defined composition rules.

    Rules:
    1. Preserve original text contents (after space normalization).
    2. When both title and body exist: title + ' ' + sep_token + ' ' + body.
    3. When title is empty/missing: body.
    4. When body is empty: empty string (flagged for exclusion).

    Args:
        title: Review title string.
        text: Review body text string.
        sep_token: Separator token from the tokenizer.

    Returns:
        Combined textual representation.
    """
    clean_title = normalize_spaces(title)
    clean_text = normalize_spaces(text)

    if not clean_text:
        return ""

    if clean_title:
        return f"{clean_title} {sep_token} {clean_text}"
    return clean_text


def map_rating_to_label(rating: float | int | None, config: AppConfig) -> int | None:
    """Map numeric rating (1-5) to binary label or None if excluded.

    Args:
        rating: Rating value.
        config: Application configuration.

    Returns:
        0 (negative), 1 (positive), or None (excluded/invalid).
    """
    if rating is None or pd.isna(rating):
        return None

    try:
        int_rating = int(round(float(rating)))
    except (ValueError, TypeError):
        return None

    neg_ratings = config.dataset.label_mapping.get("negative_ratings", [1, 2])
    pos_ratings = config.dataset.label_mapping.get("positive_ratings", [4, 5])
    exc_ratings = config.dataset.label_mapping.get("excluded_ratings", [3])

    if int_rating in exc_ratings:
        return None
    if int_rating in neg_ratings:
        return LABEL_NEGATIVE
    if int_rating in pos_ratings:
        return LABEL_POSITIVE

    return None


def preprocess_dataset(
    df: pd.DataFrame,
    config: AppConfig,
    sep_token: str = "[SEP]",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Execute preprocessing: assign binary labels, exclude rating 3, compose combined text.

    Args:
        df: Canonical DataFrame with review_id, review_title, review_text, overall_rating.
        config: Application configuration.
        sep_token: Separator token string from tokenizer (default '[SEP]').

    Returns:
        Tuple of (preprocessed_df, preprocessing_stats_dict).
    """
    initial_count = len(df)
    logger.info("Starting dataset preprocessing on %d records...", initial_count)

    work_df = df.copy()

    # Step 1: Assign binary labels based on overall_rating
    work_df[COL_LABEL] = work_df[COL_OVERALL_RATING].apply(
        lambda r: map_rating_to_label(r, config)
    )

    # Count rating 3 exclusions
    rating_3_mask = work_df[COL_OVERALL_RATING] == 3
    rating_3_count = int(rating_3_mask.sum())

    # Filter out records where label is None (rating 3 or invalid rating)
    valid_label_mask = work_df[COL_LABEL].notna()
    work_df = work_df[valid_label_mask].copy()
    work_df[COL_LABEL] = work_df[COL_LABEL].astype(int)
    after_rating_filter_count = len(work_df)

    # Step 2: Compose combined text
    work_df[COL_COMBINED_TEXT] = work_df.apply(
        lambda row: compose_input_text(
            str(row.get(COL_REVIEW_TITLE, "")),
            str(row.get(COL_REVIEW_TEXT, "")),
            sep_token=sep_token,
        ),
        axis=1,
    )

    # Step 3: Filter records with missing/empty combined text
    has_text_mask = work_df[COL_COMBINED_TEXT].str.strip() != ""
    empty_text_count = int((~has_text_mask).sum())
    work_df = work_df[has_text_mask].copy()
    after_empty_filter_count = len(work_df)

    # Step 4: Statistics on titles and lengths
    has_title_mask = work_df[COL_REVIEW_TITLE].fillna("").str.strip() != ""
    with_title_count = int(has_title_mask.sum())
    without_title_count = int((~has_title_mask).sum())

    # Character lengths and estimated token counts (approx 1 token ~= 4 chars in PT)
    combined_lengths = work_df[COL_COMBINED_TEXT].str.len()
    estimated_tokens = work_df[COL_COMBINED_TEXT].apply(lambda s: len(s.split()))
    max_len_limit = config.transformer.max_length
    # Tokens exceeding max_len_limit
    truncated_estimate_count = int((estimated_tokens > max_len_limit).sum())
    truncated_ratio = (
        float(truncated_estimate_count / len(work_df)) if len(work_df) > 0 else 0.0
    )

    stats: dict[str, Any] = {
        "initial_records": initial_count,
        "rating_3_excluded": rating_3_count,
        "rating_3_exclusion_justification": RATING_3_EXCLUSION_JUSTIFICATION,
        "after_rating_filter": after_rating_filter_count,
        "empty_text_excluded": empty_text_count,
        "final_preprocessed_records": after_empty_filter_count,
        "with_title_count": with_title_count,
        "with_title_pct": float(round((with_title_count / len(work_df)) * 100, 2)) if len(work_df) > 0 else 0.0,
        "without_title_count": without_title_count,
        "without_title_pct": float(round((without_title_count / len(work_df)) * 100, 2)) if len(work_df) > 0 else 0.0,
        "label_distribution": {
            "negative_0": int((work_df[COL_LABEL] == LABEL_NEGATIVE).sum()),
            "positive_1": int((work_df[COL_LABEL] == LABEL_POSITIVE).sum()),
        },
        "text_char_length": {
            "mean": float(combined_lengths.mean()) if not combined_lengths.empty else 0.0,
            "median": float(combined_lengths.median()) if not combined_lengths.empty else 0.0,
            "min": int(combined_lengths.min()) if not combined_lengths.empty else 0,
            "max": int(combined_lengths.max()) if not combined_lengths.empty else 0,
        },
        "word_token_count": {
            "mean": float(estimated_tokens.mean()) if not estimated_tokens.empty else 0.0,
            "median": float(estimated_tokens.median()) if not estimated_tokens.empty else 0.0,
            "min": int(estimated_tokens.min()) if not estimated_tokens.empty else 0,
            "max": int(estimated_tokens.max()) if not estimated_tokens.empty else 0,
        },
        "truncated_at_max_length_estimate": {
            "max_length": max_len_limit,
            "truncated_count": truncated_estimate_count,
            "truncated_ratio": round(truncated_ratio, 4),
        },
    }

    logger.info(
        "Preprocessing complete: %d retained, %d rating-3 excluded, %d empty excluded. "
        "Negatives: %d, Positives: %d",
        after_empty_filter_count,
        rating_3_count,
        empty_text_count,
        stats["label_distribution"]["negative_0"],
        stats["label_distribution"]["positive_1"],
    )

    return work_df, stats
