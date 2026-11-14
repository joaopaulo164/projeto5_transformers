"""Deduplication and conflict resolution before dataset splitting."""

from __future__ import annotations

import logging
import unicodedata
from typing import Any

import pandas as pd

from sentiment_transformer.constants import (
    COL_LABEL,
    COL_NORMALIZED_KEY,
    COL_REVIEW_TEXT,
    COL_REVIEW_TITLE,
)

logger = logging.getLogger(__name__)


def generate_normalized_key(title: str, text: str) -> str:
    """Generate a deterministic normalized representation used exclusively for duplicate detection.

    Steps:
    1. Unicode normalization (NFKC).
    2. Lowercase conversion.
    3. Strip and collapse repeated whitespace.
    4. Deterministic combination of normalized title and text.

    Args:
        title: Review title string.
        text: Review text string.

    Returns:
        Deterministic normalized string key.
    """
    clean_title = (
        unicodedata.normalize("NFKC", str(title) if title is not None else "")
        .strip()
        .lower()
    )
    clean_title = " ".join(clean_title.split())

    clean_text = (
        unicodedata.normalize("NFKC", str(text) if text is not None else "")
        .strip()
        .lower()
    )
    clean_text = " ".join(clean_text.split())

    return f"{clean_title} || {clean_text}"


def deduplicate_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Detect and remove duplicate records and conflicting label groups before data splitting.

    Rules:
    - Compute `normalized_duplicate_key`.
    - If identical normalized keys have conflicting labels (e.g. one marked 0 and one marked 1),
      remove the entire group to avoid ambiguous or noisy ground truth.
    - If identical normalized keys have identical labels, retain the first occurrence deterministically.

    Args:
        df: Preprocessed DataFrame.

    Returns:
        Tuple of (deduplicated_df, deduplication_stats_dict).
    """
    initial_count = len(df)
    logger.info("Starting deduplication on %d records...", initial_count)

    work_df = df.copy()

    # Create normalized duplicate key
    work_df[COL_NORMALIZED_KEY] = work_df.apply(
        lambda row: generate_normalized_key(
            str(row.get(COL_REVIEW_TITLE, "")),
            str(row.get(COL_REVIEW_TEXT, "")),
        ),
        axis=1,
    )

    # Detect label conflicts per normalized key
    # Group by normalized key and check distinct label count
    label_counts_per_key = work_df.groupby(COL_NORMALIZED_KEY)[COL_LABEL].nunique()
    conflicting_keys = set(label_counts_per_key[label_counts_per_key > 1].index)
    conflicting_mask = work_df[COL_NORMALIZED_KEY].isin(conflicting_keys)
    conflicting_records_count = int(conflicting_mask.sum())
    conflicting_groups_count = len(conflicting_keys)

    if conflicting_records_count > 0:
        logger.warning(
            "Found %d records across %d distinct text groups with conflicting sentiment labels. "
            "Excluding entire conflicting groups to prevent noise.",
            conflicting_records_count,
            conflicting_groups_count,
        )

    # Exclude conflicting groups entirely
    non_conflicting_df = work_df[~conflicting_mask].copy()

    # Now drop exact normalized duplicates (keeping first)
    deduped_df = non_conflicting_df.drop_duplicates(
        subset=[COL_NORMALIZED_KEY], keep="first"
    ).copy()

    final_count = len(deduped_df)
    total_duplicates_removed = initial_count - final_count

    stats: dict[str, Any] = {
        "initial_records": initial_count,
        "conflicting_groups_count": conflicting_groups_count,
        "conflicting_records_removed": conflicting_records_count,
        "exact_duplicates_removed": total_duplicates_removed - conflicting_records_count,
        "total_records_removed": total_duplicates_removed,
        "clean_records_retained": final_count,
        "retention_rate_pct": float(round((final_count / initial_count) * 100, 2)) if initial_count > 0 else 0.0,
    }

    logger.info(
        "Deduplication complete: %d clean records retained (%d total removed, %d conflicts removed).",
        final_count,
        total_duplicates_removed,
        conflicting_records_count,
    )

    return deduped_df, stats
