"""Sampling, stratified splitting, and partition persistence."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

from sentiment_transformer.config import AppConfig
from sentiment_transformer.constants import (
    COL_COMBINED_TEXT,
    COL_LABEL,
    COL_NORMALIZED_KEY,
    COL_OVERALL_RATING,
    COL_REVIEW_ID,
    COL_REVIEW_TEXT,
    COL_REVIEW_TITLE,
    COL_SPLIT,
    LABEL_NEGATIVE,
    LABEL_POSITIVE,
    REQUIRED_PROCESSED_COLUMNS,
    SPLIT_TEST,
    SPLIT_TRAIN,
    SPLIT_VAL,
)
from sentiment_transformer.exceptions import DataValidationError
from sentiment_transformer.paths import ensure_dir, resolve_path
from sentiment_transformer.reproducibility import calculate_sha256

logger = logging.getLogger(__name__)


def sample_balanced_dataset(df: pd.DataFrame, config: AppConfig) -> pd.DataFrame:
    """Sample balanced subset of records if sampling is enabled in configuration.

    Args:
        df: Deduplicated DataFrame.
        config: Application configuration.

    Returns:
        Sampled DataFrame (or full DataFrame if sampling is disabled).
    """
    if not config.sampling.enabled:
        logger.info("Sampling is disabled; retaining all %d deduplicated records.", len(df))
        return df

    seed = config.sampling.seed
    samples_per_class = config.sampling.samples_per_class
    total_requested = config.sampling.total_samples

    neg_df = df[df[COL_LABEL] == LABEL_NEGATIVE]
    pos_df = df[df[COL_LABEL] == LABEL_POSITIVE]

    logger.info(
        "Available before sampling: %d negative (label 0), %d positive (label 1).",
        len(neg_df),
        len(pos_df),
    )

    if config.sampling.balance_classes:
        n_neg = min(len(neg_df), samples_per_class)
        n_pos = min(len(pos_df), samples_per_class)
        target_n = min(n_neg, n_pos)

        if target_n < samples_per_class:
            logger.warning(
                "Requested %d samples per class, but only %d available in minority class.",
                samples_per_class,
                target_n,
            )

        sampled_neg = neg_df.sample(n=target_n, random_state=seed)
        sampled_pos = pos_df.sample(n=target_n, random_state=seed)
        combined = pd.concat([sampled_neg, sampled_pos], ignore_index=True)
    else:
        n_total = min(len(df), total_requested)
        combined = df.sample(n=n_total, random_state=seed)

    # Shuffle sampled dataframe deterministically
    shuffled = combined.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    logger.info(
        "Sampled %d records (%d neg, %d pos).",
        len(shuffled),
        int((shuffled[COL_LABEL] == LABEL_NEGATIVE).sum()),
        int((shuffled[COL_LABEL] == LABEL_POSITIVE).sum()),
    )
    return shuffled


def split_and_save_dataset(
    df: pd.DataFrame,
    config: AppConfig,
    output_dir: str | Path | None = None,
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    """Perform stratified train/val/test splitting and save partitions to disk.

    Partitions:
    - Train: default 70%
    - Val: default 15%
    - Test: default 15%

    Args:
        df: Deduplicated and sampled DataFrame.
        config: Application configuration.
        output_dir: Target directory to save processed CSVs (defaults to config.dataset.processed_dir).

    Returns:
        Tuple of (dict_of_split_dataframes, split_statistics_dict).

    Raises:
        DataValidationError: If partitions fail invariant checks (e.g. data leakage, missing columns).
    """
    logger.info("Executing stratified train/val/test split on %d records...", len(df))

    sampled_df = sample_balanced_dataset(df, config)

    train_ratio = config.split.train_ratio
    val_ratio = config.split.val_ratio
    test_ratio = config.split.test_ratio
    seed = config.split.seed

    # 1. First split: separate test set (15%)
    # test_size = test_ratio
    train_val_df, test_df = train_test_split(
        sampled_df,
        test_size=test_ratio,
        random_state=seed,
        stratify=sampled_df[COL_LABEL],
    )

    # 2. Second split: separate train (70%) and val (15%) from remaining (85%)
    val_relative_ratio = val_ratio / (train_ratio + val_ratio)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_relative_ratio,
        random_state=seed,
        stratify=train_val_df[COL_LABEL],
    )

    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()

    train_df[COL_SPLIT] = SPLIT_TRAIN
    val_df[COL_SPLIT] = SPLIT_VAL
    test_df[COL_SPLIT] = SPLIT_TEST

    # Verify column invariants
    for name, split_df in [(SPLIT_TRAIN, train_df), (SPLIT_VAL, val_df), (SPLIT_TEST, test_df)]:
        missing_cols = [c for c in REQUIRED_PROCESSED_COLUMNS if c not in split_df.columns]
        if missing_cols:
            raise DataValidationError(f"Partition '{name}' is missing mandatory columns: {missing_cols}")

    # Check for leakage / overlap in IDs and normalized duplicate keys
    train_ids = set(train_df[COL_REVIEW_ID])
    val_ids = set(val_df[COL_REVIEW_ID])
    test_ids = set(test_df[COL_REVIEW_ID])

    train_keys = set(train_df[COL_NORMALIZED_KEY])
    val_keys = set(val_df[COL_NORMALIZED_KEY])
    test_keys = set(test_df[COL_NORMALIZED_KEY])

    overlap_ids = (train_ids & val_ids) | (train_ids & test_ids) | (val_ids & test_ids)
    if overlap_ids:
        raise DataValidationError(f"Data leakage detected! Overlapping review_ids between splits: {len(overlap_ids)}")

    overlap_keys = (train_keys & val_keys) | (train_keys & test_keys) | (val_keys & test_keys)
    if overlap_keys:
        raise DataValidationError(
            f"Data leakage detected! Overlapping normalized texts between splits: {len(overlap_keys)}"
        )

    # Combine all processed splits
    all_processed_df = pd.concat([train_df, val_df, test_df], ignore_index=True)

    # Save to disk
    proc_dir = ensure_dir(
        resolve_path(output_dir if output_dir is not None else config.dataset.processed_dir)
    )

    train_path = proc_dir / "train.csv"
    val_path = proc_dir / "val.csv"
    test_path = proc_dir / "test.csv"
    all_path = proc_dir / "all_processed.csv"

    train_df.to_csv(train_path, index=False, encoding="utf-8")
    val_df.to_csv(val_path, index=False, encoding="utf-8")
    test_df.to_csv(test_path, index=False, encoding="utf-8")
    all_processed_df.to_csv(all_path, index=False, encoding="utf-8")

    logger.info("Saved split CSV files to: %s", proc_dir)

    split_stats: dict[str, Any] = {
        "total_records": len(all_processed_df),
        "train": {
            "count": len(train_df),
            "ratio": round(len(train_df) / len(all_processed_df), 4),
            "negative_0": int((train_df[COL_LABEL] == LABEL_NEGATIVE).sum()),
            "positive_1": int((train_df[COL_LABEL] == LABEL_POSITIVE).sum()),
            "file_path": str(train_path),
            "sha256": calculate_sha256(train_path),
        },
        "val": {
            "count": len(val_df),
            "ratio": round(len(val_df) / len(all_processed_df), 4),
            "negative_0": int((val_df[COL_LABEL] == LABEL_NEGATIVE).sum()),
            "positive_1": int((val_df[COL_LABEL] == LABEL_POSITIVE).sum()),
            "file_path": str(val_path),
            "sha256": calculate_sha256(val_path),
        },
        "test": {
            "count": len(test_df),
            "ratio": round(len(test_df) / len(all_processed_df), 4),
            "negative_0": int((test_df[COL_LABEL] == LABEL_NEGATIVE).sum()),
            "positive_1": int((test_df[COL_LABEL] == LABEL_POSITIVE).sum()),
            "file_path": str(test_path),
            "sha256": calculate_sha256(test_path),
        },
        "all_processed": {
            "count": len(all_processed_df),
            "file_path": str(all_path),
            "sha256": calculate_sha256(all_path),
        },
    }

    # Save stats json in processed dir
    stats_json_path = proc_dir / "split_statistics.json"
    with open(stats_json_path, "w", encoding="utf-8") as f:
        json.dump(split_stats, f, indent=2, ensure_ascii=False)

    splits_dict = {
        SPLIT_TRAIN: train_df,
        SPLIT_VAL: val_df,
        SPLIT_TEST: test_df,
        "all": all_processed_df,
    }

    return splits_dict, split_stats
