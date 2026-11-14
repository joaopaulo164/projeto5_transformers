"""Unit tests for sampling, stratified splitting, and partition isolation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from sentiment_transformer.config import AppConfig, SamplingConfig, SplitConfig
from sentiment_transformer.constants import (
    COL_COMBINED_TEXT,
    COL_LABEL,
    COL_NORMALIZED_KEY,
    COL_OVERALL_RATING,
    COL_REVIEW_ID,
    COL_REVIEW_TEXT,
    COL_REVIEW_TITLE,
    SPLIT_TEST,
    SPLIT_TRAIN,
    SPLIT_VAL,
)
from sentiment_transformer.data.splitting import (
    sample_balanced_dataset,
    split_and_save_dataset,
)


def test_sample_balanced_dataset() -> None:
    # 30 negatives, 70 positives
    records = []
    for i in range(100):
        lbl = 0 if i < 30 else 1
        records.append(
            {COL_REVIEW_ID: f"id_{i}", COL_LABEL: lbl, COL_COMBINED_TEXT: f"t_{i}"}
        )
    df = pd.DataFrame(records)

    cfg = AppConfig(
        sampling=SamplingConfig(
            enabled=True,
            total_samples=40,
            samples_per_class=20,
            balance_classes=True,
            seed=42,
        )
    )

    sampled = sample_balanced_dataset(df, cfg)
    assert len(sampled) == 40
    assert (sampled[COL_LABEL] == 0).sum() == 20
    assert (sampled[COL_LABEL] == 1).sum() == 20


def test_split_and_save_dataset_proportions_and_no_leakage(temp_dir: Path) -> None:
    records = []
    for i in range(100):
        lbl = 0 if i < 50 else 1
        records.append(
            {
                COL_REVIEW_ID: f"rev_{i:04d}",
                COL_REVIEW_TITLE: f"Titulo {i}",
                COL_REVIEW_TEXT: f"Texto da avaliacao {i}",
                COL_COMBINED_TEXT: f"Titulo {i} [SEP] Texto {i}",
                COL_NORMALIZED_KEY: f"titulo {i} || texto {i}",
                COL_OVERALL_RATING: 1 if lbl == 0 else 5,
                COL_LABEL: lbl,
            }
        )
    df = pd.DataFrame(records)

    cfg = AppConfig(
        sampling=SamplingConfig(enabled=False),
        split=SplitConfig(train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, seed=42),
    )

    splits, stats = split_and_save_dataset(df, cfg, output_dir=temp_dir)

    train_df = splits[SPLIT_TRAIN]
    val_df = splits[SPLIT_VAL]
    test_df = splits[SPLIT_TEST]

    # Proportions (approx 70/15/15)
    assert len(train_df) + len(val_df) + len(test_df) == 100
    assert 65 <= len(train_df) <= 75
    assert 10 <= len(val_df) <= 20
    assert len(test_df) == 15

    # Class balance preservation
    assert (train_df[COL_LABEL] == 0).sum() >= 30
    assert (train_df[COL_LABEL] == 1).sum() >= 30

    # Strict isolation / no leakage
    train_ids = set(train_df[COL_REVIEW_ID])
    val_ids = set(val_df[COL_REVIEW_ID])
    test_ids = set(test_df[COL_REVIEW_ID])

    assert len(train_ids & val_ids) == 0
    assert len(train_ids & test_ids) == 0
    assert len(val_ids & test_ids) == 0

    # Verify files created on disk
    assert (temp_dir / "train.csv").is_file()
    assert (temp_dir / "val.csv").is_file()
    assert (temp_dir / "test.csv").is_file()
    assert (temp_dir / "all_processed.csv").is_file()
    assert (temp_dir / "split_statistics.json").is_file()
