"""Invariant tests verifying foundational scientific and data integrity requirements."""

from __future__ import annotations

from pathlib import Path

from sentiment_transformer.config import AppConfig
from sentiment_transformer.constants import (
    COL_COMBINED_TEXT,
    COL_LABEL,
    COL_NORMALIZED_KEY,
    COL_OVERALL_RATING,
    COL_REVIEW_ID,
    LABEL_NEGATIVE,
    LABEL_POSITIVE,
    REQUIRED_PROCESSED_COLUMNS,
    SPLIT_TEST,
    SPLIT_TRAIN,
    SPLIT_VAL,
)
from sentiment_transformer.pipeline.phases import run_data_phase


def test_invariants_on_data_partitions(test_config: AppConfig, temp_dir: Path) -> None:
    """Verify all mandatory scientific data invariants on generated partitions."""
    run_dir = temp_dir / "invariant_run"
    run_dir.mkdir(parents=True, exist_ok=True)

    data_res = run_data_phase(config=test_config, run_dir=run_dir)
    splits = data_res["splits"]

    train_df = splits[SPLIT_TRAIN]
    val_df = splits[SPLIT_VAL]
    test_df = splits[SPLIT_TEST]

    for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        # 1. Required columns present
        for col in REQUIRED_PROCESSED_COLUMNS:
            assert col in df.columns, f"Column '{col}' missing from partition '{name}'"

        # 2. No rating 3 in partitions
        assert (
            df[COL_OVERALL_RATING] == 3
        ).sum() == 0, f"Found rating 3 in partition '{name}'"

        # 3. Labels strictly restricted to 0 and 1
        unique_labels = set(df[COL_LABEL].unique())
        assert unique_labels.issubset(
            {LABEL_NEGATIVE, LABEL_POSITIVE}
        ), f"Invalid labels in '{name}': {unique_labels}"

        # 4. Both classes present
        assert (
            df[COL_LABEL] == LABEL_NEGATIVE
        ).sum() > 0, f"No negative samples in partition '{name}'"
        assert (
            df[COL_LABEL] == LABEL_POSITIVE
        ).sum() > 0, f"No positive samples in partition '{name}'"

        # 5. Non-empty text representation
        assert (
            df[COL_COMBINED_TEXT].str.strip() == ""
        ).sum() == 0, f"Found empty text in partition '{name}'"

    # 6. ID Isolation across partitions (No leakage)
    train_ids = set(train_df[COL_REVIEW_ID])
    val_ids = set(val_df[COL_REVIEW_ID])
    test_ids = set(test_df[COL_REVIEW_ID])

    assert len(train_ids & val_ids) == 0, "Overlap found between train and val IDs!"
    assert len(train_ids & test_ids) == 0, "Overlap found between train and test IDs!"
    assert len(val_ids & test_ids) == 0, "Overlap found between val and test IDs!"

    # 7. Normalized Duplicate Key Isolation across partitions (No leakage)
    train_keys = set(train_df[COL_NORMALIZED_KEY])
    val_keys = set(val_df[COL_NORMALIZED_KEY])
    test_keys = set(test_df[COL_NORMALIZED_KEY])

    assert (
        len(train_keys & val_keys) == 0
    ), "Text overlap found between train and val splits!"
    assert (
        len(train_keys & test_keys) == 0
    ), "Text overlap found between train and test splits!"
    assert (
        len(val_keys & test_keys) == 0
    ), "Text overlap found between val and test splits!"
