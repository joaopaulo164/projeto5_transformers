"""Unit tests for normalized duplicate detection and conflicting label resolution."""

from __future__ import annotations

import pandas as pd

from sentiment_transformer.constants import (
    COL_LABEL,
    COL_REVIEW_ID,
    COL_REVIEW_TEXT,
    COL_REVIEW_TITLE,
)
from sentiment_transformer.data.deduplication import (
    deduplicate_dataset,
    generate_normalized_key,
)


def test_generate_normalized_key() -> None:
    k1 = generate_normalized_key("Excelente Produto!  ", "Adorei   muito  ")
    k2 = generate_normalized_key("excelente produto! ", "adorei muito")
    assert k1 == k2
    assert "||" in k1


def test_deduplicate_dataset_exact_and_conflicts() -> None:
    df = pd.DataFrame(
        [
            # Unique record 1 (Positive)
            {
                COL_REVIEW_ID: "1",
                COL_REVIEW_TITLE: "A",
                COL_REVIEW_TEXT: "Texto um",
                COL_LABEL: 1,
            },
            # Unique record 2 (Negative)
            {
                COL_REVIEW_ID: "2",
                COL_REVIEW_TITLE: "B",
                COL_REVIEW_TEXT: "Texto dois",
                COL_LABEL: 0,
            },
            # Duplicate record of 1 with identical label
            {
                COL_REVIEW_ID: "3",
                COL_REVIEW_TITLE: "a",
                COL_REVIEW_TEXT: "texto um",
                COL_LABEL: 1,
            },
            # Conflicting pair: same text, different labels (0 and 1) -> must both be removed!
            {
                COL_REVIEW_ID: "4",
                COL_REVIEW_TITLE: "C",
                COL_REVIEW_TEXT: "Conflito texto",
                COL_LABEL: 1,
            },
            {
                COL_REVIEW_ID: "5",
                COL_REVIEW_TITLE: "c",
                COL_REVIEW_TEXT: "conflito texto",
                COL_LABEL: 0,
            },
        ]
    )

    dedup_df, stats = deduplicate_dataset(df)

    # Expected retained: record 1 (unique/deduped) and record 2 (unique).
    # Conflicting records (4 and 5) must be removed.
    assert len(dedup_df) == 2
    assert set(dedup_df[COL_REVIEW_ID]) == {"1", "2"}
    assert stats["conflicting_records_removed"] == 2
    assert stats["conflicting_groups_count"] == 1
    assert stats["clean_records_retained"] == 2
