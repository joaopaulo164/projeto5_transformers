"""Unit tests for Portuguese negation heuristics and error analysis generation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from sentiment_transformer.evaluation.error_analysis import (
    build_error_analysis,
    detect_portuguese_negation,
)


def test_detect_portuguese_negation() -> None:
    assert detect_portuguese_negation("Não gostei do produto") is True
    assert detect_portuguese_negation("O aparelho nunca funcionou direito") is True
    assert detect_portuguese_negation("Produto sem qualidade e péssimo") is True
    assert detect_portuguese_negation("Excelente compra adorei muito") is False
    assert detect_portuguese_negation("") is False


def test_build_error_analysis(
    sample_processed_df: pd.DataFrame, temp_dir: Path
) -> None:
    preds_df = sample_processed_df.copy()
    preds_df["true_label"] = preds_df["label"]
    # Introduce some fake predictions (some correct, some errors)
    preds_df["predicted_label"] = [
        1 if i % 3 == 0 else preds_df["label"].iloc[i] for i in range(len(preds_df))
    ]
    preds_df["negative_probability"] = 0.4
    preds_df["positive_probability"] = 0.6
    preds_df["is_correct"] = (
        preds_df["true_label"] == preds_df["predicted_label"]
    ).astype(int)

    out_csv = temp_dir / "error_analysis.csv"
    error_df, stats = build_error_analysis(
        preds_df, output_csv_path=out_csv, min_samples_per_category=2
    )

    assert len(error_df) > 0
    assert out_csv.is_file()
    assert "contains_negation" in error_df.columns
    assert "error_type" in error_df.columns
    assert "manual_analysis_guidelines" in stats
